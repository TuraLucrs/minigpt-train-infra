"""用 tiny Qwen3 验证真实 Gloo/NCCL/HCCL 与 TP 数学路径。

该入口不会下载模型，也不会把 smoke 结果冒充正式性能证据。它在加载 32B 权重前验证：
rank/device 绑定、process group、分片 safetensors loader、AllReduce、AllGather、KV Cache
以及 rank 0 token broadcast。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.experiment import git_snapshot  # noqa: E402
from minigpt.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from minigpt.qwen3_tp import load_tp_qwen3_from_pretrained  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在真实 process group 上运行 tiny Qwen3 TP 正确性 smoke。"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--backend", choices=("gloo", "nccl", "hccl"), required=True)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=180)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _tiny_config_dict() -> dict[str, object]:
    """可被 TP=1/2/4/8 整除，并在 TP>KV heads 时覆盖 KV 复制。"""

    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float32",
        "initializer_range": 0.02,
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
    }


def _save_tiny_checkpoint(directory: Path) -> Qwen3ForCausalLM:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("TP hardware smoke 需要安装 safetensors") from exc

    raw = _tiny_config_dict()
    torch.manual_seed(2026)
    reference = Qwen3ForCausalLM(Qwen3Config.from_dict(raw)).eval()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = {
        name: tensor.detach().contiguous()
        for name, tensor in reference.state_dict().items()
    }
    save_file(state, directory / "model.safetensors")
    return reference


def _default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    if dtype == torch.float16:
        return 8e-3, 8e-3
    return 1e-4, 1e-4


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    denominator = max(float(expected.float().abs().max().item()), 1e-12)
    return max_abs, max_abs / denominator


def _comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, bool]:
    max_abs, max_relative_to_peak = _error_metrics(actual, expected)
    passed = bool(
        torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    )
    return max_abs, max_relative_to_peak, passed


def _run_model_checks(
    distributed: DistributedContext,
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, float | bool], list[int], float, int, int]:
    runtime = distributed.runtime
    dtype = runtime.amp_dtype or torch.float32
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "tiny-qwen3"
        reference = _save_tiny_checkpoint(model_dir)

        runtime.synchronize()
        load_started = time.perf_counter()
        tp_model = load_tp_qwen3_from_pretrained(
            model_dir,
            distributed,
            dtype=dtype,
        )
        reference = reference.to(device=runtime.device, dtype=dtype).eval()
        runtime.synchronize()
        load_seconds = time.perf_counter() - load_started

        input_ids = torch.tensor(
            [[4, 5, 6, 7], [8, 9, 0, 0]],
            dtype=torch.long,
            device=runtime.device,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]],
            dtype=torch.bool,
            device=runtime.device,
        )
        last_positions = torch.tensor([3, 1], device=runtime.device)

        with torch.inference_mode(), runtime.autocast():
            expected_full = reference(input_ids, attention_mask)
            actual_full = tp_model(input_ids, attention_mask)
            cache = tp_model.allocate_kv_cache(
                max_batch_size=2,
                max_seq_len=8,
                device=runtime.device,
                dtype=dtype,
            )
            actual_prefill = tp_model.prefill_with_cache(
                input_ids,
                attention_mask,
                cache,
                logit_positions=last_positions,
            )
            expected_prefill = expected_full[
                torch.arange(input_ids.shape[0], device=runtime.device),
                last_positions,
            ]

            next_ids = (
                torch.argmax(expected_prefill, dim=-1, keepdim=True)
                if distributed.is_primary
                else torch.zeros((2, 1), dtype=torch.long, device=runtime.device)
            )
            distributed.broadcast(next_ids, src=0)
            locally_expected_ids = torch.argmax(expected_prefill, dim=-1, keepdim=True)
            token_broadcast_passed = bool(torch.equal(next_ids, locally_expected_ids))

            actual_decode = tp_model.decode_with_cache(
                next_ids,
                torch.ones(2, dtype=torch.bool, device=runtime.device),
                cache,
            )[:, 0]
            expected_decode_rows = []
            for row in range(input_ids.shape[0]):
                history = input_ids[row][attention_mask[row]]
                extended = torch.cat((history, next_ids[row]))[None]
                expected_decode_rows.append(reference(extended)[:, -1])
            expected_decode = torch.cat(expected_decode_rows, dim=0)

        runtime.synchronize()
        full_abs, full_relative, full_passed = _comparison(
            actual_full, expected_full, atol=atol, rtol=rtol
        )
        prefill_abs, prefill_relative, prefill_passed = _comparison(
            actual_prefill,
            expected_prefill,
            atol=atol,
            rtol=rtol,
        )
        decode_abs, decode_relative, decode_passed = _comparison(
            actual_decode,
            expected_decode,
            atol=atol,
            rtol=rtol,
        )
        local_parameter_count = sum(
            parameter.numel() for parameter in tp_model.parameters()
        )
        full_parameter_count = sum(
            parameter.numel() for parameter in reference.parameters()
        )
        return (
            {
                "full_max_abs": full_abs,
                "full_max_relative_to_peak": full_relative,
                "full_passed": full_passed,
                "prefill_max_abs": prefill_abs,
                "prefill_max_relative_to_peak": prefill_relative,
                "prefill_passed": prefill_passed,
                "decode_max_abs": decode_abs,
                "decode_max_relative_to_peak": decode_relative,
                "decode_passed": decode_passed,
                "token_broadcast_passed": token_broadcast_passed,
            },
            [int(value) for value in next_ids.flatten().cpu().tolist()],
            load_seconds,
            local_parameter_count,
            full_parameter_count,
        )


def main() -> None:
    args = parse_args()
    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            args.device,
            args.precision,
            backend=args.backend,
            timeout_seconds=args.distributed_timeout_seconds,
        )
        if distributed.world_size not in {1, 2, 4, 8}:
            raise ValueError("tiny TP hardware smoke 只支持 world_size=1/2/4/8")
        dtype = distributed.runtime.amp_dtype or torch.float32
        default_atol, default_rtol = _default_tolerances(dtype)
        atol = default_atol if args.atol is None else args.atol
        rtol = default_rtol if args.rtol is None else args.rtol
        if atol < 0 or rtol < 0:
            raise ValueError("atol/rtol 不能小于 0")

        distributed.runtime.empty_cache()
        distributed.runtime.reset_peak_memory()
        checks, selected_ids, load_seconds, local_parameters, full_parameters = (
            _run_model_checks(distributed, atol=atol, rtol=rtol)
        )
        current_mb, peak_mb = distributed.runtime.memory_stats_mb()
        rank_values = distributed.all_gather_floats(
            [
                checks["full_max_abs"],
                checks["prefill_max_abs"],
                checks["decode_max_abs"],
                load_seconds,
                current_mb,
                peak_mb,
                float(checks["full_passed"]),
                float(checks["prefill_passed"]),
                float(checks["decode_passed"]),
                float(checks["token_broadcast_passed"]),
            ]
        )
        failures = [
            rank
            for rank, values in enumerate(rank_values)
            if any(value != 1.0 for value in values[6:10])
        ]
        if failures:
            raise AssertionError(
                f"TP hardware smoke 数值或 token 同步失败，ranks={failures}；"
                f"atol={atol}, rtol={rtol}"
            )
        if not distributed.is_primary:
            return

        git = git_snapshot(PROJECT_ROOT)
        real_accelerator_tp = (
            distributed.world_size >= 2
            and distributed.runtime.device.type in {"cuda", "npu"}
        )
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "tensor_parallel_hardware_smoke",
            "status": "passed",
            "evidence_class": (
                "real_accelerator_tp_correctness_smoke_not_performance_evidence"
                if real_accelerator_tp
                else "development_correctness_smoke"
            ),
            "environment": {
                "python": platform.python_version(),
                **distributed.runtime.backend_metadata(),
            },
            "distributed": distributed.metadata(),
            "model": {
                "config": asdict(Qwen3Config.from_dict(_tiny_config_dict())),
                "full_parameter_count": full_parameters,
                "local_parameter_count": local_parameters,
            },
            "tolerances": {"atol": atol, "rtol": rtol},
            "selected_token_ids": selected_ids,
            "per_rank": [
                {
                    "rank": rank,
                    "full_max_abs": values[0],
                    "prefill_max_abs": values[1],
                    "decode_max_abs": values[2],
                    "load_seconds": values[3],
                    "current_memory_mb": values[4],
                    "peak_memory_mb": values[5],
                    "full_passed": bool(values[6]),
                    "prefill_passed": bool(values[7]),
                    "decode_passed": bool(values[8]),
                    "token_broadcast_passed": bool(values[9]),
                }
                for rank, values in enumerate(rank_values)
            ],
            "provenance": {"git": git, "command": list(sys.argv)},
        }
        output = (
            project_path(args.output)
            if args.output is not None
            else PROJECT_ROOT
            / "runs"
            / f"tp_hardware_smoke_{distributed.world_size}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告：{output}")
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
