"""检查自研 Qwen3 full forward/KV Cache 与 Transformers reference 的 logits 对齐。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.qwen3 import Qwen3Config, load_qwen3_from_pretrained  # noqa: E402
from minigpt.qwen3_inference import CachedQwen3ModelRunner  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 reference/cached logits 对齐门禁。")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--output", default="runs/qwen3_parity.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "allclose": bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def main() -> None:
    args = parse_args()
    model_dir = project_path(args.model_dir)
    runtime = RuntimeContext.create(args.device, args.precision)
    dtype = runtime.amp_dtype or torch.float32
    config = Qwen3Config.from_json(model_dir / "config.json")
    if config.vocab_size < 12:
        raise ValueError("parity 输入需要 vocab_size >= 12")
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

    model = load_qwen3_from_pretrained(model_dir, device=runtime.device, dtype=dtype)
    runner = CachedQwen3ModelRunner(model, runtime)
    runner.validate_generation([4, 2], 2)
    with torch.inference_mode():
        with runtime.autocast():
            custom_full = model(input_ids, attention_mask)
        cached_prefill = runner.prefill(input_ids, attention_mask)
        next_ids = torch.argmax(custom_full[torch.arange(2), last_positions], dim=-1, keepdim=True)
        cached_decode = runner.decode(next_ids, torch.ones(2, dtype=torch.bool, device=runtime.device))
        row0 = torch.cat((input_ids[0, :4], next_ids[0])).unsqueeze(0)
        row1 = torch.cat((input_ids[1, :2], next_ids[1])).unsqueeze(0)
        with runtime.autocast():
            custom_decode = torch.cat((model(row0)[:, -1], model(row1)[:, -1]))

    valid_custom_full = custom_full[attention_mask.bool()].float().cpu()
    custom_prefill_reference = custom_full[
        torch.arange(2, device=runtime.device), last_positions
    ].float().cpu()
    cached_prefill = cached_prefill.float().cpu()
    cached_decode = cached_decode.float().cpu()
    custom_decode = custom_decode.float().cpu()
    del runner, model, custom_full
    gc.collect()
    runtime.empty_cache()

    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    reference = HFQwen3ForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(runtime.device).eval()
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
    with torch.inference_mode(), runtime.autocast():
        reference_full = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
    reference_valid = reference_full[attention_mask.bool()].float().cpu()

    checks = {
        "custom_vs_transformers_full": error_stats(
            valid_custom_full,
            reference_valid,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "cached_vs_custom_prefill": error_stats(
            cached_prefill,
            custom_prefill_reference,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "cached_vs_custom_decode": error_stats(
            cached_decode,
            custom_decode,
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    passed = all(bool(check["allclose"]) for check in checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "model_dir": str(model_dir),
        "device": str(runtime.device),
        "precision": runtime.precision,
        "atol": args.atol,
        "rtol": args.rtol,
        "checks": checks,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise AssertionError("Qwen3 parity gate failed")


if __name__ == "__main__":
    main()
