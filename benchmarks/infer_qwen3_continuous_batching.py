"""在一个 Qwen3 TP replica 上重放 v0.7 Continuous Batching workload。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.distributed import (  # noqa: E402
    DistributedContext,
    TensorParallelReplicaContext,
)
from minigpt.ascend_profiling import (  # noqa: E402
    AscendProfileProtocol,
    AscendStepProfiler,
    build_profile_manifest,
    parse_profile_ranks,
    write_profile_manifest,
)
from minigpt.experiment import build_model_directory_provenance  # noqa: E402
from minigpt.qwen3 import count_qwen3_parameters  # noqa: E402
from minigpt.qwen3_tp import load_tp_qwen3_slot_runner  # noqa: E402
from minigpt.serving import ContinuousBatchEngine  # noqa: E402
from minigpt.serving_benchmark import benchmark_trace_replay  # noqa: E402
from minigpt.replica import partition_workload_by_projected_load  # noqa: E402
from minigpt.workload import WorkloadTrace  # noqa: E402


QWEN3_32B_PARAMETERS = 32_762_123_264
WORKLOAD_CLASSES = frozenset(
    {"short_short", "long_prefill_short_decode", "mixed"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--mode", choices=("open_loop", "closed_loop"), required=True)
    parser.add_argument(
        "--closed-loop-clients",
        type=int,
        default=None,
        help="整个 layout 的 client 总数；按 replica 确定性拆分",
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--max-slots", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--ttft-slo-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-ms", type=float, default=None)
    parser.add_argument("--e2e-slo-ms", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--deterministic-open-loop",
        action="store_true",
        help="open_loop 真机模式：measured repeats 重放 warmup 记录的到达动作脚本，"
        "消除 wall-clock 抖动导致的批组成轮间差异（延迟仍按真实墙钟计量）",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--greedy-token-path",
        choices=("full_gather", "distributed_argmax"),
        default="full_gather",
        help=(
            "TP greedy token 选择路径；distributed_argmax 只交换每个 rank 的"
            "局部最大 score/token，含采样请求的 batch 自动退回完整 logits"
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "npu"), default="auto"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--backend", choices=("auto", "gloo", "nccl", "hccl"), default="auto"
    )
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--layout-id", required=True)
    parser.add_argument(
        "--logical-device-ids",
        required=True,
        help="整个 torchrun 使用的 logical device id，按 global rank 逗号分隔",
    )
    parser.add_argument("--physical-card-count", type=int, required=True)
    parser.add_argument("--chips-per-card", type=int, required=True)
    parser.add_argument("--interconnect-topology", required=True)
    parser.add_argument(
        "--cann-version",
        default=None,
        help="正式 Ascend 证据必填；填写环境快照中确认的 CANN 完整版本",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="在 measured repeats 之外增加一次 torch_npu Profiler replay",
    )
    parser.add_argument(
        "--profile-ranks",
        default="all",
        help="all 或逗号分隔的 global ranks；正式 v0.7.1 gate 必须为 all",
    )
    parser.add_argument("--profile-skip-steps", type=int, default=8)
    parser.add_argument("--profile-warmup-steps", type=int, default=2)
    parser.add_argument("--profile-active-steps", type=int, default=4)
    parser.add_argument(
        "--profile-aic-metrics",
        choices=("pipe_utilization", "memory", "arithmetic_utilization"),
        default="pipe_utilization",
    )
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_device_ids(value: str, world_size: int) -> list[int]:
    try:
        device_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("logical-device-ids 必须是逗号分隔整数") from exc
    if len(device_ids) != world_size:
        raise ValueError(
            f"logical-device-ids 数量 {len(device_ids)} 与 TP world_size "
            f"{world_size} 不一致"
        )
    if len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise ValueError("logical-device-ids 必须是互不重复的非负整数")
    return device_ids


def validate_layout(
    args: argparse.Namespace,
    global_world_size: int,
) -> list[int]:
    if args.physical_card_count <= 0 or args.chips_per_card <= 0:
        raise ValueError("physical-card-count/chips-per-card 必须大于 0")
    if args.tp_size <= 0 or global_world_size % args.tp_size != 0:
        raise ValueError("tp-size 必须为 global world_size 的正因数")
    device_ids = parse_device_ids(args.logical_device_ids, global_world_size)
    declared_total_devices = args.physical_card_count * args.chips_per_card
    if len(device_ids) != declared_total_devices:
        raise ValueError(
            "本次正式 layout 必须声明 physical-card-count × chips-per-card 个 devices"
        )
    return device_ids


def local_closed_loop_clients(
    global_clients: int | None,
    *,
    mode: str,
    replica_index: int,
    replica_count: int,
) -> int | None:
    if mode == "open_loop":
        if global_clients is not None:
            raise ValueError("open_loop 不接受 closed-loop-clients")
        return None
    if global_clients is None or global_clients < replica_count:
        raise ValueError("closed_loop client 总数必须不少于 replica 数")
    base, remainder = divmod(global_clients, replica_count)
    return base + (1 if replica_index < remainder else 0)


def _digest_words(value: str) -> list[int]:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("digest 必须是小写 SHA-256")
    return [int(value[index : index + 8], 16) for index in range(0, 64, 8)]


def validate_global_trace_consistency(
    trace: WorkloadTrace,
    source_file_sha256: str,
    distributed: DistributedContext,
) -> None:
    local = torch.tensor(
        _digest_words(trace.request_sha256)
        + _digest_words(source_file_sha256)
        + [len(trace.requests)],
        dtype=torch.long,
        device=distributed.runtime.device,
    )
    expected = local.clone() if distributed.is_primary else torch.zeros_like(local)
    distributed.broadcast(expected, src=0)
    mismatch = torch.tensor(
        [0 if torch.equal(local, expected) else 1],
        dtype=torch.int32,
        device=distributed.runtime.device,
    )
    distributed.all_reduce_sum(mismatch)
    if int(mismatch.item()) > 0:
        raise RuntimeError("global ranks 读取到的源 workload 不一致")


def validate_replica_output_consistency(
    output_sha256: str,
    distributed: TensorParallelReplicaContext,
) -> None:
    """精确比较完整 SHA-256；所有 TP ranks 先走完相同 collective。"""

    local = torch.tensor(
        _digest_words(output_sha256),
        dtype=torch.long,
        device=distributed.runtime.device,
    )
    expected = local.clone() if distributed.is_primary else torch.zeros_like(local)
    distributed.broadcast(expected, src=0)
    mismatch = torch.tensor(
        [0 if torch.equal(local, expected) else 1],
        dtype=torch.int32,
        device=distributed.runtime.device,
    )
    distributed.all_reduce_sum(mismatch)
    if int(mismatch.item()) > 0:
        raise AssertionError("不同 TP rank 的完整 trace 输出 SHA-256 不一致")


def evidence_class(
    report: dict[str, object],
    *,
    parameter_count: int,
    hash_weights: bool,
) -> str:
    git = report["provenance"]["git"]
    protocol = report["protocol"]
    environment = report["environment"]
    distributed = report["distributed"]
    workload = report["workload"]
    source_file_sha256 = str(workload.get("source_file_sha256", ""))
    formal_candidate = (
        parameter_count == QWEN3_32B_PARAMETERS
        and environment["device_type"] == "npu"
        and environment["precision"] == "bf16"
        and isinstance(environment.get("cann_version"), str)
        and bool(str(environment["cann_version"]).strip())
        and distributed["backend"] == "hccl"
        and distributed["global_world_size"] == 8
        and hash_weights
        and git["commit"] != "unknown"
        and git["dirty"] is False
        and int(protocol["warmup"]) >= 1
        and int(protocol["repeats"]) >= 3
        and workload.get("workload_class") in WORKLOAD_CLASSES
        and len(source_file_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in source_file_sha256
        )
        and all(
            protocol[field] is not None and float(protocol[field]) > 0.0
            for field in ("ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms")
        )
    )
    if formal_candidate:
        return "qwen3_32b_ascend_continuous_batching_candidate"
    if parameter_count == QWEN3_32B_PARAMETERS:
        return "qwen3_32b_continuous_batching_incomplete_evidence"
    return "correctness_or_nonformal_model"


def main() -> None:
    args = parse_args()
    global_distributed: DistributedContext | None = None
    try:
        global_distributed = DistributedContext.create(
            args.device,
            args.precision,
            backend=args.backend,
            timeout_seconds=args.distributed_timeout_seconds,
        )
        logical_device_ids = validate_layout(args, global_distributed.world_size)
        distributed = TensorParallelReplicaContext.create(
            global_distributed,
            tp_size=args.tp_size,
        )
        workload_path = project_path(args.workload)
        source_payload = workload_path.read_bytes()
        source_file_sha256 = hashlib.sha256(source_payload).hexdigest()
        source_trace = WorkloadTrace.load(workload_path)
        if source_trace.partition is not None:
            raise ValueError("真实 layout benchmark 必须输入未分片的源 workload")
        validate_global_trace_consistency(
            source_trace,
            source_file_sha256,
            global_distributed,
        )
        if distributed.replica_count == 1:
            trace = source_trace
            _partitions, routing_manifest = partition_workload_by_projected_load(
                source_trace,
                replica_count=1,
                max_slots_per_replica=args.max_slots,
            )
        else:
            partitions, routing_manifest = partition_workload_by_projected_load(
                source_trace,
                replica_count=distributed.replica_count,
                max_slots_per_replica=args.max_slots,
            )
            trace = partitions[distributed.replica_index]
        clients = local_closed_loop_clients(
            args.closed_loop_clients,
            mode=args.mode,
            replica_index=distributed.replica_index,
            replica_count=distributed.replica_count,
        )
        replica_device_ids = [
            logical_device_ids[rank] for rank in distributed.group_ranks
        ]
        model_dir = project_path(args.model_dir)
        output_dir = project_path(args.output_dir)
        profiling_session = None
        profile_protocol = None
        profile_ranks: tuple[int, ...] = ()
        profile_root = output_dir / "profiler"
        if args.profile:
            if global_distributed.runtime.device.type != "npu":
                raise ValueError("--profile 只支持真实 Ascend NPU")
            profile_ranks = parse_profile_ranks(
                args.profile_ranks,
                global_distributed.world_size,
            )
            profile_protocol = AscendProfileProtocol(
                skip_steps=args.profile_skip_steps,
                warmup_steps=args.profile_warmup_steps,
                active_steps=args.profile_active_steps,
                aic_metrics=args.profile_aic_metrics,
                profile_memory=args.profile_memory,
            )
            profile_protocol.validate()
            if profile_root.exists():
                raise FileExistsError(
                    f"profile 输出目录已经存在，拒绝混入旧数据：{profile_root}"
                )
            global_distributed.barrier()
            profiling_session = AscendStepProfiler(
                profile_root,
                global_rank=global_distributed.rank,
                logical_device_id=logical_device_ids[global_distributed.rank],
                selected=global_distributed.rank in profile_ranks,
                protocol=profile_protocol,
            )

        distributed.runtime.synchronize()
        distributed.runtime.reset_peak_memory()
        load_started = time.perf_counter()
        runner, tokenizer = load_tp_qwen3_slot_runner(
            model_dir,
            distributed,
            max_slots=args.max_slots,
            max_seq_len=args.max_seq_len,
            use_chat_template=args.chat_template,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
            greedy_token_path=args.greedy_token_path,
        )
        distributed.runtime.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        model_loaded_memory_mb, model_loaded_peak_mb = (
            distributed.runtime.memory_stats_mb()
        )
        encoded_prompt_lengths = [
            len(tokenizer.encode(request.prompt)) for request in trace.requests
        ]
        for request, prompt_length in zip(trace.requests, encoded_prompt_lengths):
            runner.validate_request(prompt_length, request.config.max_new_tokens)

        engine = ContinuousBatchEngine(
            runner,
            tokenizer,
            max_queue_size=args.max_queue_size,
            distributed=distributed,
        )
        report = benchmark_trace_replay(
            engine,
            trace,
            mode=args.mode,
            closed_loop_clients=clients,
            warmup=args.warmup,
            repeats=args.repeats,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            e2e_slo_ms=args.e2e_slo_ms,
            distributed=distributed,
            before_replay=distributed.global_barrier,
            after_replay=distributed.global_barrier,
            deterministic_open_loop=args.deterministic_open_loop,
            profiling_session=profiling_session,
        )

        local_peaks = [
            float(run["memory"]["peak_device_memory_mb"])
            for run in report["runs"]
        ]
        validate_replica_output_consistency(
            str(report["runs"][0]["output_sha256"]),
            distributed,
        )
        rank_values = distributed.all_gather_floats(
            [
                max(local_peaks),
                statistics.median(local_peaks),
                model_load_seconds,
                model_loaded_memory_mb,
                model_loaded_peak_mb,
            ]
        )

        model = runner.model
        full_parameter_count = count_qwen3_parameters(model.config)
        local_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        report["model"] = {
            "type": "TensorParallelQwen3ForCausalLM",
            "config": asdict(model.config),
            "full_parameter_count": full_parameter_count,
            "local_parameter_count": local_parameter_count,
        }
        report["protocol"]["greedy_token_path"] = args.greedy_token_path
        report["engine"]["token_selection"] = runner.token_selection_metadata()
        report["workload"]["encoded_prompt_lengths"] = encoded_prompt_lengths
        report["workload"]["source_file_sha256"] = source_file_sha256
        report["environment"]["cann_version"] = args.cann_version
        routing_payload = json.dumps(
            routing_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        report["workload"]["routing"] = {
            "router": "least_projected_load",
            "estimator": "prompt_unicode_codepoints_plus_max_new_tokens",
            "assignment_sha256": hashlib.sha256(routing_payload).hexdigest(),
            "assignments": routing_manifest,
        }
        report["distributed"] = {
            **distributed.metadata(),
            "layout_id": args.layout_id,
            "logical_device_ids": replica_device_ids,
            "global_logical_device_ids": logical_device_ids,
            "global_closed_loop_clients": args.closed_loop_clients,
            "replica_closed_loop_clients": clients,
            "physical_card_count": args.physical_card_count,
            "chips_per_card": args.chips_per_card,
            "interconnect_topology": args.interconnect_topology,
            "run_label": args.run_label,
            "rank_results_consistent": True,
            "per_rank": [
                {
                    "tp_rank": rank,
                    "global_rank": distributed.group_ranks[rank],
                    "logical_device_id": replica_device_ids[rank],
                    "max_measured_peak_mb": values[0],
                    "median_measured_peak_mb": values[1],
                    "model_load_seconds": values[2],
                    "model_loaded_memory_mb": values[3],
                    "model_loaded_peak_mb": values[4],
                }
                for rank, values in enumerate(rank_values)
            ],
        }
        if global_distributed.is_primary:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "source_workload.json").write_bytes(source_payload)
            provenance = build_model_directory_provenance(
                PROJECT_ROOT,
                model_dir,
                sys.argv,
                hash_weights=args.hash_weights,
            )
            (output_dir / "layout_provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        global_distributed.barrier()
        profile_manifest_artifact = None
        if args.profile:
            assert profile_protocol is not None
            if global_distributed.is_primary:
                provenance = json.loads(
                    (output_dir / "layout_provenance.json").read_text(
                        encoding="utf-8"
                    )
                )
                profile_manifest = build_profile_manifest(
                    profile_root,
                    layout_id=args.layout_id,
                    workload_class=source_trace.workload_class,
                    mode=args.mode,
                    source_workload_sha256=source_trace.request_sha256,
                    source_workload_file_sha256=source_file_sha256,
                    git_commit=str(provenance["git"]["commit"]),
                    selected_ranks=profile_ranks,
                    logical_device_ids=logical_device_ids,
                    protocol=profile_protocol,
                )
                profile_manifest_path = output_dir / "profile_manifest.json"
                write_profile_manifest(profile_manifest_path, profile_manifest)
            global_distributed.barrier()
            profile_manifest_path = output_dir / "profile_manifest.json"
            profile_manifest_payload = profile_manifest_path.read_bytes()
            profile_manifest_artifact = {
                "name": profile_manifest_path.name,
                "sha256": hashlib.sha256(profile_manifest_payload).hexdigest(),
                "size_bytes": len(profile_manifest_payload),
            }
            report["profiling"]["profile_manifest"] = dict(
                profile_manifest_artifact
            )
        if distributed.is_primary:
            report["provenance"] = json.loads(
                (output_dir / "layout_provenance.json").read_text(encoding="utf-8")
            )
            report["evidence_class"] = evidence_class(
                report,
                parameter_count=full_parameter_count,
                hash_weights=args.hash_weights,
            )
            output = output_dir / f"replica-{distributed.replica_index:02d}.json"
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            summary = report["summary"]
            print("=" * 80)
            print("Qwen3 Continuous Batching Benchmark（measured repeats 中位数）")
            print(
                f"layout / replica : {args.layout_id} / "
                f"{distributed.replica_index}"
            )
            print(f"TP size          : {distributed.world_size}")
            print(f"workload         : {trace.workload_id}")
            print(f"mode             : {args.mode}")
            print(
                "request/s        : "
                f"{summary['completed_requests_per_second']['median']:.3f}"
            )
            print(
                "goodput request/s: "
                f"{summary['goodput_requests_per_second']['median']:.3f}"
            )
            print(
                "output token/s   : "
                f"{summary['output_tokens_per_second']['median']:.3f}"
            )
            print(f"evidence class   : {report['evidence_class']}")
            print(f"report           : {output}")
        global_distributed.barrier()
        if global_distributed.is_primary:
            report_files = []
            for replica_index in range(distributed.replica_count):
                report_path = output_dir / f"replica-{replica_index:02d}.json"
                payload = report_path.read_bytes()
                report_files.append(
                    {
                        "replica_index": replica_index,
                        "name": report_path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                )
            manifest = {
                "schema_version": 1,
                "layout_id": args.layout_id,
                "tp_size": distributed.world_size,
                "replica_count": distributed.replica_count,
                "global_world_size": distributed.global_world_size,
                "global_logical_device_ids": logical_device_ids,
                "source_workload_sha256": source_trace.request_sha256,
                "source_workload_file_sha256": source_file_sha256,
                "workload_class": source_trace.workload_class,
                "source_workload": {
                    "name": "source_workload.json",
                    "sha256": source_file_sha256,
                    "size_bytes": len(source_payload),
                },
                "routing_assignment_sha256": hashlib.sha256(
                    routing_payload
                ).hexdigest(),
                "reports": report_files,
            }
            if profile_manifest_artifact is not None:
                manifest["profile"] = profile_manifest_artifact
            (output_dir / "layout_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        global_distributed.barrier()
    finally:
        if global_distributed is not None:
            global_distributed.close()


if __name__ == "__main__":
    main()
