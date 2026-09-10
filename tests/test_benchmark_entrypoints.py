"""Real offline CLI integration for v0.9; tiny values are never hardware evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.backends.profiling import summarize_profile
from minigpt.benchmark import generation_output_digest
from minigpt.benchmark_contract import validate_device_mapping
from minigpt.distributed import DistributedContext
from minigpt.inference import GenerationConfig
from minigpt.serving import RequestSpec
from minigpt.serving_layout import load_layout_manifest, summarize_serving_layout
from minigpt.workload import WorkloadTrace


def run(command: list[str], *, timeout: int = 180, success: bool = True) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if success and result.returncode:
        raise AssertionError(f"CLI failed: {command}\n{result.stdout}\n{result.stderr}")
    if not success:
        assert result.returncode != 0, command
    return result


def test_device_identity() -> None:
    context = DistributedContext.create("cpu", "fp32")
    assert validate_device_mapping(context, [0])["visible_accelerator_count"] == 0
    try:
        validate_device_mapping(context, [2])
    except ValueError:
        pass
    else:
        raise AssertionError("CPU ranks cannot impersonate accelerator IDs")

    class FakeRuntime:
        class device:
            type = "cuda"
            index = 0

        def visible_device_count(self):
            return 2

    context.runtime = FakeRuntime()
    context.world_size = 2
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5"}):
        assert validate_device_mapping(context, [3, 5])["verified"]
        try:
            validate_device_mapping(context, [0, 1])
        except ValueError:
            pass
        else:
            raise AssertionError("metadata cannot replace real device visibility")


def check_cli(root: Path) -> None:
    model = root / "model"
    run(["scripts/create_tiny_qwen3_fixture.py", "--output", str(model)])
    config = GenerationConfig(max_new_tokens=5, seed=2026)
    trace = WorkloadTrace(workload_id="v09-cli-correctness", workload_class="short_short",
                          requests=tuple(RequestSpec(request_id=f"请求-{index}", prompt="hello world",
                                                     config=config, arrival_time_ms=0.0) for index in range(2)),
                          metadata={"purpose": "correctness only"})
    workload = root / "workload.json"
    trace.save(workload)
    single_trace = WorkloadTrace(workload_id="single", workload_class="short_short", requests=(trace.requests[0],))
    single_workload = root / "single.json"
    single_trace.save(single_workload)
    common = ["--model-dir", str(model), "--device", "cpu", "--precision", "fp32", "--backend", "gloo",
              "--warmup", "1", "--repeats", "2", "--logical-device-ids", "0", "--physical-card-count", "0",
              "--chips-per-card", "0", "--interconnect-topology", "cpu_processes", "--hash-weights"]
    profile = ["--profile", "--profile-format", "portable", "--profile-skip-steps", "0",
               "--profile-warmup-steps", "0", "--profile-active-steps", "4"]
    reports = {}
    for label, source, mode in (("single", single_workload, "kv_cache"),
                                ("static", workload, "kv_cache"), ("recompute", workload, "recompute")):
        output = root / label / "report.json"
        run(["benchmarks/infer_qwen3_tp.py", *common, *profile, "--workload", str(source),
             "--decode-mode", mode, "--output", str(output)])
        report = json.loads(output.read_text(encoding="utf-8"))
        reports[label] = report
        assert report["evidence_class"] == "correctness_or_nonformal_model"
        memory = report["memory_measurements"]
        assert memory["supported"] is False and memory["max_rank_peak_allocated_bytes"] is None
        assert memory["per_rank"][0]["peak_allocated_bytes"] is None
        assert report["workload"]["source_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert report["distributed"]["device_mapping"]["kind"] == "cpu_process_ranks"
        assert len(report["runs"]) == 2
        for measured in report["runs"]:
            assert measured["output_sha256"] == generation_output_digest(measured["requests"])
            assert measured["requests"][0]["request_id"] == "请求-0"
            assert all(row["output_tokens"] == 5 and row["stop_reason"] == "length" for row in measured["requests"])
            assert measured["memory_snapshot"]["peak_allocated_bytes"] is None
        replay = report["profiling"]["replay"]
        assert report["profiling"]["measurement_excluded"] is True
        assert replay["output_sha256"] == generation_output_digest(replay["requests"])
        assert replay["output_sha256"] == report["runs"][0]["output_sha256"]
        summary = summarize_profile(output.parent / "profile_manifest.json")
        assert summary["complete"], summary["incomplete_reasons"]
        assert summary["aggregate_step_fractions"]["computing"] is None
        print(f"v0.9 {label} CLI capture and evidence passed", flush=True)
    for index in range(2):
        assert reports["static"]["runs"][0]["requests"][index]["generated_ids"] == reports["recompute"]["runs"][0]["requests"][index]["generated_ids"]

    output_dir = root / "continuous"
    run(["benchmarks/infer_qwen3_continuous_batching.py", *common, *profile,
         "--workload", str(workload), "--mode", "open_loop", "--deterministic-open-loop",
         "--tp-size", "1", "--max-slots", "2", "--max-seq-len", "64", "--max-queue-size", "4",
         "--layout-id", "tp1", "--output-dir", str(output_dir)])
    layout_id, loaded = load_layout_manifest(output_dir / "layout_manifest.json")
    layout = summarize_serving_layout(layout_id, loaded)
    assert layout["all_replica_reports_formal_candidates"] is False
    assert layout["summary"]["completed_requests_per_second"]["median"] > 0.0
    report = loaded[0][1]
    assert report["memory_measurements"]["max_rank_peak_allocated_bytes"] is None
    assert report["environment"]["device_type"] == "cpu"
    assert summarize_profile(output_dir / "profile_manifest.json")["complete"]
    static_rows = {row["request_id"]: row for row in reports["static"]["runs"][0]["requests"]}
    for row in report["runs"][0]["serving"]["requests"]:
        assert row["generated_ids"] == static_rows[row["request_id"]]["generated_ids"]
        assert row["stop_reason"] == static_rows[row["request_id"]]["stop_reason"]
    # Existing capture directories may not be mixed with a second run.
    run(["benchmarks/infer_qwen3_tp.py", *common, *profile, "--workload", str(workload),
         "--output", str(root / "static" / "report.json")], success=False)
    run(["benchmarks/infer_qwen3_tp.py", *common, "--workload", str(workload), "--max-new-tokens", "6",
         "--output", str(root / "conflicting.json")], success=False)

    if os.environ.get("MINIGPT_RUN_GLOO_TESTS") == "1":
        for kind in ("static", "continuous"):
            output = root / ("gloo_" + kind)
            distributed_common = copy.copy(common)
            distributed_common[distributed_common.index("--logical-device-ids") + 1] = "0,1"
            launcher = ["-m", "torch.distributed.run", "--standalone", "--nnodes", "1", "--nproc-per-node", "2"]
            if kind == "static":
                command = ["benchmarks/infer_qwen3_tp.py", *distributed_common, *profile, "--workload", str(workload),
                           "--output", str(output / "report.json")]
            else:
                command = ["benchmarks/infer_qwen3_continuous_batching.py", *distributed_common, *profile,
                           "--workload", str(workload), "--mode", "open_loop", "--deterministic-open-loop",
                           "--tp-size", "2", "--max-slots", "2", "--max-seq-len", "64", "--max-queue-size", "4",
                           "--layout-id", "tp2", "--output-dir", str(output)]
            run([*launcher, *command])
            assert summarize_profile(output / "profile_manifest.json")["complete"]


def main() -> None:
    test_device_identity()
    with tempfile.TemporaryDirectory() as temporary:
        check_cli(Path(temporary))
    print("v0.9 real benchmark entrypoint integration tests passed.")


if __name__ == "__main__":
    main()
