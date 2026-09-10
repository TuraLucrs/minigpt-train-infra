"""Portable inference timing: real per-row delivery and isolated profiling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.benchmark import (  # noqa: E402
    benchmark_generation, benchmark_static_batch, generation_output_digest, timed_static_batch,
)
from minigpt.backends.profiling import (  # noqa: E402
    ProfileProtocol, build_profile_manifest, create_step_profiler,
    summarize_profile, write_profile_manifest,
)
from minigpt.inference import GenerationConfig  # noqa: E402
from test_kv_cache import build_engines  # noqa: E402


def check_real_generation_and_row_eos() -> None:
    recompute, cached, _model, tokenizer = build_engines()
    prompts = ["ab", "abcdef", "abcd"]
    for engine in (recompute, cached):
        for config in (
            GenerationConfig(max_new_tokens=4),
            GenerationConfig(max_new_tokens=4, strategy="sample", top_k=4, top_p=0.8, seed=19),
        ):
            reference = engine.generate_batch(prompts, config)
            measured = timed_static_batch(engine, prompts, config)
            assert [result.generated_ids for result in measured.results] == [result.generated_ids for result in reference]
            for row, request in enumerate(measured.requests):
                assert request["request_id"] == str(row)
                assert request["prompt_ids"] == tokenizer.encode(prompts[row])
                assert request["input_tokens"] == len(tokenizer.encode(prompts[row]))
                assert request["output_tokens"] == 4
                assert request["state"] == "finished" and request["stop_reason"] == "length"
                assert request["e2e_latency_ms"] >= request["ttft_ms"] > 0
                assert request["tpot_ms"] is not None
                assert abs(request["tpot_ms"] * 3 - sum(request["decode_step_ms"])) < 1e-6
            assert measured.metrics["e2e_latency_ms"] >= max(row["e2e_latency_ms"] for row in measured.requests)

    candidates = ["a", "ab", "abc", "abcd", "abcdef", "m", "mnop", "abcdefgh"]
    first = cached.generate_batch(candidates, GenerationConfig(max_new_tokens=1))
    pair = next(
        (i, j) for i in range(len(candidates)) for j in range(len(candidates))
        if len(candidates[i]) != len(candidates[j]) and first[i].generated_ids != first[j].generated_ids
    )
    prompts = [candidates[index] for index in pair]
    eos = first[pair[0]].generated_ids[0]
    config = GenerationConfig(max_new_tokens=4, eos_token_id=eos)
    reference = cached.generate_batch(prompts, config)
    records: list[dict[str, object]] = []
    measured = timed_static_batch(cached, prompts, config, after_step=records.append)
    assert [row.generated_ids for row in measured.results] == [row.generated_ids for row in reference]
    early, later = measured.requests
    assert early["stop_reason"] == "eos" and early["output_tokens"] == 1
    assert early["tpot_ms"] is None and early["decode_step_ms"] == []
    assert later["output_tokens"] > 1 and later["tpot_ms"] is not None
    assert early["e2e_latency_ms"] < later["e2e_latency_ms"]
    assert early["e2e_latency_ms"] < measured.metrics["e2e_latency_ms"]
    assert records[0] == {"prefill_batch_size": 2, "decode_batch_size": 0}
    assert all(record == {"prefill_batch_size": 0, "decode_batch_size": 1} for record in records[1:])


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += 0.001
        return value


class SlowProfiler:
    def __init__(self, clock: Clock, runtime: object) -> None:
        self.clock = clock
        self.runtime = runtime
        self.started = False
        self.finished = False
        self.records: list[dict[str, object]] = []

    def __enter__(self) -> "SlowProfiler":
        self.started = True
        return self

    def __exit__(self, *args: object) -> None:
        self.finished = True

    def step(self, record: dict[str, object]) -> None:
        assert self.started and not self.finished
        self.records.append(record)
        self.clock.now += 10.0

    def metadata(self) -> dict[str, object]:
        assert self.finished
        return {"collector": "test_delayed_profiler", "selected": True}


def check_profile_isolation_and_measured_memory() -> None:
    _reference, engine, _model, _tokenizer = build_engines()
    config = GenerationConfig(max_new_tokens=4)
    for static in (False, True):
        benchmark = benchmark_static_batch if static else benchmark_generation
        prompt = ["ab", "abcdef"] if static else "ab"
        baseline_clock = Clock()
        with patch("minigpt.benchmark.time.perf_counter", baseline_clock):
            baseline = benchmark(engine, prompt, config, warmup=0, repeats=2)
        clock = Clock()
        profiler = SlowProfiler(clock, engine.runner.runtime)
        snapshots: list[bool] = []
        runtime_type = type(engine.runner.runtime)
        original_snapshot = runtime_type.memory_snapshot

        def snapshot(runtime):
            snapshots.append(profiler.started)
            return original_snapshot(runtime)

        with patch("minigpt.benchmark.time.perf_counter", clock), patch.object(runtime_type, "memory_snapshot", snapshot):
            profiled = benchmark(engine, prompt, config, warmup=0, repeats=2, profiling_session=profiler)
        assert len(profiled["runs"]) == 2
        assert profiled["profiling"]["measurement_excluded"] is True
        assert profiled["profiling"]["replay"]["output_sha256"] == baseline["runs"][0]["output_sha256"]
        assert profiled["profiling"]["replay"]["scheduler_steps"] == 4
        assert len(profiler.records) == 4
        assert profiled["profiling"]["replay"]["wall_time_ms"] > 1000 * profiled["summary"]["e2e_latency_ms"]["max"]
        for field in baseline["summary"]:
            first, second = baseline["summary"][field], profiled["summary"][field]
            if first is None:
                assert second is None
            else:
                assert first.keys() == second.keys()
                for key in first:
                    assert abs(float(first[key]) - float(second[key])) < 1e-8, (field, key)
        for run in profiled["runs"]:
            assert run["memory_snapshot"]["supported"] is False
            assert run["memory_snapshot"]["peak_allocated_bytes"] is None
        # The first calls belong to measured runs; profile-only calls occur later.
        assert snapshots and snapshots[0] is False and any(snapshots)
        assert snapshots == sorted(snapshots)


def check_no_decode_metrics() -> None:
    _reference, engine, _model, _tokenizer = build_engines()
    single = benchmark_generation(engine, "ab", GenerationConfig(max_new_tokens=1), warmup=0, repeats=1)
    batch = benchmark_static_batch(engine, ["ab", "abcdef"], GenerationConfig(max_new_tokens=1), warmup=0, repeats=1)
    assert single["runs"][0]["tpot_ms"] is None
    assert single["runs"][0]["decode_tokens_per_second"] is None
    assert single["runs"][0]["requests"][0]["tpot_ms"] is None
    assert batch["summary"]["tpot_ms"] is None
    assert all(row["tpot_ms"] is None and row["decode_step_ms"] == [] for row in batch["runs"][0]["requests"])
    original_digest = single["runs"][0]["output_sha256"]
    single["runs"][0]["requests"][0]["request_id"] = "真实请求-1"
    assert generation_output_digest(single["runs"][0]["requests"]) != original_digest


def check_real_cpu_profile(root: Path) -> None:
    _reference, engine, _model, _tokenizer = build_engines()
    protocol = ProfileProtocol(skip_steps=0, warmup_steps=1, active_steps=3)
    for static in (False, True):
        case_root = root / ("static" if static else "single")
        profiler = create_step_profiler(engine.runner.runtime, case_root / "profiler", global_rank=0, logical_device_id=0, protocol=protocol)
        benchmark = benchmark_static_batch if static else benchmark_generation
        report = benchmark(engine, ["ab", "abcdef"] if static else "ab", GenerationConfig(max_new_tokens=4), warmup=0, repeats=1, profiling_session=profiler)
        assert report["profiling"]["measurement_excluded"] is True
        assert report["runs"][0]["output_sha256"] == report["profiling"]["replay"]["output_sha256"]
        payload = build_profile_manifest(
            case_root / "profiler", runtime=engine.runner.runtime, layout_id="cpu_tiny",
            workload_class="mixed", mode="static_batch" if static else "single_request", source_workload_sha256="a" * 64,
            source_workload_file_sha256="b" * 64, git_commit="c" * 40,
            selected_ranks=[0], logical_device_ids=[0], protocol=protocol,
        )
        path = case_root / "profile_manifest.json"
        write_profile_manifest(path, payload)
        profile = summarize_profile(path)
        assert profile["complete"] is True, profile["incomplete_reasons"]
        assert profile["aggregate_step_fractions"]["communication_not_overlapped"] is None
        names = {row["name"] for row in profile["ranks"][0]["host_annotations"]}
        assert "minigpt::decode_phase" in names and "minigpt::decode_model" in names
        json.dumps(report, allow_nan=False)


def main() -> None:
    torch.set_num_threads(1)
    check_real_generation_and_row_eos()
    check_profile_isolation_and_measured_memory()
    check_no_decode_metrics()
    with tempfile.TemporaryDirectory() as directory:
        check_real_cpu_profile(Path(directory))
    print("Portable benchmark per-request timing and profiler isolation tests passed.")


if __name__ == "__main__":
    main()
