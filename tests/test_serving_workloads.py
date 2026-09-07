"""v0.7 workload、replay 与 least-loaded 多副本路由正确性门。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.inference import (  # noqa: E402
    GenerationConfig,
    SlotCachedMiniGPTModelRunner,
)
from minigpt.model import MiniGPT, MiniGPTConfig  # noqa: E402
from minigpt.replica import (  # noqa: E402
    LeastLoadedRouter,
    MultiReplicaServing,
    ReplicaSnapshot,
    partition_workload_by_projected_load,
)
from minigpt.replay import OfflineTraceReplayer  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402
from minigpt.serving import ContinuousBatchEngine, RequestSpec, RequestState  # noqa: E402
from minigpt.serving_benchmark import benchmark_trace_replay  # noqa: E402
from minigpt.tokenizer import CharTokenizer  # noqa: E402
from minigpt.workload import WorkloadTrace, generate_workload  # noqa: E402


class ManualClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("不能让时钟倒退")
        self.seconds += seconds


class AdvancingRunner:
    """为 deterministic replay 测试给每次模型调用增加固定虚拟耗时。"""

    def __init__(
        self,
        delegate: SlotCachedMiniGPTModelRunner,
        clock: ManualClock,
    ) -> None:
        self.delegate = delegate
        self.clock = clock
        self.runtime = delegate.runtime
        self.implementation_name = f"timed_{delegate.implementation_name}"
        self.max_slots = delegate.max_slots
        self.max_seq_len = delegate.max_seq_len

    def validate_request(self, prompt_length: int, max_new_tokens: int) -> None:
        self.delegate.validate_request(prompt_length, max_new_tokens)

    def prefill_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.delegate.prefill_slots(slot_ids, input_ids, attention_mask)
        self.clock.advance(0.001)
        return logits

    def decode_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.delegate.decode_slots(slot_ids, input_ids)
        self.clock.advance(0.001)
        return logits

    def release_slots(self, slot_ids: Sequence[int]) -> None:
        self.delegate.release_slots(slot_ids)

    def cache_lengths(self, slot_ids: Sequence[int]) -> list[int]:
        return self.delegate.cache_lengths(slot_ids)


def build_engine(
    clock: ManualClock,
    *,
    seed: int,
    max_slots: int = 2,
    max_queue_size: int = 16,
) -> ContinuousBatchEngine:
    torch.manual_seed(seed)
    tokenizer = CharTokenizer.train_from_text("abcdefghijklmnopqrstuvwxyz \n")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=24,
            n_layer=1,
            n_head=2,
            n_embd=24,
            dropout=0.0,
        )
    ).eval()
    runtime = RuntimeContext.create("cpu", "fp32")
    delegate = SlotCachedMiniGPTModelRunner(model, runtime, max_slots=max_slots)
    return ContinuousBatchEngine(
        AdvancingRunner(delegate, clock),
        tokenizer,
        max_queue_size=max_queue_size,
        clock=clock,
    )


def tiny_trace(*, arrival_interval_ms: float = 2.0) -> WorkloadTrace:
    requests = tuple(
        RequestSpec(
            request_id=f"request-{index}",
            prompt="ab" + "c" * (index % 3),
            config=GenerationConfig(max_new_tokens=2 + index % 3),
            arrival_time_ms=index * arrival_interval_ms,
        )
        for index in range(8)
    )
    trace = WorkloadTrace(
        workload_id="tiny-mixed",
        workload_class="mixed",
        requests=requests,
        metadata={"purpose": "cpu correctness"},
    )
    trace.validate()
    return trace


def check_workload_schema_and_presets() -> None:
    for preset in ("short_short", "long_prefill_short_decode", "mixed"):
        generated = generate_workload(
            preset,
            request_count=8,
            arrival_interval_ms=1.5,
            seed=7,
        )
        assert generated.workload_class == preset
        assert len(generated.request_sha256) == 64

    trace = tiny_trace()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.json"
        trace.save(path)
        restored = WorkloadTrace.load(path)
        assert restored == trace
        assert restored.request_sha256 == trace.request_sha256

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["requests"][0]["prompt"] = "changed"
        try:
            WorkloadTrace.from_dict(tampered)
        except ValueError as exc:
            assert "SHA" in str(exc)
        else:
            raise AssertionError("被篡改的 workload 必须被 digest 门禁拒绝")


def check_router_scoring_and_live_replicas() -> None:
    router = LeastLoadedRouter()
    selected = router.choose(
        [
            ReplicaSnapshot("replica-b", 1, 0, 2, 10),
            ReplicaSnapshot("replica-a", 1, 0, 2, 10),
            ReplicaSnapshot("replica-c", 2, 1, 2, 30),
        ]
    )
    assert selected == "replica-a"

    clock = ManualClock()
    serving = MultiReplicaServing(
        {
            "replica-0": build_engine(clock, seed=11),
            "replica-1": build_engine(clock, seed=11),
        },
        clock=clock,
    )
    for request in tiny_trace(arrival_interval_ms=0.0).requests:
        serving.submit(request, now_ms=0.0)
    routed = list(serving.request_to_replica.values())
    assert set(routed) == {"replica-0", "replica-1"}
    assert abs(routed.count("replica-0") - routed.count("replica-1")) <= 1
    serving.run_until_idle()
    assert all(
        request.state == RequestState.FINISHED
        for engine in serving.replicas.values()
        for request in engine.requests.values()
    )
    report = serving.report()
    assert report["replica_count"] == 2
    assert report["completed_requests_per_second"] > 0.0


def check_open_and_closed_loop_replay() -> None:
    trace = tiny_trace(arrival_interval_ms=5.0)
    open_clock = ManualClock()
    open_engine = build_engine(open_clock, seed=17, max_slots=1)
    open_result = OfflineTraceReplayer(
        open_engine,
        trace,
        mode="open_loop",
        clock=open_clock,
        sleeper=open_clock.advance,
    ).run()
    assert open_result.submitted_requests == len(trace.requests)
    assert open_result.wait_count > 0
    assert open_result.scheduler_steps > 0
    assert open_engine.is_idle
    assert all(
        request.state == RequestState.FINISHED
        for request in open_engine.requests.values()
    )

    closed_clock = ManualClock()
    closed_serving = MultiReplicaServing(
        {
            "replica-0": build_engine(closed_clock, seed=23, max_slots=1),
            "replica-1": build_engine(closed_clock, seed=23, max_slots=1),
        },
        clock=closed_clock,
    )
    closed_result = OfflineTraceReplayer(
        closed_serving,
        trace,
        mode="closed_loop",
        closed_loop_clients=3,
        clock=closed_clock,
        sleeper=closed_clock.advance,
    ).run()
    assert closed_result.submitted_requests == len(trace.requests)
    assert closed_result.closed_loop_clients == 3
    assert closed_serving.is_idle
    assert len(closed_serving.routes) == len(trace.requests)


def check_projected_load_partitions() -> None:
    trace = tiny_trace()
    partitions, manifest = partition_workload_by_projected_load(
        trace,
        replica_count=4,
        max_slots_per_replica=2,
    )
    assert len(partitions) == 4
    assert len(manifest) == len(trace.requests)
    assert {partition.partition.replica_index for partition in partitions} == {
        0,
        1,
        2,
        3,
    }
    assert all(
        partition.source_sha256 == trace.request_sha256
        for partition in partitions
    )
    recovered_ids = {
        request.request_id
        for partition in partitions
        for request in partition.requests
    }
    assert recovered_ids == {request.request_id for request in trace.requests}


def check_replay_benchmark_protocol() -> None:
    clock = ManualClock()
    engine = build_engine(clock, seed=31, max_slots=2)
    report = benchmark_trace_replay(
        engine,
        tiny_trace(arrival_interval_ms=1.0),
        mode="open_loop",
        closed_loop_clients=None,
        warmup=1,
        repeats=2,
        ttft_slo_ms=1_000.0,
        tpot_slo_ms=1_000.0,
        e2e_slo_ms=1_000.0,
        clock=clock,
        sleeper=clock.advance,
    )
    assert report["protocol"]["warmup"] == 1
    assert len(report["runs"]) == 2
    assert report["runs"][0]["output_sha256"] == report["runs"][1]["output_sha256"]
    assert report["summary"]["goodput_requests_per_second"]["median"] > 0.0
    assert report["summary"]["ttft_ms"]["p95"] >= 0.0
    assert report["summary"]["active_batch_size"]["p99"] >= 1.0


def main() -> None:
    check_workload_schema_and_presets()
    check_router_scoring_and_live_replicas()
    check_open_and_closed_loop_replay()
    check_projected_load_partitions()
    check_replay_benchmark_protocol()
    print("v0.7 workload replay and multi-replica routing tests passed.")


if __name__ == "__main__":
    main()
