"""最小多副本路由层；不包含 HTTP、服务发现或跨机容错。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping, Sequence

from .serving import ContinuousBatchEngine, RequestSpec, ServingRequest
from .workload import WorkloadPartition, WorkloadTrace


@dataclass(frozen=True)
class ReplicaSnapshot:
    replica_id: str
    running_requests: int
    waiting_requests: int
    max_slots: int
    outstanding_token_work: int

    def validate(self) -> None:
        if not self.replica_id:
            raise ValueError("replica_id 不能为空")
        if self.max_slots <= 0:
            raise ValueError("max_slots 必须大于 0")
        if min(
            self.running_requests,
            self.waiting_requests,
            self.outstanding_token_work,
        ) < 0:
            raise ValueError("replica load 不能为负数")


class LeastLoadedRouter:
    """按归一化剩余 token work 选副本，确定性 replica_id 打破平局。"""

    @staticmethod
    def score(snapshot: ReplicaSnapshot) -> tuple[float, float, float, str]:
        snapshot.validate()
        capacity = float(snapshot.max_slots)
        return (
            snapshot.outstanding_token_work / capacity,
            (snapshot.running_requests + snapshot.waiting_requests) / capacity,
            snapshot.running_requests / capacity,
            snapshot.replica_id,
        )

    def choose(self, snapshots: Sequence[ReplicaSnapshot]) -> str:
        if not snapshots:
            raise ValueError("至少需要一个 replica snapshot")
        return min(snapshots, key=self.score).replica_id


def _remaining_work(request: ServingRequest) -> int:
    generated_remaining = max(
        request.spec.config.max_new_tokens - len(request.generated_ids),
        0,
    )
    prompt_work = len(request.prompt_ids) if not request.generated_ids else 0
    return prompt_work + generated_remaining


class MultiReplicaServing:
    """在多个独立 ContinuousBatchEngine 前提供最小 least-loaded router。"""

    def __init__(
        self,
        replicas: Mapping[str, ContinuousBatchEngine],
        *,
        router: LeastLoadedRouter | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not replicas:
            raise ValueError("至少需要一个 serving replica")
        if any(not replica_id for replica_id in replicas):
            raise ValueError("replica_id 不能为空")
        if len(set(replicas)) != len(replicas):
            raise ValueError("replica_id 不能重复")
        self.replicas = dict(replicas)
        self.router = router or LeastLoadedRouter()
        self._clock = clock
        self._origin = clock()
        self.request_to_replica: dict[str, str] = {}
        self.routes: list[dict[str, object]] = []
        self.steps: list[dict[str, object]] = []

    def now_ms(self) -> float:
        return (self._clock() - self._origin) * 1000.0

    @property
    def is_idle(self) -> bool:
        return all(engine.is_idle for engine in self.replicas.values())

    @property
    def running_count(self) -> int:
        return sum(engine.running_count for engine in self.replicas.values())

    @property
    def waiting_count(self) -> int:
        return sum(engine.waiting_count for engine in self.replicas.values())

    def snapshots(self) -> list[ReplicaSnapshot]:
        snapshots: list[ReplicaSnapshot] = []
        for replica_id, engine in sorted(self.replicas.items()):
            outstanding = sum(
                _remaining_work(request)
                for request in engine.requests.values()
                if not request.is_terminal
            )
            snapshots.append(
                ReplicaSnapshot(
                    replica_id=replica_id,
                    running_requests=engine.running_count,
                    waiting_requests=engine.waiting_count,
                    max_slots=engine.runner.max_slots,
                    outstanding_token_work=outstanding,
                )
            )
        return snapshots

    def submit(
        self,
        spec: RequestSpec,
        *,
        now_ms: float | None = None,
    ) -> ServingRequest:
        if spec.request_id in self.request_to_replica:
            raise ValueError(f"重复 request_id：{spec.request_id}")
        snapshots = self.snapshots()
        eligible_ids = {
            replica_id
            for replica_id, engine in self.replicas.items()
            if engine.waiting_count < engine.max_queue_size
        }
        eligible = [
            snapshot
            for snapshot in snapshots
            if snapshot.replica_id in eligible_ids
        ]
        replica_id = self.router.choose(eligible or snapshots)
        timestamp = self.now_ms() if now_ms is None else float(now_ms)
        request = self.replicas[replica_id].submit(spec, now_ms=timestamp)
        self.request_to_replica[spec.request_id] = replica_id
        self.routes.append(
            {
                "timestamp_ms": timestamp,
                "request_id": spec.request_id,
                "replica_id": replica_id,
                "state_after_submit": request.state.value,
            }
        )
        return request

    def cancel(self, request_id: str, *, now_ms: float | None = None) -> bool:
        try:
            replica_id = self.request_to_replica[request_id]
        except KeyError as exc:
            raise KeyError(f"未知 request_id：{request_id}") from exc
        return self.replicas[replica_id].cancel(request_id, now_ms=now_ms)

    def step(self) -> dict[str, object]:
        started_at = self.now_ms()
        replica_steps: dict[str, object] = {}
        for replica_id, engine in sorted(self.replicas.items()):
            if not engine.is_idle:
                replica_steps[replica_id] = engine.step()
        record = {
            "step": len(self.steps),
            "started_at_ms": started_at,
            "ended_at_ms": self.now_ms(),
            "replicas": replica_steps,
            "running_after": self.running_count,
            "waiting_after": self.waiting_count,
        }
        self.steps.append(record)
        return record

    def run_until_idle(self, *, max_steps: int = 1_000_000) -> None:
        steps = 0
        while not self.is_idle:
            if steps >= max_steps:
                raise RuntimeError("多副本调度超过 max_steps")
            self.step()
            steps += 1

    def reset(self) -> None:
        if not self.is_idle:
            raise RuntimeError("只能在所有 replicas idle 时 reset")
        for engine in self.replicas.values():
            engine.reset()
        self.request_to_replica.clear()
        self.routes.clear()
        self.steps.clear()
        self._origin = self._clock()

    def report(self, **slo: float | None) -> dict[str, object]:
        replica_reports = {
            replica_id: engine.report(**slo)
            for replica_id, engine in sorted(self.replicas.items())
        }
        requests = [
            request
            for report in replica_reports.values()
            for request in report["requests"]
        ]
        starts = [float(item["submitted_at_ms"]) for item in requests]
        ends = [
            float(item["finished_at_ms"])
            for item in requests
            if item["finished_at_ms"] is not None
        ]
        duration_ms = max(ends) - min(starts) if starts and ends else 0.0
        duration_seconds = max(duration_ms / 1000.0, 1e-12)
        completed = [item for item in requests if item["state"] == "finished"]
        good = [item for item in completed if item["slo_met"]]
        state_counts = {
            state: sum(item["state"] == state for item in requests)
            for state in (
                "waiting",
                "prefilling",
                "decoding",
                "finished",
                "cancelled",
                "rejected",
                "failed",
            )
        }
        return {
            "schema_version": 1,
            "router": "least_loaded",
            "replica_count": len(self.replicas),
            "duration_ms": duration_ms,
            "requests": len(requests),
            "state_counts": state_counts,
            "completed_requests_per_second": len(completed) / duration_seconds,
            "goodput_requests_per_second": len(good) / duration_seconds,
            "input_tokens_per_second": sum(
                int(item["input_tokens"]) for item in completed
            )
            / duration_seconds,
            "output_tokens_per_second": sum(
                int(item["output_tokens"]) for item in completed
            )
            / duration_seconds,
            "request_to_replica": dict(self.request_to_replica),
            "routes": list(self.routes),
            "steps": list(self.steps),
            "replicas": replica_reports,
        }


def partition_workload_by_projected_load(
    trace: WorkloadTrace,
    *,
    replica_count: int,
    max_slots_per_replica: int,
) -> tuple[list[WorkloadTrace], list[dict[str, object]]]:
    """离线为独立进程副本分片；使用累计字符+输出 token 作为显式估计。"""

    trace.validate()
    if replica_count <= 0 or max_slots_per_replica <= 0:
        raise ValueError("replica_count/max_slots_per_replica 必须大于 0")
    router = LeastLoadedRouter()
    loads = [0 for _ in range(replica_count)]
    assignments: list[list[RequestSpec]] = [[] for _ in range(replica_count)]
    manifest: list[dict[str, object]] = []
    for request in trace.requests:
        snapshots = [
            ReplicaSnapshot(
                replica_id=str(index),
                running_requests=0,
                waiting_requests=len(assignments[index]),
                max_slots=max_slots_per_replica,
                outstanding_token_work=loads[index],
            )
            for index in range(replica_count)
        ]
        replica_index = int(router.choose(snapshots))
        estimated_work = len(request.prompt) + request.config.max_new_tokens
        loads[replica_index] += estimated_work
        assignments[replica_index].append(request)
        manifest.append(
            {
                "request_id": request.request_id,
                "replica_index": replica_index,
                "estimated_work": estimated_work,
                "projected_replica_work": loads[replica_index],
            }
        )
    partitions = [
        WorkloadTrace(
            workload_id=f"{trace.workload_id}-replica-{index}",
            workload_class=trace.workload_class,
            requests=tuple(requests),
            metadata={
                **trace.metadata,
                "partition_method": "least_projected_character_plus_output_work",
            },
            partition=WorkloadPartition(
                source_sha256=trace.request_sha256,
                replica_count=replica_count,
                replica_index=index,
            ),
        )
        for index, requests in enumerate(assignments)
    ]
    if any(not partition.requests for partition in partitions):
        raise ValueError("请求数必须不少于 replica_count，不能产生空 partition")
    for partition in partitions:
        partition.validate()
    manifested_work = sum(int(item["estimated_work"]) for item in manifest)
    if not math.isclose(sum(loads), manifested_work):
        raise AssertionError("partition projected load 汇总不一致")
    return partitions, manifest
