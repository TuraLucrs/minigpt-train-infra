"""离线 open-loop / closed-loop trace replay。"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol

import torch

from .serving import RequestSpec, ServingRequest
from .workload import WorkloadTrace


class ReplayTarget(Protocol):
    is_idle: bool

    def submit(
        self,
        spec: RequestSpec,
        *,
        now_ms: float | None = None,
    ) -> ServingRequest: ...

    def step(self) -> dict[str, object]: ...


class ReplayCollectives(Protocol):
    is_primary: bool

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReplaySummary:
    mode: str
    closed_loop_clients: int | None
    submitted_requests: int
    scheduler_steps: int
    wait_count: int
    started_at_unix_ns: int
    ended_at_unix_ns: int
    wall_time_ms: float


class OfflineTraceReplayer:
    """由 rank 0 决定到达/step；所有 TP ranks 执行相同 action 序列。"""

    STEP = 1
    WAIT = 2
    DONE = 3

    def __init__(
        self,
        target: ReplayTarget,
        trace: WorkloadTrace,
        *,
        mode: str,
        closed_loop_clients: int | None = None,
        distributed: ReplayCollectives | None = None,
        control_device: torch.device | str = "cpu",
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        trace.validate()
        if mode not in {"open_loop", "closed_loop"}:
            raise ValueError("mode 必须是 open_loop 或 closed_loop")
        if mode == "closed_loop":
            if closed_loop_clients is None or closed_loop_clients <= 0:
                raise ValueError("closed_loop 必须提供正数 closed_loop_clients")
        elif closed_loop_clients is not None:
            raise ValueError("open_loop 不接受 closed_loop_clients")
        self.target = target
        self.trace = trace
        self.mode = mode
        self.closed_loop_clients = closed_loop_clients
        self.distributed = distributed
        self.control_device = torch.device(control_device)
        self.clock = clock
        self.sleeper = sleeper

    def _is_primary(self) -> bool:
        return self.distributed is None or self.distributed.is_primary

    def _broadcast_action(
        self,
        submit_count: int,
        action: int,
        wait_microseconds: int,
    ) -> tuple[int, int, int]:
        values = (
            [submit_count, action, wait_microseconds]
            if self._is_primary()
            else [0, 0, 0]
        )
        control = torch.tensor(values, dtype=torch.long, device=self.control_device)
        if self.distributed is not None:
            self.distributed.broadcast(control, src=0)
        submitted, selected_action, wait_us = [
            int(value) for value in control.cpu().tolist()
        ]
        if submitted < 0 or wait_us < 0 or selected_action not in {
            self.STEP,
            self.WAIT,
            self.DONE,
        }:
            raise RuntimeError("收到非法 trace replay control action")
        return submitted, selected_action, wait_us

    def run(self, *, max_steps: int = 1_000_000) -> ReplaySummary:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        monotonic_start = self.clock()
        unix_start = time.time_ns()
        cursor = 0
        submitted_requests: list[ServingRequest] = []
        scheduler_steps = 0
        wait_count = 0

        while True:
            if self._is_primary():
                elapsed_ms = (self.clock() - monotonic_start) * 1000.0
                if self.mode == "open_loop":
                    due_cursor = cursor
                    while (
                        due_cursor < len(self.trace.requests)
                        and self.trace.requests[due_cursor].arrival_time_ms <= elapsed_ms
                    ):
                        due_cursor += 1
                    submit_count = due_cursor - cursor
                else:
                    active = sum(not request.is_terminal for request in submitted_requests)
                    submit_count = min(
                        int(self.closed_loop_clients) - active,
                        len(self.trace.requests) - cursor,
                    )

                will_have_work = not self.target.is_idle or submit_count > 0
                if will_have_work:
                    action = self.STEP
                    wait_us = 0
                elif cursor + submit_count >= len(self.trace.requests):
                    action = self.DONE
                    wait_us = 0
                else:
                    action = self.WAIT
                    next_arrival = self.trace.requests[cursor].arrival_time_ms
                    wait_us = max(int((next_arrival - elapsed_ms) * 1000.0), 1)
            else:
                elapsed_ms = 0.0
                submit_count = 0
                action = self.DONE
                wait_us = 0

            submit_count, action, wait_us = self._broadcast_action(
                submit_count,
                action,
                wait_us,
            )
            if cursor + submit_count > len(self.trace.requests):
                raise RuntimeError("rank 0 要求提交超过 trace 末尾的请求")
            submit_at_ms = (self.clock() - monotonic_start) * 1000.0
            for request in self.trace.requests[cursor : cursor + submit_count]:
                submitted_requests.append(
                    self.target.submit(request, now_ms=submit_at_ms)
                )
            cursor += submit_count

            if action == self.STEP:
                if scheduler_steps >= max_steps:
                    raise RuntimeError("trace replay 超过 max_steps")
                self.target.step()
                scheduler_steps += 1
            elif action == self.WAIT:
                self.sleeper(wait_us / 1_000_000.0)
                wait_count += 1
            else:
                if cursor != len(self.trace.requests) or not self.target.is_idle:
                    raise RuntimeError("trace replay 在请求完成前收到 DONE")
                break

        ended = self.clock()
        unix_end = time.time_ns()
        return ReplaySummary(
            mode=self.mode,
            closed_loop_clients=self.closed_loop_clients,
            submitted_requests=len(submitted_requests),
            scheduler_steps=scheduler_steps,
            wait_count=wait_count,
            started_at_unix_ns=unix_start,
            ended_at_unix_ns=unix_end,
            wall_time_ms=(ended - monotonic_start) * 1000.0,
        )
