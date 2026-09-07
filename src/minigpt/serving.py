"""Continuous Batching、请求状态机和固定 KV Cache slot 生命周期。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq
import math
import statistics
import time
from typing import Callable, Protocol, Sequence

import torch

from .inference import GenerationConfig, InferenceEngine, TextTokenizer
from .runtime import RuntimeContext


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


TERMINAL_REQUEST_STATES = frozenset(
    {
        RequestState.FINISHED,
        RequestState.CANCELLED,
        RequestState.REJECTED,
        RequestState.FAILED,
    }
)


@dataclass(frozen=True)
class RequestSpec:
    """一个可重放请求；arrival_time_ms 相对 workload 起点。"""

    request_id: str
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    arrival_time_ms: float = 0.0
    deadline_ms: float | None = None

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id 不能为空")
        if not self.prompt:
            raise ValueError("prompt 不能为空")
        if not math.isfinite(self.arrival_time_ms) or self.arrival_time_ms < 0.0:
            raise ValueError("arrival_time_ms 必须是非负有限值")
        if self.deadline_ms is not None and (
            not math.isfinite(self.deadline_ms) or self.deadline_ms <= 0.0
        ):
            raise ValueError("deadline_ms 必须是正有限值，或者设为 None")
        self.config.validate()


@dataclass
class ServingRequest:
    """请求在调度器中的可变状态；slot_id 只在运行期间有效。"""

    spec: RequestSpec
    prompt_ids: list[int]
    sequence_id: int
    state: RequestState = RequestState.WAITING
    generated_ids: list[int] = field(default_factory=list)
    slot_id: int | None = None
    submitted_at_ms: float = 0.0
    admitted_at_ms: float | None = None
    token_ready_at_ms: list[float] = field(default_factory=list)
    finished_at_ms: float | None = None
    stop_reason: str | None = None
    error: str | None = None
    generator: torch.Generator | None = field(default=None, repr=False)

    @property
    def request_id(self) -> str:
        return self.spec.request_id

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_REQUEST_STATES


class SlotModelRunner(Protocol):
    """调度器依赖的窄模型接口；MiniGPT、Qwen3、TP 共用。"""

    runtime: RuntimeContext
    implementation_name: str
    max_slots: int
    max_seq_len: int

    def validate_request(self, prompt_length: int, max_new_tokens: int) -> None: ...

    def prefill_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def decode_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
    ) -> torch.Tensor: ...

    def release_slots(self, slot_ids: Sequence[int]) -> None: ...

    def cache_lengths(self, slot_ids: Sequence[int]) -> list[int]: ...


class SchedulerCollectives(Protocol):
    """TP scheduler 热路径所需的最小 collective 接口。"""

    is_primary: bool
    world_size: int

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor: ...

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor: ...


class KVSlotAllocator:
    """固定数量 slot 的 ownership 管理；不会隐藏排队或超卖容量。"""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("KV slot capacity 必须大于 0")
        self.capacity = capacity
        self._free = list(range(capacity))
        heapq.heapify(self._free)
        self._owners: dict[int, str] = {}
        self.peak_used = 0
        self.allocations = 0
        self.releases = 0

    @property
    def used(self) -> int:
        return len(self._owners)

    @property
    def free(self) -> int:
        return len(self._free)

    @property
    def owners(self) -> dict[int, str]:
        return dict(self._owners)

    @property
    def available_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._free))

    def allocate(self, request_id: str, *, slot_id: int | None = None) -> int:
        if request_id in self._owners.values():
            raise RuntimeError(f"请求 {request_id!r} 已经持有 KV slot")
        if not self._free:
            raise RuntimeError("没有可用 KV slot")
        if slot_id is None:
            slot_id = heapq.heappop(self._free)
        else:
            if not 0 <= slot_id < self.capacity:
                raise RuntimeError(f"KV slot {slot_id} 越界")
            try:
                self._free.remove(slot_id)
            except ValueError as exc:
                raise RuntimeError(f"KV slot {slot_id} 不可用") from exc
            heapq.heapify(self._free)
        self._owners[slot_id] = request_id
        self.allocations += 1
        self.peak_used = max(self.peak_used, self.used)
        return slot_id

    def release(self, slot_id: int, request_id: str) -> None:
        owner = self._owners.get(slot_id)
        if owner is None:
            raise RuntimeError(f"KV slot {slot_id} 当前没有 owner")
        if owner != request_id:
            raise RuntimeError(
                f"KV slot {slot_id} 属于 {owner!r}，不能由 {request_id!r} 释放"
            )
        del self._owners[slot_id]
        heapq.heappush(self._free, slot_id)
        self.releases += 1


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("分位数至少需要一个样本")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
    }


class ContinuousBatchEngine:
    """同步 reference scheduler；每个 step 先 Decode，再接纳并 Prefill 新请求。"""

    def __init__(
        self,
        runner: SlotModelRunner,
        tokenizer: TextTokenizer,
        *,
        pad_token_id: int | None = None,
        max_queue_size: int = 1024,
        distributed: SchedulerCollectives | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size 必须大于 0")
        if runner.max_slots <= 0 or runner.max_seq_len <= 0:
            raise ValueError("runner 必须提供正数 max_slots/max_seq_len")
        self.runner = runner
        self.tokenizer = tokenizer
        self.max_queue_size = max_queue_size
        self.distributed = distributed
        self._clock = clock
        self._origin = clock()
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            try:
                pad_token_id = tokenizer.stoi[tokenizer.unk_token]  # type: ignore[attr-defined]
            except AttributeError as exc:
                raise ValueError("tokenizer 必须提供 pad_token_id") from exc
        self.pad_token_id = int(pad_token_id)
        self.allocator = KVSlotAllocator(runner.max_slots)
        self.requests: dict[str, ServingRequest] = {}
        self._waiting: deque[str] = deque()
        self._running: dict[str, ServingRequest] = {}
        self._next_sequence_id = 0
        self.events: list[dict[str, object]] = []
        self.steps: list[dict[str, object]] = []

    def now_ms(self) -> float:
        return (self._clock() - self._origin) * 1000.0

    @property
    def is_idle(self) -> bool:
        return not self._waiting and not self._running

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    def _event(
        self,
        event: str,
        request: ServingRequest,
        timestamp_ms: float,
        **details: object,
    ) -> None:
        record: dict[str, object] = {
            "timestamp_ms": timestamp_ms,
            "event": event,
            "request_id": request.request_id,
            "state": request.state.value,
            "slot_id": request.slot_id,
        }
        record.update(details)
        self.events.append(record)

    def _reject(
        self,
        spec: RequestSpec,
        prompt_ids: list[int],
        sequence_id: int,
        now_ms: float,
        reason: str,
        error: str,
    ) -> ServingRequest:
        request = ServingRequest(
            spec=spec,
            prompt_ids=prompt_ids,
            sequence_id=sequence_id,
            state=RequestState.REJECTED,
            submitted_at_ms=now_ms,
            finished_at_ms=now_ms,
            stop_reason=reason,
            error=error,
        )
        self.requests[spec.request_id] = request
        self._event("rejected", request, now_ms, reason=reason, error=error)
        return request

    def submit(self, spec: RequestSpec, *, now_ms: float | None = None) -> ServingRequest:
        if spec.request_id in self.requests:
            raise ValueError(f"重复 request_id：{spec.request_id}")
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        timestamp = self.now_ms() if now_ms is None else float(now_ms)
        prompt_ids: list[int] = []
        try:
            spec.validate()
            prompt_ids = [int(token_id) for token_id in self.tokenizer.encode(spec.prompt)]
            if not prompt_ids:
                raise ValueError("tokenizer 不能把非空 prompt 编码为空序列")
            stop_ids = spec.config.stop_token_ids()
            if any(token_id >= self.tokenizer.vocab_size for token_id in stop_ids):
                raise ValueError("停止 token 超出 tokenizer 词表")
            self.runner.validate_request(len(prompt_ids), spec.config.max_new_tokens)
        except ValueError as exc:
            return self._reject(
                spec,
                prompt_ids,
                sequence_id,
                timestamp,
                "invalid_request",
                str(exc),
            )
        if len(self._waiting) >= self.max_queue_size:
            return self._reject(
                spec,
                prompt_ids,
                sequence_id,
                timestamp,
                "queue_full",
                f"等待队列已达到上限 {self.max_queue_size}",
            )

        generator = None
        if spec.config.strategy == "sample" and (
            self.distributed is None or self.distributed.is_primary
        ):
            generator = torch.Generator(device=self.runner.runtime.device)
            generator.manual_seed(spec.config.seed)
        request = ServingRequest(
            spec=spec,
            prompt_ids=prompt_ids,
            sequence_id=sequence_id,
            submitted_at_ms=timestamp,
            generator=generator,
        )
        self.requests[spec.request_id] = request
        self._waiting.append(spec.request_id)
        self._event("submitted", request, timestamp)
        return request

    def _release(self, request: ServingRequest) -> None:
        if request.slot_id is None:
            return
        slot_id = request.slot_id
        self.runner.release_slots([slot_id])
        self.allocator.release(slot_id, request.request_id)
        request.slot_id = None
        self._running.pop(request.request_id, None)

    def cancel(self, request_id: str, *, now_ms: float | None = None) -> bool:
        request = self.requests.get(request_id)
        if request is None:
            raise KeyError(f"未知 request_id：{request_id}")
        if request.is_terminal:
            return False
        timestamp = self.now_ms() if now_ms is None else float(now_ms)
        if request.state == RequestState.WAITING:
            self._waiting.remove(request_id)
        else:
            self._event("cancelling", request, timestamp)
            self._release(request)
        request.state = RequestState.CANCELLED
        request.finished_at_ms = timestamp
        request.stop_reason = "cancelled"
        self._event("cancelled", request, timestamp)
        return True

    def _encode_prefill_batch(
        self,
        requests: Sequence[ServingRequest],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(request.prompt_ids) for request in requests)
        device = self.runner.runtime.device
        input_ids = torch.full(
            (len(requests), width),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, request in enumerate(requests):
            length = len(request.prompt_ids)
            input_ids[row, :length] = torch.tensor(
                request.prompt_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, :length] = True
        return input_ids, attention_mask

    def _select_tokens(
        self,
        logits: torch.Tensor,
        requests: Sequence[ServingRequest],
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[0] != len(requests):
            raise ValueError("slot runner 必须返回与请求一一对应的 [B,V] logits")
        if self.distributed is not None and not self.distributed.is_primary:
            next_ids = torch.zeros(
                (len(requests), 1),
                dtype=torch.long,
                device=logits.device,
            )
        else:
            next_ids = torch.cat(
                [
                    InferenceEngine.select_next_token(
                        logits[row : row + 1],
                        request.spec.config,
                        request.generator,
                    )
                    for row, request in enumerate(requests)
                ]
            )
        if self.distributed is not None:
            next_ids = self.distributed.broadcast(next_ids, src=0)
        return next_ids

    def _finish(
        self,
        request: ServingRequest,
        *,
        timestamp_ms: float,
        stop_reason: str,
    ) -> None:
        request.state = RequestState.FINISHED
        request.finished_at_ms = timestamp_ms
        request.stop_reason = stop_reason
        self._event("finished", request, timestamp_ms, reason=stop_reason)
        self._release(request)

    def _record_token(
        self,
        request: ServingRequest,
        token_id: int,
        timestamp_ms: float,
    ) -> None:
        request.generated_ids.append(token_id)
        request.token_ready_at_ms.append(timestamp_ms)
        stop_ids = frozenset(request.spec.config.stop_token_ids())
        if token_id in stop_ids:
            self._finish(request, timestamp_ms=timestamp_ms, stop_reason="eos")
        elif len(request.generated_ids) >= request.spec.config.max_new_tokens:
            self._finish(request, timestamp_ms=timestamp_ms, stop_reason="length")

    def _fail_requests(
        self,
        requests: Sequence[ServingRequest],
        exc: Exception,
        timestamp_ms: float,
    ) -> None:
        for request in requests:
            request.state = RequestState.FAILED
            request.finished_at_ms = timestamp_ms
            request.stop_reason = "error"
            request.error = f"{type(exc).__name__}: {exc}"
            self._event("failed", request, timestamp_ms, error=request.error)
            self._release(request)

    @staticmethod
    def _request_fingerprint(request: ServingRequest) -> int:
        config = request.spec.config
        canonical = repr(
            (
                request.request_id,
                tuple(request.prompt_ids),
                config.max_new_tokens,
                config.strategy,
                config.temperature,
                config.top_k,
                config.top_p,
                config.seed,
                config.eos_token_id,
                config.eos_token_ids,
                request.spec.deadline_ms,
            )
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    def _raise_if_any_rank_invalid(
        self,
        local_error: str | None,
        *,
        phase: str,
    ) -> None:
        if self.distributed is None:
            if local_error is not None:
                raise RuntimeError(local_error)
            return
        flag = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.runner.runtime.device,
        )
        self.distributed.all_reduce_sum(flag)
        if int(flag.item()) > 0:
            rank = getattr(self.distributed, "rank", "unknown")
            detail = local_error or "另一个 rank 的本地 scheduler 状态不一致"
            raise RuntimeError(
                f"{phase} control plan 在至少一个 rank 无效；"
                f"当前 rank={rank}：{detail}"
            )

    def _broadcast_control_plan(
        self,
        phase: str,
        entries: Sequence[tuple[int, int, int]],
        *,
        population: int,
    ) -> tuple[list[tuple[int, int, int]], int]:
        if self.distributed is None:
            return list(entries), population

        phase_codes = {"decode": 1, "prefill": 2}
        expected_phase = phase_codes[phase]
        if self.distributed.is_primary:
            header = torch.tensor(
                [expected_phase, len(entries), population],
                dtype=torch.long,
                device=self.runner.runtime.device,
            )
        else:
            header = torch.zeros(
                3,
                dtype=torch.long,
                device=self.runner.runtime.device,
            )
        self.distributed.broadcast(header, src=0)
        actual_phase, count, primary_population = [
            int(value) for value in header.cpu().tolist()
        ]
        header_error = None
        if actual_phase != expected_phase:
            header_error = (
                f"phase code 不一致：expected={expected_phase}, actual={actual_phase}"
            )
        elif count < 0 or count > self.runner.max_slots:
            header_error = f"control plan count 越界：{count}"
        elif primary_population < count or primary_population > (
            self.max_queue_size + self.runner.max_slots
        ):
            header_error = f"control plan population 越界：{primary_population}"
        self._raise_if_any_rank_invalid(header_error, phase=phase)

        if self.distributed.is_primary:
            payload = torch.tensor(
                entries,
                dtype=torch.long,
                device=self.runner.runtime.device,
            ).reshape(count, 3)
        else:
            payload = torch.zeros(
                (count, 3),
                dtype=torch.long,
                device=self.runner.runtime.device,
            )
        if count:
            self.distributed.broadcast(payload, src=0)
        return [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in payload.cpu().tolist()
        ], primary_population

    def _requests_by_sequence(self) -> dict[int, ServingRequest]:
        return {
            request.sequence_id: request
            for request in self.requests.values()
            if not request.is_terminal
        }

    def _synchronized_decode_requests(self) -> list[ServingRequest]:
        local = sorted(
            self._running.values(),
            key=lambda request: int(request.slot_id),
        )
        entries = [
            (
                request.sequence_id,
                int(request.slot_id),
                self._request_fingerprint(request),
            )
            for request in local
        ]
        primary_entries = (
            entries
            if self.distributed is None or self.distributed.is_primary
            else []
        )
        plan, population = self._broadcast_control_plan(
            "decode",
            primary_entries,
            population=len(local),
        )

        by_sequence = self._requests_by_sequence()
        resolved: list[ServingRequest] = []
        error = None
        if len(local) != population:
            error = (
                f"running request 数不一致：local={len(local)}, "
                f"primary={population}"
            )
        elif len({sequence for sequence, _slot, _fingerprint in plan}) != len(plan):
            error = "Decode plan 包含重复 request sequence"
        elif len({slot for _sequence, slot, _fingerprint in plan}) != len(plan):
            error = "Decode plan 包含重复 KV slot"
        else:
            for sequence, slot_id, fingerprint in plan:
                request = by_sequence.get(sequence)
                if request is None:
                    error = f"本 rank 缺少 Decode request sequence={sequence}"
                    break
                if (
                    request.state != RequestState.DECODING
                    or request.slot_id != slot_id
                    or request.request_id not in self._running
                ):
                    error = (
                        f"Decode request {request.request_id!r} 状态/slot 不一致"
                    )
                    break
                if self._request_fingerprint(request) != fingerprint:
                    error = f"Decode request {request.request_id!r} 内容或配置不一致"
                    break
                resolved.append(request)
            if error is None and {
                request.sequence_id for request in local
            } != {sequence for sequence, _slot, _fingerprint in plan}:
                error = "Decode running 集合与 rank 0 control plan 不一致"
        self._raise_if_any_rank_invalid(error, phase="decode")
        return resolved

    def _synchronized_admissions(self) -> list[ServingRequest]:
        if self.distributed is None or self.distributed.is_primary:
            count = min(self.waiting_count, self.allocator.free)
            free_slots = self.allocator.available_slots[:count]
            request_ids = list(self._waiting)[:count]
            entries = [
                (
                    self.requests[request_id].sequence_id,
                    slot_id,
                    self._request_fingerprint(self.requests[request_id]),
                )
                for request_id, slot_id in zip(request_ids, free_slots)
            ]
        else:
            entries = []
        plan, population = self._broadcast_control_plan(
            "prefill",
            entries,
            population=self.waiting_count,
        )

        by_sequence = self._requests_by_sequence()
        waiting_ids = set(self._waiting)
        available_slots = set(self.allocator.available_slots)
        resolved: list[ServingRequest] = []
        error = None
        if self.waiting_count != population:
            error = (
                f"waiting request 数不一致：local={self.waiting_count}, "
                f"primary={population}"
            )
        elif len({sequence for sequence, _slot, _fingerprint in plan}) != len(plan):
            error = "Prefill plan 包含重复 request sequence"
        elif len({slot for _sequence, slot, _fingerprint in plan}) != len(plan):
            error = "Prefill plan 包含重复 KV slot"
        else:
            for sequence, slot_id, fingerprint in plan:
                request = by_sequence.get(sequence)
                if request is None:
                    error = f"本 rank 缺少 Prefill request sequence={sequence}"
                    break
                if request.state != RequestState.WAITING or request.request_id not in waiting_ids:
                    error = f"Prefill request {request.request_id!r} 不在 waiting 队列"
                    break
                if slot_id not in available_slots:
                    error = f"Prefill plan 指定的 KV slot {slot_id} 在本 rank 不可用"
                    break
                if self._request_fingerprint(request) != fingerprint:
                    error = f"Prefill request {request.request_id!r} 内容或配置不一致"
                    break
                resolved.append(request)
        self._raise_if_any_rank_invalid(error, phase="prefill")

        for request, (_sequence, slot_id, _fingerprint) in zip(resolved, plan):
            self._waiting.remove(request.request_id)
            self.allocator.allocate(request.request_id, slot_id=slot_id)
            request.slot_id = slot_id
            request.state = RequestState.PREFILLING
            request.admitted_at_ms = self.now_ms()
            self._running[request.request_id] = request
            self._event("admitted", request, request.admitted_at_ms)
        return resolved

    def _decode_running(self) -> tuple[list[str], list[str]]:
        requests = self._synchronized_decode_requests()
        if not requests:
            return [], []
        slot_ids = [int(request.slot_id) for request in requests]
        input_ids = torch.tensor(
            [[request.generated_ids[-1]] for request in requests],
            dtype=torch.long,
            device=self.runner.runtime.device,
        )
        try:
            logits = self.runner.decode_slots(slot_ids, input_ids)
            next_ids = self._select_tokens(logits, requests)
            self.runner.runtime.synchronize()
        except Exception as exc:
            failed_at = self.now_ms()
            self._fail_requests(requests, exc, failed_at)
            raise
        ready_at = self.now_ms()
        finished: list[str] = []
        for row, request in enumerate(requests):
            self._record_token(request, int(next_ids[row, 0].item()), ready_at)
            if request.state == RequestState.FINISHED:
                finished.append(request.request_id)
        return [request.request_id for request in requests], finished

    def _admit_and_prefill(self) -> tuple[list[str], list[str]]:
        requests = self._synchronized_admissions()
        if not requests:
            return [], []

        input_ids, attention_mask = self._encode_prefill_batch(requests)
        slot_ids = [int(request.slot_id) for request in requests]
        try:
            logits = self.runner.prefill_slots(slot_ids, input_ids, attention_mask)
            next_ids = self._select_tokens(logits, requests)
            self.runner.runtime.synchronize()
        except Exception as exc:
            failed_at = self.now_ms()
            self._fail_requests(requests, exc, failed_at)
            raise
        ready_at = self.now_ms()
        finished: list[str] = []
        for row, request in enumerate(requests):
            request.state = RequestState.DECODING
            self._record_token(request, int(next_ids[row, 0].item()), ready_at)
            if request.state == RequestState.FINISHED:
                finished.append(request.request_id)
            else:
                self._event("first_token", request, ready_at)
        return [request.request_id for request in requests], finished

    @torch.inference_mode()
    def step(self) -> dict[str, object]:
        """执行一个调度周期；没有请求时返回空 step，但不制造模型调用。"""

        started_at = self.now_ms()
        active_before = self.running_count
        waiting_before = self.waiting_count
        decoded, decode_finished = self._decode_running()
        admitted, prefill_finished = self._admit_and_prefill()
        ended_at = self.now_ms()
        active_slots = sorted(self.allocator.owners)
        used_tokens = sum(self.runner.cache_lengths(active_slots))
        record: dict[str, object] = {
            "step": len(self.steps),
            "started_at_ms": started_at,
            "ended_at_ms": ended_at,
            "duration_ms": ended_at - started_at,
            "active_before": active_before,
            "waiting_before": waiting_before,
            "decode_batch_size": len(decoded),
            "prefill_batch_size": len(admitted),
            "decoded_request_ids": decoded,
            "admitted_request_ids": admitted,
            "finished_request_ids": decode_finished + prefill_finished,
            "active_after": self.running_count,
            "waiting_after": self.waiting_count,
            "kv_slots_used": self.allocator.used,
            "kv_slots_free": self.allocator.free,
            "kv_used_tokens": used_tokens,
            "kv_reserved_tokens": self.allocator.used * self.runner.max_seq_len,
        }
        self.steps.append(record)
        return record

    def run_until_idle(self, *, max_steps: int = 1_000_000) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        steps = 0
        while True:
            if self.distributed is None:
                should_step = not self.is_idle
            else:
                should_step_on_primary = (
                    self.distributed.is_primary and not self.is_idle
                )
                primary_decision = torch.tensor(
                    [1 if should_step_on_primary else 0],
                    dtype=torch.int32,
                    device=self.runner.runtime.device,
                )
                self.distributed.broadcast(primary_decision, src=0)
                should_step = bool(primary_decision.item())
                if not should_step:
                    local_error = (
                        None
                        if self.is_idle
                        else "rank 0 已 idle，但本 rank 仍有 waiting/running 请求"
                    )
                    self._raise_if_any_rank_invalid(
                        local_error,
                        phase="run_until_idle",
                    )
            if not should_step:
                break
            if steps >= max_steps:
                raise RuntimeError("调度器超过 max_steps，可能存在无法结束的请求")
            self.step()
            steps += 1

    def reset(self) -> None:
        if not self.is_idle:
            raise RuntimeError("只能在没有 waiting/running 请求时 reset")
        self.runner.release_slots(list(range(self.runner.max_slots)))
        self.allocator = KVSlotAllocator(self.runner.max_slots)
        self.requests.clear()
        self.events.clear()
        self.steps.clear()
        self._next_sequence_id = 0
        self._origin = self._clock()

    @staticmethod
    def _request_metrics(request: ServingRequest) -> dict[str, object]:
        admitted = request.admitted_at_ms
        first_token = request.token_ready_at_ms[0] if request.token_ready_at_ms else None
        finished = request.finished_at_ms
        tpot_ms = None
        if len(request.token_ready_at_ms) > 1:
            tpot_ms = (
                request.token_ready_at_ms[-1] - request.token_ready_at_ms[0]
            ) / (len(request.token_ready_at_ms) - 1)
        return {
            "request_id": request.request_id,
            "sequence_id": request.sequence_id,
            "state": request.state.value,
            "stop_reason": request.stop_reason,
            "error": request.error,
            "scheduled_arrival_ms": request.spec.arrival_time_ms,
            "submitted_at_ms": request.submitted_at_ms,
            "admitted_at_ms": admitted,
            "first_token_at_ms": first_token,
            "finished_at_ms": finished,
            "queue_ms": None if admitted is None else admitted - request.submitted_at_ms,
            "ttft_ms": None if first_token is None else first_token - request.submitted_at_ms,
            "tpot_ms": tpot_ms,
            "e2e_latency_ms": None if finished is None else finished - request.submitted_at_ms,
            "deadline_ms": request.spec.deadline_ms,
            "input_tokens": len(request.prompt_ids),
            "output_tokens": len(request.generated_ids),
            "prompt_ids": list(request.prompt_ids),
            "generated_ids": list(request.generated_ids),
        }

    def report(
        self,
        *,
        ttft_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
        e2e_slo_ms: float | None = None,
    ) -> dict[str, object]:
        request_metrics = [
            self._request_metrics(request) for request in self.requests.values()
        ]
        completed = [
            item for item in request_metrics if item["state"] == RequestState.FINISHED.value
        ]
        starts = [float(item["submitted_at_ms"]) for item in request_metrics]
        ends = [
            float(item["finished_at_ms"])
            for item in request_metrics
            if item["finished_at_ms"] is not None
        ]
        duration_ms = max(ends) - min(starts) if starts and ends else 0.0
        duration_seconds = max(duration_ms / 1000.0, 1e-12)

        def numeric(field: str) -> list[float]:
            return [float(item[field]) for item in completed if item[field] is not None]

        def meets_slo(item: dict[str, object]) -> bool:
            request_e2e_limit = item["deadline_ms"] or e2e_slo_ms
            checks = (
                (ttft_slo_ms, item["ttft_ms"]),
                (tpot_slo_ms, item["tpot_ms"]),
                (request_e2e_limit, item["e2e_latency_ms"]),
            )
            return all(
                limit is None or (value is not None and float(value) <= float(limit))
                for limit, value in checks
            )

        good_requests = sum(meets_slo(item) for item in completed)
        input_tokens = sum(int(item["input_tokens"]) for item in completed)
        output_tokens = sum(int(item["output_tokens"]) for item in completed)
        peak_used_tokens = max(
            (int(step["kv_used_tokens"]) for step in self.steps),
            default=0,
        )
        peak_reserved_tokens = max(
            (int(step["kv_reserved_tokens"]) for step in self.steps),
            default=0,
        )
        peak_internal_waste_tokens = max(
            (
                int(step["kv_reserved_tokens"]) - int(step["kv_used_tokens"])
                for step in self.steps
            ),
            default=0,
        )
        state_counts = {
            state.value: sum(item["state"] == state.value for item in request_metrics)
            for state in RequestState
        }
        return {
            "schema_version": 1,
            "engine": {
                "runner": self.runner.implementation_name,
                "device": str(self.runner.runtime.device),
                "precision": self.runner.runtime.precision,
                "max_queue_size": self.max_queue_size,
                "max_slots": self.runner.max_slots,
                "max_seq_len": self.runner.max_seq_len,
            },
            "slo": {
                "ttft_ms": ttft_slo_ms,
                "tpot_ms": tpot_slo_ms,
                "e2e_ms": e2e_slo_ms,
            },
            "summary": {
                "requests": len(request_metrics),
                "state_counts": state_counts,
                "duration_ms": duration_ms,
                "completed_requests_per_second": len(completed) / duration_seconds,
                "goodput_requests_per_second": good_requests / duration_seconds,
                "input_tokens_per_second": input_tokens / duration_seconds,
                "output_tokens_per_second": output_tokens / duration_seconds,
                "queue_ms": _latency_summary(numeric("queue_ms")),
                "ttft_ms": _latency_summary(numeric("ttft_ms")),
                "tpot_ms": _latency_summary(numeric("tpot_ms")),
                "e2e_latency_ms": _latency_summary(numeric("e2e_latency_ms")),
            },
            "kv_cache": {
                "slot_capacity": self.allocator.capacity,
                "peak_slots_used": self.allocator.peak_used,
                "allocations": self.allocator.allocations,
                "releases": self.allocator.releases,
                "token_capacity": self.allocator.capacity * self.runner.max_seq_len,
                "peak_used_tokens": peak_used_tokens,
                "peak_reserved_tokens": peak_reserved_tokens,
                "peak_internal_waste_tokens": peak_internal_waste_tokens,
                "external_fragmentation_tokens": 0,
            },
            "requests": request_metrics,
            "steps": list(self.steps),
            "events": list(self.events),
        }
