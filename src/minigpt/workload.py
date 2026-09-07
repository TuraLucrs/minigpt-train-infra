"""Continuous Batching 可重放 workload schema 与确定性预设。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping

from .inference import GenerationConfig
from .serving import RequestSpec


WORKLOAD_SCHEMA_VERSION = 1
WORKLOAD_PRESETS = ("short_short", "long_prefill_short_decode", "mixed")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _request_to_dict(spec: RequestSpec) -> dict[str, object]:
    return {
        "request_id": spec.request_id,
        "prompt": spec.prompt,
        "arrival_time_ms": spec.arrival_time_ms,
        "deadline_ms": spec.deadline_ms,
        "generation": asdict(spec.config),
    }


def _request_from_dict(raw: object) -> RequestSpec:
    if not isinstance(raw, dict):
        raise ValueError("workload request 必须是对象")
    required = {"request_id", "prompt", "arrival_time_ms", "generation"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"workload request 缺少字段：{sorted(missing)}")
    generation = raw["generation"]
    if not isinstance(generation, dict):
        raise ValueError("request.generation 必须是对象")
    allowed_generation = {
        "max_new_tokens",
        "strategy",
        "temperature",
        "top_k",
        "top_p",
        "seed",
        "eos_token_id",
        "eos_token_ids",
    }
    unknown_generation = set(generation) - allowed_generation
    if unknown_generation:
        raise ValueError(
            f"request.generation 包含未知字段：{sorted(unknown_generation)}"
        )
    generation_values = dict(generation)
    eos_token_ids = generation_values.get("eos_token_ids")
    if eos_token_ids is not None:
        if not isinstance(eos_token_ids, list):
            raise ValueError("generation.eos_token_ids 必须是数组或 null")
        generation_values["eos_token_ids"] = tuple(
            int(value) for value in eos_token_ids
        )
    config = GenerationConfig(**generation_values)
    deadline = raw.get("deadline_ms")
    spec = RequestSpec(
        request_id=str(raw["request_id"]),
        prompt=str(raw["prompt"]),
        config=config,
        arrival_time_ms=float(raw["arrival_time_ms"]),
        deadline_ms=None if deadline is None else float(deadline),
    )
    spec.validate()
    return spec


@dataclass(frozen=True)
class WorkloadPartition:
    """同一源 workload 被拆给多个副本时的身份。"""

    source_sha256: str
    replica_count: int
    replica_index: int

    def validate(self) -> None:
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("partition.source_sha256 必须是小写 SHA-256")
        if self.replica_count <= 0:
            raise ValueError("partition.replica_count 必须大于 0")
        if not 0 <= self.replica_index < self.replica_count:
            raise ValueError("partition.replica_index 必须位于 [0, replica_count)")


@dataclass(frozen=True)
class WorkloadTrace:
    """按 arrival_time_ms 排序、可以保存和严格校验的请求序列。"""

    workload_id: str
    workload_class: str
    requests: tuple[RequestSpec, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    partition: WorkloadPartition | None = None

    def validate(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id 不能为空")
        if not self.workload_class:
            raise ValueError("workload_class 不能为空")
        if not self.requests:
            raise ValueError("workload 至少需要一个请求")
        request_ids: set[str] = set()
        previous_arrival = -1.0
        for request in self.requests:
            request.validate()
            if request.request_id in request_ids:
                raise ValueError(f"重复 workload request_id：{request.request_id}")
            if request.arrival_time_ms < previous_arrival:
                raise ValueError("workload requests 必须按 arrival_time_ms 非递减排序")
            request_ids.add(request.request_id)
            previous_arrival = request.arrival_time_ms
        if self.partition is not None:
            self.partition.validate()

    @property
    def request_sha256(self) -> str:
        payload = [_request_to_dict(request) for request in self.requests]
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def source_sha256(self) -> str:
        if self.partition is not None:
            return self.partition.source_sha256
        return self.request_sha256

    def to_dict(self) -> dict[str, object]:
        self.validate()
        partition = None
        if self.partition is not None:
            partition = asdict(self.partition)
        return {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "workload_id": self.workload_id,
            "workload_class": self.workload_class,
            "request_sha256": self.request_sha256,
            "metadata": dict(self.metadata),
            "partition": partition,
            "requests": [_request_to_dict(request) for request in self.requests],
        }

    @classmethod
    def from_dict(cls, raw: object) -> "WorkloadTrace":
        if not isinstance(raw, dict):
            raise ValueError("workload 根节点必须是对象")
        if raw.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"只支持 workload schema_version={WORKLOAD_SCHEMA_VERSION}"
            )
        requests_raw = raw.get("requests")
        if not isinstance(requests_raw, list):
            raise ValueError("workload.requests 必须是数组")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("workload.metadata 必须是对象")
        partition_raw = raw.get("partition")
        partition = None
        if partition_raw is not None:
            if not isinstance(partition_raw, dict):
                raise ValueError("workload.partition 必须是对象或 null")
            try:
                partition = WorkloadPartition(
                    source_sha256=str(partition_raw["source_sha256"]),
                    replica_count=int(partition_raw["replica_count"]),
                    replica_index=int(partition_raw["replica_index"]),
                )
            except KeyError as exc:
                raise ValueError(f"workload.partition 缺少字段：{exc.args[0]}") from exc
        trace = cls(
            workload_id=str(raw.get("workload_id", "")),
            workload_class=str(raw.get("workload_class", "")),
            requests=tuple(_request_from_dict(item) for item in requests_raw),
            metadata=dict(metadata),
            partition=partition,
        )
        trace.validate()
        declared_digest = raw.get("request_sha256")
        if declared_digest != trace.request_sha256:
            raise ValueError("workload request_sha256 与实际请求内容不一致")
        return trace

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadTrace":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def generate_workload(
    preset: str,
    *,
    request_count: int,
    arrival_interval_ms: float,
    seed: int = 2026,
) -> WorkloadTrace:
    """生成三类固定语义 workload；正式报告仍保存 tokenizer 实测长度。"""

    if preset not in WORKLOAD_PRESETS:
        raise ValueError(f"preset 必须是 {WORKLOAD_PRESETS} 之一")
    if request_count <= 0:
        raise ValueError("request_count 必须大于 0")
    if not math.isfinite(arrival_interval_ms) or arrival_interval_ms < 0.0:
        raise ValueError("arrival_interval_ms 必须是非负有限值")
    rng = random.Random(seed)
    short_prompts = (
        "用一句话解释 KV Cache。",
        "什么是 TTFT？",
        "说明 Prefill 和 Decode 的区别。",
        "为什么推理服务需要 batching？",
    )
    long_unit = (
        "请沿着请求进入调度器、完成 Prefill、逐 token Decode、"
        "释放 KV Cache 的顺序，"
        "解释每一步的数据形状、状态变化和可能失败的位置。"
    )
    requests: list[RequestSpec] = []
    for index in range(request_count):
        if preset == "short_short":
            prompt = short_prompts[index % len(short_prompts)]
            output_tokens = 16
        elif preset == "long_prefill_short_decode":
            prompt = long_unit * 24
            output_tokens = 8
        elif rng.random() < 0.45:
            prompt = long_unit * (12 + index % 13)
            output_tokens = 8 + index % 9
        else:
            prompt = short_prompts[index % len(short_prompts)]
            output_tokens = 16 + index % 17
        requests.append(
            RequestSpec(
                request_id=f"{preset}-{index:05d}",
                prompt=prompt,
                config=GenerationConfig(
                    max_new_tokens=output_tokens,
                    strategy="greedy",
                    seed=seed + index,
                ),
                arrival_time_ms=index * arrival_interval_ms,
            )
        )
    trace = WorkloadTrace(
        workload_id=(
            f"{preset}-n{request_count}-interval{arrival_interval_ms:g}-seed{seed}"
        ),
        workload_class=preset,
        requests=tuple(requests),
        metadata={
            "generator": "minigpt.workload.generate_workload",
            "request_count": request_count,
            "arrival_interval_ms": arrival_interval_ms,
            "seed": seed,
            "prompt_lengths_are": "tokenizer-dependent; report encoded lengths",
        },
    )
    trace.validate()
    return trace
