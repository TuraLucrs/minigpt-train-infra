"""Shared experiment identity and measurement contracts for the v0.9 CLIs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from .backends.runtime_base import DeviceMemorySnapshot


def parse_device_ids(value: str, world_size: int) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("logical-device-ids 必须是逗号分隔整数") from exc
    if len(ids) != world_size or len(set(ids)) != len(ids) or min(ids, default=-1) < 0:
        raise ValueError("logical-device-ids 必须按 rank 列出 world_size 个不同的非负整数")
    return ids


def validate_device_mapping(distributed, logical_ids: Sequence[int]) -> dict[str, object]:
    """Bind rank metadata to actual single-node accelerator visibility.

    Labels alone never select devices. The launcher must set device visibility
    before importing PyTorch, or explicitly use its default identity mapping.
    CPU IDs are process positions and never imply accelerator availability.
    """

    ids = parse_device_ids(",".join(str(value) for value in logical_ids), distributed.world_size)
    if distributed.local_rank != distributed.rank:
        raise ValueError("v0.9 当前矩阵要求单机 torchrun，LOCAL_RANK 必须等于 RANK")
    runtime = distributed.runtime
    device_type = runtime.device.type
    environment_variable = {"cuda": "CUDA_VISIBLE_DEVICES", "npu": "ASCEND_RT_VISIBLE_DEVICES"}.get(device_type)
    if device_type == "cpu":
        if ids != list(range(distributed.world_size)):
            raise ValueError("CPU logical IDs 必须为进程编号 0..world_size-1")
        return {"verified": True, "kind": "cpu_process_ranks", "logical_device_ids": ids,
                "visible_accelerator_count": 0, "environment_variable": None,
                "environment_value": None, "local_device_index": None}
    visible = runtime.visible_device_count()
    if visible < distributed.world_size:
        raise ValueError("实际可见 accelerator 数量少于 torchrun world_size")
    value = os.environ.get(environment_variable)
    if value is None:
        if ids != list(range(distributed.world_size)):
            raise ValueError(f"非默认设备映射必须在启动 Python 前设置 {environment_variable}")
    else:
        actual_ids = parse_device_ids(value, visible)
        if actual_ids[:distributed.world_size] != ids:
            raise ValueError(f"logical-device-ids 与实际 {environment_variable} 映射不一致")
    if runtime.device.index != distributed.local_rank:
        raise ValueError("实际 runtime device index 与 LOCAL_RANK 不一致")
    return {"verified": True, "kind": "accelerator_visibility", "logical_device_ids": ids,
            "visible_accelerator_count": visible, "environment_variable": environment_variable,
            "environment_value": value, "local_device_index": runtime.device.index}


def validate_rank_digest(distributed, digest: str) -> None:
    """Compare all 256 bits using fixed-shape collectives on every TP rank."""

    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("expected a lowercase SHA-256 digest")
    words = [int(digest[index:index + 8], 16) for index in range(0, 64, 8)]
    local = torch.tensor(words, device=distributed.runtime.device, dtype=torch.int64)
    expected = local.clone() if distributed.is_primary else torch.zeros_like(local)
    distributed.broadcast(expected, src=0)
    mismatch = torch.tensor([int(not torch.equal(local, expected))], device=local.device, dtype=torch.int32)
    distributed.all_reduce_sum(mismatch)
    if mismatch.item():
        raise RuntimeError("TP ranks 的输入或输出 SHA-256 不一致")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def artifact_reference(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def measured_memory(distributed, logical_ids: Sequence[int], snapshot) -> dict[str, object]:
    """Gather exact int64 allocator bytes; unsupported counters stay null."""

    fields = ("allocated_bytes", "peak_allocated_bytes", "reserved_bytes", "peak_reserved_bytes", "total_bytes")
    values = [int(snapshot.supported)] + [getattr(snapshot, field) if getattr(snapshot, field) is not None else -1 for field in fields]
    tensor = torch.tensor(values, dtype=torch.int64, device=distributed.runtime.device)
    gathered = distributed.all_gather_last_dim(tensor).reshape(distributed.world_size, len(values)).cpu().tolist()
    per_rank = []
    for rank, row in enumerate(gathered):
        per_rank.append({"rank": rank, "logical_device_id": int(logical_ids[rank]),
                         "supported": bool(row[0]), **{field: value if value >= 0 else None for field, value in zip(fields, row[1:])}})
    supported = all(row["supported"] for row in per_rank)
    peaks = [row["peak_allocated_bytes"] for row in per_rank]
    return {"unit": "bytes", "measurement_type": "measured" if supported else "unsupported",
            "source": "pytorch_allocator" if supported else "unavailable", "supported": supported,
            "scope": "measured_repeats", "per_rank": per_rank,
            "max_rank_peak_allocated_bytes": max(peaks) if supported and None not in peaks else None,
            "sum_rank_peak_allocated_bytes": sum(peaks) if supported and None not in peaks else None}


def memory_from_measured_runs(distributed, logical_ids: Sequence[int], runs: list[dict[str, object]]) -> dict[str, object]:
    snapshots = [run["memory_snapshot"] for run in runs]
    if not snapshots:
        raise ValueError("measured memory needs measured runs")
    supported = all(snapshot["supported"] for snapshot in snapshots)
    fields = ("allocated_bytes", "peak_allocated_bytes", "reserved_bytes", "peak_reserved_bytes", "total_bytes")
    values = {}
    for field in fields:
        numbers = [snapshot[field] for snapshot in snapshots]
        values[field] = (max(numbers) if field.startswith("peak_") else numbers[-1]) if None not in numbers else None
    snapshot = DeviceMemorySnapshot(supported=supported, **values)
    return measured_memory(distributed, logical_ids, snapshot)
