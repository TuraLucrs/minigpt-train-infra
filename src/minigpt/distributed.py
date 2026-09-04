"""CPU/Gloo、CUDA/NCCL、Ascend/HCCL 共用的窄分布式运行时边界。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
import socket
from typing import Callable

import torch
import torch.distributed as dist

from .runtime import RuntimeContext


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，收到 {value!r}") from exc


def _default_backend(device_type: str) -> str:
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


@dataclass
class DistributedContext:
    """一次 torchrun 任务解析后的 rank、device 与 process group。"""

    runtime: RuntimeContext
    backend: str
    rank: int
    local_rank: int
    world_size: int
    owns_process_group: bool

    @classmethod
    def create(
        cls,
        requested_device: str,
        requested_precision: str,
        *,
        backend: str = "auto",
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        init_method: str | None = None,
        timeout_seconds: int = 180,
        warn: Callable[[str], None] = print,
    ) -> "DistributedContext":
        rank_value = _environment_int("RANK", 0) if rank is None else rank
        world_size_value = (
            _environment_int("WORLD_SIZE", 1) if world_size is None else world_size
        )
        local_rank_value = (
            _environment_int("LOCAL_RANK", rank_value)
            if local_rank is None
            else local_rank
        )
        if world_size_value <= 0:
            raise ValueError("world_size 必须大于 0")
        if not 0 <= rank_value < world_size_value:
            raise ValueError("rank 必须位于 [0, world_size)")
        if local_rank_value < 0:
            raise ValueError("local_rank 不能小于 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        accelerator_index = (
            local_rank_value
            if requested_device.lower() in {"auto", "cuda", "npu"}
            else None
        )
        requested_device_name = requested_device.lower()
        runtime = RuntimeContext.create(
            requested_device,
            requested_precision,
            warn=warn,
            device_index=accelerator_index,
        )
        if (
            requested_device_name in {"cuda", "npu"}
            and runtime.device.type != requested_device_name
        ):
            raise RuntimeError(
                f"分布式任务明确请求 {requested_device_name}，禁止静默回退到 "
                f"{runtime.device.type}"
            )

        backend_requested = backend.lower()
        backend_name = (
            _default_backend(runtime.device.type)
            if backend_requested == "auto"
            else backend_requested
        )
        expected_backend = _default_backend(runtime.device.type)
        if backend_name != expected_backend:
            raise ValueError(
                f"当前 {runtime.device.type} TP 路径要求 backend={expected_backend}，"
                f"收到 {backend_name}"
            )

        owns_process_group = False
        if dist.is_available() and dist.is_initialized():
            if (
                dist.get_rank() != rank_value
                or dist.get_world_size() != world_size_value
            ):
                raise RuntimeError("现有 process group 与请求的 rank/world_size 不一致")
            initialized_backend = str(dist.get_backend()).lower()
            if initialized_backend != backend_name:
                raise RuntimeError(
                    f"现有 process group backend={initialized_backend}，"
                    f"请求 backend={backend_name}"
                )
        elif world_size_value > 1:
            if not dist.is_available():
                raise RuntimeError("当前 PyTorch 未启用 torch.distributed")
            backend_checker = getattr(dist, "is_backend_available", None)
            if callable(backend_checker) and not backend_checker(backend_name):
                raise RuntimeError(
                    f"当前 PyTorch 不支持 distributed backend={backend_name}"
                )
            try:
                dist.init_process_group(
                    backend=backend_name,
                    init_method=init_method or "env://",
                    rank=rank_value,
                    world_size=world_size_value,
                    timeout=timedelta(seconds=timeout_seconds),
                )
            except Exception:
                if dist.is_initialized():
                    dist.destroy_process_group()
                raise
            else:
                owns_process_group = True

        return cls(
            runtime=runtime,
            backend=backend_name,
            rank=rank_value,
            local_rank=local_rank_value,
            world_size=world_size_value,
            owns_process_group=owns_process_group,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def _require_process_group(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            raise RuntimeError("distributed collective 需要已初始化的 process group")

    def barrier(self) -> None:
        if self.is_distributed:
            self._require_process_group()
            dist.barrier()

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_distributed:
            self._require_process_group()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed:
            return tensor
        self._require_process_group()
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)

    def all_gather_floats(self, values: list[float]) -> list[list[float]]:
        """在所有 rank 收集少量测量值；用于报告而非模型热路径。"""

        if not self.is_distributed:
            return [list(values)]
        self._require_process_group()
        # HCCL/NCCL 的通用交集对 FP32 支持最好；这些值只用于报告，不进入数值计算。
        local = torch.tensor(values, dtype=torch.float32, device=self.runtime.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        return [tensor.cpu().tolist() for tensor in gathered]

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not 0 <= src < self.world_size:
            raise ValueError("broadcast src 必须位于 [0, world_size)")
        if self.is_distributed:
            self._require_process_group()
            dist.broadcast(tensor, src=src)
        return tensor

    def metadata(self) -> dict[str, str | int | bool]:
        if self.runtime.device.type == "cuda":
            visible_device_count = torch.cuda.device_count()
        elif self.runtime.device.type == "npu":
            visible_device_count = torch.npu.device_count()  # type: ignore[attr-defined]
        else:
            visible_device_count = 0
        return {
            "backend": self.backend,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "is_primary": self.is_primary,
            "process_group_initialized": dist.is_available() and dist.is_initialized(),
            "hostname": socket.gethostname(),
            "visible_device_count": visible_device_count,
        }

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_process_group = False

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
