"""Runtime compatibility, optional SDK isolation, and accelerator API contracts."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.backends.runtime import get_runtime_backend  # noqa: E402
from minigpt.backends.runtime_ascend import AscendRuntimeBackend  # noqa: E402
from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.runtime import (  # noqa: E402
    DeviceIntervalTimer,
    RuntimeCapabilities,
    RuntimeContext,
)


class _FakeEvent:
    def __init__(self, calls: list[tuple[object, ...]], index: int) -> None:
        self.calls = calls
        self.index = index

    def record(self) -> None:
        self.calls.append(("event.record", self.index))

    def synchronize(self) -> None:
        self.calls.append(("event.synchronize", self.index))

    def elapsed_time(self, other: "_FakeEvent") -> float:
        self.calls.append(("event.elapsed_time", self.index, other.index))
        return 12.5


class _AcceleratorAPI:
    """An SDK-shaped fixture; model/Tensor arithmetic is never mocked."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.events: list[_FakeEvent] = []
        self.amp_dtypes = [torch.float16, torch.bfloat16]
        self.nccl = SimpleNamespace(version=lambda: (2, 23, 1))

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def set_device(self, device: object) -> None:
        self.calls.append(("set_device", device))

    def is_bf16_supported(self) -> bool:
        return torch.bfloat16 in self.amp_dtypes

    def get_amp_supported_dtype(self) -> list[torch.dtype]:
        return list(self.amp_dtypes)

    def synchronize(self, device: object) -> None:
        self.calls.append(("synchronize", device))

    def manual_seed_all(self, seed: int) -> None:
        self.calls.append(("manual_seed_all", seed))

    def reset_peak_memory_stats(self, device: object) -> None:
        self.calls.append(("reset_peak_memory_stats", device))

    def memory_allocated(self, device: object) -> int:
        self.calls.append(("memory_allocated", device))
        return 2 * 1024 * 1024

    def max_memory_allocated(self, device: object) -> int:
        return 3 * 1024 * 1024

    def memory_reserved(self, device: object) -> int:
        return 4 * 1024 * 1024

    def max_memory_reserved(self, device: object) -> int:
        return 5 * 1024 * 1024

    def get_device_name(self, device: object) -> str:
        return "contract-fixture-device"

    def get_device_properties(self, device: object) -> object:
        return SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024)

    def empty_cache(self) -> None:
        self.calls.append(("empty_cache",))

    def Event(self, *, enable_timing: bool) -> _FakeEvent:
        if not enable_timing:
            raise AssertionError("runtime timing events must enable timing")
        event = _FakeEvent(self.calls, len(self.events))
        self.events.append(event)
        return event


class RuntimeBackendTests(unittest.TestCase):
    def test_cpu_imports_and_execution_do_not_load_optional_npu(self) -> None:
        script = f"""
import importlib.abc
import sys
import torch
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
class BlockNpu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise AssertionError('CPU execution attempted to import torch_npu')
sys.meta_path.insert(0, BlockNpu())
from minigpt.runtime import RuntimeContext
from minigpt.distributed import DistributedContext
import minigpt.ascend_profiling
import minigpt.backends.runtime_ascend
runtime = RuntimeContext.create('cpu', 'fp32')
runtime.synchronize()
runtime.reset_peak_memory()
runtime.empty_cache()
assert runtime.memory_snapshot().supported is False
assert runtime.backend_metadata()['torch_npu'] is None
with DistributedContext.create('cpu', 'fp32', rank=0, local_rank=0, world_size=1) as context:
    assert context.backend == 'gloo'
    assert context.metadata()['visible_device_count'] == 0
    assert context.all_reduce_sum(torch.tensor([2])).item() == 2
print('CPU optional SDK isolation passed')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cpu_runtime_keeps_correctness_and_legacy_memory_contract(self) -> None:
        runtime = RuntimeContext.create("cpu", "fp32")
        self.assertEqual(runtime.backend.distributed_backend, "gloo")
        self.assertEqual(runtime.device_name(), "CPU")
        self.assertEqual(runtime.visible_device_count(), 0)
        self.assertEqual(runtime.memory_stats_mb(), (0.0, 0.0))
        snapshot = runtime.memory_snapshot().to_dict()
        self.assertIs(snapshot["supported"], False)
        for name in (
            "allocated_bytes", "peak_allocated_bytes", "reserved_bytes",
            "peak_reserved_bytes", "total_bytes",
        ):
            self.assertIsNone(snapshot[name])
        self.assertEqual(snapshot["source"], "unavailable")
        runtime.manual_seed(42)
        first = torch.rand(8)
        runtime.manual_seed(42)
        with runtime.autocast():
            second = torch.rand(8)
            self.assertEqual(second.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))
        timer = DeviceIntervalTimer(runtime)
        with self.assertRaises(RuntimeError):
            timer.elapsed_seconds()
        timer.start()
        self.assertGreaterEqual(timer.elapsed_seconds(), 0.0)
        self.assertIsNone(timer.last_device_seconds)

    def test_fallback_is_explicit_and_precision_policy_is_preserved(self) -> None:
        warnings: list[str] = []
        context = RuntimeContext.create("cpu", "bf16", warn=warnings.append)
        self.assertEqual(context.precision, "fp32")
        self.assertIsNone(context.amp_dtype)
        self.assertEqual(len(warnings), 1)
        with self.assertRaises(RuntimeError):
            RuntimeContext.create("cpu", "bf16", allow_precision_fallback=False)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                DistributedContext.create("cuda", "fp32")
        with self.assertRaises(ValueError):
            DistributedContext.create("cpu", "fp32", backend="nccl")
        with self.assertRaises(ValueError):
            RuntimeContext.create("cpu", "fp32", device_index=-1)
        with self.assertRaises(ValueError):
            get_runtime_backend("unrecognized")

    def test_auto_cpu_survives_unavailable_optional_shared_libraries(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(AscendRuntimeBackend, "_extension", side_effect=OSError("missing SDK")),
        ):
            context = RuntimeContext.create("auto", "fp32")
            self.assertEqual(context.device.type, "cpu")
            with self.assertRaises(RuntimeError):
                RuntimeContext.create("npu", "fp32", allow_accelerator_fallback=False)

    def test_cuda_dispatch_preserves_device_precision_and_memory_units(self) -> None:
        api = _AcceleratorAPI()
        with (
            patch.object(torch, "cuda", api),
            patch.object(torch.amp, "autocast", return_value=nullcontext()) as autocast,
            patch.object(AscendRuntimeBackend, "is_available", side_effect=AssertionError("unexpected NPU probe")),
        ):
            runtime = RuntimeContext.create("auto", "bf16", device_index=1)
            self.assertEqual(runtime.device, torch.device("cuda:1"))
            self.assertEqual(runtime.backend.distributed_backend, "nccl")
            self.assertTrue(runtime.capabilities.supports_bf16)
            with runtime.autocast():
                pass
            autocast.assert_called_with(device_type="cuda", dtype=torch.bfloat16)
            runtime.synchronize()
            runtime.reset_peak_memory()
            runtime.empty_cache()
            self.assertEqual(runtime.memory_stats_mb(), (2.0, 3.0))
            snapshot = runtime.memory_snapshot()
            self.assertEqual(snapshot.reserved_bytes, 4 * 1024 * 1024)
            self.assertEqual(snapshot.peak_reserved_bytes, 5 * 1024 * 1024)
            self.assertEqual(snapshot.total_bytes, 16 * 1024 * 1024 * 1024)
            self.assertIn(("set_device", runtime.device), api.calls)
            self.assertIn(("synchronize", runtime.device), api.calls)
            self.assertEqual(runtime.backend_metadata()["nccl_version"], "2.23.1")
            with self.assertRaises(ValueError):
                RuntimeContext.create("cuda", "fp32", device_index=2)

    def test_device_interval_synchronizes_only_the_end_event(self) -> None:
        api = _AcceleratorAPI()
        with patch.object(torch, "cuda", api):
            runtime = RuntimeContext.create("cuda", "fp32", device_index=0)
            timer = DeviceIntervalTimer(runtime)
            timer.start()
            with self.assertRaises(RuntimeError):
                timer.start()
            self.assertGreaterEqual(timer.elapsed_seconds(), 0.0)
            self.assertAlmostEqual(timer.last_device_seconds, 0.0125)
            self.assertEqual(
                [call for call in api.calls if str(call[0]).startswith("event.")],
                [
                    ("event.record", 0), ("event.record", 1),
                    ("event.synchronize", 1), ("event.elapsed_time", 0, 1),
                ],
            )
            self.assertFalse(any(call[0] == "synchronize" for call in api.calls))

    def test_ascend_sdk_contract_and_missing_reserved_statistics(self) -> None:
        api = _AcceleratorAPI()
        device = SimpleNamespace(type="npu", index=1)
        runtime = RuntimeContext(
            device=device,
            precision="bf16",
            amp_dtype=torch.bfloat16,
            capabilities=RuntimeCapabilities(True, True, True, True, False),
        )
        extension = SimpleNamespace(__version__="contract-fixture-version")
        with (
            patch.object(AscendRuntimeBackend, "_extension", return_value=extension),
            patch.object(torch, "npu", api, create=True),
            patch.object(torch.amp, "autocast", return_value=nullcontext()) as autocast,
            patch.dict(os.environ, {"CANN_VERSION": "fixture-cann", "HCCL_VERSION": "fixture-hccl"}),
        ):
            backend = runtime.backend
            self.assertTrue(backend.is_available())
            self.assertEqual(backend.distributed_backend, "hccl")
            self.assertEqual(backend.device_count(), 2)
            backend.set_device(device)
            backend.manual_seed_all(77)
            runtime.synchronize()
            runtime.reset_peak_memory()
            runtime.empty_cache()
            with runtime.autocast():
                pass
            autocast.assert_called_with(device_type="npu", dtype=torch.bfloat16)
            self.assertEqual(runtime.memory_stats_mb(), (2.0, 3.0))
            self.assertIn(("manual_seed_all", 77), api.calls)
            self.assertIn(("synchronize", device), api.calls)
            self.assertEqual(backend.version_metadata()["cann_version"], "fixture-cann")
            self.assertEqual(backend.version_metadata()["hccl_version"], "fixture-hccl")
            api.amp_dtypes = [torch.bfloat16]
            self.assertEqual(backend.supported_autocast_dtypes(), frozenset({torch.bfloat16}))
            with (
                patch.object(api, "memory_reserved", None),
                patch.object(api, "max_memory_reserved", None),
            ):
                snapshot = runtime.memory_snapshot()
                self.assertIsNone(snapshot.reserved_bytes)
                self.assertIsNone(snapshot.peak_reserved_bytes)
                self.assertEqual(snapshot.allocated_bytes, 2 * 1024 * 1024)
            with patch.object(api, "get_amp_supported_dtype", None):
                self.assertEqual(
                    backend.supported_autocast_dtypes(),
                    frozenset({torch.float16, torch.bfloat16}),
                )


if __name__ == "__main__":
    unittest.main()
