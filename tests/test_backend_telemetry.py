"""Pre-launch idle evidence contracts; no accelerator is fabricated by these tests."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.backends import telemetry  # noqa: E402


CUDA_DEVICES = [
    {"logical_device_id": 3, "physical_card_id": 3, "chip_id": 0},
    {"logical_device_id": 5, "physical_card_id": 5, "chip_id": 0},
]
CUDA_IDLE = "3, 100, 1000, 5\n5, 50, 1000, 0\n"
NPU_DEVICES = [
    {"logical_device_id": 4, "physical_card_id": 2, "chip_id": 0},
    {"logical_device_id": 5, "physical_card_id": 2, "chip_id": 1},
]
NPU_IDLE = """NPU ID : 2
Chip Count : 2
Chip ID : 0
HBM Usage Rate(%) : 3
AICore Usage Rate(%) : 0
Chip ID : 1
HBM Usage Rate(%) : 9
AICore Usage Rate(%) : 5
"""


class _Clock:
    def __init__(self) -> None:
        self.elapsed_ns = 0
        self.sleeps: list[float] = []

    def unix_ns(self) -> int:
        return 1_700_000_000_000_000_000 + self.elapsed_ns

    def monotonic_ns(self) -> int:
        return 100_000_000_000 + self.elapsed_ns

    def advance(self, seconds: float) -> None:
        self.elapsed_ns += round(seconds * 1e9)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


@contextmanager
def _commands(outputs: list[object]):
    """Only the external management command and clock are simulated."""
    clock = _Clock()
    remaining = iter(outputs)

    def run(command, **kwargs):
        clock.advance(0.01)
        output = next(remaining)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, tuple):
            return subprocess.CompletedProcess(command, output[0], output[1], output[2])
        return subprocess.CompletedProcess(command, 0, output, "")

    with (
        patch.object(telemetry.time, "time_ns", side_effect=clock.unix_ns),
        patch.object(telemetry.time, "monotonic_ns", side_effect=clock.monotonic_ns),
        patch.object(telemetry.time, "sleep", side_effect=clock.sleep),
        patch.object(telemetry.subprocess, "run", side_effect=run) as command,
    ):
        yield clock, command


class DevicePreflightTests(unittest.TestCase):
    def test_real_cpu_path_is_explicitly_unsupported_and_imports_no_vendor_sdk(self) -> None:
        script = f"""
import importlib.abc
import json
import sys
import torch
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
class RejectVendorSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch_npu', 'pynvml'}}:
            raise AssertionError('preflight imported vendor runtime: ' + fullname)
sys.meta_path.insert(0, RejectVendorSdk())
from minigpt.backends.telemetry import collect_device_preflight, summarize_device_preflight
from minigpt.runtime import RuntimeContext
runtime = RuntimeContext.create('cpu', 'fp32')
result = collect_device_preflight(runtime.device.type, [{{'logical_device_id': 0, 'physical_card_id': None, 'chip_id': None}}])
assert result['supported'] is False
assert result['clean'] is None
assert result['complete'] is False
assert result['rounds'] == []
assert result['device_maxima'] == []
assert result['collector'] == 'unavailable'
assert result['capabilities']['device_memory_utilization'] is False
assert summarize_device_preflight(result)['clean'] is None
json.dumps(result, allow_nan=False)
print('CPU preflight remains unsupported without loading an accelerator SDK')
"""
        completed = subprocess.run([sys.executable, "-c", script], check=False,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_cuda_captures_raw_rounds_and_explicit_nvml_mapping_at_inclusive_limits(self) -> None:
        with _commands([CUDA_IDLE] * 3) as (clock, command):
            record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES, query_timeout_seconds=2.0)
        self.assertTrue(record["supported"])
        self.assertTrue(record["complete"], record["errors"])
        self.assertTrue(record["clean"], record["violations"])
        self.assertGreaterEqual(record["sample_span_seconds"], 1.0)
        self.assertEqual(clock.sleeps, [0.5, 0.5])
        self.assertEqual(command.call_count, 3)
        for call in command.call_args_list:
            self.assertEqual(call.args[0], ["nvidia-smi", telemetry._CUDA_QUERY,
                                           "--format=csv,noheader,nounits", "-i", "3,5"])
            self.assertIs(call.kwargs["shell"], False)
            self.assertEqual(call.kwargs["timeout"], 2.0)
        for round_record in record["rounds"]:
            self.assertEqual(round_record["queries"][0]["stdout"], CUDA_IDLE)
            self.assertEqual(round_record["queries"][0]["returncode"], 0)
        maxima = record["device_maxima"]
        self.assertEqual([row["nvml_index"] for row in maxima], [3, 5])
        self.assertEqual([row["runtime_visible_ordinal"] for row in maxima], [0, 1])
        self.assertEqual(maxima[0]["max_memory_usage_percent"], 10.0)
        self.assertEqual(maxima[0]["max_compute_utilization_percent"], 5.0)
        self.assertEqual([row["sample_count"] for row in maxima], [3, 3])
        self.assertEqual(record["errors"], [])
        json.dumps(record, allow_nan=False)

    def test_late_or_transient_load_fails_using_window_maximum(self) -> None:
        for busy_round in (1, 2):
            with self.subTest(busy_round=busy_round):
                outputs = [CUDA_IDLE] * 3
                outputs[busy_round] = "3, 101, 1000, 0\n5, 0, 1000, 6\n"
                with _commands(outputs):
                    record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES)
                self.assertTrue(record["complete"], record["errors"])
                self.assertFalse(record["clean"])
                self.assertAlmostEqual(record["device_maxima"][0]["max_memory_usage_percent"], 10.1)
                self.assertEqual(record["device_maxima"][1]["max_compute_utilization_percent"], 6.0)
                self.assertEqual(len(record["violations"]), 2)

    def test_missing_duplicate_unexpected_or_invalid_cuda_device_data_is_incomplete(self) -> None:
        bad_outputs = [
            "3, 0, 1000, 0\n",  # Late disappearance of one requested device.
            "3, 0, 1000, 0\n3, 0, 1000, 0\n",
            "3, 0, 1000, 0\n4, 0, 1000, 0\n",
            "3, N/A, 1000, 0\n5, 0, 1000, 0\n",
            "3, 0, 0, 0\n5, 0, 1000, 0\n",
            "3, 1001, 1000, 0\n5, 0, 1000, 0\n",
            "3, -1, 1000, 0\n5, 0, 1000, 0\n",
            "3, 0, 1000, nan\n5, 0, 1000, 0\n",
            "3, 0, 1000, 101\n5, 0, 1000, 0\n",
            "3, 0, 1000\n5, 0, 1000, 0\n",
        ]
        for output in bad_outputs:
            with self.subTest(output=output):
                with _commands([CUDA_IDLE, CUDA_IDLE, output]) as (_, command):
                    record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES)
                self.assertEqual(command.call_count, 3)
                self.assertFalse(record["complete"])
                self.assertFalse(record["clean"])
                self.assertTrue(record["errors"])
                self.assertEqual(record["rounds"][-1]["queries"][0]["stdout"], output)

    def test_failed_or_timed_out_management_queries_keep_evidence_and_fail_closed(self) -> None:
        failures = [
            (9, "partial stdout", "driver unavailable"),
            FileNotFoundError("nvidia-smi is missing"),
            subprocess.TimeoutExpired("nvidia-smi", 2.0, output=b"partial query", stderr=b"timeout detail"),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with _commands([CUDA_IDLE, failure, CUDA_IDLE]):
                    record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES, query_timeout_seconds=2.0)
                self.assertFalse(record["complete"])
                self.assertFalse(record["clean"])
                query = record["rounds"][1]["queries"][0]
                self.assertIsNotNone(query["error"])
                if isinstance(failure, subprocess.TimeoutExpired):
                    self.assertEqual(query["stdout"], "partial query")
                    self.assertEqual(query["stderr"], "timeout detail")
                json.dumps(record, allow_nan=False)

    def test_npu_uses_real_common_parser_for_each_explicit_chip_and_keeps_original_samples(self) -> None:
        with _commands([NPU_IDLE] * 6) as (_, command):
            record = telemetry.collect_device_preflight("npu", NPU_DEVICES, query_timeout_seconds=1.5)
        self.assertTrue(record["clean"], record["errors"])
        self.assertEqual(command.call_count, 6)
        for call in command.call_args_list:
            self.assertEqual(call.args[0], ["npu-smi", "info", "-t", "common", "-i", "2"])
            self.assertEqual(call.kwargs["timeout"], 1.5)
        self.assertFalse(record["capabilities"]["raw_command_output"])
        maxima = record["device_maxima"]
        self.assertEqual([row["max_memory_usage_percent"] for row in maxima], [3.0, 9.0])
        self.assertEqual([row["max_compute_utilization_percent"] for row in maxima], [0.0, 5.0])
        for round_record in record["rounds"]:
            for index, query in enumerate(round_record["queries"]):
                raw = query["raw_sample"]
                self.assertEqual((raw["logical_device_id"], raw["npu_id"], raw["chip_id"]), (index + 4, 2, index))
                self.assertIn("timestamp_unix_ns", raw)
                self.assertIn("hbm_usage_percent", raw)
                self.assertIn("aicore_usage_percent", raw)

    def test_npu_missing_hbm_aicore_or_chip_and_late_busy_chip_cannot_pass(self) -> None:
        bad_outputs = [
            NPU_IDLE.replace("HBM Usage Rate(%)", "Memory Usage Rate(%)"),
            NPU_IDLE.replace("AICore Usage Rate(%)", "Unknown metric"),
            NPU_IDLE.replace("Chip ID : 1", "Chip ID : 9"),
        ]
        for output in bad_outputs:
            with self.subTest(output=output):
                with _commands([NPU_IDLE] * 5 + [output]):
                    record = telemetry.collect_device_preflight("npu", NPU_DEVICES)
                self.assertFalse(record["complete"])
                self.assertFalse(record["clean"])
                self.assertTrue(record["errors"])
        with _commands([NPU_IDLE] * 5 + [NPU_IDLE.replace("HBM Usage Rate(%) : 9", "HBM Usage Rate(%) : 11")]):
            busy = telemetry.collect_device_preflight("npu", NPU_DEVICES)
        self.assertTrue(busy["complete"])
        self.assertFalse(busy["clean"])
        self.assertEqual(busy["device_maxima"][1]["max_memory_usage_percent"], 11.0)

    def test_mapping_ambiguities_fail_before_any_hardware_query(self) -> None:
        bad_maps = [[], [CUDA_DEVICES[0], CUDA_DEVICES[0]],
                    [{"logical_device_id": 3, "physical_card_id": 0, "chip_id": 0}],
                    [{"logical_device_id": 3, "physical_card_id": 3, "chip_id": 1}],
                    [{"logical_device_id": 3, "physical_card_id": 3}],
                    [{"logical_device_id": True, "physical_card_id": 1, "chip_id": 0}]]
        with patch.object(telemetry.subprocess, "run", side_effect=AssertionError("unexpected query")) as command:
            for devices in bad_maps:
                with self.subTest(devices=devices):
                    record = telemetry.collect_device_preflight("cuda", devices)
                    self.assertFalse(record["clean"])
                    self.assertFalse(record["complete"])
                    self.assertTrue(record["errors"])
            command.assert_not_called()

    def test_recomputation_ignores_stored_clean_and_maxima_and_rejects_short_actual_windows(self) -> None:
        with _commands([CUDA_IDLE, CUDA_IDLE, "3, 500, 1000, 80\n5, 0, 1000, 0\n"]):
            record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES)
        record["clean"] = True
        record["complete"] = True
        record["device_maxima"] = []
        saved = deepcopy(record)
        summary = telemetry.summarize_device_preflight(record)
        self.assertFalse(summary["clean"])
        self.assertEqual(summary["device_maxima"][0]["max_memory_usage_percent"], 50.0)
        self.assertEqual(record, saved, "summarization must not mutate evidence")
        with _commands([CUDA_IDLE] * 3):
            idle = telemetry.collect_device_preflight("cuda", CUDA_DEVICES)
        for round_record in idle["rounds"]:
            round_record["started"]["monotonic_ns"] = idle["started"]["monotonic_ns"]
            round_record["ended"]["monotonic_ns"] = idle["started"]["monotonic_ns"]
            for query in round_record["queries"]:
                query["started"]["monotonic_ns"] = idle["started"]["monotonic_ns"]
                query["ended"]["monotonic_ns"] = idle["started"]["monotonic_ns"]
        self.assertFalse(telemetry.summarize_device_preflight(idle)["clean"])

    def test_fixed_protocol_requires_three_rounds_one_second_and_bounded_queries(self) -> None:
        for kwargs in ({"samples": 2}, {"samples": True}, {"interval_seconds": 0.1},
                       {"interval_seconds": float("inf")}, {"query_timeout_seconds": 0},
                       {"query_timeout_seconds": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                telemetry.collect_device_preflight("cuda", CUDA_DEVICES, **kwargs)
        with _commands([CUDA_IDLE] * 3):
            record = telemetry.collect_device_preflight("cuda", CUDA_DEVICES)
        record["protocol"]["memory_limit_percent"] = 100.0
        self.assertFalse(telemetry.summarize_device_preflight(record)["clean"])


if __name__ == "__main__":
    unittest.main()
