"""Executable v0.9 A/B matrices, immutable attempts, and evidence validation.

Each point runs a real benchmark in a new process. Planning never marks a
point complete. Resume verifies code, model, configuration, and every saved
artifact before it reuses a successful attempt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import threading
import uuid
from typing import Any, Callable, Mapping, Sequence

from .experiment import build_model_directory_provenance, git_snapshot, sha256_file
from .inference import GenerationConfig
from .qwen3 import Qwen3Config, count_qwen3_parameters
from .qwen3_tp import Qwen3TensorParallelPlan
from .serving import RequestSpec
from .workload import WorkloadTrace


MATRIX_SCHEMA_VERSION = 1
QWEN3_32B_PARAMETERS = 32_762_123_264
FAMILY_COUNTS = {
    "ascend_a3": (2, 4, 8, 16),
    "ascend_a5": (1, 2, 4, 8),
    "cuda": (1, 2, 4, 8),
    "cpu": (1, 2, 4),
}
TERMINAL_FAILURES = {"failed", "timed_out", "interrupted", "cleanup_blocked"}
_VISIBILITY_VARIABLES = {"cuda": "CUDA_VISIBLE_DEVICES", "npu": "ASCEND_RT_VISIBLE_DEVICES"}
_METRICS = ("prefill_ms", "decode_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms", "makespan_ms",
            "output_tokens_per_second", "goodput_requests_per_second", "peak_device_memory_mb")
_IDENTITY_ENVIRONMENT = ("CANN_VERSION", "HCCL_VERSION", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH",
                         "CUDA_HOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "NCCL_ALGO", "NCCL_PROTO",
                         "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
                         "HCCL_WHITELIST_DISABLE", "GLOO_SOCKET_IFNAME", "CUBLAS_WORKSPACE_CONFIG",
                         "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NPU_ALLOC_CONF")
_SOFTWARE_CODE = """
import importlib.metadata, json, os, platform, sys
packages = {}
for name in ('torch', 'torch-npu', 'numpy', 'safetensors', 'transformers', 'tokenizers'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': platform.python_version(), 'executable': os.path.realpath(sys.executable),
                  'platform': platform.platform(), 'machine': platform.machine(), 'host': platform.node(),
                  'packages': packages}, sort_keys=True))
"""


def _public_version(value: object) -> str | None:
    # The report side records torch.__version__ while the matrix identity records
    # importlib.metadata; the two can disagree on the local segment for one install
    # (e.g. "2.10.0+cpu" vs "2.10.0"), so compare the public version.
    if value is None:
        return None
    return str(value).split("+", 1)[0]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a nonempty filename-safe identifier")
    return value


def validate_matrix_config(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported matrix config schema")
    required = {"schema_version", "matrix_id", "hardware_family", "device", "precision",
                "device_counts", "chips_per_card", "batch_size", "sessions_per_variant",
                "warmup", "repeats", "timeout_seconds", "max_seq_len", "profile", "slo",
                "workloads", "seed"}
    allowed = required | {"description", "max_session_cv", "case_selection"}
    if required - set(raw) or set(raw) - allowed:
        raise ValueError(f"invalid matrix fields: missing={sorted(required - set(raw))}, unknown={sorted(set(raw) - allowed)}")
    config = json.loads(json.dumps(raw))
    _identifier(config["matrix_id"], "matrix_id")
    family = config["hardware_family"]
    if family not in FAMILY_COUNTS:
        raise ValueError("unknown hardware_family")
    expected_device = "npu" if family.startswith("ascend") else family
    if config["device"] != expected_device:
        raise ValueError("hardware_family and device disagree")
    if config["device_counts"] != list(FAMILY_COUNTS[family]):
        raise ValueError(f"{family} matrix must declare all device counts {FAMILY_COUNTS[family]}; select execution with --case/--point")
    for count in config["device_counts"]:
        _integer(count, "device_counts")
    expected_chips = 2 if family == "ascend_a3" else 0 if family == "cpu" else 1
    if config["chips_per_card"] != expected_chips:
        raise ValueError("chips_per_card does not match the selected matrix family")
    _integer(config["chips_per_card"], "chips_per_card", 0)
    if config["precision"] != ("fp32" if family == "cpu" else "bf16"):
        raise ValueError("CPU correctness uses fp32; formal accelerator matrices use bf16")
    _integer(config["batch_size"], "batch_size", 2)
    if config["sessions_per_variant"] not in (1, 2):
        raise ValueError("sessions_per_variant must be 1 (AB) or 2 (ABBA)")
    _integer(config["sessions_per_variant"], "sessions_per_variant")
    if family != "cpu" and config["sessions_per_variant"] != 2:
        raise ValueError("hardware matrices require two independent sessions per variant in ABBA order")
    _integer(config["warmup"], "warmup", 1)
    _integer(config["repeats"], "repeats", 1 if family == "cpu" else 3)
    _integer(config["max_seq_len"], "max_seq_len")
    _integer(config["seed"], "seed", 0)
    _positive(config["timeout_seconds"], "timeout_seconds")
    _positive(config.get("max_session_cv", 0.1), "max_session_cv")
    profile = config["profile"]
    if not isinstance(profile, dict) or set(profile) != {"enabled", "skip_steps", "warmup_steps", "active_steps"}:
        raise ValueError("profile requires enabled/skip_steps/warmup_steps/active_steps")
    if type(profile["enabled"]) is not bool:
        raise ValueError("profile.enabled must be boolean")
    if family != "cpu" and not profile["enabled"]:
        raise ValueError("hardware matrices require separate profile replay")
    for field in ("skip_steps", "warmup_steps"):
        _integer(profile[field], f"profile.{field}", 0)
    _integer(profile["active_steps"], "profile.active_steps")
    if profile["enabled"] and (profile["skip_steps"] != 0 or profile["warmup_steps"] != 0 or profile["active_steps"] < 2):
        raise ValueError("matrix profile replay must capture Prefill step 0 and at least one Decode step; use skip=0, warmup=0, active>=2")
    slo = config["slo"]
    if not isinstance(slo, dict) or set(slo) != {"ttft_ms", "tpot_ms", "e2e_ms"}:
        raise ValueError("slo requires ttft_ms/tpot_ms/e2e_ms")
    for field, value in slo.items():
        _positive(value, f"slo.{field}")
    workloads = config["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be nonempty")
    seen: set[str] = set()
    for workload in workloads:
        if not isinstance(workload, dict) or set(workload) != {"id", "workload_class", "prompts", "prompt_repeat", "max_new_tokens"}:
            raise ValueError("invalid workload specification")
        name = _identifier(workload["id"], "workload.id")
        if name in seen:
            raise ValueError("duplicate workload id")
        seen.add(name)
        if workload["workload_class"] not in {"short_short", "long_prefill_short_decode", "mixed"}:
            raise ValueError("unsupported workload_class")
        if not isinstance(workload["prompts"], list) or not workload["prompts"] or not all(isinstance(value, str) and value for value in workload["prompts"]):
            raise ValueError("workload prompts must be nonempty strings")
        _integer(workload["prompt_repeat"], "prompt_repeat")
        _integer(workload["max_new_tokens"], "max_new_tokens", 2)
        required_steps = sum(profile[key] for key in ("skip_steps", "warmup_steps", "active_steps"))
        if profile["enabled"] and workload["max_new_tokens"] < required_steps:
            raise ValueError("fixed output length is shorter than the requested profile window")
    if family != "cpu" and not {"short_short", "long_prefill_short_decode"}.issubset({row["workload_class"] for row in workloads}):
        raise ValueError("hardware matrix needs both short and long-prefill workloads")
    selection = config.get("case_selection")
    if selection is not None:
        if not isinstance(selection, list) or not selection:
            raise ValueError("case_selection must be a nonempty list")
        for case_id in selection:
            _identifier(case_id, "case_selection")
        if len(set(selection)) != len(selection):
            raise ValueError("case_selection contains duplicate case IDs")
    return config


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    return validate_matrix_config(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MatrixPoint:
    point_id: str
    case_id: str
    axis: str
    role: str
    session: int
    workload_id: str
    batch_size: int
    tp_size: int
    entrypoint: str
    decode_mode: str

    @property
    def workload_key(self) -> str:
        return f"{self.workload_id}-n{self.batch_size}"


def expand_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_matrix_config(dict(config))
    points: list[MatrixPoint] = []
    cases: list[dict[str, Any]] = []
    order = ("baseline", "candidate") if config["sessions_per_variant"] == 1 else ("baseline", "candidate", "candidate", "baseline")

    def add_case(case_id: str, axis: str, workload_id: str, batch_size: int,
                 baseline: tuple[int, str, str], candidate: tuple[int, str, str]) -> None:
        counts = {"baseline": 0, "candidate": 0}
        ids: list[str] = []
        for role in order:
            index = counts[role]
            counts[role] += 1
            size, entrypoint, decode_mode = baseline if role == "baseline" else candidate
            point_id = f"{case_id}-{'a' if role == 'baseline' else 'b'}{index}"
            point = MatrixPoint(point_id, case_id, axis, role, index, workload_id, batch_size, size, entrypoint, decode_mode)
            points.append(point)
            ids.append(point_id)
        cases.append({"case_id": case_id, "axis": axis, "workload_id": workload_id,
                      "batch_size": batch_size, "point_ids": ids,
                      "phase_metrics": ["prefill_ms", "decode_ms"] if axis in {"kv", "tp"} else []})

    for workload in config["workloads"]:
        name = workload["id"]
        for size in config["device_counts"]:
            add_case(f"kv-{name}-tp{size}", "kv", name, 1,
                     (size, "tp", "recompute"), (size, "tp", "kv_cache"))
            add_case(f"batching-{name}-tp{size}", "batching", name, config["batch_size"],
                     (size, "tp", "kv_cache"), (size, "continuous", "kv_cache"))
        baseline_size = config["device_counts"][0]
        for size in config["device_counts"][1:]:
            add_case(f"tp-{name}-{baseline_size}-vs-{size}", "tp", name, 1,
                     (baseline_size, "tp", "kv_cache"), (size, "tp", "kv_cache"))
    if "case_selection" in config:
        cases_by_id = {case["case_id"]: case for case in cases}
        unknown = set(config["case_selection"]) - set(cases_by_id)
        if unknown:
            raise ValueError(f"unknown case_selection IDs: {sorted(unknown)}")
        cases = [cases_by_id[case_id] for case_id in config["case_selection"]]
        points_by_id = {point.point_id: point for point in points}
        points = [points_by_id[point_id] for case in cases for point_id in case["point_ids"]]
    required_counts = sorted({point.tp_size for point in points})
    return {"schema_version": MATRIX_SCHEMA_VERSION, "matrix_id": config["matrix_id"],
            "hardware_family": config["hardware_family"], "device_counts": config["device_counts"],
            "required_device_counts": required_counts,
            "session_order": list(order), "cases": cases, "points": [asdict(point) for point in points]}


def make_workload(config: Mapping[str, Any], point: MatrixPoint) -> WorkloadTrace:
    specification = next(row for row in config["workloads"] if row["id"] == point.workload_id)
    generation = GenerationConfig(max_new_tokens=specification["max_new_tokens"], strategy="greedy", seed=config["seed"])
    prompts = specification["prompts"]
    return WorkloadTrace(
        workload_id=point.workload_key, workload_class=specification["workload_class"],
        requests=tuple(RequestSpec(request_id=f"{point.workload_id}-{index:04d}",
                                   prompt=prompts[index % len(prompts)] * specification["prompt_repeat"],
                                   config=generation, arrival_time_ms=0.0)
                       for index in range(point.batch_size)),
        metadata={"generator": "v09_fixed_work_matrix", "eos_policy": "fixed_output_length",
                  "arrival_policy": "all_requests_ready_at_start"},
    )


def load_device_map(path: str | Path | None, config: Mapping[str, Any]) -> list[dict[str, int | None]]:
    required_counts = expand_matrix(config)["required_device_counts"]
    count = max(required_counts)
    if config["device"] == "cpu":
        if path is not None:
            raise ValueError("CPU correctness runs use process ranks, not a physical device map")
        return [{"logical_device_id": index, "physical_card_id": None, "chip_id": None} for index in range(count)]
    if path is None:
        raise ValueError("accelerator execution requires an explicit --device-map JSON")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1 or not isinstance(raw.get("devices"), list):
        raise ValueError("invalid device-map schema")
    devices = raw["devices"]
    if len(devices) < count:
        raise ValueError(f"device map must contain at least {count} logical devices in launch order")
    # A verified full-machine map may be reused by a compact acceptance preset.
    # Only the prefix required by this plan becomes part of the run identity.
    devices = devices[:count]
    logical: set[int] = set()
    targets: set[tuple[int, int]] = set()
    for entry in devices:
        if not isinstance(entry, dict) or set(entry) != {"logical_device_id", "physical_card_id", "chip_id"}:
            raise ValueError("each mapped device requires logical_device_id/physical_card_id/chip_id")
        for field, value in entry.items():
            _integer(value, field, 0)
        if config["device"] == "cuda" and (entry["physical_card_id"] != entry["logical_device_id"] or entry["chip_id"] != 0):
            raise ValueError("CUDA device map requires full-GPU host NVML indices as both logical and physical IDs, with chip_id=0")
        if entry["logical_device_id"] in logical or (entry["physical_card_id"], entry["chip_id"]) in targets:
            raise ValueError("device map contains duplicate logical or physical targets")
        logical.add(entry["logical_device_id"])
        targets.add((entry["physical_card_id"], entry["chip_id"]))
    for size in required_counts:
        selected = devices[:size]
        cards = {entry["physical_card_id"] for entry in selected}
        if len(cards) * config["chips_per_card"] != size:
            raise ValueError("each device-count prefix must contain whole physical cards")
        for card in cards:
            chips = {entry["chip_id"] for entry in selected if entry["physical_card_id"] == card}
            if chips != set(range(config["chips_per_card"])):
                raise ValueError("device map chip indices do not match chips_per_card")
    return devices


def _source_identity(project_root: Path) -> dict[str, Any]:
    paths = sorted((project_root / "src" / "minigpt").rglob("*.py"))
    paths += [project_root / "benchmarks" / name for name in (
        "infer_qwen3_tp.py", "infer_qwen3_continuous_batching.py", "run_v09_matrix.py",
        "summarize_v09_matrix.py")]
    paths += [project_root / "pyproject.toml"]
    files = {path.relative_to(project_root).as_posix(): sha256_file(path) for path in paths}
    return {"git": git_snapshot(project_root), "files_sha256": files, "content_sha256": canonical_sha256(files)}


def _model_identity(project_root: Path, model_dir: Path) -> dict[str, Any]:
    provenance = build_model_directory_provenance(project_root, model_dir, [], hash_weights=True)
    config = Qwen3Config.from_json(model_dir / "config.json")
    result = {key: provenance[key] for key in ("config_sha256", "metadata_sha256", "weights", "index_sha256")}
    result["config"] = asdict(config)
    result["full_parameter_count"] = count_qwen3_parameters(config)
    result["is_formal_model"] = result["full_parameter_count"] == QWEN3_32B_PARAMETERS and not (model_dir / "fixture_manifest.json").exists()
    return result


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _software_identity(python_executable: str, env: Mapping[str, str], project_root: Path) -> dict[str, Any]:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.run([python_executable, "-c", _SOFTWARE_CODE], cwd=project_root, env=dict(env),
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                             timeout=30.0, **options)
    if process.returncode != 0:
        raise ValueError("cannot establish selected Python interpreter/dependency identity: " + process.stderr[-2000:])
    result = json.loads(process.stdout)
    result["environment"] = {key: env.get(key) for key in _IDENTITY_ENVIRONMENT}
    return result


def launch_environment(config: Mapping[str, Any], point: MatrixPoint,
                       devices: Sequence[Mapping[str, Any]], project_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    overrides = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1",
                 "PYTHONDONTWRITEBYTECODE": "1"}
    if config["device"] != "cpu":
        overrides[_VISIBILITY_VARIABLES[config["device"]]] = ",".join(str(row["logical_device_id"]) for row in devices[:point.tp_size])
    if config["device"] == "cuda":
        overrides["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if config["device"] == "cpu" and os.name != "nt" and "GLOO_SOCKET_IFNAME" not in os.environ:
        overrides["GLOO_SOCKET_IFNAME"] = "lo"
    if os.name == "nt":
        overrides["USE_LIBUV"] = "0"
    if point.tp_size == 1:
        overrides.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
    env = dict(os.environ)
    env.update(overrides)
    env["PYTHONPATH"] = str(project_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env, overrides


def build_command(config: Mapping[str, Any], point: MatrixPoint, *, project_root: Path,
                  model_dir: Path, workload_path: Path, output_dir: Path, python_executable: str,
                  devices: Sequence[Mapping[str, Any]], interconnect_topology: str, port: int) -> list[str]:
    selected = devices[:point.tp_size]
    cards = 0 if config["device"] == "cpu" else len({row["physical_card_id"] for row in selected})
    command = [python_executable]
    if point.tp_size > 1:
        command += ["-m", "torch.distributed.run", "--nnodes=1",
                    f"--nproc-per-node={point.tp_size}", "--master-addr=127.0.0.1", f"--master-port={port}"]
    entrypoint = "infer_qwen3_tp.py" if point.entrypoint == "tp" else "infer_qwen3_continuous_batching.py"
    command += [str(project_root / "benchmarks" / entrypoint),
                "--model-dir", str(model_dir), "--workload", str(workload_path),
                "--device", config["device"], "--precision", config["precision"],
                "--backend", {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[config["device"]],
                "--warmup", str(config["warmup"]), "--repeats", str(config["repeats"]),
                "--hash-weights", "--logical-device-ids", ",".join(str(row["logical_device_id"]) for row in selected),
                "--physical-card-count", str(cards), "--chips-per-card", str(config["chips_per_card"]),
                "--interconnect-topology", interconnect_topology, "--run-label", point.point_id,
                "--distributed-timeout-seconds", str(max(30, min(int(config["timeout_seconds"]), 600)))]
    if config["device"] == "npu" and os.environ.get("CANN_VERSION"):
        command += ["--cann-version", os.environ["CANN_VERSION"]]
    if point.entrypoint == "tp":
        command += ["--decode-mode", point.decode_mode, "--output", str(output_dir / "report.json")]
    else:
        command += ["--mode", "closed_loop", "--closed-loop-clients", str(point.batch_size),
                    "--tp-size", str(point.tp_size), "--max-slots", str(point.batch_size),
                    "--max-seq-len", str(config["max_seq_len"]), "--max-queue-size", str(max(16, point.batch_size)),
                    "--layout-id", point.point_id, "--ttft-slo-ms", str(config["slo"]["ttft_ms"]),
                    "--tpot-slo-ms", str(config["slo"]["tpot_ms"]), "--e2e-slo-ms", str(config["slo"]["e2e_ms"]),
                    "--greedy-token-path", "full_gather", "--output-dir", str(output_dir)]
    if config["profile"]["enabled"]:
        command += ["--profile", "--profile-format", "portable", "--profile-ranks", "all",
                    "--profile-skip-steps", str(config["profile"]["skip_steps"]),
                    "--profile-warmup-steps", str(config["profile"]["warmup_steps"]),
                    "--profile-active-steps", str(config["profile"]["active_steps"])]
    return command


class ProcessCleanupBlocked(RuntimeError):
    """The Linux supervisor remains alive and owns the lock until cleanup finishes."""


def _linux_process_identity(pid: int) -> dict[str, Any] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = data[data.rfind(")") + 2:].split()
    return {"pid": pid, "birth_marker": fields[19], "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]), "session_id": int(fields[3]), "state": fields[0]}


def _linux_owned_children(pid: int, *, required: bool = False) -> set[int]:
    """Read only this already-owned process's task children, never a global PID scan."""
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        if required:
            raise RuntimeError("cannot observe the live Linux supervisor's task children")
        return set()
    children: set[int] = set()
    observed_main_task = False
    for task in tasks:
        try:
            children.update(int(value) for value in (task / "children").read_text(encoding="utf-8").split())
            observed_main_task |= task.name == str(pid)
        except (FileNotFoundError, ProcessLookupError):
            if required and task.name == str(pid):
                raise RuntimeError("cannot read the live Linux supervisor's main-task children")
            continue
    if required and not observed_main_task:
        raise RuntimeError("the Linux supervisor's main task is not observable")
    return children


class _OwnedLinuxProcesses:
    """A subreaper's descendants, including children adopted after a launcher exits."""

    def __init__(self) -> None:
        self.guard_pid = os.getpid()
        self.guard_birth = _process_birth(self.guard_pid)
        if self.guard_birth is None:
            raise RuntimeError("Linux supervision requires a positive supervisor process identity")
        _linux_owned_children(self.guard_pid, required=True)
        self.identities: dict[tuple[int, str], dict[str, Any]] = {}
        self.pidfds: dict[tuple[int, str], int] = {}

    def refresh(self, launcher: subprocess.Popen[Any] | None) -> list[dict[str, Any]]:
        guard = _linux_process_identity(self.guard_pid)
        if guard is None or guard["birth_marker"] != self.guard_birth:
            raise RuntimeError("cannot verify the live Linux supervisor process identity")
        if launcher is not None:
            launcher.poll()  # Reap the direct Popen child through its own API.
        parents = [(self.guard_pid, self.guard_birth)]
        parents.extend((pid, birth) for (pid, birth) in self.identities
                       if (current := _linux_process_identity(pid)) is not None and current["birth_marker"] == birth)
        visited: set[tuple[int, str]] = set()
        while parents:
            parent, parent_birth = parents.pop()
            current_parent = _linux_process_identity(parent)
            if current_parent is None or current_parent["birth_marker"] != parent_birth or (parent, parent_birth) in visited:
                continue
            visited.add((parent, parent_birth))
            for pid in _linux_owned_children(parent, required=parent == self.guard_pid):
                current = _linux_process_identity(pid)
                parent_after = _linux_process_identity(parent)
                if (current is None or current["parent_pid"] != parent or parent_after is None
                        or parent_after["birth_marker"] != parent_birth):
                    continue  # A concurrent orphan will be rediscovered under this subreaper.
                key = (pid, current["birth_marker"])
                if key not in self.identities:
                    self.identities[key] = {name: value for name, value in current.items() if name != "state"}
                    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                        try:
                            self.pidfds[key] = os.pidfd_open(pid)
                        except (OSError, ProcessLookupError):
                            pass  # Older vendor kernels use the verified birth-marker path below.
                parents.append(key)
        live: list[dict[str, Any]] = []
        for key, identity in self.identities.items():
            pid, birth = key
            current = _linux_process_identity(pid)
            if current is None or current["birth_marker"] != birth:
                continue
            if current["state"] in {"Z", "X"}:
                if current["parent_pid"] == self.guard_pid and (launcher is None or pid != launcher.pid):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except (ChildProcessError, ProcessLookupError):
                        pass
                continue
            live.append(identity)
        return live

    def send(self, identity: Mapping[str, Any], signum: int) -> None:
        pid, birth = identity["pid"], identity["birth_marker"]
        current = _linux_process_identity(pid)
        if current is None or current["birth_marker"] != birth or current["state"] in {"Z", "X"}:
            return
        try:
            descriptor = self.pidfds.get((pid, birth))
            if descriptor is not None:
                signal.pidfd_send_signal(descriptor, signum)
            else:
                os.kill(pid, signum)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        for descriptor in self.pidfds.values():
            os.close(descriptor)


def _linux_guard_main(request_path: str) -> None:
    """Own one launcher tree until every process exits, regardless of worker sessions.

    This helper is deliberately separate from the matrix controller. It inherits
    the matrix lock and receives SIGTERM when its controller dies, including a
    controller SIGKILL. A killed launcher reparents its workers to this subreaper.
    The controller must never SIGKILL this helper during incomplete cleanup.
    """
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    state_path = Path(request["state_path"])
    state: dict[str, Any] = {"schema_version": 1, "kind": "linux_child_subreaper",
                             "guard": _linux_process_identity(os.getpid()), "controller": request["controller"],
                             "command": request["command"], "status": "starting", "cleanup_complete": False,
                             "launcher": None, "processes": [], "live_processes": [], "errors": []}
    _save(state_path, state)  # No child may be created before this durable registration.
    requested_stop: list[int] = []

    def stop(signum, frame):
        requested_stop.append(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if state["guard"] is None or not state["guard"].get("birth_marker") or not request["controller"].get("birth_marker"):
            raise RuntimeError("Linux supervision requires positive guard and controller process identities")
        _linux_owned_children(os.getpid(), required=True)
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = (ctypes.c_int,) + (ctypes.c_ulong,) * 4
        libc.prctl.restype = ctypes.c_int
        for option, value in ((36, 1), (1, int(signal.SIGTERM))):  # CHILD_SUBREAPER / PDEATHSIG
            if libc.prctl(option, value, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot establish Linux child supervision")
        # The controller blocks these signals only across Popen ownership transfer.
        # Neither the actual launcher nor its workers may inherit that block.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
        controller = request["controller"]
        if os.getppid() != controller["pid"] or _process_birth(controller["pid"]) != controller["birth_marker"]:
            requested_stop.append(int(signal.SIGTERM))
    except BaseException as exc:
        state.update(status="setup_failed", cleanup_complete=True, errors=[f"{type(exc).__name__}: {exc}"])
        _save(state_path, state)
        raise SystemExit(125)

    owned = _OwnedLinuxProcesses()
    launcher: subprocess.Popen[Any] | None = None
    cleanup_started: float | None = None
    launcher_exited: float | None = None
    orphaned = False
    failure = False
    last_saved: str | None = None
    signaled: set[tuple[int, str, int]] = set()

    def persist() -> None:
        nonlocal failure
        try:
            _save(state_path, state)
        except OSError as exc:
            # Keep supervising and cleaning even when the evidence volume fails.
            # The durable pre-launch registration remains incomplete, so reuse
            # fails closed if the final complete record cannot be written.
            failure = True
            requested_stop.append(int(signal.SIGTERM))
            message = f"supervisor journal write failed: {exc}"
            if message not in state["errors"]:
                state["errors"].append(message)

    try:
        if not requested_stop:
            state["status"] = "launching"
            _save(state_path, state)
            try:
                launcher = subprocess.Popen(request["command"], cwd=request["cwd"], stdin=subprocess.DEVNULL,
                                            start_new_session=True)
                state["launcher"] = _linux_process_identity(launcher.pid)
            except BaseException as exc:
                state["errors"].append(f"launcher creation failed: {type(exc).__name__}: {exc}")
                failure = True
                requested_stop.append(int(signal.SIGTERM))
        while True:
            now = time.monotonic()
            try:
                live = owned.refresh(launcher)
            except BaseException as exc:
                # Inability to establish ownership is never permission to release
                # the lock or fall back to signaling a guessed/global process set.
                state.update(status="cleanup_blocked", cleanup_complete=False)
                message = f"process ownership unavailable: {type(exc).__name__}: {exc}"
                if message not in state["errors"]:
                    state["errors"].append(message)
                    persist()
                requested_stop.append(int(signal.SIGTERM))
                time.sleep(0.2)
                continue
            if launcher is not None and launcher.returncode is not None and launcher_exited is None:
                launcher_exited = now
            if launcher_exited is not None and live and now - launcher_exited >= 0.2:
                orphaned = True
                requested_stop.append(int(signal.SIGTERM))
            if requested_stop and cleanup_started is None:
                cleanup_started = now
            state.update(processes=list(owned.identities.values()), live_processes=live,
                         launcher_exit_code=None if launcher is None else launcher.returncode,
                         orphaned_workers=orphaned, stop_requested=bool(requested_stop))
            # A launcher that exits early cannot make its adopted workers vanish
            # from this check: they remain direct children of the subreaper.
            if not live and (launcher is None or launcher.returncode is not None):
                # A parent may have died after the breadth-first pass visited
                # this guard, adopting a previously unseen grandchild. Check
                # the kernel child list again before declaring the tree empty.
                # ECHILD is the kernel's proof that no adopted child remains;
                # an empty /proc observation alone is insufficient.
                children_remain = False
                while True:
                    try:
                        child_pid, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if child_pid == 0:
                        children_remain = True
                        break
                if children_remain:
                    requested_stop.append(int(signal.SIGTERM))
                    state["status"] = "cleanup_blocked"
                    persist()
                    time.sleep(0.05)
                    continue
                state.update(status="complete", cleanup_complete=True, completed_at_utc=_utc())
                persist()
                break
            if cleanup_started is not None:
                signum = signal.SIGKILL if now - cleanup_started >= 5.0 else signal.SIGTERM
                state["status"] = "cleanup_blocked" if now - cleanup_started >= 15.0 else "stopping"
                for identity in reversed(live):
                    key = (identity["pid"], identity["birth_marker"], int(signum))
                    if key not in signaled:
                        try:
                            owned.send(identity, signum)
                            signaled.add(key)
                        except OSError as exc:
                            message = f"owned process signal failed: {identity['pid']}: {exc}"
                            if message not in state["errors"]:
                                state["errors"].append(message)
            else:
                state["status"] = "running"
            encoded = canonical_sha256(state)
            if encoded != last_saved:
                persist()
                last_saved = encoded
            time.sleep(0.05 if cleanup_started is not None else 0.2)
    finally:
        owned.close()
    # Preserve actual launcher failures (for example 7 or 23) instead of turning
    # every child failure into a generic supervisor exit 1.
    code = launcher.returncode if launcher is not None and not (requested_stop or orphaned or failure) else 1
    if code is None or code < 0:
        code = 1
    raise SystemExit(code)


def _guard_registration(log_path: Path, lock_fd: int | None, command: Sequence[str],
                         project_root: Path) -> tuple[Path, Path]:
    matrix_root = Path(os.readlink(f"/proc/self/fd/{lock_fd}")).parent if lock_fd is not None else log_path.parent
    directory = matrix_root / ".process_guards"
    directory.mkdir(parents=True, exist_ok=True)
    key = uuid.uuid4().hex
    request_path, state_path = directory / f"{key}.request.json", directory / f"{key}.state.json"
    controller_birth = _process_birth(os.getpid())
    if controller_birth is None:
        raise RuntimeError("Linux supervision requires an observable controller process identity")
    _save(request_path, {"schema_version": 1, "command": list(command), "cwd": str(project_root),
                         "controller": {"pid": os.getpid(), "birth_marker": controller_birth},
                         "state_path": str(state_path)})
    return request_path, state_path


def _guard_snapshot(state_path: Path, log_path: Path) -> dict[str, Any] | None:
    if not state_path.is_file():
        return None  # The guard has not yet registered, and cannot have launched a child.
    snapshot = json.loads(state_path.read_text(encoding="utf-8"))
    _save(log_path.with_suffix(".process_tree.json"), snapshot)
    return snapshot


def _terminate_tree(process: subprocess.Popen[Any], *, guard_state: Path | None = None) -> None:
    if guard_state is not None:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired as exc:
                # The guard must keep its inherited lock while any worker remains.
                raise ProcessCleanupBlocked("Linux process cleanup is still active; its supervisor retains the matrix lock") from exc
        snapshot = json.loads(guard_state.read_text(encoding="utf-8")) if guard_state.exists() else None
        if snapshot is not None and snapshot.get("cleanup_complete") is not True:
            raise ProcessCleanupBlocked("Linux supervisor exited without proving all owned workers stopped; resume remains blocked")
        return
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=15)


def _execute(command: Sequence[str], *, env: Mapping[str, str], project_root: Path,
             log_path: Path, timeout_seconds: float, lock_fd: int | None = None,
             on_started: Callable[[int], None] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    started_unix_ns = time.time_ns()
    options: dict[str, Any] = {"cwd": project_root, "env": dict(env), "stdin": subprocess.DEVNULL}
    guard_state: Path | None = None
    launched_command = list(command)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
        if lock_fd is not None:
            options["pass_fds"] = (lock_fd,)
        if sys.platform.startswith("linux"):
            request_path, guard_state = _guard_registration(log_path, lock_fd, command, project_root)
            launched_command = [sys.executable, "-c",
                                "import sys; sys.path.insert(0, sys.argv[1]); from minigpt.experiment_matrix import _linux_guard_main; _linux_guard_main(sys.argv[2])",
                                str(Path(__file__).resolve().parents[1]), str(request_path)]
    with log_path.open("wb") as log:
        process: subprocess.Popen[Any] | None = None
        old_mask = None
        status = "succeeded"
        try:
            # Ownership must be assigned inside the exception boundary. On Linux
            # also defer INT/TERM while Popen itself transfers that ownership.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}) if guard_state is not None else None
            try:
                process = subprocess.Popen(launched_command, stdout=log, stderr=subprocess.STDOUT, **options)
            finally:
                if old_mask is not None:
                    restore_mask, old_mask = old_mask, None
                    signal.pthread_sigmask(signal.SIG_SETMASK, restore_mask)
            if on_started is not None:
                on_started(process.pid)
            returncode = process.wait(timeout=timeout_seconds)
            if guard_state is not None:
                snapshot = _guard_snapshot(guard_state, log_path)
                if snapshot is not None and snapshot.get("cleanup_complete") is not True:
                    raise ProcessCleanupBlocked("Linux supervisor stopped with unconfirmed worker cleanup; resume remains blocked")
                if (snapshot is not None and snapshot.get("launcher_exit_code") is not None
                        and not snapshot.get("stop_requested") and not snapshot.get("orphaned_workers")):
                    returncode = snapshot["launcher_exit_code"]
            if returncode != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            assert process is not None
            _terminate_tree(process, guard_state=guard_state)
            status, returncode = "timed_out", process.returncode
        except BaseException:
            if process is not None:
                _terminate_tree(process, guard_state=guard_state)
            raise
        finally:
            try:
                if guard_state is not None:
                    _guard_snapshot(guard_state, log_path)
            finally:
                # A Python exception can occur at the first line after Popen,
                # even before the inner finally restores its deferred signals.
                if old_mask is not None:
                    restore_mask, old_mask = old_mask, None
                    signal.pthread_sigmask(signal.SIG_SETMASK, restore_mask)
    return {"status": status, "exit_code": returncode, "elapsed_seconds": time.monotonic() - started,
            "process_started_at_unix_ns": started_unix_ns, "process_ended_at_unix_ns": time.time_ns()}


@contextmanager
def _matrix_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "matrix_runner.lock").open("a+b")
    locked = False
    try:
        if stream.tell() == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("another matrix controller or its benchmark process still owns this output directory") from exc
        # A guard unexpectedly killed by an external actor cannot certify that
        # its last snapshot included every just-forked worker. Refuse reuse even
        # if the OS lock itself was released; never guess that the tree is empty.
        for path in (root / ".process_guards").glob("*.state.json"):
            registration = json.loads(path.read_text(encoding="utf-8"))
            if registration.get("cleanup_complete") is not True:
                raise RuntimeError("previous Linux supervisor has not confirmed worker cleanup; matrix resume is blocked")
        yield stream.fileno()
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                # Do not explicitly unlock: an orphan child may still hold this
                # shared open-file description after an interrupted controller.
                pass
        stream.close()


@contextmanager
def _handle_termination():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("matrix controller received SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _process_birth(pid: int) -> str | None:
    """Detect a surviving owned process without signaling a possibly reused PID."""
    if os.name != "nt":
        try:
            data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            tail = data[data.rfind(")") + 2:].split()
            return None if tail[0] == "Z" else tail[19]
        except FileNotFoundError:
            return None
    else:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return None
            raise RuntimeError("cannot verify whether a previous benchmark process has exited")
        try:
            exit_code = wintypes.DWORD()
            stamps = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or not kernel.GetProcessTimes(handle, *(ctypes.byref(stamp) for stamp in stamps)):
                raise RuntimeError("cannot read previous benchmark process identity")
            return str((stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime) if exit_code.value == 259 else None
        finally:
            kernel.CloseHandle(handle)


_PROBE_CODE = """
import json, sys
from pathlib import Path
from minigpt.runtime import RuntimeContext
runtime = RuntimeContext.create(sys.argv[1], sys.argv[2], device_index=0, allow_accelerator_fallback=False, allow_precision_fallback=False)
count = runtime.visible_device_count()
expected = int(sys.argv[3])
if runtime.device.type != 'cpu' and count != expected:
    raise RuntimeError(f'visible device count {count} does not match selected mapping {expected}')
payload = {'environment': runtime.backend_metadata(), 'visible_device_count': count, 'expected_process_count': expected,
           'memory': runtime.memory_snapshot().to_dict(), 'visibility_count_verified': True}
Path(sys.argv[4]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
"""


def artifact(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes matrix directory")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def verify_artifacts(root: Path, artifacts: Sequence[Mapping[str, Any]]) -> None:
    if not artifacts:
        raise ValueError("successful attempt has no hash-bound artifacts")
    seen: set[str] = set()
    for row in artifacts:
        value = row.get("path")
        if not isinstance(value, str) or not value or value in seen or Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value or ":" in value:
            raise ValueError("invalid or duplicate matrix artifact path")
        seen.add(value)
        path = root / value
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"matrix artifact missing or outside root: {value}")
        if type(row.get("size_bytes")) is not int or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"matrix artifact hash mismatch: {value}")


def _attempt_times(attempt: Mapping[str, Any]) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(attempt["started_at_utc"])
    end = datetime.fromisoformat(attempt["ended_at_utc"])
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError("invalid successful attempt timestamps")
    return start, end


def _verified_attempt(root: Path, point: MatrixPoint, config: Mapping[str, Any],
                      identity: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    if attempt.get("status") != "succeeded" or attempt.get("exit_code") != 0 or attempt.get("preflight", {}).get("status") != "succeeded":
        raise ValueError("saved process outcome did not succeed")
    _attempt_times(attempt)
    attempt_number = _integer(attempt.get("attempt"), "attempt", 0)
    attempt_dir = root / "points" / point.point_id / f"attempt-{attempt_number:03d}"
    workload_path = root / "workloads" / f"{point.workload_key}.json"
    verify_artifacts(root, attempt["artifacts"])
    required_paths = {path.relative_to(root).as_posix() for path in attempt_dir.rglob("*") if path.is_file()}
    required_paths.add(workload_path.relative_to(root).as_posix())
    if {row["path"] for row in attempt["artifacts"]} != required_paths:
        raise ValueError("attempt artifact manifest does not cover exactly its complete raw files and source workload")
    if attempt.get("software") != identity["software"]:
        raise ValueError("attempt interpreter or dependency identity differs from the matrix")
    result = inspect_point_output(point, config, attempt_dir / "output", workload_path, identity["model"], identity=identity)
    probe = json.loads((attempt_dir / "runtime_probe.json").read_text(encoding="utf-8"))
    if probe.get("expected_process_count") != point.tp_size or probe.get("visibility_count_verified") is not True:
        raise ValueError("runtime probe does not verify the planned device count")
    if probe.get("visible_device_count") != (0 if config["device"] == "cpu" else point.tp_size):
        raise ValueError("runtime probe observed a different accelerator count")
    if not isinstance(probe.get("environment"), dict) or not {"device_type", "precision", "torch", "device_name"}.issubset(probe["environment"]):
        raise ValueError("runtime probe is missing its actual backend environment")
    if any(result["environment"].get(key) != value for key, value in probe.get("environment", {}).items()):
        raise ValueError("runtime probe and actual benchmark backend environments disagree")
    selected_devices = identity["devices"][:point.tp_size]
    if config["device"] == "cuda" and attempt.get("environment_overrides", {}).get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("CUDA visibility order was not bound to the device preflight index order")
    if config["device"] != "cpu":
        from .backends.telemetry import summarize_device_preflight
        preflight = json.loads((attempt_dir / "device_preflight.json").read_text(encoding="utf-8"))
        if preflight.get("device_type") != config["device"] or preflight.get("devices") != selected_devices:
            raise ValueError("device idle preflight is not bound to the selected physical map")
        start, end = _attempt_times(attempt)
        if not (int(start.timestamp() * 1e9) <= preflight["started"]["unix_ns"]
                <= preflight["ended"]["unix_ns"] <= attempt["process_started_at_unix_ns"]
                <= attempt["process_ended_at_unix_ns"] <= int(end.timestamp() * 1e9) + 1000):
            raise ValueError("device preflight was not collected within this attempt before benchmark launch")
        if summarize_device_preflight(preflight).get("clean") is not True:
            raise ValueError("device idle preflight is incomplete or reports device contention")
    if result != attempt.get("result"):
        raise ValueError("saved point summary differs from verified raw reports")
    return result


def _workload_hash(report: Mapping[str, Any]) -> str | None:
    for name in ("workload", "workload_fingerprint"):
        source = report.get(name, {})
        if isinstance(source, dict):
            for key in ("source_file_sha256", "source_workload_file_sha256", "workload_file_sha256"):
                if source.get(key):
                    return str(source[key])
    return None


def _all_requests(report: Mapping[str, Any], run: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(run.get("serving"), dict):
        return list(run["serving"].get("requests", []))
    if isinstance(run.get("requests"), list):
        return list(run["requests"])
    result = report.get("result", {})
    if isinstance(result.get("requests"), list):
        return list(result["requests"])
    return []


def _measured_peak_mib(memory: Mapping[str, Any], runs: Sequence[Mapping[str, Any]],
                       config: Mapping[str, Any], point: MatrixPoint, logical_ids: Sequence[int]) -> float | None:
    supported = config["device"] != "cpu"
    if (memory.get("supported") is not supported or memory.get("unit") != "bytes"
            or memory.get("scope") != "measured_repeats"
            or memory.get("source") != ("pytorch_allocator" if supported else "unavailable")
            or memory.get("measurement_type") != ("measured" if supported else "unsupported")):
        raise ValueError("memory evidence lacks measured allocator bytes or explicit unsupported metadata")
    ranks = memory.get("per_rank")
    if not isinstance(ranks, list) or len(ranks) != point.tp_size:
        raise ValueError("memory evidence lacks exact TP-rank coverage")
    fields = ("allocated_bytes", "peak_allocated_bytes", "reserved_bytes", "peak_reserved_bytes", "total_bytes")

    def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("supported") is not supported:
            raise ValueError("memory snapshot capability does not match the selected device")
        for field in fields:
            value = snapshot.get(field)
            if supported and not (field in {"reserved_bytes", "peak_reserved_bytes"} and value is None):
                _integer(value, field, 0)
            elif not supported and value is not None:
                raise ValueError("unsupported CPU device memory must be null")
        if supported and (snapshot["peak_allocated_bytes"] < snapshot["allocated_bytes"]
                          or (snapshot["peak_reserved_bytes"] is not None and snapshot["reserved_bytes"] is not None
                              and snapshot["peak_reserved_bytes"] < snapshot["reserved_bytes"])
                          or snapshot["total_bytes"] <= 0):
            raise ValueError("memory snapshot peaks/totals are inconsistent")

    for index, rank in enumerate(ranks):
        if rank.get("rank") != index or rank.get("logical_device_id") != logical_ids[index]:
            raise ValueError("memory rank mapping differs from actual device visibility")
        validate_snapshot(rank)
    snapshots = [run.get("memory_snapshot", {}) for run in runs]
    for snapshot in snapshots:
        validate_snapshot(snapshot)
    for field in fields:
        numbers = [snapshot[field] for snapshot in snapshots]
        expected = ((max(numbers) if field.startswith("peak_") else numbers[-1])
                    if supported and None not in numbers else None)
        if ranks[0].get(field) != expected:
            raise ValueError(f"primary rank memory {field} differs from the measured raw snapshots")
    peaks = [rank["peak_allocated_bytes"] for rank in ranks]
    expected_max, expected_sum = (max(peaks), sum(peaks)) if supported else (None, None)
    if memory.get("max_rank_peak_allocated_bytes") != expected_max or memory.get("sum_rank_peak_allocated_bytes") != expected_sum:
        raise ValueError("saved memory aggregates differ from exact per-rank allocator bytes")
    return expected_max / (1024 * 1024) if supported else None


def inspect_point_output(point: MatrixPoint, config: Mapping[str, Any], output_dir: Path,
                         workload_path: Path, expected_model: Mapping[str, Any], *,
                         identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report_paths = [output_dir / "report.json"] if point.entrypoint == "tp" else sorted(output_dir.glob("replica-*.json"))
    if len(report_paths) != 1:
        raise ValueError("this matrix requires exactly one TP replica report per point")
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    environment = report.get("environment", {})
    if environment.get("device_type") != config["device"] or environment.get("precision") != config["precision"]:
        raise ValueError("benchmark silently changed the requested backend or precision")
    distributed = report.get("distributed", {})
    size = distributed.get("world_size") if point.entrypoint == "tp" else distributed.get("global_world_size")
    if size != point.tp_size or distributed.get("backend") != {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[config["device"]]:
        raise ValueError("benchmark ran with the wrong process count or collective backend")
    if distributed.get("rank_results_consistent") is not True:
        raise ValueError("benchmark did not pass rank output consistency")
    expected_runner = "qwen3_tp_" + (point.decode_mode if point.entrypoint == "tp" else "slot_kv_cache")
    expected_mode = ("single_request" if point.batch_size == 1 else "static_batch") if point.entrypoint == "tp" else "closed_loop"
    protocol = report.get("protocol", {})
    expected_protocol = {"warmup": config["warmup"], "repeats": config["repeats"], "mode": expected_mode}
    if point.entrypoint == "tp":
        expected_protocol["decode_mode"] = point.decode_mode
        expected_benchmark = f"single_request_{config['device']}_{expected_runner}" if point.batch_size == 1 else f"static_batch_{expected_runner}"
        if report.get("benchmark") != expected_benchmark:
            raise ValueError("benchmark runner/entrypoint does not match the planned decode implementation")
    else:
        expected_protocol.update(closed_loop_clients=point.batch_size, greedy_token_path="full_gather",
                                 ttft_slo_ms=config["slo"]["ttft_ms"], tpot_slo_ms=config["slo"]["tpot_ms"],
                                 e2e_slo_ms=config["slo"]["e2e_ms"])
        engine = report.get("engine", {})
        if report.get("benchmark") != "continuous_batching_trace_replay" or engine.get("runner") != expected_runner:
            raise ValueError("continuous benchmark did not use the planned slot-cache runner")
        for key, value in (("max_slots", point.batch_size), ("max_seq_len", config["max_seq_len"]),
                           ("max_queue_size", max(16, point.batch_size))):
            if engine.get(key) != value:
                raise ValueError(f"continuous benchmark changed {key}")
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("benchmark protocol differs from planned warmup/repeats/decode/workload mode")
    provenance = report.get("provenance", {})
    for key in ("config_sha256", "metadata_sha256", "weights", "index_sha256"):
        if provenance.get(key) != expected_model.get(key):
            raise ValueError(f"benchmark model provenance differs in {key}")
    if provenance.get("weight_hashes_included") is not True:
        raise ValueError("benchmark has no verified model weight manifest")
    if identity is not None:
        software = identity["software"]
        if environment.get("python") != software["python"] or _public_version(environment.get("torch")) != _public_version(software["packages"]["torch"]):
            raise ValueError("benchmark software environment differs from the saved interpreter identity")
        if provenance.get("git", {}).get("commit") != identity["source"]["git"].get("commit"):
            raise ValueError("benchmark Git commit differs from the matrix identity")
        selected_devices = identity["devices"][:point.tp_size]
        logical_ids = [row["logical_device_id"] for row in selected_devices]
        mapping = distributed.get("device_mapping", {})
        if mapping.get("verified") is not True or mapping.get("logical_device_ids") != logical_ids:
            raise ValueError("benchmark device visibility does not match the matrix device map")
        if distributed.get("logical_device_ids") != logical_ids or distributed.get("global_logical_device_ids") != logical_ids:
            raise ValueError("benchmark rank device IDs differ from the planned single replica")
        cards = 0 if config["device"] == "cpu" else len({row["physical_card_id"] for row in selected_devices})
        for key, value in (("physical_card_count", cards), ("chips_per_card", config["chips_per_card"]),
                           ("interconnect_topology", identity["interconnect_topology"]), ("run_label", point.point_id)):
            if distributed.get(key) != value:
                raise ValueError(f"benchmark physical topology or point identity differs in {key}")
    if _workload_hash(report) != sha256_file(workload_path):
        raise ValueError("benchmark does not bind the exact source workload file")
    trace = WorkloadTrace.load(workload_path)
    if trace.to_dict() != make_workload(config, point).to_dict():
        raise ValueError("source workload no longer matches the matrix point")
    source_copy = output_dir / "source_workload.json"
    if not source_copy.is_file() or sha256_file(source_copy) != sha256_file(workload_path):
        raise ValueError("benchmark lacks its byte-identical source workload copy")
    expected_ids = {request.request_id for request in trace.requests}
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != config["repeats"]:
        raise ValueError("benchmark measured-repeat count is incomplete")
    work_shapes: list[str] = []
    output_digests: list[str] = []
    values: dict[str, list[float]] = {name: [] for name in _METRICS}
    for run in runs:
        requests = _all_requests(report, run)
        if len(requests) != point.batch_size or {row.get("request_id") for row in requests} != expected_ids:
            raise ValueError("benchmark lacks the complete original per-request results")
        canonical_rows = []
        outputs = []
        for row in sorted(requests, key=lambda value: value["request_id"]):
            if row.get("state") != "finished" or row.get("stop_reason") not in {"length", "max_new_tokens"}:
                raise ValueError("fixed-work matrix request did not finish at the requested length")
            generated = row.get("generated_ids")
            expected = next(request for request in trace.requests if request.request_id == row["request_id"])
            if not isinstance(generated, list) or not all(type(token) is int and token >= 0 for token in generated) or len(generated) != expected.config.max_new_tokens or row.get("output_tokens") != len(generated):
                raise ValueError("request output length does not match fixed workload")
            input_tokens = _integer(row.get("input_tokens"), "request.input_tokens")
            prompt_ids = row.get("prompt_ids")
            if not isinstance(prompt_ids, list) or len(prompt_ids) != input_tokens or not all(type(token) is int and token >= 0 for token in prompt_ids):
                raise ValueError("request lacks its exact encoded prompt IDs")
            canonical_rows.append({"request_id": row["request_id"], "state": "finished", "stop_reason": "length",
                                   "input_tokens": input_tokens, "prompt_ids": prompt_ids, "output_tokens": len(generated)})
            outputs.append({"request_id": row["request_id"], "generated_ids": generated})
            for name in ("ttft_ms", "tpot_ms", "e2e_latency_ms"):
                values[name].append(_positive(row.get(name), name))
            if row["ttft_ms"] + (len(generated) - 1) * row["tpot_ms"] > row["e2e_latency_ms"] + max(0.001, row["e2e_latency_ms"] * 1e-6):
                raise ValueError("request delivery timing is inconsistent with TTFT/TPOT/E2E")
        if point.entrypoint == "tp":
            from .benchmark import generation_output_digest
            digest = generation_output_digest(requests)
        else:
            from .serving_benchmark import serving_output_digest
            digest = serving_output_digest(run["serving"])
        if run.get("output_sha256") != digest:
            raise ValueError("measured output digest differs from the raw request records")
        work_shapes.append(canonical_sha256(canonical_rows))
        output_digests.append(canonical_sha256(outputs))
        if point.entrypoint == "continuous":
            wall_ms = _positive(run["replay"]["wall_time_ms"], "wall_time_ms")
            steps = run["serving"]["steps"]
            prefill = sum(float(step["prefill_phase_ms"]) for step in steps)
            decode = sum(float(step["decode_phase_ms"]) for step in steps)
        else:
            wall_ms = _positive(run["e2e_latency_ms"], "e2e_latency_ms")
            prefill, decode = run.get("prefill_ms"), run.get("decode_ms")
        values["makespan_ms"].append(wall_ms)
        values["output_tokens_per_second"].append(sum(len(row["generated_ids"]) for row in requests) / (wall_ms / 1000.0))
        for name, value in (("prefill_ms", prefill), ("decode_ms", decode)):
            if value is not None:
                values[name].append(_positive(value, name))
        good = 0
        measurable_goodput = True
        for row in requests:
            ttft, tpot, e2e = row.get("ttft_ms"), row.get("tpot_ms"), row.get("e2e_latency_ms")
            if any(value is None for value in (ttft, tpot, e2e)):
                measurable_goodput = False
                continue
            good += int(ttft <= config["slo"]["ttft_ms"] and tpot <= config["slo"]["tpot_ms"] and e2e <= config["slo"]["e2e_ms"])
        if measurable_goodput:
            values["goodput_requests_per_second"].append(good / (wall_ms / 1000.0))
    if len(set(work_shapes)) != 1 or len(set(output_digests)) != 1:
        raise ValueError("fixed-work measured repetitions changed outputs or work shape")
    if point.axis in {"kv", "tp"} and (not values["prefill_ms"] or not values["decode_ms"]):
        raise ValueError("Prefill/Decode A/B point has no measured phase timing")
    profile_summary = None
    if config["profile"]["enabled"]:
        from .backends.profiling import summarize_profile

        profiling = report.get("profiling", {})
        if profiling.get("measurement_excluded") is not True:
            raise ValueError("profile replay is not explicitly excluded from measured results")
        manifest = output_dir / "profile_manifest.json"
        binding = profiling.get("profile_manifest", {})
        if binding.get("sha256") != sha256_file(manifest) or binding.get("size_bytes") != manifest.stat().st_size:
            raise ValueError("benchmark does not hash-bind the profile manifest")
        raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        for key, expected in (("schema_version", 2), ("backend", config["device"]),
                              ("layout_id", f"tp{point.tp_size}" if point.entrypoint == "tp" else point.point_id),
                              ("mode", expected_mode), ("global_world_size", point.tp_size),
                              ("source_workload_sha256", trace.request_sha256),
                              ("source_workload_file_sha256", sha256_file(workload_path)),
                              ("git_commit", provenance["git"]["commit"]),
                              ("logical_device_ids", distributed["logical_device_ids"]),
                              ("selected_ranks", list(range(point.tp_size)))):
            if raw_manifest.get(key) != expected:
                raise ValueError(f"profile manifest differs from the benchmark in {key}")
        from .backends.profiling import ProfileProtocol
        expected_profile = ProfileProtocol(**{key: config["profile"][key] for key in ("skip_steps", "warmup_steps", "active_steps")}).as_dict()
        if raw_manifest.get("protocol") != expected_profile:
            raise ValueError("profile replay changed the requested protocol")
        replay = profiling.get("replay", {})
        if replay.get("output_sha256") not in {run["output_sha256"] for run in runs}:
            raise ValueError("profile replay is not output-equivalent to measured runs")
        if point.entrypoint == "tp" and generation_output_digest(replay.get("requests", [])) != replay.get("output_sha256"):
            raise ValueError("profile replay output digest differs from its raw request records")
        if point.entrypoint == "continuous" and serving_output_digest(replay.get("serving", {})) != replay.get("output_sha256"):
            raise ValueError("continuous profile replay output digest differs from its raw serving records")
        profile_summary = summarize_profile(manifest)
        if profile_summary.get("complete") is not True:
            raise ValueError(f"profile collection incomplete: {profile_summary.get('incomplete_reasons')}")
    memory = report.get("memory_measurements", {})
    peak_mib = _measured_peak_mib(memory, runs, config, point, distributed["logical_device_ids"])
    if peak_mib is not None:
        values["peak_device_memory_mb"] = [peak_mib]
    summaries = {name: statistics.median(samples) if samples else None for name, samples in values.items()}
    if config["device"] == "cpu":
        summaries["peak_device_memory_mb"] = None
    return {"metrics": summaries, "metric_samples": values, "work_shape_sha256": work_shapes[0],
            "output_sha256": output_digests[0], "source_workload_sha256": trace.request_sha256,
            "source_workload_file_sha256": sha256_file(workload_path), "environment": environment,
            "distributed": distributed, "git": provenance.get("git", {}),
            "measurement_scope": report.get("measurement_scope", {}),
            "memory_measurements": memory,
            "protocol": protocol,
            "phase_timing_scope": "scheduler_phase_total" if point.entrypoint == "continuous" else "synchronous_generation_phase",
            "profile": None if profile_summary is None else {
                "complete": profile_summary["complete"], "backend": profile_summary["backend"],
                "aggregate_step_fractions": profile_summary["aggregate_step_fractions"],
                "metric_capabilities": profile_summary["metric_capabilities"]}}


def run_matrix(config_path: str | Path, model_dir: str | Path, output_dir: str | Path, *,
               project_root: str | Path, device_map_path: str | Path | None = None,
               interconnect_topology: str | None = None, python_executable: str = sys.executable,
               resume: bool = False, dry_run: bool = False, selected_cases: Sequence[str] = (),
               selected_points: Sequence[str] = ()) -> dict[str, Any]:
    with _matrix_lock(Path(output_dir).resolve()) as lock_fd, _handle_termination():
        return _run_matrix(config_path, model_dir, output_dir, project_root=project_root,
                           device_map_path=device_map_path, interconnect_topology=interconnect_topology,
                           python_executable=python_executable, resume=resume, dry_run=dry_run,
                           selected_cases=selected_cases, selected_points=selected_points, lock_fd=lock_fd)


def _run_matrix(config_path: str | Path, model_dir: str | Path, output_dir: str | Path, *,
                project_root: str | Path, device_map_path: str | Path | None,
                interconnect_topology: str | None, python_executable: str,
                resume: bool, dry_run: bool, selected_cases: Sequence[str],
                selected_points: Sequence[str], lock_fd: int) -> dict[str, Any]:
    project = Path(project_root).resolve()
    model = Path(model_dir).resolve()
    root = Path(output_dir).resolve()
    config = load_matrix_config(config_path)
    python_executable = str(Path(shutil.which(python_executable) or python_executable).resolve())
    if not Path(python_executable).is_file():
        raise ValueError("selected Python executable does not exist")
    plan = expand_matrix(config)
    devices = load_device_map(device_map_path, config)
    topology = interconnect_topology or ("cpu_processes" if config["device"] == "cpu" else "")
    if not topology.strip():
        raise ValueError("accelerator matrices require a recorded --interconnect-topology")
    points = [MatrixPoint(**row) for row in plan["points"]]
    known_cases = {point.case_id for point in points}
    known_points = {point.point_id for point in points}
    if set(selected_cases) - known_cases or set(selected_points) - known_points:
        raise ValueError("unknown --case or --point selector")
    selected = [point for point in points if (not selected_cases or point.case_id in selected_cases) and
                (not selected_points or point.point_id in selected_points)]
    if not selected:
        raise ValueError("selection contains no matrix points")
    model_config = Qwen3Config.from_json(model / "config.json")
    for size in plan["required_device_counts"]:
        for rank in range(size):
            Qwen3TensorParallelPlan.create(model_config, rank, size)
    model_identity = _model_identity(project, model)
    if config["device"] != "cpu" and not model_identity["is_formal_model"]:
        raise ValueError("hardware matrix requires genuine Qwen3-32B dense weights; tiny fixtures are CPU correctness only")
    identity_env, _ = launch_environment(config, selected[0], devices, project)
    identity = {"config_sha256": canonical_sha256(config), "source": _source_identity(project),
                "model": model_identity, "devices": devices, "interconnect_topology": topology,
                "python_executable": python_executable, "software": _software_identity(python_executable, identity_env, project)}
    identity_hash = canonical_sha256(identity)
    state_path = root / "matrix_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("matrix output exists; use --resume to verify and continue it")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != MATRIX_SCHEMA_VERSION or state.get("identity_sha256") != identity_hash or state.get("identity") != identity:
            raise ValueError("resume rejected: code/model/config/device mapping/interpreter identity changed")
        if state.get("config") != config or state.get("plan") != plan or set(state.get("points", {})) != known_points:
            raise ValueError("resume rejected: stored plan or point set changed")
        for point_id, record in state["points"].items():
            if record["status"] == "succeeded":
                _verified_attempt(root, next(point for point in points if point.point_id == point_id), config,
                                  identity, record["attempts"][-1])
    else:
        if any(path.name != "matrix_runner.lock" for path in root.iterdir()):
            raise FileExistsError("new matrix output directory must be empty")
        root.mkdir(parents=True, exist_ok=True)
        state = {"schema_version": MATRIX_SCHEMA_VERSION, "created_at_utc": _utc(),
                 "identity_sha256": identity_hash, "identity": identity, "config": config, "plan": plan,
                 "points": {point.point_id: {"status": "pending", "attempts": []} for point in points}}
        _save(root / "matrix_config.json", config)
        _save(root / "matrix_plan.json", plan)
    for point in points:
        workload_path = root / "workloads" / f"{point.workload_key}.json"
        trace = make_workload(config, point)
        if workload_path.exists():
            if WorkloadTrace.load(workload_path).to_dict() != trace.to_dict():
                raise ValueError("saved workload changed; resume cannot replace evidence")
        else:
            trace.save(workload_path)
    _save(state_path, state)
    if dry_run:
        return state
    ordered_cases = {case["case_id"]: case["point_ids"] for case in plan["cases"]}
    for point in selected:
        record = state["points"][point.point_id]
        if record["status"] == "succeeded":
            print(f"verified existing point: {point.point_id}", flush=True)
            continue
        case_ids = ordered_cases[point.case_id]
        predecessors = case_ids[:case_ids.index(point.point_id)]
        if any(state["points"][point_id]["status"] != "succeeded" for point_id in predecessors):
            print(f"deferred {point.point_id}: complete earlier sessions in {point.case_id} first", flush=True)
            continue
        if _source_identity(project)["content_sha256"] != identity["source"]["content_sha256"]:
            raise RuntimeError("source files changed while the matrix was running")
        if record["status"] == "running":
            active = record["attempts"][-1].get("active_process")
            if active and active.get("birth_marker") is not None and _process_birth(active["pid"]) == active["birth_marker"]:
                raise RuntimeError("previous benchmark is still alive; resume cannot start a competing process")
            record["attempts"][-1].update(status="interrupted", ended_at_utc=_utc(),
                                          reason="previous matrix process stopped before recording an outcome")
        attempt_number = max((attempt["attempt"] for attempt in record["attempts"]), default=-1) + 1
        attempt_dir = root / "points" / point.point_id / f"attempt-{attempt_number:03d}"
        # Preserve a directory created before a killed controller journaled it.
        # It is incomplete evidence and must never be overwritten or reused.
        while attempt_dir.exists():
            record["attempts"].append({"attempt": attempt_number, "status": "interrupted",
                                       "reason": "preserved unjournaled attempt directory from an interrupted controller",
                                       "ended_at_utc": _utc(),
                                       "artifacts": [artifact(path, root) for path in sorted(attempt_dir.rglob("*")) if path.is_file()]})
            attempt_number += 1
            attempt_dir = root / "points" / point.point_id / f"attempt-{attempt_number:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        benchmark_output = attempt_dir / "output"
        benchmark_output.mkdir()
        workload_path = root / "workloads" / f"{point.workload_key}.json"
        env, overrides = launch_environment(config, point, devices, project)
        command = build_command(config, point, project_root=project, model_dir=model, workload_path=workload_path,
                                output_dir=benchmark_output, python_executable=python_executable, devices=devices,
                                interconnect_topology=topology, port=_free_port())
        attempt: dict[str, Any] = {"attempt": attempt_number, "status": "running", "started_at_utc": _utc(),
                                  "command": command, "environment_overrides": overrides, "cwd": str(project),
                                  "artifacts": [], "reason": None}
        record["attempts"].append(attempt)
        record["status"] = "running"
        _save(state_path, state)
        print(f"running {point.point_id} (attempt {attempt_number}, TP={point.tp_size})", flush=True)

        def record_process(pid: int) -> None:
            attempt["active_process"] = {"pid": pid, "birth_marker": _process_birth(pid)}
            _save(state_path, state)

        try:
            attempt["software"] = _software_identity(python_executable, env, project)
            if attempt["software"] != identity["software"]:
                raise ValueError("Python dependencies or library environment changed during the matrix")
            probe_command = [python_executable, "-c", _PROBE_CODE, config["device"], config["precision"],
                             str(point.tp_size), str(attempt_dir / "runtime_probe.json")]
            probe = _execute(probe_command, env=env, project_root=project, log_path=attempt_dir / "preflight.log",
                             timeout_seconds=min(float(config["timeout_seconds"]), 120.0), lock_fd=lock_fd,
                             on_started=record_process)
            attempt["preflight"] = probe
            if probe["status"] != "succeeded":
                attempt.update(status=probe["status"], exit_code=probe["exit_code"], reason="runtime preflight failed; see preflight.log")
            else:
                if config["device"] != "cpu":
                    from .backends.telemetry import collect_device_preflight
                    preflight = collect_device_preflight(config["device"], devices[:point.tp_size])
                    _save(attempt_dir / "device_preflight.json", preflight)
                    if preflight.get("supported") is not True or preflight.get("clean") is not True:
                        raise ValueError("device preflight rejected incomplete telemetry or contention; see device_preflight.json")
                outcome = _execute(command, env=env, project_root=project, log_path=attempt_dir / "benchmark.log",
                                   timeout_seconds=float(config["timeout_seconds"]), lock_fd=lock_fd,
                                   on_started=record_process)
                attempt.update(outcome)
                if outcome["status"] == "succeeded":
                    if _source_identity(project)["content_sha256"] != identity["source"]["content_sha256"]:
                        raise ValueError("source files changed while this benchmark was running")
                    attempt["result"] = inspect_point_output(point, config, benchmark_output, workload_path, model_identity, identity=identity)
                else:
                    attempt["reason"] = "benchmark did not complete successfully; see benchmark.log"
        except ProcessCleanupBlocked as exc:
            attempt.update(status="cleanup_blocked", reason=str(exc))
            raise  # Do not start another point while a supervisor still owns workers.
        except KeyboardInterrupt:
            attempt.update(status="interrupted", reason="matrix run interrupted")
            raise
        except Exception as exc:
            attempt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        finally:
            attempt["ended_at_utc"] = _utc()
            attempt["artifacts"] = [artifact(path, root) for path in sorted(attempt_dir.rglob("*")) if path.is_file()]
            attempt["artifacts"].append(artifact(workload_path, root))
            if attempt["status"] == "succeeded":
                try:
                    _verified_attempt(root, point, config, identity, attempt)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    attempt.update(status="failed", reason=f"evidence validation failed: {exc}")
            record["status"] = attempt["status"]
            state["updated_at_utc"] = _utc()
            _save(state_path, state)
        print(f"{point.point_id}: {record['status']}", flush=True)
    return state


def _cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return None if mean <= 0.0 else statistics.pstdev(values) / mean


def summarize_matrix(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    state = json.loads((root / "matrix_state.json").read_text(encoding="utf-8"))
    config = validate_matrix_config(state["config"])
    plan = expand_matrix(config)
    if state.get("schema_version") != MATRIX_SCHEMA_VERSION or state.get("plan") != plan:
        raise ValueError("matrix state contains a modified plan")
    if set(state.get("points", {})) != {point["point_id"] for point in plan["points"]}:
        raise ValueError("matrix state contains a modified point set")
    identity = state["identity"]
    if state.get("identity_sha256") != canonical_sha256(identity) or identity.get("config_sha256") != canonical_sha256(config):
        raise ValueError("matrix identity does not match its configuration")
    for filename, expected in (("matrix_config.json", config), ("matrix_plan.json", plan)):
        if json.loads((root / filename).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"saved {filename} differs from the matrix identity")
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    failed_attempt_count = 0
    for raw_point in plan["points"]:
        point = MatrixPoint(**raw_point)
        record = state["points"].get(point.point_id, {"status": "pending", "attempts": []})
        failed_attempt_count += sum(attempt.get("status") in TERMINAL_FAILURES for attempt in record.get("attempts", []))
        row = {**raw_point, "status": record["status"], "attempt_count": len(record.get("attempts", [])), "result": None}
        try:
            if record["status"] != "succeeded" or not record["attempts"]:
                raise ValueError(f"point status is {record['status']}")
            attempt = record["attempts"][-1]
            row["result"] = _verified_attempt(root, point, config, identity, attempt)
            row["execution"] = {key: attempt[key] for key in ("attempt", "started_at_utc", "ended_at_utc")}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["status"] = "incomplete"
            row["reason"] = str(exc)
            reasons.append(f"{point.point_id}: {exc}")
        rows.append(row)
        by_id[point.point_id] = row
    cases: list[dict[str, Any]] = []
    for case in plan["cases"]:
        members = [by_id[point_id] for point_id in case["point_ids"]]
        complete = all(row["status"] == "succeeded" and row["result"] for row in members)
        issues: list[str] = []
        metrics: dict[str, Any] = {}
        if complete:
            times = [_attempt_times(row["execution"]) for row in members]
            if any(current[0] < previous[1] for previous, current in zip(times, times[1:])):
                issues.append("successful sessions did not execute in the required AB/ABBA order")
            if len({canonical_sha256(row["result"]["environment"]) for row in members}) != 1:
                issues.append("backend hardware/software environment changed within the A/B case")
            if len({canonical_sha256(row["result"]["git"]) for row in members}) != 1:
                issues.append("Git revision/dirty state changed within the A/B case")
            if len({row["result"]["source_workload_sha256"] for row in members}) != 1:
                issues.append("source workloads differ")
            if len({row["result"]["work_shape_sha256"] for row in members}) != 1:
                issues.append("fixed request work shape differs")
            if len({row["result"]["output_sha256"] for row in members}) != 1:
                issues.append("fixed-batch greedy outputs differ")
            for metric in _METRICS:
                groups = {
                    role: [row["result"]["metrics"][metric] for row in members if row["role"] == role]
                    for role in ("baseline", "candidate")
                }
                observed = all(value is not None for values in groups.values() for value in values)
                same_scope = metric not in {"prefill_ms", "decode_ms"} or len({row["result"]["phase_timing_scope"] for row in members}) == 1
                baseline = statistics.median(groups["baseline"]) if observed else None
                candidate = statistics.median(groups["candidate"]) if observed else None
                metrics[metric] = {
                    "baseline": baseline, "candidate": candidate, "comparable": observed and same_scope,
                    "candidate_over_baseline": candidate / baseline if observed and same_scope and baseline > 0.0 else None,
                    "reason": None if observed and same_scope else "metric unavailable or measurement scopes differ",
                }
            for role in ("baseline", "candidate"):
                throughput = [row["result"]["metrics"]["output_tokens_per_second"] for row in members if row["role"] == role]
                cv = _cv(throughput)
                if config["device"] != "cpu" and (cv is None or cv > config.get("max_session_cv", 0.1)):
                    issues.append(f"{role} independent-session throughput stability gate failed")
        else:
            issues.append("one or more required independent sessions are missing or failed")
        case_complete = complete and not issues
        if not case_complete:
            reasons.append(f"{case['case_id']}: {'; '.join(issues)}")
        cases.append({**case, "complete": case_complete, "incomplete_reasons": issues, "metrics": metrics})
    complete = not reasons
    git = identity["source"].get("git", {})
    formal = (complete and config["device"] != "cpu" and identity["model"]["is_formal_model"]
              and git.get("dirty") is False and bool(re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", "")))))
    if config["device"] != "cpu":
        for row in rows:
            environment = (row["result"] or {}).get("environment", {})
            if (row["result"] or {}).get("git", {}).get("dirty") is not False:
                formal = False
            if config["device"] == "npu" and not (environment.get("torch_npu") and environment.get("cann_version")):
                formal = False
            if config["device"] == "cuda" and not environment.get("cuda_version"):
                formal = False
    return {"schema_version": MATRIX_SCHEMA_VERSION, "created_at_utc": _utc(), "matrix_id": config["matrix_id"],
            "hardware_family": config["hardware_family"], "device": config["device"], "complete": complete,
            "formal_performance_evidence": formal,
            "evidence_class": "qwen3_32b_hardware_matrix" if formal else "cpu_correctness_matrix" if complete and config["device"] == "cpu" else "incomplete_or_development_matrix",
            "expected_points": len(rows), "successful_points": sum(row["status"] == "succeeded" for row in rows),
            "failed_attempt_count": failed_attempt_count, "incomplete_reasons": reasons,
            "metric_units": {name: "MiB" if name == "peak_device_memory_mb" else "tokens/second" if name == "output_tokens_per_second" else "requests/second" if name == "goodput_requests_per_second" else "milliseconds" for name in _METRICS},
            "identity": identity, "points": rows, "cases": cases,
            "comparison_contract": {key: config[key] for key in ("precision", "warmup", "repeats", "sessions_per_variant", "max_seq_len", "slo", "profile", "seed")},
            "source_state": artifact(root / "matrix_state.json", root)}


def summarize_matrices(roots: Sequence[str | Path]) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one matrix directory is required")
    matrices = [summarize_matrix(root) for root in roots]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for matrix in matrices:
        for row in matrix["points"]:
            if row["status"] != "succeeded":
                continue
            key = (row["entrypoint"], row["decode_mode"], row["tp_size"], row["batch_size"],
                   row["result"]["source_workload_sha256"], row["role"])
            grouped.setdefault(key, []).append({"matrix_id": matrix["matrix_id"], "hardware_family": matrix["hardware_family"],
                                               "device": matrix["device"], "point_id": row["point_id"], "result": row["result"],
                                               "model_sha256": canonical_sha256(matrix["identity"]["model"]),
                                               "code_sha256": matrix["identity"]["source"]["content_sha256"],
                                               "contract_sha256": canonical_sha256(matrix["comparison_contract"]),
                                               "formal": matrix["formal_performance_evidence"]})
    comparisons: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        if len({row["hardware_family"] for row in rows}) < 2:
            continue
        comparable = (len({row["model_sha256"] for row in rows}) == 1
                      and len({row["code_sha256"] for row in rows}) == 1
                      and len({row["contract_sha256"] for row in rows}) == 1
                      and len({row["result"]["work_shape_sha256"] for row in rows}) == 1
                      and all(row["formal"] for row in rows))
        comparisons.append({"entrypoint": key[0], "decode_mode": key[1], "tp_size": key[2], "batch_size": key[3],
                            "source_workload_sha256": key[4], "role": key[5], "comparable_hardware_evidence": comparable,
                            "reason": None if comparable else "requires complete formal matrices with identical model, source code, precision, warmup/repeats, SLO/profile/capacity protocol and fixed request work",
                            "observations": [{"matrix_id": row["matrix_id"], "hardware_family": row["hardware_family"],
                                              "device": row["device"], "point_id": row["point_id"],
                                              "metrics": row["result"]["metrics"],
                                              "profile": row["result"]["profile"]} for row in rows]})
    return {"schema_version": MATRIX_SCHEMA_VERSION, "created_at_utc": _utc(),
            "complete": all(matrix["complete"] for matrix in matrices), "matrices": matrices,
            "cross_backend_comparisons": comparisons,
            "profiler_comparison_rule": "Compare declared metric semantics/time bases; native Ascend Stage and Kineto host-step intervals are not identical denominators."}


def write_matrix_markdown(summary: Mapping[str, Any], output: str | Path) -> None:
    lines = ["# v0.9 matrix execution report", "",
             "All metrics below come from verified raw benchmark artifacts. Missing hardware points remain incomplete.", ""]
    for matrix in summary["matrices"]:
        lines += [f"## {matrix['matrix_id']}", "",
                  f"Evidence class: {matrix['evidence_class']}. Successful points: {matrix['successful_points']}/{matrix['expected_points']}.",
                  f"Preserved failed/interrupted attempts: {matrix['failed_attempt_count']}.", "",
                  "| A/B case | Complete | Baseline token/s | Candidate token/s |",
                  "|---|---|---:|---:|"]
        for case in matrix["cases"]:
            metric = case["metrics"].get("output_tokens_per_second", {})
            fmt = lambda value: "unavailable" if value is None else f"{value:.6g}"
            lines.append(f"| {case['case_id']} | {case['complete']} | {fmt(metric.get('baseline'))} | {fmt(metric.get('candidate'))} |")
        if matrix["incomplete_reasons"]:
            lines += ["", "Incomplete evidence:"]
            lines += [f"- {reason}" for reason in matrix["incomplete_reasons"]]
        lines += [""]
    lines += ["Cross-backend profiler ratios require matching metric definitions; unknown metrics are unavailable, never zero.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
