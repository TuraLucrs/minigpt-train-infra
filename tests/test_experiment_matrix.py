"""Executable matrix contracts and real offline CPU evidence integration.

Run this file directly. --unit-only skips model creation and benchmark runs.
MINIGPT_RUN_GLOO_TESTS=1 additionally completes all 16 CPU points with actual
TP2/TP4 Gloo subprocesses; Linux CI owns that optional transport integration.
Management-command fixtures never stand in for a measured matrix result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt import experiment_matrix as matrix  # noqa: E402


CONFIG_NAMES = {
    "cpu": "v09_cpu_correctness.json",
    "cuda": "v09_cuda.json",
    "ascend_a3": "v09_ascend_a3.json",
    "ascend_a5": "v09_ascend_a5.json",
}

FULL_CONFIG_NAMES = {
    "cuda": "v09_cuda_full.json",
    "ascend_a3": "v09_ascend_a3_full.json",
    "ascend_a5": "v09_ascend_a5_full.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def expect_error(action: Callable[[], object], text: str = "", *, errors=(ValueError,)) -> Exception:
    try:
        action()
    except errors as exc:
        assert text.lower() in str(exc).lower(), (text, type(exc).__name__, str(exc))
        return exc
    raise AssertionError(f"expected {errors} containing {text!r}")


def load_config(family: str = "cpu") -> dict:
    return matrix.load_matrix_config(PROJECT_ROOT / "configs" / CONFIG_NAMES[family])


def points(config: dict) -> list[matrix.MatrixPoint]:
    return [matrix.MatrixPoint(**row) for row in matrix.expand_matrix(config)["points"]]


def check_plans_and_fixed_work() -> None:
    expected_counts = {"cpu": [1, 2, 4], "cuda": [1, 2, 4, 8],
                       "ascend_a3": [2, 4, 8, 16], "ascend_a5": [1, 2, 4, 8]}
    comparison_fields = {"tp_size", "entrypoint", "decode_mode", "batch_size", "workload_id"}
    expected_changes = {"kv": {"decode_mode"}, "batching": {"entrypoint"}, "tp": {"tp_size"}}
    for family, counts in expected_counts.items():
        config = load_config(family)
        original = deepcopy(config)
        plan = matrix.expand_matrix(config)
        assert config == original, "planning must not rewrite the provided configuration"
        assert plan["device_counts"] == counts
        expected_required = counts if family == "cpu" else ([2, 8] if family == "ascend_a3" else [1, 8])
        assert {row["tp_size"] for row in plan["points"]} == set(expected_required)
        session_order = ["baseline", "candidate"] if family == "cpu" else ["baseline", "candidate", "candidate", "baseline"]
        assert plan["session_order"] == session_order
        expected_cases = 8 if family == "cpu" else 3
        assert len(plan["cases"]) == expected_cases
        assert len(plan["points"]) == expected_cases * len(session_order)
        assert plan["required_device_counts"] == expected_required
        assert len({row["point_id"] for row in plan["points"]}) == len(plan["points"])
        by_id = {row["point_id"]: row for row in plan["points"]}
        for case in plan["cases"]:
            rows = [by_id[point_id] for point_id in case["point_ids"]]
            assert [row["role"] for row in rows] == session_order
            assert [row["session"] for row in rows] == ([0, 0] if family == "cpu" else [0, 0, 1, 1])
            baseline, candidate = rows[0], rows[1]
            changed = {key for key in comparison_fields if baseline[key] != candidate[key]}
            assert changed == expected_changes[case["axis"]], (case["case_id"], changed)
            if case["axis"] == "kv":
                assert (baseline["decode_mode"], candidate["decode_mode"]) == ("recompute", "kv_cache")
            elif case["axis"] == "batching":
                assert (baseline["entrypoint"], candidate["entrypoint"]) == ("tp", "continuous")
                assert baseline["decode_mode"] == candidate["decode_mode"] == "kv_cache"
                assert baseline["batch_size"] == config["batch_size"]
            else:
                assert baseline["tp_size"] == counts[0] < candidate["tp_size"]
            traces = [matrix.make_workload(config, matrix.MatrixPoint(**row)) for row in rows]
            assert all(trace.to_dict() == traces[0].to_dict() for trace in traces)
            assert len(traces[0].requests) == case["batch_size"]
            for request in traces[0].requests:
                assert request.arrival_time_ms == 0.0
                assert request.config.strategy == "greedy"
                assert request.config.eos_token_ids is None
                assert request.config.max_new_tokens >= config["profile"]["active_steps"]
    assert len(matrix.expand_matrix(load_config())["points"]) == 16
    for filename in FULL_CONFIG_NAMES.values():
        full = matrix.load_matrix_config(PROJECT_ROOT / "configs" / filename)
        full_plan = matrix.expand_matrix(full)
        assert len(full_plan["cases"]) == 22
        assert len(full_plan["points"]) == 88
        assert full_plan["required_device_counts"] == full["device_counts"]
    abba_cpu = load_config()
    abba_cpu["sessions_per_variant"] = 2
    assert len(matrix.expand_matrix(abba_cpu)["points"]) == 32


def check_invalid_configurations() -> None:
    cpu = load_config()
    mutations = [
        lambda value: value.pop("seed"),
        lambda value: value.update(unrecognized=True),
        lambda value: value.update(device_counts=[1]),
        lambda value: value.update(device="cuda"),
        lambda value: value.update(chips_per_card=1),
        lambda value: value.update(precision="bf16"),
        lambda value: value.update(batch_size=1),
        lambda value: value.update(warmup=True),
        lambda value: value.update(repeats=0),
        lambda value: value.update(timeout_seconds=float("nan")),
        lambda value: value.update(max_session_cv=0),
        lambda value: value.update(matrix_id="../../outside"),
        lambda value: value["profile"].update(enabled="true"),
        lambda value: value["profile"].update(skip_steps=1),
        lambda value: value["profile"].update(warmup_steps=1),
        lambda value: value["profile"].update(active_steps=1),
        lambda value: value["slo"].update(ttft_ms=0),
        lambda value: value["workloads"][0].update(prompts=[]),
        lambda value: value["workloads"][0].update(max_new_tokens=1),
        lambda value: value["workloads"][0].update(max_new_tokens=2),
        lambda value: value["workloads"].append(deepcopy(value["workloads"][0])),
    ]
    for index, mutation in enumerate(mutations):
        invalid = deepcopy(cpu)
        mutation(invalid)
        try:
            expect_error(lambda: matrix.validate_matrix_config(invalid))
        except AssertionError as exc:
            raise AssertionError(f"invalid configuration mutation {index} was accepted") from exc
    for field, invalid_value in (("sessions_per_variant", 1), ("repeats", 2), ("precision", "fp32")):
        invalid = load_config("cuda")
        invalid[field] = invalid_value
        expect_error(lambda: matrix.validate_matrix_config(invalid))
    invalid = load_config("ascend_a3")
    invalid["profile"]["enabled"] = False
    expect_error(lambda: matrix.validate_matrix_config(invalid), "profile")
    invalid = load_config("cuda")
    invalid["workloads"] = [row for row in invalid["workloads"] if row["workload_class"] == "short_short"]
    expect_error(lambda: matrix.validate_matrix_config(invalid), "both short and long")
    invalid = load_config("cuda")
    invalid["case_selection"] = ["not-a-real-case"]
    expect_error(lambda: matrix.expand_matrix(invalid), "unknown case_selection")
    invalid = load_config("cuda")
    invalid["case_selection"] = ["kv-long-tp1", "kv-long-tp1"]
    expect_error(lambda: matrix.validate_matrix_config(invalid), "duplicate")


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def check_mapping_environment_and_commands(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for family in ("cuda", "ascend_a3", "ascend_a5"):
        config = load_config(family)
        map_path = PROJECT_ROOT / "configs" / f"v09_{family}.devices.example.json"
        devices = matrix.load_device_map(map_path, config)
        assert len(devices) == max(matrix.expand_matrix(config)["required_device_counts"])
        expect_error(lambda: matrix.load_device_map(None, config), "explicit")
        invalid_maps = []
        invalid_maps.append(devices[:-1])
        duplicate = deepcopy(devices)
        duplicate[-1] = deepcopy(duplicate[0])
        invalid_maps.append(duplicate)
        missing_field = deepcopy(devices)
        missing_field[0].pop("chip_id")
        invalid_maps.append(missing_field)
        bool_id = deepcopy(devices)
        bool_id[0]["logical_device_id"] = True
        invalid_maps.append(bool_id)
        if family == "ascend_a3":
            partial_cards = deepcopy(devices)
            partial_cards[1], partial_cards[2] = partial_cards[2], partial_cards[1]
            invalid_maps.append(partial_cards)
        for index, invalid_devices in enumerate(invalid_maps):
            invalid_path = directory / f"invalid-{family}-{index}.json"
            write_json(invalid_path, {"schema_version": 1, "devices": invalid_devices})
            expect_error(lambda: matrix.load_device_map(invalid_path, config))

    cpu = load_config()
    cpu_devices = matrix.load_device_map(None, cpu)
    assert [row["logical_device_id"] for row in cpu_devices] == [0, 1, 2, 3]
    assert all(row["physical_card_id"] is None and row["chip_id"] is None for row in cpu_devices)
    expect_error(lambda: matrix.load_device_map(directory / "any.json", cpu), "process ranks")
    for family in CONFIG_NAMES:
        config = load_config(family)
        if family == "cpu":
            devices = cpu_devices
        else:
            devices = matrix.load_device_map(PROJECT_ROOT / "configs" / f"v09_{family}.devices.example.json", config)
        if family == "cuda":
            devices = [{"logical_device_id": value, "physical_card_id": value, "chip_id": 0}
                       for value in (3, 5, 7, 8, 9, 10, 11, 12)]
        for point in points(config):
            with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "99",
                                         "ASCEND_RT_VISIBLE_DEVICES": "99"}):
                env, overrides = matrix.launch_environment(config, point, devices, PROJECT_ROOT)
            assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
            assert env["PYTHONPATH"].split(os.pathsep)[0] == str(PROJECT_ROOT / "src")
            selected_ids = ",".join(str(row["logical_device_id"]) for row in devices[:point.tp_size])
            if family == "cuda":
                assert env["CUDA_VISIBLE_DEVICES"] == overrides["CUDA_VISIBLE_DEVICES"] == selected_ids
                assert env["CUDA_DEVICE_ORDER"] == overrides["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
            elif family.startswith("ascend"):
                assert env["ASCEND_RT_VISIBLE_DEVICES"] == selected_ids
            else:
                assert "CUDA_VISIBLE_DEVICES" not in overrides and "ASCEND_RT_VISIBLE_DEVICES" not in overrides
            model_path = directory / "model with spaces"
            workload_path = directory / "workload with spaces.json"
            output_path = directory / "output with spaces"
            command = matrix.build_command(config, point, project_root=PROJECT_ROOT, model_dir=model_path,
                                           workload_path=workload_path, output_dir=output_path,
                                           python_executable=sys.executable, devices=devices,
                                           interconnect_topology="test_topology", port=29173)
            assert command[0] == sys.executable
            if point.tp_size == 1:
                assert "torch.distributed.run" not in command
                assert command[1].endswith(".py")
                assert env["RANK"] == env["LOCAL_RANK"] == "0"
                assert env["WORLD_SIZE"] == "1"
            else:
                assert command[1:3] == ["-m", "torch.distributed.run"]
                assert f"--nproc-per-node={point.tp_size}" in command
                assert "--master-port=29173" in command
            assert _option(command, "--device") == config["device"]
            assert _option(command, "--backend") == {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[config["device"]]
            assert _option(command, "--logical-device-ids") == selected_ids
            assert _option(command, "--model-dir") == str(model_path)
            assert _option(command, "--workload") == str(workload_path)
            assert _option(command, "--warmup") == str(config["warmup"])
            assert _option(command, "--repeats") == str(config["repeats"])
            assert _option(command, "--profile-format") == "portable"
            assert _option(command, "--profile-ranks") == "all"
            assert _option(command, "--profile-active-steps") == str(config["profile"]["active_steps"])
            assert "--hash-weights" in command
            cards = 0 if family == "cpu" else len({row["physical_card_id"] for row in devices[:point.tp_size]})
            assert _option(command, "--physical-card-count") == str(cards)
            if point.entrypoint == "tp":
                assert _option(command, "--decode-mode") == point.decode_mode
                assert _option(command, "--output") == str(output_path / "report.json")
            else:
                assert _option(command, "--mode") == "closed_loop"
                assert _option(command, "--closed-loop-clients") == str(point.batch_size)
                assert _option(command, "--max-slots") == str(point.batch_size)
                assert _option(command, "--greedy-token-path") == "full_gather"
                assert _option(command, "--layout-id") == point.point_id


def _child_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def check_real_subprocess_outcomes_and_lock(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "failed.log"
    outcome = matrix._execute([sys.executable, "-u", "-c", "import sys; print('actual exit seven'); sys.exit(7)"],
                              env=os.environ, project_root=PROJECT_ROOT, log_path=log, timeout_seconds=10)
    assert outcome["status"] == "failed" and outcome["exit_code"] == 7
    assert "actual exit seven" in log.read_text(encoding="utf-8")
    child_pids: list[int] = []
    started = time.monotonic()
    timed = matrix._execute([sys.executable, "-u", "-c", "import time; print('actual sleeping process'); time.sleep(30)"],
                            env=os.environ, project_root=PROJECT_ROOT, log_path=directory / "timeout.log",
                            timeout_seconds=0.25, on_started=child_pids.append)
    assert timed["status"] == "timed_out" and timed["exit_code"] != 0
    assert time.monotonic() - started < 20, "a short subprocess timeout did not remain bounded"
    assert len(child_pids) == 1 and matrix._process_birth(child_pids[0]) is None
    lock_root = directory / "locked matrix"
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from minigpt.experiment_matrix import _matrix_lock
try:
    with _matrix_lock(Path(sys.argv[2])):
        print('acquired')
except RuntimeError as exc:
    print('rejected: ' + str(exc))
"""
    command = [sys.executable, "-c", script, str(PROJECT_ROOT / "src"), str(lock_root)]
    with matrix._matrix_lock(lock_root):
        second = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30, **_child_options())
        assert second.returncode == 0, second.stdout + second.stderr
        assert "rejected:" in second.stdout and "acquired" not in second.stdout
    after_release = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30, **_child_options())
    assert after_release.returncode == 0 and after_release.stdout.strip() == "acquired", after_release.stdout + after_release.stderr


def check_interrupt_immediately_after_spawn(directory: Path) -> None:
    """Interrupt the first Python line with a live Popen, before on_started runs."""
    directory.mkdir(parents=True, exist_ok=True)
    spawned: list[subprocess.Popen] = []
    callbacks: list[int] = []
    previous_trace = sys.gettrace()
    original_signal_mask = (signal.pthread_sigmask(signal.SIG_BLOCK, set())
                            if sys.platform.startswith("linux") else None)

    def interrupt_after_spawn(frame, event, _argument):
        if event == "line" and frame.f_code is matrix._execute.__code__ and not spawned:
            process = frame.f_locals.get("process")
            if isinstance(process, subprocess.Popen):
                spawned.append(process)
                raise KeyboardInterrupt("injected interruption immediately after Popen")
        return interrupt_after_spawn

    try:
        with matrix._matrix_lock(directory) as lock_fd:
            try:
                sys.settrace(interrupt_after_spawn)
                matrix._execute([sys.executable, "-u", "-c", "import time; time.sleep(60)"],
                                env=os.environ, project_root=PROJECT_ROOT,
                                log_path=directory / "interrupted-start.log", timeout_seconds=60,
                                lock_fd=lock_fd, on_started=callbacks.append)
            except KeyboardInterrupt as exc:
                assert str(exc) == "injected interruption immediately after Popen"
            else:
                raise AssertionError("the deterministic post-Popen interrupt was not propagated")
            finally:
                sys.settrace(previous_trace)
        if original_signal_mask is not None:
            assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_signal_mask, \
                "post-Popen interruption changed the controller's signal mask"
        assert len(spawned) == 1, "the post-Popen interruption was not exercised"
        assert callbacks == [], "the interrupt arrived after the startup callback"
        assert spawned[0].poll() is not None, "an interruption immediately after Popen leaked its child"
        assert matrix._process_birth(spawned[0].pid) is None
        with matrix._matrix_lock(directory):
            pass
    finally:
        sys.settrace(previous_trace)
        try:
            for process in spawned:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
        finally:
            if original_signal_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)


def check_artifact_rejection(directory: Path) -> None:
    config = load_config()
    point = points(config)[0]
    attempt_dir = directory / "points" / point.point_id / "attempt-000"
    raw_path = attempt_dir / "output" / "report.json"
    workload_path = directory / "workloads" / f"{point.workload_key}.json"
    # These are malformed evidence fixtures only; no synthetic successful run is constructed.
    write_json(raw_path, {"negative_evidence_fixture": True})
    write_json(workload_path, {"negative_evidence_fixture": True})
    artifacts = [matrix.artifact(raw_path, directory), matrix.artifact(workload_path, directory)]
    matrix.verify_artifacts(directory, artifacts)
    expect_error(lambda: matrix.verify_artifacts(directory, []), "no hash-bound")
    expect_error(lambda: matrix.verify_artifacts(directory, artifacts + artifacts[:1]), "duplicate")
    for path in ("../outside.json", str(raw_path.resolve()), "points\\escape.json", "drive:escape"):
        invalid = deepcopy(artifacts)
        invalid[0]["path"] = path
        expect_error(lambda: matrix.verify_artifacts(directory, invalid), "path")
    invalid = deepcopy(artifacts)
    invalid[0]["sha256"] = "0" * 64
    expect_error(lambda: matrix.verify_artifacts(directory, invalid), "hash mismatch")
    attempt = {"attempt": 0, "status": "succeeded", "exit_code": 0,
               "preflight": {"status": "succeeded"}, "started_at_utc": "2026-01-01T00:00:00+00:00",
               "ended_at_utc": "2026-01-01T00:00:01+00:00", "artifacts": artifacts[1:]}
    expect_error(lambda: matrix._verified_attempt(directory, point, config, {"software": {}}, attempt), "cover exactly")
    attempt["status"] = "failed"
    expect_error(lambda: matrix._verified_attempt(directory, point, config, {}, attempt), "outcome")


def check_real_sigterm_cleanup(directory: Path) -> None:
    """Linux must terminate the owned child and release its inherited matrix lock."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "owned-child.json"
    script = """
import json, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from minigpt.experiment_matrix import _execute, _handle_termination, _matrix_lock, _process_birth
root, marker = Path(sys.argv[2]), Path(sys.argv[3])
def started(pid):
    pending = marker.with_suffix('.pending')
    pending.write_text(json.dumps({'pid': pid, 'birth_marker': _process_birth(pid)}), encoding='utf-8')
    pending.replace(marker)
try:
    with _matrix_lock(root) as lock_fd, _handle_termination():
        _execute([sys.executable, '-u', '-c', 'import time; time.sleep(60)'],
                 env=os.environ, project_root=Path(sys.argv[4]), log_path=root / 'child.log',
                 timeout_seconds=60, lock_fd=lock_fd, on_started=started)
except KeyboardInterrupt:
    print('controller handled SIGTERM and cleaned its child', flush=True)
    raise SystemExit(128 + signal.SIGTERM)
raise SystemExit('controller was not interrupted by the test')
"""
    controller = subprocess.Popen([sys.executable, "-u", "-c", script, str(PROJECT_ROOT / "src"),
                                   str(directory), str(marker), str(PROJECT_ROOT)],
                                  cwd=PROJECT_ROOT, env=dict(os.environ), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, start_new_session=True)
    child: dict | None = None
    try:
        deadline = time.monotonic() + 30
        while not marker.is_file() and controller.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if not marker.is_file():
            if controller.poll() is not None:
                stdout, stderr = controller.communicate(timeout=5)
                raise AssertionError(f"controller exited before starting its child: {stdout}\n{stderr}")
            raise AssertionError("controller did not publish its owned child PID within 30 seconds")
        child = read_json(marker)
        assert child["birth_marker"] is not None
        assert matrix._process_birth(child["pid"]) == child["birth_marker"]
        controller.send_signal(signal.SIGTERM)
        stdout, stderr = controller.communicate(timeout=20)
        assert controller.returncode == 128 + signal.SIGTERM, stdout + stderr
        assert "handled SIGTERM" in stdout
        assert matrix._process_birth(child["pid"]) is None, "SIGTERM left the owned benchmark child alive"
        with matrix._matrix_lock(directory):
            pass
    finally:
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)
            try:
                controller.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                controller.kill()
                controller.communicate(timeout=10)
        # Cleanup is limited to the known child identity if a failing test left it alive.
        if child and child["birth_marker"] is not None and matrix._process_birth(child["pid"]) == child["birth_marker"]:
            try:
                os.killpg(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


_OWNED_WORKER_HELPER = """
import json, os, signal, subprocess, sys, time
from pathlib import Path

role, root, source, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]

def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    fields = stat[stat.rfind(')') + 2:].split()
    return {'pid': pid, 'birth_marker': fields[19],
            'pgid': os.getpgid(pid), 'session_id': os.getsid(pid)}

def publish(name, value):
    path = root / (name + '.json')
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value), encoding='utf-8')
    pending.replace(path)

if role in ('worker', 'unrelated'):
    if role == 'worker':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    publish(role, identity(os.getpid()))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(0.05)
elif role == 'launcher':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('launcher received TERM')
    signal.signal(signal.SIGTERM, interrupted)
    worker = subprocess.Popen([sys.executable, '-u', __file__, 'worker', str(root), source, mode],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while not (root / 'worker.json').is_file():
            if worker.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('detached worker did not become ready')
            time.sleep(0.01)
        publish('launcher', identity(os.getpid()))
        worker.wait(timeout=120)
    except KeyboardInterrupt:
        # Match torchrun: a worker gets its own session and a 30-second TERM grace.
        worker.terminate()
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=10)
        raise SystemExit(128 + signal.SIGTERM)
elif role == 'controller':
    sys.path.insert(0, source)
    from minigpt.experiment_matrix import _execute, _handle_termination, _matrix_lock
    try:
        with _matrix_lock(root) as lock_fd, _handle_termination():
            result = _execute([sys.executable, '-u', __file__, 'launcher', str(root), source, mode],
                              env=os.environ, project_root=Path(source).parent,
                              log_path=root / 'launcher.log',
                              timeout_seconds=8 if mode == 'timeout' else 60,
                              lock_fd=lock_fd, on_started=lambda pid: publish('guard', identity(pid)))
            publish('outcome', result)
    except KeyboardInterrupt:
        print('controller cancellation completed', flush=True)
        raise SystemExit(128 + signal.SIGTERM)
else:
    raise ValueError(role)
"""


def _wait_helper_identity(path: Path, owner: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 30
    while not path.is_file():
        if owner.poll() is not None:
            stdout, stderr = owner.communicate(timeout=5)
            raise AssertionError(f"helper exited before publishing {path.name}: {stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"helper did not publish {path.name} within 30 seconds")
        time.sleep(0.02)
    identity = read_json(path)
    assert identity["birth_marker"] is not None
    assert matrix._process_birth(identity["pid"]) == identity["birth_marker"], identity
    return identity


def _wait_owned_helper_exit(identity: dict, *, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while matrix._process_birth(identity["pid"]) == identity["birth_marker"]:
        if time.monotonic() >= deadline:
            raise AssertionError(f"owned helper remained alive after {timeout} seconds: {identity}")
        time.sleep(0.02)


def _kill_known_helper(identity: dict) -> None:
    if matrix._process_birth(identity["pid"]) != identity["birth_marker"]:
        return
    try:
        assert os.getpgid(identity["pid"]) == identity["pgid"]
        if matrix._process_birth(identity["pid"]) != identity["birth_marker"]:
            return
        # Every signaled group was created by this test and is still led by its recorded PID.
        if identity["pgid"] == identity["pid"]:
            os.killpg(identity["pid"], signal.SIGKILL)
        else:
            os.kill(identity["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass


def _assert_matrix_lock_rejected(directory: Path) -> None:
    try:
        with matrix._matrix_lock(directory):
            pass
    except RuntimeError:
        return
    raise AssertionError("matrix lock admitted another controller while an owned worker was alive")


def check_real_owned_worker_lifecycle(directory: Path) -> None:
    """Use real detached workers; successful launcher cleanup alone is insufficient."""
    directory.mkdir(parents=True, exist_ok=True)
    helper = directory / "owned_worker_helper.py"
    helper.write_text(_OWNED_WORKER_HELPER, encoding="utf-8")
    for mode in ("timeout", "sigterm", "orphan"):
        root = directory / mode
        root.mkdir()
        base = [sys.executable, "-u", str(helper)]
        suffix = [str(root), str(PROJECT_ROOT / "src"), mode]
        unrelated = subprocess.Popen([*base, "unrelated", *suffix], cwd=PROJECT_ROOT,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        controller: subprocess.Popen | None = None
        identities: dict[str, dict] = {}
        try:
            identities["unrelated"] = _wait_helper_identity(root / "unrelated.json", unrelated)
            controller = subprocess.Popen([*base, "controller", *suffix], cwd=PROJECT_ROOT,
                                          env=dict(os.environ), stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, start_new_session=True)
            for role in ("guard", "launcher", "worker"):
                identities[role] = _wait_helper_identity(root / f"{role}.json", controller)
            worker = identities["worker"]
            launcher = identities["launcher"]
            assert worker["pgid"] == worker["session_id"] == worker["pid"]
            assert worker["session_id"] != launcher["session_id"]
            _assert_matrix_lock_rejected(root)
            if mode == "sigterm":
                controller.send_signal(signal.SIGTERM)
            elif mode == "orphan":
                # The guard must retain ownership even after both ancestor processes disappear.
                controller.kill()
                controller.communicate(timeout=5)
                assert controller.returncode == -signal.SIGKILL
                _kill_known_helper(launcher)
                _wait_owned_helper_exit(launcher, timeout=5)
                assert matrix._process_birth(worker["pid"]) == worker["birth_marker"]
                _assert_matrix_lock_rejected(root)
                expect_error(
                    lambda: matrix.run_matrix(PROJECT_ROOT / "configs" / "v09_cpu_correctness.json",
                                              root / "model-must-not-be-read", root,
                                              project_root=PROJECT_ROOT, resume=True),
                    errors=(RuntimeError,),
                )
            if mode != "orphan":
                stdout, stderr = controller.communicate(timeout=45)
                if mode == "sigterm":
                    assert controller.returncode == 128 + signal.SIGTERM, stdout + stderr
                    assert "controller cancellation completed" in stdout
                else:
                    assert controller.returncode == 0, stdout + stderr
                    result = read_json(root / "outcome.json")
                    assert result["status"] == "timed_out" and result["exit_code"] != 0
            for role in ("worker", "launcher", "guard"):
                _wait_owned_helper_exit(identities[role])
            with matrix._matrix_lock(root):
                pass
            assert unrelated.poll() is None, f"{mode} cleanup killed an unrelated process"
            assert matrix._process_birth(unrelated.pid) == identities["unrelated"]["birth_marker"]
            print(f"Real detached-worker {mode} cleanup, lock, and unrelated-process checks passed.", flush=True)
        finally:
            if controller is not None:
                if controller.poll() is None:
                    controller.kill()
                controller.communicate(timeout=10)
            for role in ("worker", "launcher", "guard"):
                marker = root / f"{role}.json"
                identity = identities.get(role) or (read_json(marker) if marker.is_file() else None)
                if identity is not None:
                    _kill_known_helper(identity)
                    _wait_owned_helper_exit(identity, timeout=10)
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=10)


def _tree_snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    return {path.relative_to(directory).as_posix(): (matrix.sha256_file(path), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@contextmanager
def preserved_files(*paths: Path):
    original = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, payload in original.items():
            path.write_bytes(payload)


def _assert_selected_success(state: dict, selected: list[matrix.MatrixPoint]) -> None:
    failures = {point.point_id: state["points"][point.point_id] for point in selected
                if state["points"][point.point_id]["status"] != "succeeded"}
    assert not failures, json.dumps(failures, ensure_ascii=False, indent=2)


def _inspect(state: dict, root: Path, point: matrix.MatrixPoint) -> dict:
    attempt = state["points"][point.point_id]["attempts"][-1]
    output = root / "points" / point.point_id / f"attempt-{attempt['attempt']:03d}" / "output"
    return matrix.inspect_point_output(point, state["config"], output,
                                       root / "workloads" / f"{point.workload_key}.json",
                                       state["identity"]["model"], identity=state["identity"])


def check_real_evidence_rejection(state: dict, root: Path, selected: list[matrix.MatrixPoint],
                                  config_path: Path, model_dir: Path) -> None:
    cached = next(point for point in selected if point.axis == "kv" and point.role == "candidate")
    attempt = state["points"][cached.point_id]["attempts"][-1]
    output = root / "points" / cached.point_id / f"attempt-{attempt['attempt']:03d}" / "output"
    report_path, manifest_path = output / "report.json", output / "profile_manifest.json"
    with preserved_files(report_path):
        report = read_json(report_path)
        report["protocol"]["decode_mode"] = "recompute"
        write_json(report_path, report)
        expect_error(lambda: _inspect(state, root, cached), "protocol")
    with preserved_files(report_path):
        report = read_json(report_path)
        report["runs"][0]["requests"][0]["ttft_ms"] = None
        write_json(report_path, report)
        expect_error(lambda: _inspect(state, root, cached), "ttft_ms")
    with preserved_files(report_path):
        report = read_json(report_path)
        report["runs"][0]["requests"][0].pop("state")
        write_json(report_path, report)
        expect_error(lambda: _inspect(state, root, cached), "finish")
    with preserved_files(report_path, manifest_path):
        manifest = read_json(manifest_path)
        manifest["source_workload_sha256"] = "0" * 64
        write_json(manifest_path, manifest)
        report = read_json(report_path)
        # Rebind the changed file so this exercises identity validation, not only stale hashes.
        report["profiling"]["profile_manifest"].update(sha256=matrix.sha256_file(manifest_path),
                                                       size_bytes=manifest_path.stat().st_size)
        write_json(report_path, report)
        expect_error(lambda: _inspect(state, root, cached), "source_workload_sha256")
    with preserved_files(report_path):
        report = read_json(report_path)
        report["profiling"]["replay"]["requests"][0]["generated_ids"][0] += 1
        write_json(report_path, report)
        expect_error(lambda: _inspect(state, root, cached), "replay output digest")
    continuous = next(point for point in selected if point.entrypoint == "continuous")
    continuous_attempt = state["points"][continuous.point_id]["attempts"][-1]
    continuous_path = root / "points" / continuous.point_id / f"attempt-{continuous_attempt['attempt']:03d}" / "output" / "replica-00.json"
    with preserved_files(continuous_path):
        report = read_json(continuous_path)
        report["profiling"]["replay"]["serving"]["requests"][0]["generated_ids"][0] += 1
        write_json(continuous_path, report)
        expect_error(lambda: _inspect(state, root, continuous), "replay")
    omitted = deepcopy(attempt)
    report_relative = report_path.relative_to(root).as_posix()
    omitted["artifacts"] = [row for row in omitted["artifacts"] if row["path"] != report_relative]
    expect_error(lambda: matrix._verified_attempt(root, cached, state["config"], state["identity"], omitted), "cover exactly")
    changed_summary = deepcopy(attempt)
    changed_summary["result"]["metrics"]["output_tokens_per_second"] *= 2
    expect_error(lambda: matrix._verified_attempt(root, cached, state["config"], state["identity"], changed_summary), "summary differs")
    state_path = root / "matrix_state.json"
    with preserved_files(state_path):
        raw_state = read_json(state_path)
        raw_state["points"].pop(next(point.point_id for point in points(state["config"]) if point not in selected))
        write_json(state_path, raw_state)
        expect_error(lambda: matrix.summarize_matrix(root), "point set")
    with preserved_files(state_path):
        raw_state = read_json(state_path)
        raw_state["points"][cached.point_id]["attempts"][-1] = omitted
        write_json(state_path, raw_state)
        summary = matrix.summarize_matrix(root)
        assert next(row for row in summary["points"] if row["point_id"] == cached.point_id)["status"] == "incomplete"
        expect_error(lambda: matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                                              resume=True, selected_points=[point.point_id for point in selected]), "cover exactly")
    log_path = report_path.parent.parent / "benchmark.log"
    with preserved_files(log_path):
        with log_path.open("ab") as stream:
            stream.write(b"\nmodified after the original attempt\n")
        expect_error(lambda: matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                                              resume=True, selected_points=[point.point_id for point in selected]), "hash mismatch")
    with preserved_files(config_path):
        changed_config = read_json(config_path)
        changed_config["timeout_seconds"] += 1
        write_json(config_path, changed_config)
        expect_error(lambda: matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                                              resume=True, selected_points=[point.point_id for point in selected]), "identity changed")


def check_real_cpu_matrix(directory: Path, *, full_gloo: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = directory / "tiny model"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    fixture = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "create_tiny_qwen3_fixture.py"),
                              "--output", str(model_dir)], cwd=PROJECT_ROOT, env=env, check=False,
                             capture_output=True, text=True, encoding="utf-8", timeout=90, **_child_options())
    assert fixture.returncode == 0, fixture.stdout + fixture.stderr
    assert read_json(model_dir / "fixture_manifest.json")["formal_performance_model"] is False
    config = load_config()
    config_path = directory / "complete cpu config.json"
    write_json(config_path, config)
    root = directory / "matrix output"
    selected = [point for point in points(config) if point.tp_size == 1 and point.axis in {"kv", "batching"}]
    assert len(selected) == 4
    selected_ids = [point.point_id for point in selected]
    state = matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT, dry_run=True)
    assert len(state["points"]) == 16 and all(row["status"] == "pending" and row["attempts"] == [] for row in state["points"].values())
    assert matrix.summarize_matrix(root)["successful_points"] == 0

    orphan_point = next(point for point in selected if point.axis == "batching" and point.role == "baseline")
    orphan_file = root / "points" / orphan_point.point_id / "attempt-000" / "interrupted-controller.txt"
    orphan_file.parent.mkdir(parents=True)
    orphan_file.write_text("Preserve this pre-journaled attempt exactly.\n", encoding="utf-8")
    orphan_digest = matrix.sha256_file(orphan_file)
    # The real preflight child exits nonzero once. No benchmark outputs or success statuses are mocked.
    with patch.object(matrix, "_PROBE_CODE", "import sys; print('intentional real preflight failure', flush=True); sys.exit(23)"):
        failed = matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                                    resume=True, selected_points=[selected_ids[0]])
    failed_attempt = failed["points"][selected_ids[0]]["attempts"][0]
    assert failed_attempt["status"] == "failed" and failed_attempt["exit_code"] == 23
    failed_log = root / "points" / selected_ids[0] / "attempt-000" / "preflight.log"
    assert "intentional real preflight failure" in failed_log.read_text(encoding="utf-8")
    failed_digest = matrix.sha256_file(failed_log)

    state = matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                              resume=True, selected_points=selected_ids)
    _assert_selected_success(state, selected)
    assert matrix.sha256_file(failed_log) == failed_digest
    assert matrix.sha256_file(orphan_file) == orphan_digest
    assert [row["status"] for row in state["points"][selected_ids[0]]["attempts"]] == ["failed", "succeeded"]
    assert [row["status"] for row in state["points"][orphan_point.point_id]["attempts"]] == ["interrupted", "succeeded"]
    before_summary = _tree_snapshot(root)
    summary = matrix.summarize_matrix(root)
    assert _tree_snapshot(root) == before_summary, "summarization changed saved evidence"
    assert summary["successful_points"] == 4 and summary["expected_points"] == 16
    assert summary["complete"] is False and summary["formal_performance_evidence"] is False
    assert summary["failed_attempt_count"] == 2
    selected_rows = [row for row in summary["points"] if row["point_id"] in selected_ids]
    assert all(row["status"] == "succeeded" for row in selected_rows)
    assert all(row["result"]["metrics"]["peak_device_memory_mb"] is None for row in selected_rows)
    assert all(row["result"]["profile"]["complete"] is True and row["result"]["profile"]["backend"] == "cpu" for row in selected_rows)
    assert all(row["result"]["memory_measurements"]["supported"] is False for row in selected_rows)
    completed_cases = [case for case in summary["cases"] if set(case["point_ids"]).issubset(selected_ids)]
    assert len(completed_cases) == 2 and all(case["complete"] for case in completed_cases), completed_cases
    before_reuse = _tree_snapshot(root / "points")
    attempts_before = {point_id: deepcopy(state["points"][point_id]["attempts"]) for point_id in selected_ids}
    with patch.object(matrix, "_execute", side_effect=AssertionError("resume launched a previously successful point")):
        reused = matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT,
                                   resume=True, selected_points=selected_ids)
    assert _tree_snapshot(root / "points") == before_reuse
    assert {point_id: reused["points"][point_id]["attempts"] for point_id in selected_ids} == attempts_before
    check_real_evidence_rejection(reused, root, selected, config_path, model_dir)
    assert matrix.summarize_matrix(root)["successful_points"] == 4
    if full_gloo:
        print("Running all remaining CPU TP2/TP4 points with actual Gloo processes.", flush=True)
        full_state = matrix.run_matrix(config_path, model_dir, root, project_root=PROJECT_ROOT, resume=True)
        _assert_selected_success(full_state, points(config))
        complete = matrix.summarize_matrix(root)
        assert complete["complete"] is True and complete["successful_points"] == complete["expected_points"] == 16, complete["incomplete_reasons"]
        assert complete["evidence_class"] == "cpu_correctness_matrix"
        assert complete["formal_performance_evidence"] is False
        assert all(row["result"]["metrics"]["peak_device_memory_mb"] is None for row in complete["points"])
        assert all(row["result"]["profile"]["complete"] is True for row in complete["points"])
        assert all(case["complete"] for case in complete["cases"])
    else:
        print("Actual CPU TP1 four-point integration passed; TP2/TP4 require MINIGPT_RUN_GLOO_TESTS=1 on Linux.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true", help="skip tiny-model creation and real benchmark integration")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="minigpt-matrix-tests-") as temporary:
        directory = Path(temporary)
        check_plans_and_fixed_work()
        check_invalid_configurations()
        check_mapping_environment_and_commands(directory / "maps")
        check_real_subprocess_outcomes_and_lock(directory / "processes")
        check_interrupt_immediately_after_spawn(directory / "interrupted-start")
        if sys.platform.startswith("linux"):
            check_real_sigterm_cleanup(directory / "sigterm")
            check_real_owned_worker_lifecycle(directory / "owned-workers")
        else:
            print("Skipped real SIGTERM/detached-worker cleanup: this integration requires Linux.", flush=True)
        check_artifact_rejection(directory / "artifacts")
        print("Matrix plan, mapping, command, real subprocess/lock, and artifact unit checks passed.", flush=True)
        if not args.unit_only:
            check_real_cpu_matrix(directory / "integration", full_gloo=os.environ.get("MINIGPT_RUN_GLOO_TESTS") == "1")
    print("Experiment matrix tests passed.", flush=True)


if __name__ == "__main__":
    main()
