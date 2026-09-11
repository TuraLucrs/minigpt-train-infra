"""Run the counterbalanced A3 SLO-aware TP/replica research pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.decode_critical_path import summarize_npu_preflight  # noqa: E402
from minigpt.experiment import git_snapshot, sha256_file  # noqa: E402
from minigpt.serving_layout import load_layout_manifest  # noqa: E402
from minigpt.serving_telemetry import load_telemetry  # noqa: E402
from minigpt.slo_layout_pilot import (  # noqa: E402
    build_session_plan,
    canonical_sha256,
    load_pilot_config,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


def _artifact(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _run(command: Sequence[str], *, env: Mapping[str, str], log: Path | None = None) -> None:
    if log is None:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=dict(env))
        if completed.returncode:
            raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        except BaseException:
            _stop_process_group(process)
            raise
    if returncode:
        raise RuntimeError(f"benchmark failed ({returncode}); see {log}")


def _stop_process_group(process: subprocess.Popen[object], timeout: float = 30.0) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def _start_sampler(command: Sequence[str], *, env: Mapping[str, str], log: Path) -> tuple[subprocess.Popen[object], object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=dict(env),
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def _load_device_map(path: Path) -> list[dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("devices"), list):
        raise ValueError("invalid A3 device map")
    devices = raw["devices"][:8]
    if len(devices) != 8:
        raise ValueError("A3 pilot requires eight mapped logical devices")
    normalized = []
    for row in devices:
        if not isinstance(row, dict) or set(row) != {"logical_device_id", "physical_card_id", "chip_id"}:
            raise ValueError("each device-map row requires logical_device_id/physical_card_id/chip_id")
        if any(type(row[field]) is not int or row[field] < 0 for field in row):
            raise ValueError("device-map identifiers must be nonnegative integers")
        normalized.append({field: int(row[field]) for field in row})
    if {row["logical_device_id"] for row in normalized} != set(range(8)):
        raise ValueError("this A3 pilot requires the verified logical device IDs 0..7")
    targets = {(row["physical_card_id"], row["chip_id"]) for row in normalized}
    if len(targets) != 8:
        raise ValueError("physical card/chip targets must be unique")
    cards = {row["physical_card_id"] for row in normalized}
    if len(cards) != 4 or any(
        {row["chip_id"] for row in normalized if row["physical_card_id"] == card} != {0, 1}
        for card in cards
    ):
        raise ValueError("A3 pilot requires four complete two-chip cards")
    return normalized


def _telemetry_args(devices: Sequence[Mapping[str, int]]) -> list[str]:
    result = []
    for row in devices:
        result += [
            "--target",
            f"{row['logical_device_id']}={row['physical_card_id']}:{row['chip_id']}",
        ]
    return result


def _extract_workloads(output_root: Path) -> Path:
    target = output_root / "frozen_workloads"
    target.mkdir(parents=True, exist_ok=True)
    archive = PROJECT_ROOT / "v0.7_ascend_evidence.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"missing frozen workload archive: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        for name in ("short_short", "mixed"):
            member_name = f"runs/v07/workloads/{name}.json"
            member = handle.getmember(member_name)
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read {member_name}")
            payload = stream.read()
            destination = target / f"{name}.json"
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError(f"frozen workload changed: {destination}")
            destination.write_bytes(payload)
    return target


def _validate_completed_session(
    session_dir: Path,
    *,
    expected: Mapping[str, object],
    config_sha256: str,
) -> bool:
    marker_path = session_dir / "session_complete.json"
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != 1 or marker.get("config_sha256") != config_sha256:
        return False
    for field in ("scenario_id", "round", "layout_id", "session_id"):
        if marker.get(field) != expected.get(field):
            return False
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            return False
        path = session_dir.parent.parent.parent / str(artifact.get("path", ""))
        if not path.is_file() or path.stat().st_size != artifact.get("size_bytes"):
            return False
        if sha256_file(path) != artifact.get("sha256"):
            return False
    layout_id, _reports = load_layout_manifest(session_dir / "layout_manifest.json")
    if layout_id != expected["layout_id"]:
        return False
    load_telemetry(session_dir / "telemetry.json")
    preflight = load_telemetry(session_dir / "device_preflight.json")
    return bool(
        summarize_npu_preflight(
            preflight, expected_logical_device_ids=range(8)
        )["clean"]
    )


def _archive_output(output_root: Path, archive: Path) -> None:
    checksums = output_root / "SHA256SUMS"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != checksums:
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    checksums.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_root, arcname=output_root.name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/slo_layout_pilot_a3.json")
    parser.add_argument("--model-dir")
    parser.add_argument("--device-map")
    parser.add_argument("--interconnect-topology")
    parser.add_argument("--cann-version", default=os.environ.get("CANN_VERSION"))
    parser.add_argument("--output-dir", default="runs/slo_layout_pilot_a3")
    parser.add_argument("--archive", default="slo_layout_pilot_a3_evidence.tar.gz")
    parser.add_argument("--master-port-base", type=int, default=29820)
    parser.add_argument("--telemetry-interval-ms", type=float, default=200.0)
    parser.add_argument("--host-telemetry-interval-ms", type=float, default=500.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-sessions", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_pilot_config(config_path)
    plan = build_session_plan(config)
    if args.list_sessions:
        for row in plan:
            print(
                f"{row['position']:02d} {row['scenario_id']} round={row['round']} "
                f"order={row['order_in_round']} layout={row['layout_id']}"
            )
        print(f"total: {len(plan)} independent benchmark processes")
        return 0
    for option in ("model_dir", "device_map", "interconnect_topology"):
        if not getattr(args, option):
            parser.error(f"--{option.replace('_', '-')} is required for execution")
    if not args.cann_version:
        parser.error("--cann-version or CANN_VERSION is required")
    model_dir = Path(str(args.model_dir)).resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    device_map_path = Path(str(args.device_map)).resolve()
    devices = _load_device_map(device_map_path)
    logical_ids = ",".join(str(row["logical_device_id"]) for row in devices)
    output_root = Path(args.output_dir).resolve()
    archive = Path(args.archive).resolve()
    config_sha256 = canonical_sha256(config)
    source = git_snapshot(PROJECT_ROOT)
    if source["dirty"] is not False:
        raise SystemExit("formal pilot requires a clean tracked worktree")
    run_identity = {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "config_sha256": config_sha256,
        "git": source,
        "device_map": devices,
        "interconnect_topology": args.interconnect_topology,
        "cann_version": args.cann_version,
        "model_dir": str(model_dir),
        "plan": plan,
    }
    identity_path = output_root / "pilot_run_manifest.json"
    if output_root.exists() and not args.resume:
        raise SystemExit(f"output exists; use a new directory or --resume: {output_root}")
    if args.resume:
        if not identity_path.is_file():
            raise SystemExit(f"resume manifest is missing: {identity_path}")
        saved = json.loads(identity_path.read_text(encoding="utf-8"))
        comparable_saved = dict(saved)
        comparable_saved.pop("created_at_utc", None)
        if comparable_saved != run_identity:
            raise SystemExit("resume identity changed (config/commit/model/devices/environment)")
    else:
        output_root.mkdir(parents=True)
        _save_json(identity_path, {**run_identity, "created_at_utc": _utc()})
    workload_dir = _extract_workloads(output_root)
    layouts = {str(row["id"]): row for row in config["layouts"]}
    scenarios = {str(row["id"]): row for row in config["scenarios"]}
    telemetry_common = _telemetry_args(devices)
    env = dict(os.environ)
    env.update({
        "ASCEND_RT_VISIBLE_DEVICES": logical_ids,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
    })

    for planned in plan:
        scenario_id = str(planned["scenario_id"])
        layout_id = str(planned["layout_id"])
        scenario = scenarios[scenario_id]
        layout = layouts[layout_id]
        session_dir = (
            output_root / scenario_id / f"round-{int(planned['round']):02d}" /
            str(planned["session_id"])
        )
        if args.resume and session_dir.exists() and _validate_completed_session(
            session_dir, expected=planned, config_sha256=config_sha256
        ):
            print(f"SKIP verified {planned['position']:02d}/12 {scenario_id} {layout_id}")
            continue
        if session_dir.exists():
            quarantine = output_root / "_incomplete" / f"{session_dir.name}-{time.time_ns()}"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(session_dir), str(quarantine))
            print(f"preserved incomplete attempt at {quarantine}")
        session_dir.mkdir(parents=True)
        print(
            f"===== {planned['position']:02d}/12 {scenario_id} "
            f"round={planned['round']} layout={layout_id} ====="
        )
        preflight_path = session_dir / "device_preflight.json"
        preflight_command = [
            sys.executable, "benchmarks/sample_npu_telemetry.py", *telemetry_common,
            "--interval-ms", str(args.telemetry_interval_ms), "--query-type", "common",
            "--duration-seconds", "1.0", "--min-samples", "3", "--output", str(preflight_path),
        ]
        _run(preflight_command, env=env)
        _run(
            [sys.executable, "benchmarks/check_v08_npu_idle.py", "--telemetry", str(preflight_path)],
            env=env,
        )
        _run(
            [sys.executable, "benchmarks/capture_v071_host_snapshot.py", "--phase", "before", "--output", str(session_dir / "host_before.json")],
            env=env,
        )

        telemetry_process, telemetry_log = _start_sampler(
            [
                sys.executable, "benchmarks/sample_npu_telemetry.py", *telemetry_common,
                "--interval-ms", str(args.telemetry_interval_ms), "--query-type", "common",
                "--output", str(session_dir / "telemetry.json"),
            ],
            env=env,
            log=session_dir / "telemetry_sampler.log",
        )
        host_process, host_log = _start_sampler(
            [
                sys.executable, "benchmarks/sample_host_telemetry.py",
                "--interval-ms", str(args.host_telemetry_interval_ms),
                "--output", str(session_dir / "host_telemetry.json"),
            ],
            env=env,
            log=session_dir / "host_telemetry_sampler.log",
        )
        slo = scenario["slo"]
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={args.master_port_base + int(planned['position']) - 1}",
            "--nproc-per-node=8",
            "benchmarks/infer_qwen3_continuous_batching.py",
            "--model-dir", str(model_dir),
            "--workload", str(workload_dir / f"{scenario['workload_class']}.json"),
            "--mode", str(scenario["mode"]),
            "--tp-size", str(layout["tp_size"]),
            "--max-slots", str(layout["per_replica_max_slots"]),
            "--max-seq-len", str(config["max_seq_len"]),
            "--max-queue-size", str(layout["per_replica_max_queue_size"]),
            "--ttft-slo-ms", str(slo["ttft_ms"]),
            "--tpot-slo-ms", str(slo["tpot_ms"]),
            "--e2e-slo-ms", str(slo["e2e_ms"]),
            "--warmup", str(config["warmup"]),
            "--repeats", str(config["repeats"]),
            "--chat-template", "--device", "npu", "--precision", "bf16",
            "--backend", "hccl", "--distributed-timeout-seconds", "1800",
            "--hash-weights", "--layout-id", layout_id,
            "--logical-device-ids", logical_ids,
            "--physical-card-count", str(config["physical_card_count"]),
            "--chips-per-card", str(config["chips_per_card"]),
            "--interconnect-topology", args.interconnect_topology,
            "--cann-version", args.cann_version,
            "--run-label", f"research-{config['pilot_id']}-{planned['session_id']}",
            "--output-dir", str(session_dir),
        ]
        if scenario["mode"] == "open_loop":
            command.append("--deterministic-open-loop")
        else:
            command += ["--closed-loop-clients", str(scenario["closed_loop_clients"])]
        benchmark_error: BaseException | None = None
        try:
            _run(command, env=env, log=session_dir / "benchmark.log")
        except BaseException as exc:
            benchmark_error = exc
        telemetry_status = _stop_process_group(telemetry_process)
        host_status = _stop_process_group(host_process)
        telemetry_log.close()
        host_log.close()
        _run(
            [sys.executable, "benchmarks/capture_v071_host_snapshot.py", "--phase", "after", "--output", str(session_dir / "host_after.json")],
            env=env,
        )
        if benchmark_error is not None:
            raise benchmark_error
        if telemetry_status != 0 or host_status != 0:
            raise RuntimeError(
                f"sampler failure: npu={telemetry_status}, host={host_status} in {session_dir}"
            )
        layout_manifest = session_dir / "layout_manifest.json"
        loaded_layout, _reports = load_layout_manifest(layout_manifest)
        if loaded_layout != layout_id:
            raise RuntimeError(f"saved layout mismatch in {layout_manifest}")
        load_telemetry(session_dir / "telemetry.json")
        artifact_names = [
            "layout_manifest.json", "source_workload.json", "device_preflight.json",
            "telemetry.json", "host_telemetry.json", "host_before.json", "host_after.json",
            "benchmark.log", "telemetry_sampler.log", "host_telemetry_sampler.log",
        ]
        artifact_names += sorted(
            path.name for path in session_dir.glob("replica-*.json")
        )
        if (session_dir / "layout_provenance.json").is_file():
            artifact_names.append("layout_provenance.json")
        marker = {
            "schema_version": 1,
            "completed_at_utc": _utc(),
            "config_sha256": config_sha256,
            **planned,
            "git_commit": source["commit"],
            "artifacts": {
                name: _artifact(session_dir / name, relative_to=output_root)
                for name in artifact_names
            },
        }
        _save_json(session_dir / "session_complete.json", marker)

    summary_json = output_root / "slo_layout_pilot_summary.json"
    report_md = output_root / "RUN_LOG.md"
    _run(
        [
            sys.executable, "benchmarks/summarize_slo_layout_pilot.py",
            "--root", str(output_root), "--config", str(config_path),
            "--output", str(summary_json), "--markdown", str(report_md),
        ],
        env=env,
    )
    _archive_output(output_root, archive)
    print(f"summary: {summary_json}")
    print(f"run log: {report_md}")
    print(f"archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
