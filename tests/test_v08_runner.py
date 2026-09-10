"""用真实 Bash/git/tar 验证 v0.8 runner 生命周期；Python/NPU 命令使用临时 mock。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MOCK_DRIVER = r'''import json
import os
from pathlib import Path
import sys
import time

kind, arguments = sys.argv[1], sys.argv[2:]
scenario = os.environ.get("V08_MOCK_FAILURE", "none")

def option(name):
    return arguments[arguments.index(name) + 1]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mock_lifecycle_only": True, **value}) + "\n", encoding="utf-8")

with Path(os.environ["V08_MOCK_CALL_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": kind, "arguments": arguments}) + "\n")

if kind == "sample":
    write_json(option("--output"), {"complete": True, "samples": []})
elif kind == "preflight":
    if scenario == "preflight":
        raise SystemExit(17)
elif kind == "benchmark":
    output = Path(option("--output-dir"))
    ready = output / "telemetry.json.ready"
    deadline = time.monotonic() + 10
    while not ready.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("mock telemetry sampler did not start")
        time.sleep(.01)
    write_json(output / "mock_benchmark.json", {"greedy_path": option("--greedy-token-path")})
    print("MOCK benchmark: no model or accelerator was used", flush=True)
    if scenario in {"benchmark", "benchmark_and_summary"}:
        raise SystemExit(23)
elif kind == "profile":
    write_json(option("--output"), {"complete": False})
elif kind == "summary":
    if scenario in {"summary", "benchmark_and_summary"}:
        raise SystemExit(29)
    write_json(option("--output"), {"complete": False, "decision": {"status": "mock_lifecycle_only"}})
    Path(option("--markdown-output")).write_text("Lifecycle fixture only; no hardware evidence.\n", encoding="utf-8")
elif kind != "workload":
    raise RuntimeError(f"unexpected mock command: {kind}")
'''

MOCK_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-" ]]; then
  if [[ "${2:-}" == */session_status.json ]]; then
    exec "${V08_REAL_PYTHON}" "$@"
  fi
  cat >/dev/null
  exec "${V08_REAL_PYTHON}" "${V08_MOCK_DRIVER}" workload "${2:-}"
fi
command="$1"
shift
case "${command}" in
  benchmarks/sample_npu_telemetry.py)
    output=""
    finite=0
    arguments=("$@")
    for ((i=0; i<${#arguments[@]}; i++)); do
      [[ "${arguments[i]}" != --output ]] || output="${arguments[i+1]}"
      [[ "${arguments[i]}" != --duration-seconds ]] || finite=1
    done
    if [[ "${finite}" == 1 ]]; then
      exec "${V08_REAL_PYTHON}" "${V08_MOCK_DRIVER}" sample "$@"
    fi
    trap 'exit 0' TERM INT
    "${V08_REAL_PYTHON}" "${V08_MOCK_DRIVER}" sample "$@"
    printf 'ready\n' > "${output}.ready"
    while :; do sleep .05; done
    ;;
  benchmarks/check_v08_npu_idle.py) kind=preflight ;;
  benchmarks/summarize_v071_profile.py) kind=profile ;;
  benchmarks/summarize_v08_decode_ab.py) kind=summary ;;
  *) echo "unexpected Python command: ${command}" >&2; exit 97 ;;
esac
exec "${V08_REAL_PYTHON}" "${V08_MOCK_DRIVER}" "${kind}" "$@"
'''


def bash_executable() -> str:
    configured = os.environ.get("MINIGPT_TEST_BASH")
    candidates = [configured] if configured else []
    if os.name == "nt":
        candidates.append(r"D:\Git\bin\bash.exe")
    candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("v0.8 runner integration test requires Bash (MINIGPT_TEST_BASH)")


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def prepare_fixture(directory: Path) -> None:
    (directory / "scripts").mkdir(parents=True)
    (directory / "mock_model").mkdir()
    (directory / "workloads").mkdir()
    (directory / "mock_bin").mkdir()
    for workload in ("short_short", "mixed"):
        (directory / "workloads" / f"{workload}.json").write_text("{}\n", encoding="utf-8")
    _write_executable(
        directory / "scripts" / "run_v08_decode_vocab_ab.sh",
        (ROOT / "scripts" / "run_v08_decode_vocab_ab.sh").read_text(encoding="utf-8"),
    )
    (directory / "mock_driver.py").write_text(MOCK_DRIVER, encoding="utf-8", newline="\n")
    _write_executable(directory / "mock_bin" / "python", MOCK_PYTHON)
    _write_executable(
        directory / "mock_bin" / "torchrun",
        '#!/usr/bin/env bash\nexec "${V08_REAL_PYTHON}" "${V08_MOCK_DRIVER}" benchmark "$@"\n',
    )
    _write_executable(
        directory / "mock_bin" / "npu-smi",
        '#!/usr/bin/env bash\necho "unexpected direct NPU access in lifecycle fixture" >&2\nexit 98\n',
    )
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=directory,
        check=True,
        capture_output=True,
        creationflags=CREATE_NO_WINDOW,
    )


def run_fixture(directory: Path, bash: str, failure: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        PATH=str(directory / "mock_bin") + os.pathsep + environment.get("PATH", ""),
        V08_REAL_PYTHON=Path(sys.executable).as_posix(),
        V08_MOCK_DRIVER=(directory / "mock_driver.py").as_posix(),
        V08_MOCK_CALL_LOG=(directory / "mock_calls.jsonl").as_posix(),
        V08_MOCK_FAILURE=failure,
        MODEL_DIR="mock_model",
        CANN_VERSION="mock-lifecycle-only",
        INTERCONNECT_TOPOLOGY="mock-lifecycle-only",
        WORKLOAD_DIR="workloads",
        V08_OUTPUT_ROOT="runs/evidence",
        V08_ARCHIVE="evidence.tar.gz",
    )
    return subprocess.run(
        [bash, "scripts/run_v08_decode_vocab_ab.sh"],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        creationflags=CREATE_NO_WINDOW,
    )


def verify_archive(directory: Path, exit_code: int) -> None:
    output = directory / "runs" / "evidence"
    assert (output / "EXIT_STATUS").read_text().strip() == str(exit_code)
    archive_path = directory / "evidence.tar.gz"
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert (directory / "evidence.tar.gz.sha256").read_text().split()[0] == digest
    with tarfile.open(archive_path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                files[member.name] = stream.read()
    assert files["evidence/EXIT_STATUS"].decode().strip() == str(exit_code)
    checksums = files["evidence/SHA256SUMS"].decode()
    checked = set()
    for line in checksums.splitlines():
        # GNU sha256sum 在 Windows 默认用 '*' 标记二进制文件，POSIX 通常用空格。
        assert len(line) > 66 and line[64:66] in {"  ", " *"}, line
        expected, relative = line[:64], line[66:]
        name = "evidence/" + relative.removeprefix("./")
        assert hashlib.sha256(files[name]).hexdigest() == expected
        checked.add(name)
    assert checked == set(files) - {"evidence/SHA256SUMS"}


def check_lifecycle(directory: Path, bash: str, failure: str, exit_code: int) -> None:
    prepare_fixture(directory)
    result = run_fixture(directory, bash, failure)
    assert result.returncode == exit_code, (failure, result.returncode, result.stdout, result.stderr)
    verify_archive(directory, exit_code)
    statuses = sorted((directory / "runs" / "evidence").glob("*/*/session_status.json"))
    all_sessions_finished = failure in {"none", "summary"}
    assert len(statuses) == (8 if all_sessions_finished else 1)
    for status_path in statuses:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["started_at_unix_ns"] < status["ended_at_unix_ns"]
        assert status["exit_code"] == (0 if all_sessions_finished else exit_code)
    calls = [json.loads(line) for line in (directory / "mock_calls.jsonl").read_text().splitlines()]
    benchmarks = [call for call in calls if call["kind"] == "benchmark"]
    if failure == "preflight":
        assert not benchmarks
    if all_sessions_finished:
        assert len(benchmarks) == 8
        paths = [call["arguments"][call["arguments"].index("--greedy-token-path") + 1] for call in benchmarks]
        assert paths == ["full_gather", "distributed_argmax", "distributed_argmax", "full_gather"] * 2


def check_existing_output_untouched(directory: Path, bash: str, existing: str) -> None:
    prepare_fixture(directory)
    if existing == "directory":
        output = directory / "runs" / "evidence"
        output.mkdir(parents=True)
        (output / "sentinel.bin").write_bytes(b"existing evidence must remain byte-identical\x00")
    elif existing == "archive":
        (directory / "evidence.tar.gz").write_bytes(b"existing archive must remain byte-identical\x00")
    elif existing == "checksum":
        (directory / "evidence.tar.gz.sha256").write_bytes(b"existing checksum must remain byte-identical\x00")
    else:
        raise ValueError(f"unknown existing output: {existing}")
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = run_fixture(directory, bash, "none")
    assert result.returncode != 0, (existing, result.stdout, result.stderr)
    after = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert before == after, (existing, result.stdout, result.stderr)
    assert not (directory / "mock_calls.jsonl").exists()


def main() -> None:
    bash = bash_executable()
    scratch = ROOT / "work"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v08 runner ", dir=scratch) as temporary:
        root = Path(temporary)
        assert root.resolve().is_relative_to(scratch.resolve())
        for failure, code in (("none", 0), ("preflight", 17), ("benchmark", 23),
                              ("summary", 29), ("benchmark_and_summary", 23)):
            check_lifecycle(root / failure, bash, failure, code)
            print(f"v0.8 runner lifecycle {failure}: passed", flush=True)
        for existing in ("directory", "archive", "checksum"):
            check_existing_output_untouched(root / existing, bash, existing)
            print(f"v0.8 runner existing {existing}: untouched", flush=True)
    print("v0.8 runner integration passed; all generated data was mock lifecycle evidence.")


if __name__ == "__main__":
    main()
