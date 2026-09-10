#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${MODEL_DIR:?请设置 Qwen3-32B 模型目录 MODEL_DIR}"
: "${CANN_VERSION:?请设置完整 CANN_VERSION}"
: "${INTERCONNECT_TOPOLOGY:?请设置 INTERCONNECT_TOPOLOGY}"

output_root="${V08_OUTPUT_ROOT:-runs/v08_decode_vocab_ab}"
archive_output="${V08_ARCHIVE:-${repo_root}/v0.8_decode_vocab_ab_evidence.tar.gz}"
port_base="${MASTER_PORT_BASE:-29810}"
logical_devices="0,1,2,3,4,5,6,7"
workload_dir="${WORKLOAD_DIR:-${repo_root}/runs/v08_frozen_workloads}"
telemetry_pid=""
active_session_dir=""
output_owned=0

telemetry_args=(
  --target 0=0:0
  --target 1=0:1
  --target 2=1:0
  --target 3=1:1
  --target 4=2:0
  --target 5=2:1
  --target 6=3:0
  --target 7=3:1
  --interval-ms 200
  --query-type common
)

stop_telemetry() {
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" || true
  fi
  telemetry_pid=""
}
record_session_status() {
  python - "${active_session_dir}/session_status.json" "$1" "${2:-0}" <<'PY'
import json
from pathlib import Path
import sys
import time

path, phase, code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
status = {"started_at_unix_ns": time.time_ns(), "ended_at_unix_ns": None, "exit_code": None}
if phase == "finish":
    status = json.loads(path.read_text(encoding="utf-8"))
    status.update(ended_at_unix_ns=time.time_ns(), exit_code=code)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

finalize() {
  local status=$?
  local summary_status archive_status
  trap - EXIT INT TERM
  set +e
  stop_telemetry
  # 只归档本次创建的目录；前置校验失败不能触碰已有证据。
  if [[ "${output_owned}" -eq 1 ]]; then
    if [[ -n "${active_session_dir}" ]]; then
      record_session_status finish "${status}"
    fi
    python benchmarks/summarize_v08_decode_ab.py \
      --root "${output_root}" \
      --output "${output_root}/decode_vocab_ab.json" \
      --markdown-output "${output_root}/RUN_LOG.md"
    summary_status=$?
    if [[ "${status}" -eq 0 && "${summary_status}" -ne 0 ]]; then
      status=${summary_status}
    fi
    printf '%s\n' "${status}" > "${output_root}/EXIT_STATUS"
    (
      cd "${output_root}" || exit 1
      find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS
    )
    archive_status=$?
    if [[ "${archive_status}" -eq 0 ]]; then
      tar -czf "${archive_output}" \
        -C "$(dirname "${output_root}")" "$(basename "${output_root}")"
      archive_status=$?
    fi
    if [[ "${archive_status}" -eq 0 ]]; then
      sha256sum "${archive_output}" > "${archive_output}.sha256"
      archive_status=$?
    fi
    if [[ "${archive_status}" -ne 0 ]]; then
      echo "证据归档失败，原始文件保留于 ${output_root}" >&2
      [[ "${status}" -ne 0 ]] || status=${archive_status}
    else
      echo "v0.8 A/B 证据（exit=${status}）：${archive_output}"
    fi
  fi
  exit "${status}"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR 不存在：${MODEL_DIR}" >&2
  exit 1
fi
if [[ -e "${output_root}" || -e "${archive_output}" || -e "${archive_output}.sha256" ]]; then
  echo "A/B 输出已经存在，拒绝覆盖或混入旧证据" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "正式 v0.8 A/B 要求 tracked worktree clean" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" && -n "${WORKLOAD_DIR:-}" ]]; then
  echo "WORKLOAD_DIR 不存在：${workload_dir}" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" ]]; then
  mkdir -p "${workload_dir}"
  for workload in short_short mixed; do
    tar -xOf "${repo_root}/v0.7_ascend_evidence.tar.gz" \
      "runs/v07/workloads/${workload}.json" > "${workload_dir}/${workload}.json"
  done
fi
for workload in short_short mixed; do
  if [[ ! -f "${workload_dir}/${workload}.json" ]]; then
    echo "缺少冻结 workload：${workload_dir}/${workload}.json" >&2
    exit 1
  fi
done
python - "${workload_dir}" <<'PY'
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, "src")
from minigpt.decode_critical_path import FORMAL_CASES

for case in FORMAL_CASES.values():
    path = Path(sys.argv[1]) / (case["workload_class"] + ".json")
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["source_file_sha256"]:
        raise SystemExit(f"冻结 workload 哈希不匹配：{path}")
PY

run_session() {
  local case_id="$1"
  local workload_name="$2"
  local position="$3"
  local greedy_path="$4"
  local master_port="$5"
  local profile_skip_steps="$6"
  local profile_warmup_steps="$7"
  local session_id
  local session_dir
  local run_status
  local telemetry_status

  printf -v session_id 'session-%02d-%s' "${position}" "${greedy_path}"
  session_dir="${output_root}/${case_id}/${session_id}"
  mkdir -p "${session_dir}"
  active_session_dir="${session_dir}"
  record_session_status start

  python benchmarks/sample_npu_telemetry.py \
    "${telemetry_args[@]}" \
    --duration-seconds 1.0 \
    --min-samples 3 \
    --output "${session_dir}/telemetry_before.json"
  python benchmarks/check_v08_npu_idle.py \
    --telemetry "${session_dir}/telemetry_before.json"

  python benchmarks/sample_npu_telemetry.py \
    "${telemetry_args[@]}" \
    --output "${session_dir}/telemetry.json" \
    > "${session_dir}/telemetry_sampler.log" 2>&1 &
  telemetry_pid=$!

  set +e
  torchrun \
    --master-addr 127.0.0.1 \
    --master-port "${master_port}" \
    --nproc-per-node 8 \
    benchmarks/infer_qwen3_continuous_batching.py \
    --model-dir "${MODEL_DIR}" \
    --workload "${workload_dir}/${workload_name}.json" \
    --mode open_loop \
    --deterministic-open-loop \
    --tp-size 8 \
    --max-slots 32 \
    --max-seq-len 4096 \
    --max-queue-size 128 \
    --ttft-slo-ms 15000 \
    --tpot-slo-ms 500 \
    --e2e-slo-ms 30000 \
    --warmup 1 \
    --repeats 3 \
    --chat-template \
    --greedy-token-path "${greedy_path}" \
    --device npu \
    --precision bf16 \
    --backend hccl \
    --distributed-timeout-seconds 1800 \
    --hash-weights \
    --layout-id tp8 \
    --logical-device-ids "${logical_devices}" \
    --physical-card-count 4 \
    --chips-per-card 2 \
    --interconnect-topology "${INTERCONNECT_TOPOLOGY}" \
    --cann-version "${CANN_VERSION}" \
    --run-label "v0.8-${case_id}-${session_id}" \
    --profile \
    --profile-ranks all \
    --profile-skip-steps "${profile_skip_steps}" \
    --profile-warmup-steps "${profile_warmup_steps}" \
    --profile-active-steps 4 \
    --profile-aic-metrics pipe_utilization \
    --output-dir "${session_dir}" \
    2>&1 | tee "${session_dir}/console.log"
  run_status=${PIPESTATUS[0]}
  set -e

  if kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${telemetry_pid}"
  telemetry_status=$?
  set -e
  telemetry_pid=""
  if [[ "${run_status}" -ne 0 ]]; then
    echo "${case_id}/${session_id} benchmark 失败：${run_status}" >&2
    return "${run_status}"
  fi
  if [[ "${telemetry_status}" -ne 0 ]]; then
    echo "${case_id}/${session_id} telemetry 失败：${telemetry_status}" >&2
    return "${telemetry_status}"
  fi

  python benchmarks/summarize_v071_profile.py \
    --layout-manifest "${session_dir}/layout_manifest.json" \
    --output "${session_dir}/profile_summary.json"
  record_session_status finish 0
  active_session_dir=""
}

mkdir -p "${output_root}"
output_owned=1

# 每个 workload 都采用 full/candidate/candidate/full，抵消单调机器状态漂移。
for position in 1 2 3 4; do
  paths=(full_gather distributed_argmax distributed_argmax full_gather)
  run_session short_decode short_short "${position}" "${paths[position - 1]}" \
    "$((port_base + position - 1))" 8 2
done
for position in 1 2 3 4; do
  paths=(full_gather distributed_argmax distributed_argmax full_gather)
  run_session mixed_decode mixed "${position}" "${paths[position - 1]}" \
    "$((port_base + 4 + position - 1))" 14 1
done

# EXIT handler 对成功、中断和失败执行相同的摘要与归档流程。
