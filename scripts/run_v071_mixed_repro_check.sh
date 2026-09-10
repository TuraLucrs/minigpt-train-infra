#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${MODEL_DIR:?请设置 Qwen3-32B 模型目录 MODEL_DIR}"
: "${CANN_VERSION:?请设置完整 CANN_VERSION}"
: "${INTERCONNECT_TOPOLOGY:?请设置 INTERCONNECT_TOPOLOGY}"

output_root="${REPRO_OUTPUT_ROOT:-runs/v071_mixed_repro_check}"
archive_output="${REPRO_ARCHIVE:-${repo_root}/v0.7.1_mixed_repro_evidence.tar.gz}"
port_base="${MASTER_PORT_BASE:-29710}"
telemetry_interval_ms="${NPU_TELEMETRY_INTERVAL_MS:-200}"
host_telemetry_interval_ms="${HOST_TELEMETRY_INTERVAL_MS:-500}"
session_gap_seconds="${SESSION_GAP_SECONDS:-0}"
logical_devices="0,1,2,3,4,5,6,7"
workload_dir="${WORKLOAD_DIR:-${repo_root}/runs/v071_repro_frozen_workload}"
sequence=(tp8 4xtp2 4xtp2 tp8 4xtp2 tp8 tp8 4xtp2)
telemetry_pid=""
host_telemetry_pid=""

stop_samplers() {
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" || true
  fi
  if [[ -n "${host_telemetry_pid}" ]] && kill -0 "${host_telemetry_pid}" 2>/dev/null; then
    kill -TERM "${host_telemetry_pid}" 2>/dev/null || true
    wait "${host_telemetry_pid}" || true
  fi
  telemetry_pid=""
  host_telemetry_pid=""
}
trap stop_samplers EXIT
trap 'stop_samplers; exit 130' INT
trap 'stop_samplers; exit 143' TERM

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR 不存在：${MODEL_DIR}" >&2
  exit 1
fi
if [[ -e "${output_root}" ]]; then
  echo "输出目录已经存在，拒绝混入旧证据：${output_root}" >&2
  exit 1
fi
if [[ -e "${archive_output}" ]]; then
  echo "证据归档已经存在，拒绝覆盖：${archive_output}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "正式复现检查要求 tracked worktree clean" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" && -n "${WORKLOAD_DIR:-}" ]]; then
  echo "WORKLOAD_DIR 不存在：${workload_dir}" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" ]]; then
  mkdir -p "${workload_dir}"
  tar -xOf "${repo_root}/v0.7_ascend_evidence.tar.gz" \
    "runs/v07/workloads/mixed.json" > "${workload_dir}/mixed.json"
fi
if [[ ! -f "${workload_dir}/mixed.json" ]]; then
  echo "缺少冻结 mixed workload：${workload_dir}/mixed.json" >&2
  exit 1
fi

telemetry_args=(
  --target 0=0:0
  --target 1=0:1
  --target 2=1:0
  --target 3=1:1
  --target 4=2:0
  --target 5=2:1
  --target 6=3:0
  --target 7=3:1
  --interval-ms "${telemetry_interval_ms}"
  --query-type common
)

run_session() {
  local position="$1"
  local layout_id="$2"
  local tp_size
  local max_slots
  local max_queue_size
  local session_id
  local session_dir
  local run_status
  local telemetry_status
  local host_telemetry_status

  if [[ "${layout_id}" == "tp8" ]]; then
    tp_size=8
    max_slots=32
    max_queue_size=128
  elif [[ "${layout_id}" == "4xtp2" ]]; then
    tp_size=2
    max_slots=8
    max_queue_size=32
  else
    echo "未知 layout：${layout_id}" >&2
    return 2
  fi

  printf -v session_id 'session-%02d-%s' "${position}" "${layout_id}"
  session_dir="${output_root}/${session_id}"
  mkdir -p "${session_dir}"
  python benchmarks/capture_v071_host_snapshot.py \
    --phase before \
    --output "${session_dir}/host_before.json"

  python benchmarks/sample_npu_telemetry.py \
    "${telemetry_args[@]}" \
    --output "${session_dir}/telemetry.json" \
    > "${session_dir}/telemetry_sampler.log" 2>&1 &
  telemetry_pid=$!
  python benchmarks/sample_host_telemetry.py \
    --interval-ms "${host_telemetry_interval_ms}" \
    --output "${session_dir}/host_telemetry.json" \
    > "${session_dir}/host_telemetry_sampler.log" 2>&1 &
  host_telemetry_pid=$!

  set +e
  torchrun \
    --master-addr 127.0.0.1 \
    --master-port "$((port_base + position - 1))" \
    --nproc-per-node 8 \
    benchmarks/infer_qwen3_continuous_batching.py \
    --model-dir "${MODEL_DIR}" \
    --workload "${workload_dir}/mixed.json" \
    --mode open_loop \
    --deterministic-open-loop \
    --tp-size "${tp_size}" \
    --max-slots "${max_slots}" \
    --max-seq-len 4096 \
    --max-queue-size "${max_queue_size}" \
    --ttft-slo-ms 15000 \
    --tpot-slo-ms 500 \
    --e2e-slo-ms 30000 \
    --warmup 1 \
    --repeats 3 \
    --chat-template \
    --device npu \
    --precision bf16 \
    --backend hccl \
    --distributed-timeout-seconds 1800 \
    --hash-weights \
    --layout-id "${layout_id}" \
    --logical-device-ids "${logical_devices}" \
    --physical-card-count 4 \
    --chips-per-card 2 \
    --interconnect-topology "${INTERCONNECT_TOPOLOGY}" \
    --cann-version "${CANN_VERSION}" \
    --run-label "v0.7.1-mixed-repro-${session_id}" \
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

  if kill -0 "${host_telemetry_pid}" 2>/dev/null; then
    kill -TERM "${host_telemetry_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${host_telemetry_pid}"
  host_telemetry_status=$?
  set -e
  host_telemetry_pid=""

  python benchmarks/capture_v071_host_snapshot.py \
    --phase after \
    --output "${session_dir}/host_after.json"
  if [[ "${run_status}" -ne 0 ]]; then
    echo "${session_id} benchmark 失败：${run_status}" >&2
    return "${run_status}"
  fi
  if [[ "${telemetry_status}" -ne 0 ]]; then
    echo "${session_id} telemetry 失败：${telemetry_status}" >&2
    return "${telemetry_status}"
  fi
  if [[ "${host_telemetry_status}" -ne 0 ]]; then
    echo "${session_id} host telemetry 失败：${host_telemetry_status}" >&2
    return "${host_telemetry_status}"
  fi
  if [[ "${session_gap_seconds}" != "0" ]]; then
    sleep "${session_gap_seconds}"
  fi
}

mkdir -p "${output_root}"
for index in "${!sequence[@]}"; do
  position=$((index + 1))
  echo "===== ${position}/8 ${sequence[index]} ====="
  run_session "${position}" "${sequence[index]}"
done

python benchmarks/summarize_v071_mixed_repro.py \
  --root "${output_root}" \
  --output "${output_root}/mixed_repro_summary.json" \
  --markdown-output "${output_root}/RUN_LOG.md"

(
  cd "${output_root}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
tar -czf "${archive_output}" \
  -C "$(dirname "${output_root}")" \
  "$(basename "${output_root}")"
sha256sum "${archive_output}" > "${archive_output}.sha256"

echo "复现摘要：${output_root}/mixed_repro_summary.json"
echo "运行日志：${output_root}/RUN_LOG.md"
echo "证据归档：${archive_output}"
