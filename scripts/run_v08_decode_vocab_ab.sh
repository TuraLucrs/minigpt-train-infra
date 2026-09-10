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
trap stop_telemetry EXIT
trap 'stop_telemetry; exit 130' INT
trap 'stop_telemetry; exit 143' TERM

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR 不存在：${MODEL_DIR}" >&2
  exit 1
fi
if [[ -e "${output_root}" || -e "${archive_output}" ]]; then
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

  python benchmarks/sample_npu_telemetry.py \
    "${telemetry_args[@]}" \
    --duration-seconds 0.01 \
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
}

mkdir -p "${output_root}"

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

python benchmarks/summarize_v08_decode_ab.py \
  --root "${output_root}" \
  --output "${output_root}/decode_vocab_ab.json" \
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

echo "v0.8 A/B 摘要：${output_root}/decode_vocab_ab.json"
echo "v0.8 A/B 日志：${output_root}/RUN_LOG.md"
echo "v0.8 A/B 证据：${archive_output}"
