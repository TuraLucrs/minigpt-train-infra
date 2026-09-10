#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${MODEL_DIR:?请设置 Qwen3-32B 模型目录 MODEL_DIR}"
: "${CANN_VERSION:?请设置完整 CANN_VERSION}"
: "${INTERCONNECT_TOPOLOGY:?请设置 INTERCONNECT_TOPOLOGY}"

output_root="${PROFILE_OUTPUT_ROOT:-runs/v071_profiling_gate}"
port_base="${MASTER_PORT_BASE:-29610}"
logical_devices="0,1,2,3,4,5,6,7"
workload_dir="${WORKLOAD_DIR:-${repo_root}/runs/v071_frozen_workloads}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR 不存在：${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" && -n "${WORKLOAD_DIR:-}" ]]; then
  echo "WORKLOAD_DIR 不存在：${workload_dir}" >&2
  exit 1
fi
if [[ ! -d "${workload_dir}" ]]; then
  archive="${repo_root}/v0.7_ascend_evidence.tar.gz"
  if [[ ! -f "${archive}" ]]; then
    echo "缺少 v0.7 冻结证据归档：${archive}" >&2
    exit 1
  fi
  mkdir -p "${workload_dir}"
  for workload in short_short long_prefill_short_decode mixed; do
    tar -xOf "${archive}" "runs/v07/workloads/${workload}.json" \
      > "${workload_dir}/${workload}.json"
  done
fi
for workload in short_short long_prefill_short_decode mixed; do
  if [[ ! -f "${workload_dir}/${workload}.json" ]]; then
    echo "缺少冻结 workload：${workload_dir}/${workload}.json" >&2
    exit 1
  fi
done
if [[ -e "${output_root}" ]]; then
  echo "输出目录已经存在，拒绝混入旧证据：${output_root}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "正式 profiling 要求 tracked worktree clean" >&2
  exit 1
fi

run_point() {
  local case_id="$1"
  local workload_name="$2"
  local mode="$3"
  local layout_id="$4"
  local tp_size="$5"
  local max_slots="$6"
  local max_queue_size="$7"
  local master_port="$8"
  local profile_skip_steps="$9"
  local profile_warmup_steps="${10}"
  local profile_active_steps="${11}"
  local point_dir="${output_root}/${case_id}/${layout_id}"
  local -a mode_args=()

  if [[ "${mode}" == "open_loop" ]]; then
    mode_args+=(--deterministic-open-loop)
  else
    mode_args+=(--closed-loop-clients 32)
  fi

  torchrun \
    --master-addr 127.0.0.1 \
    --master-port "${master_port}" \
    --nproc-per-node 8 \
    benchmarks/infer_qwen3_continuous_batching.py \
    --model-dir "${MODEL_DIR}" \
    --workload "${workload_dir}/${workload_name}.json" \
    --mode "${mode}" \
    "${mode_args[@]}" \
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
    --run-label "v0.7.1-${case_id}-${layout_id}" \
    --profile \
    --profile-ranks all \
    --profile-skip-steps "${profile_skip_steps}" \
    --profile-warmup-steps "${profile_warmup_steps}" \
    --profile-active-steps "${profile_active_steps}" \
    --profile-aic-metrics pipe_utilization \
    --output-dir "${point_dir}"

  python benchmarks/summarize_v071_profile.py \
    --layout-manifest "${point_dir}/layout_manifest.json" \
    --output "${point_dir}/profile_summary.json"
}

mkdir -p "${output_root}"

# 三个问题、每题两个 layout；每个点都是独立 8 进程作业。
run_point short_decode_replica short_short open_loop 2xtp4 4 16 64 "$((port_base + 0))" 8 2 4
run_point short_decode_replica short_short open_loop 4xtp2 2 8 32 "$((port_base + 1))" 8 2 4
run_point long_prefill_scaling long_prefill_short_decode closed_loop tp8 8 32 128 "$((port_base + 2))" 6 1 4
run_point long_prefill_scaling long_prefill_short_decode closed_loop 4xtp2 2 8 32 "$((port_base + 3))" 6 1 4
run_point mixed_overload mixed open_loop tp8 8 32 128 "$((port_base + 4))" 14 1 4
run_point mixed_overload mixed open_loop 4xtp2 2 8 32 "$((port_base + 5))" 14 1 4

python benchmarks/summarize_v071_profiling_gate.py \
  --point "short_decode_replica/2xtp4=${output_root}/short_decode_replica/2xtp4/layout_manifest.json" \
  --point "short_decode_replica/4xtp2=${output_root}/short_decode_replica/4xtp2/layout_manifest.json" \
  --point "long_prefill_scaling/tp8=${output_root}/long_prefill_scaling/tp8/layout_manifest.json" \
  --point "long_prefill_scaling/4xtp2=${output_root}/long_prefill_scaling/4xtp2/layout_manifest.json" \
  --point "mixed_overload/tp8=${output_root}/mixed_overload/tp8/layout_manifest.json" \
  --point "mixed_overload/4xtp2=${output_root}/mixed_overload/4xtp2/layout_manifest.json" \
  --output "${output_root}/profiling_gate.json"

echo "v0.7.1 Profiling Gate 输出：${output_root}/profiling_gate.json"
