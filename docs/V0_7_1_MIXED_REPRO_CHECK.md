# v0.7.1 mixed open-loop 复现检查

## 1. 要回答的问题

冻结的 v0.7 在同一 8 个 logical devices 上记录到：

| 布局 | goodput 中位数 |
|---|---:|
| TP8 | 1.179 req/s |
| 4×TP2 | 5.004 req/s |

因此当时的比值为 4.24×。v0.7.1 使用同一 source workload digest、同一 64 请求、同一
TTFT/TPOT/E2E SLO 再测时，结果变为 TP8 4.005 req/s、4×TP2 4.719 req/s，即 1.178×。

旧 TP8 三个 measured repeats 本身为 1.254、1.179、0.531 req/s，波动明显；旧遥测中的
AICore 当前频率始终为 1800 MHz，温度范围约 32–42°C。因此不能把差异直接归因于降频或
过热，也不能继续把 4.24× 当作稳定的布局倍率。

本检查只复现并定位这项差异，不修改 `v0.7.1-ascend-profiling-gate` tag，不启动 v0.8。

## 2. KV 浪费率与整机显存不是一个指标

当前内部浪费率按活动 slot 计算：

```text
(slot 预留 tokens - slot 已用 tokens) / slot 预留 tokens
```

每个静态 slot 都按 `max_seq_len=4096` 预留。short 请求只使用很少的 token，因此浪费率可
接近 99%；它不由模型参数量或整机总显存直接决定。

模型大小、TP 切分和整机显存决定的是这种浪费是否形成现实容量压力。Atlas A3 有 16 个
64 GB logical devices，整机约 1024 GB；正式布局比较为保持设备预算相同，只使用 logical
devices 0–7，即约 512 GB。TP8 的单 rank 峰值约 13.2 GB，因此当前实验确有很大显存余量。
所以现有证据能证明静态 KV 低效，不能证明 Paged KV 是当前 goodput 的第一瓶颈。

## 3. 为什么使用 ABBA + BAAB

正式顺序固定为：

```text
TP8, 4×TP2, 4×TP2, TP8, 4×TP2, TP8, TP8, 4×TP2
```

它由一组 ABBA 和一组反向 BAAB 组成。每种布局出现四次，并分布在实验前后；这样可以把
布局效应与缓存逐渐变热、机器负载变化、温度变化等单调时间效应分开。

每个 session 是一个独立 8 进程作业，包含：

- 1 次 warmup；
- 3 次 measured replay；
- frozen mixed workload；
- open-loop deterministic admission；
- TP8 为 32 slots/128 queue，4×TP2 为每副本 8 slots/32 queue；
- 全局仍为 32 slots/128 queue；
- logical devices 恰好为 0–7；
- 不启用 Profiler，避免额外 replay 改变实验时长和机器热状态；
- measured 区间内连续采集 AICore 利用率、频率、温度、功耗、HBM/带宽等 `npu-smi` 指标；
- measured 区间内连续采集 Host CPU、IO wait、load、可用内存和 PSI；
- 每个 session 前后采集 Host load、内存、CPU ticks 与 PSI 快照。

## 4. 真机运行

必须检出远端分支 `investigate/v0.7.1-mixed-repro` 的干净提交，然后执行：

```bash
export MODEL_DIR=/data/models/Qwen3-32B
export CANN_VERSION='9.2.0-V100R001C12B080'
export INTERCONNECT_TOPOLOGY='Atlas A3 cards0-3, 2 chips/card, HCCS intra-card + cross-card board interconnect'

bash scripts/run_v071_mixed_repro_check.sh
```

可选环境变量：

- `REPRO_OUTPUT_ROOT`：默认 `runs/v071_mixed_repro_check`；
- `REPRO_ARCHIVE`：默认仓库根目录 `v0.7.1_mixed_repro_evidence.tar.gz`；
- `MASTER_PORT_BASE`：默认 29710；
- `NPU_TELEMETRY_INTERVAL_MS`：默认 200；
- `HOST_TELEMETRY_INTERVAL_MS`：默认 500；
- `SESSION_GAP_SECONDS`：默认 0，正式首轮不要修改；
- `WORKLOAD_DIR`：仅在明确使用另一份、digest 相同的 frozen workload 时设置。

脚本结束后会生成：

```text
runs/v071_mixed_repro_check/
  session-01-tp8/
  ...
  session-08-4xtp2/
  mixed_repro_summary.json
  RUN_LOG.md
  SHA256SUMS
v0.7.1_mixed_repro_evidence.tar.gz
v0.7.1_mixed_repro_evidence.tar.gz.sha256
```

## 5. 判定边界

自动判定使用三个层次：

1. 会话内：每个 session 的三个 measured repeats 是否稳定；
2. 跨会话：同布局四次独立作业是否稳定，前半程与后半程是否存在顺序效应；
3. 状态关联：按布局中位数归一化 goodput 后，检查它与 NPU 频率、温度、功耗、AICore
   利用率以及 Host CPU、IO wait、load 和 PSI 的相关性。

可能状态：

- `historical_v07_tp8_slowdown_not_reproduced`：当前 TP8 稳定接近 v0.7.1，旧慢值未复现；
- `v071_tp8_speedup_not_reproduced`：当前 TP8 稳定接近 v0.7，v0.7.1 快值未复现；
- `stable_third_regime`：得到第三种稳定状态；
- `order_or_machine_state_effect`：前后半程差异明显；
- `unstable_unexplained`：仍存在无法由现有状态量解释的波动。

相关性只用于确定下一轮取证方向，不冒充物理因果。如果结果稳定地支持当前 v0.7.1 状态，
可以确认 4.24× 是旧 TP8 单次运行状态造成的历史倍率；如果仍不稳定，再只对异常 session
补定向 HCCL/Host trace，而不是重跑完整六点 Profiler。
