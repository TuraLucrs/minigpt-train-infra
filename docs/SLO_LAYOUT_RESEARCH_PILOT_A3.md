# A3 SLO-aware TP / Replica / Routing 先导实验

## 目标与边界

这是一组紧接 v0.9 的独立科研先导实验，不是 v0.9 重跑，也不是尚未验证的 v1.0
控制器。Router 固定为已有的 `least_projected_load`，本轮不比较新的 routing 算法。它只回答
一个最小研究问题：在固定 8 个 A3 logical devices、同一 Qwen3-32B
和相同全局调度容量下，最优的 TP/副本布局是否会随 workload/SLO operating point 改变。

当前正在执行的紧凑 v0.9 三个 case 不包含在本实验中：

- 不再运行 `kv-long-tp2` 的 recompute/KV A/B；
- 不再运行 `tp-short-2-vs-8` 的单请求 TP A/B；
- 不再运行 `batching-short-tp8` 的 static/continuous A/B；
- 不重复 v0.9 portable Profiler replay。

v0.9 负责后端、Profiler、KV/TP/Batching 基线；本实验只运行 Continuous Batching
服务布局，减少最后机时中的重复采集。

## 预注册矩阵

配置的唯一事实源是 `configs/slo_layout_pilot_a3.json`。

| operating point | workload / replay | SLO（TTFT / TPOT / E2E） | 目的 |
|---|---|---|---|
| `short_strict_closed` | 冻结 `short_short` / closed-loop 64 clients | 2000 / 200 / 5000 ms | 延迟敏感短请求 |
| `mixed_standard_open` | 冻结 `mixed` / deterministic open-loop | 15000 / 500 / 30000 ms | 混合请求服务 goodput |

每个 operating point 比较 `TP8`、`2×TP4`、`4×TP2`。三者始终使用 logical
devices 0–7，并保持全局 32 slots、128 queue entries：副本越多，仅按副本数等分容量。
每个布局包含 1 次 warmup、3 次 measured repeats 和 2 个独立 session。

两个 session 使用反向顺序控制时间漂移：

1. `TP8 → 2×TP4 → 4×TP2`
2. `4×TP2 → 2×TP4 → TP8`

因此总量是 `2 operating points × 3 layouts × 2 sessions = 12` 个独立 benchmark
进程，与紧凑 v0.9 同为 12 sessions。每个 session 启动前先对 8 个 device 做至少 3
轮空闲采样；HBM 超过 10%、AICore 超过 5%、缺 device 或采集失败都会阻断。正式测量期间
保留 200ms NPU 遥测、500ms Host 遥测、前后快照、完整 replica reports 和哈希链。

## A3 紧接 v0.9 执行

先让当前 `runs/v09_a3` 完成。随后切换本分支，继续使用刚才 v0.9 的同一个模型目录、
同一份已核验 device map 和同一拓扑描述；使用新的输出目录：

```bash
git switch research/slo-layout-pilot-a3
git pull --ff-only

export CANN_VERSION='与 v0.9 相同的完整版本'
python benchmarks/run_slo_layout_pilot.py \
  --config configs/slo_layout_pilot_a3.json \
  --model-dir /path/to/Qwen3-32B \
  --device-map /path/to/verified-a3-device-map.json \
  --interconnect-topology '与 v0.9 完全相同的拓扑说明' \
  --output-dir runs/slo_layout_pilot_a3 \
  --archive slo_layout_pilot_a3_evidence.tar.gz
```

不要把 `runs/v09_a3` 传给本入口，也不要在当前 v0.9 尚有 worker 或 HBM 占用时强行绕过
preflight。若进程中断，保留原目录，用完全相同的命令追加 `--resume`；已完成 session 会先
重验 marker、文件大小、SHA-256、manifest、telemetry 和 preflight，再决定是否跳过。不完整
attempt 会移动到 `_incomplete/`，不会覆盖。

完成后会生成：

- `runs/slo_layout_pilot_a3/slo_layout_pilot_summary.json`；
- `runs/slo_layout_pilot_a3/RUN_LOG.md`；
- `runs/slo_layout_pilot_a3/SHA256SUMS`；
- `slo_layout_pilot_a3_evidence.tar.gz` 及其 SHA-256 文件。

## 可得结论

汇总先对每个 round 重新验证三布局 manifest、原始 reports、同模型/提交/硬件、相同请求、
调度总容量、八设备遥测覆盖和 replica 起点偏差，再计算跨 session CV。默认 CV 上限 10%，
winner 相对第二名至少领先 5% 才算清晰。

- `layout_flip_supported`：两个 operating point 的稳定最优布局不同，支持继续实现
  workload/SLO-aware selector；仍不是论文级结论。
- `layout_flip_weak_margin`：观察到翻转但领先不足 5%，先扩大重复或负载点，不能急着写控制器。
- `same_layout_wins_tested_points`：当前两个点不支持自适应布局假设，应停止这条最小假设或
  扩大 workload/SLO 范围后再判断。
- `inconclusive_*variation`：机器状态或测量波动仍过大，不能选择研究实现方向。

两次独立 session 只够决定下一步候选，不足以声称可以泛化到其他模型、硬件、到达率或 SLO。
