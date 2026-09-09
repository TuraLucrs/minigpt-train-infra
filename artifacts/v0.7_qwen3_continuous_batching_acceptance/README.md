# v0.7 Qwen3-32B Continuous Batching 昇腾验收证据

本目录保存 `upgrade/v0.7-continuous-batching` 在真实 Ascend Atlas A3 上的精选、可直接审阅
证据。仓库根目录的 `v0.7_ascend_evidence.tar.gz` 保存完整原始证据，解压后约 59 MB；精选
文件不是替代品，而是方便 GitHub 直接浏览的索引。

## 验收对象

| 项目 | 值 |
|---|---|
| 日期 | 2026-09-09（UTC+8） |
| 实机运行提交 | `0bad74ef01e083b8e23d388ed145a80dc81dc309` |
| 模型 | Qwen3-32B dense，32,762,123,264 参数 |
| 权重 | BF16 safetensors，17 个分片；报告内嵌完整 SHA-256 清单 |
| 设备 | Atlas A3，4 张物理卡 × 2 芯片；使用 logical device 0–7 |
| 软件栈 | Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post5.dev20260821、CANN 9.2.0-V100R001C12B080 |
| 分布式后端 | HCCL |
| 比较布局 | TP8、2×TP4、4×TP2；三者均使用同一 8 个 logical devices |

## 证据完整性

- 完整压缩包 SHA-256：`0e26af133429bb3e02cb8738ab9495e6daa6ebc13b3bed4984608b5818389d94`；
- 解压后共 225 个文件，其中 `SHA256SUMS` 覆盖的 143 个核心文件全部通过；
- 正式矩阵为 3 类 workload × open/closed loop × 3 种布局，共 18 个布局运行；
- 每个正式报告包含 1 次 warmup、3 次 measured repeat、逐请求 token/延迟、逐步 batch、
  KV slot、逐 rank HBM、Git/权重 provenance；
- 每个布局都有 manifest、源 workload 和 8-device 遥测；六份 comparison 均能从原始文件
  重算为 `formal_qwen3_32b_tp8_vs_2xtp4_vs_4xtp2`；
- 最终 `v0.7_acceptance.json` 为
  `formal_v0.7_qwen3_32b_ascend_continuous_batching_acceptance`，矩阵完整且无缺失原因。

仓库 CI 会解压完整包、执行 `SHA256SUMS`，再由最终源码读取 manifest、原始逐副本报告、
workload 与 telemetry 重新计算六份 comparison 和最终验收，避免只信任归档中的 formal 标签。

## 正式结果

下表是三次 measured repeat 的中位数。括号内是相对同 workload/mode 的 TP8 baseline 的
goodput 倍数。

| Workload | 模式 | TP8 goodput req/s | 2×TP4 goodput req/s | 4×TP2 goodput req/s | 最优布局 |
|---|---|---:|---:|---:|---|
| short-short | closed | 9.756（1.00×） | **10.044（1.03×）** | 7.810（0.80×） | 2×TP4 |
| short-short | open | 7.834（1.00×） | **8.528（1.09×）** | 6.572（0.84×） | 2×TP4 |
| long-prefill/short-decode | closed | 8.070（1.00×） | 9.343（1.16×） | **10.440（1.29×）** | 4×TP2 |
| long-prefill/short-decode | open | 5.360（1.00×） | 4.269（0.80×） | **6.055（1.13×）** | 4×TP2 |
| mixed | closed | 2.497（1.00×） | 3.263（1.31×） | **3.491（1.40×）** | 4×TP2 |
| mixed | open | 1.179（1.00×） | 2.872（2.44×） | **5.004（4.24×）** | 4×TP2 |

对应的最优输出吞吐为：short-short closed/open 分别 160.709/136.445 tok/s；
long-prefill closed/open 分别 83.517/50.819 tok/s；mixed closed/open 分别
67.921/97.348 tok/s。

这组结果证明 v0.6 的多卡容量能力已经在 v0.7 转化为真实服务吞吐收益：短请求更适合
2×TP4；长 prefill 和 mixed 更适合 4×TP2，尤其 mixed open-loop goodput 达到 TP8 的
4.24 倍。它不表示 4×TP2 会让单个请求快 4.24 倍。

## 测量协议与边界

正式参数在完整矩阵前冻结：全局 slots/queue 为 32/128、`max_seq_len=4096`、closed-loop
clients=64、open-loop trace 到达间隔 50 ms、TTFT/TPOT/E2E SLO 为
15000/500/30000 ms，遥测间隔 200 ms。

NPU BF16 的 greedy 输出会在批形状变化、logit 近平局时出现逐位漂移。正式 open-loop 先在
warmup 记录 `(submit_count, action, wait_us)`，measured repeats 重放同一 admission action
script，使各 repeat 的批组成和输出可审计一致；吞吐与延迟仍使用每轮真实墙钟。报告以
`protocol.open_loop_admission_scripted=true` 明示此口径。因此它是确定性的离线 open-loop
准入重放，不应被描述为独立线程驱动的线上流量发生器。

完整运行中记录过外部 HBM 占用导致的 OOM 重试，以及本次实验自身的端口冲突。受影响产物
没有混入最终 18/18 矩阵；第一次被取代的运行保存在 `_superseded_first_matrix/`，过程详见
`RUN_LOG.md`。其中“`0bad74e` 未推送”是采集当时的历史状态，该提交现已进入 GitHub 分支。

## 文件索引

- `v0.7_acceptance.json`：最终六组合验收和每组最佳布局；
- `comparisons/*.json`：六份可审阅的三布局完整比较；
- `SLO_FREEZE.md`：正式运行前冻结的参数与依据；
- `RUN_LOG.md`：运行顺序、故障、重试和最终矩阵；
- `SHA256SUMS.txt`：本精选目录内文件的可移植哈希；
- `../../v0.7_ascend_evidence.tar.gz`：逐副本报告、manifest、workload、完整 telemetry、
  控制台日志、校准与 superseded 证据。
