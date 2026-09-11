# PUSH_NOTES — v0.8 Decode A/B Atlas A3 证据包（分层精选版）

本分支基于 `upgrade/v0.8-decode-critical-path` @ `2a661b1`，提交 Atlas A3
（8 逻辑 NPU，TP8）上 v0.8 decode A/B 的证据。文件按三个层次组织；
实机完整归档（含 PROF_*/FRAMEWORK 中间产物，3.1GB）保留在实机容器未推送，
本包只含结论复现所需内容。

## 一、必须要看（5 分钟读完）

| 文件 | 内容 |
|---|---|
| `CHECKER_DIAGNOSIS.md` | 诊断：官方 summarize 判 incomplete 的唯一原因是 checker 期望值写错（期望类名 vs 管线记录 implementation_name）；含修正后的五门禁核算与研究判定 |
| `independent_ab_verification.json` | 逐 session 数值 + 门禁核算（机器可读） |
| `official/decode_vocab_ab.json` | 官方汇总（complete=false + 唯一 incomplete 原因原文） |
| `official/RUN_LOG.md` | 官方 markdown 报告 |
| `official/EXIT_STATUS` | 官方退出码（2，由 checker bug 所致） |

## 二、过程内容，但复现/验证链必需

| 内容 | 作用 |
|---|---|
| `sessions/<case>/<session>/replica-00.json` | 3 次 measured 的逐 run 原始指标、路径行数、KV/内存、模型身份与权重哈希（§4 表格与门禁核算的数据源） |
| `sessions/.../profile_summary.json` | 非重叠通信占比等 profiler 汇总（门禁 3 数据源） |
| `sessions/.../layout_manifest.json` | 8 rank 汇聚指纹（协议/容量/分布式拓扑核验） |
| `sessions/.../telemetry_before.json` | 每次启动前 NPU 空闲门禁证据（8 次全部 HBM 4.0%/AICore 0.0%） |
| `sessions/.../telemetry.json` | 作业期间 200ms NPU 采样（时间窗与 measured runs 对齐核验） |
| `sessions/.../session_status.json` | session 起止 UNIX ns 与退出码（8/8 = 0） |
| `sessions/.../console.log` | benchmark 控制台输出（goodput 摘要与 profile 解析记录） |
| `sessions/.../source_workload.json` | 实际使用的冻结 workload（与 FORMAL_CASES sha256 核对） |
| `frozen_workloads/` | 从 v0.7 证据包解出的冻结 workload 源文件 |
| `sessions/.../profile/rank_*__*.gz` | 每 rank 解析后产物（trace_view.json + kernel/operator/task/step/communication 等 CSV/JSON，gzip 逐文件）——通信占比与 trace 结论的原始数据 |
| `official/SHA256SUMS.script` | 实机完整证据目录的脚本原始清单（11,210 文件） |
| `SHA256SUMS` | 本包全量清单（覆盖包内每个文件的 sha256） |

## 三、有意排除（复现结论不需要）

- `PROF_*`：CANN 解析前的原始中间格式（实机归档内保留，推送包不含）；
- `FRAMEWORK`：host 侧取证数据（同上）；
- profiler `logs/`、`*.db`（analysis.db / ascend_pytorch_profiler_0.db 为中间
  sqlite，CSV 已承载同等信息）。

## 结论一句话

8/8 session 运行正常（exit 0，0 OOM，预检 HBM 4.0%/AICore 0.0%）；官方
incomplete 的唯一原因是 `FORMAL_CAPACITY["runner"]` 期望类名而报告记录
implementation_name（`serving.py:1113` vs `decode_critical_path.py:56`）。
修正后按文档 §5 判定 `full_vocab_gather_not_end_to_end_bottleneck`：
候选路径无端到端收益（short -1.1%、mixed +0.2%，通信占比不降），应停止扩展。
