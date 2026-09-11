# v0.8 Decode A/B Atlas A3 实机运行报告与 checker 诊断（外部验收记录）

- 日期：2026-09-11
- 分支/精确提交：`upgrade/v0.8-decode-critical-path` @ `2a661b1372b521d04c26fefc91cfc6080c72054b`
  （`git rev-parse HEAD` 核验一致；运行全程未修改任何 tracked 文件）
- 机器：Atlas A3（4 物理卡 × 2 芯片），TP8，logical devices 0-7（npu-smi 26.2.rc1.b021）
- CANN：9.2.0-V100R001C12B080；torch_npu 2.10.0.post5.dev20260821；torch 2.10.0+cpu；Python 3.12.13
- 模型：`/data/models/Qwen3-32B`（bf16，`--hash-weights`）
- 驱动：`bash scripts/run_v08_decode_vocab_ab.sh`（标准三环境变量；
  torchrun 端口 29810-29817；`HCCL_NPU_SOCKET_PORT_RANGE=60000-60050`）
- CPU 回归：6/6 通过（5 个脚本式测试 + `MINIGPT_RUN_GLOO_TESTS=1` 的 Gloo 多进程回归）

## 1. 运行结果摘要（8/8 session 全部正常结束，0 OOM）

每个 session：NPU 空闲预检（1s 窗口 24 样本，8 次全部 HBM max 4.0% / AICore max 0.0%）
→ torchrun TP8 open-loop（1 warmup + 3 measured + 有界 profile replay）→ 200ms NPU
遥测全程 → `profile_summary.json`。8 个 session `session_status.json` 全部
started/ended/exit_code 完整，退出码全 0。

goodput（=completed，请求/秒，session 内 3 次 measured 的中位数）：

| case | session | greedy path | goodput | 非重叠通信占比 | TPOT 中位（ms） |
|---|---|---|---|---|---|
| short_decode | 01 | full_gather | 7.610 | 0.524 | 245.7 |
| short_decode | 02 | distributed_argmax | 7.333 | 0.550 | 250.8 |
| short_decode | 03 | distributed_argmax | 7.823 | 0.545 | 240.9 |
| short_decode | 04 | full_gather | 7.718 | 0.394 | 240.9 |
| mixed_decode | 01 | full_gather | 3.668 | 0.457 | 338.9 |
| mixed_decode | 02 | distributed_argmax | 3.804 | 0.425 | 335.2 |
| mixed_decode | 03 | distributed_argmax | 3.834 | 0.482 | 342.4 |
| mixed_decode | 04 | full_gather | 3.898 | 0.433 | 328.6 |

基线与 v0.7.1 门禁历史一致（tp8 short_short ~7.5-8 req/s；tp8 mixed ~3.7-4.0），
基线无机器状态漂移异常。

## 2. 官方汇总 incomplete 的唯一原因：checker 期望值字符串写错

`benchmarks/summarize_v08_decode_ab.py` 判定 `complete=False`、`EXIT_STATUS=2`，
8 个 session 的 incomplete 原因完全相同且仅此一条：

```text
正式 scheduler capacity runner 必须为 'SlotCachedTensorParallelQwen3ModelRunner'
```

证据链：

1. 报告中 `engine.runner` 由 `src/minigpt/serving.py:1113` 写入，取
   `self.runner.implementation_name`。这是 v0.7 血统一贯的记录方式，真实运行记录的值是
   `qwen3_tp_slot_kv_cache`；
2. 该 implementation_name 属于的类正是 `SlotCachedTensorParallelQwen3ModelRunner`
   （`src/minigpt/qwen3_tp.py:1040-1043`）——**运行侧 runner 完全正确**；
3. 2a661b1 新增的 `src/minigpt/decode_critical_path.py:56`
   `FORMAL_CAPACITY["runner"] = "SlotCachedTensorParallelQwen3ModelRunner"` 期望的是
   **类名字符串**，而真实 runner 报告从不含类名 → 8/8 session 必然命中；
4. `tests/test_decode_critical_path.py` 的合成 fixture 未覆盖
   `scheduler_capacity.runner` 字段，因此 CPU 回归发现不了该不一致。

即：这不是运行故障，而是 checker 侧常量与报告 schema 的单字符串不一致。
除此之外，其余全部门禁在 8 个 session 上通过（协议、容量数值、workload 哈希、
遥测设备映射与时间窗、模型身份、logical devices、互连拓扑、路径行数等）。

## 3. 修正该字符串后的门禁核算（从原始产物独立复核）

| 门禁 | 阈值 | short_decode | mixed_decode |
|---|---|---|---|
| 1. 同路径跨 session CV ≤10% | 0.10 | full 0.7% / cand 3.2% ✅ | full 2.3% / cand 0.4% ✅ |
| 2. 完成吞吐提升 ≥+3% | 1.03 | **-1.1% ✗** | **+0.2% ✗** |
| 3. 非重叠通信占比下降 ≥3pp | -0.03 | **+8.8pp ✗** | **+0.9pp ✗** |
| 4. goodput 回退 ≤2% | 0.02 | -1.1% ✅ | +0.9% ✅ |
| 5. 终态/长度/路径一致 | 一致 | ✅ | ✅ |

路径行数核验：每个 measured run 的 `serving.token_selection.actual_rows_by_path`
100% 落在配置路径（short 1024 行、mixed 1245 行，与输出 token 数精确相等）。

## 4. 研究判定（按 docs/V0_8_DECODE_CRITICAL_PATH.md §5）

候选 collective 输入字节缩小 4748×（估算口径，`measurement_type=estimate`），
但两个场景完成吞吐均无 ≥3% 提升、非重叠通信占比均未下降 ≥3pp、TPOT 持平。
没有任何场景达到收益门槛，按文档预设状态应为
**`full_vocab_gather_not_end_to_end_bottleneck`**：full-vocab AllGather 不是当前
Decode 端到端主瓶颈，应停止扩展这条优化，后续转向计算、Host launch 或同步空洞
（§1 终止条件触发）。

## 5. 归档说明（两份不同范围的证据）

**实机完整归档（脚本官方产物，保留在实机容器，未推送）**：
`v0.8_decode_vocab_ab_evidence.tar.gz`，SHA-256 =
`05526042bfeb713aa3047c15ab591104fb554e21c3bbf9a21ad4a86fabdc83b4`，11,869 条目，
含全部 CANN profiler 原始产物（含 PROF_* 解析前中间格式与 FRAMEWORK host 侧数据）。
`EXIT_STATUS=2` 由上述 checker bug 所致；8/8 session 本身 exit 0。

**本推送包（按"必看 / 验证链"分层的精简证据包）**：即本仓库的 `v08_evidence/` 目录，
内容与取舍见包内 `PUSH_NOTES.md`。与完整归档相比仅排除了
`PROF_*`（CANN 解析前中间格式）与 `FRAMEWORK`（host 取证备用）两类中间产物，
结论复现不需要它们；`trace_view.json` 与全部分析 CSV/JSON 以 gzip 逐文件收录，
包内 `SHA256SUMS` 覆盖全部文件，官方脚本原始清单保留为 `official/SHA256SUMS.script`。

## 6. 修复建议

1. `decode_critical_path.py` 的 `FORMAL_CAPACITY["runner"]` 改为记录管线实际写入的
   `"qwen3_tp_slot_kv_cache"`；或让 `serving.py` 报告同时记录
   `runner_class`，checker 接受二者之一；
2. `tests/test_decode_critical_path.py` 补一个覆盖 `scheduler_capacity.runner`
   的端到端 fixture，防止同类 fixture/真实报告脱节。
