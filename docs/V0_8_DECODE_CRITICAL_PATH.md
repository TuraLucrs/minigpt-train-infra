# v0.8 Decode 词表通信关键路径

## 1. 研究问题

v0.7.1 的 short workload 中 Decode wall time 占比约 85%，Profiler 的非重叠通信占比最高
达到 35.8%。当前 TP Qwen3 在每个 Prefill/Decode 模型调用后执行：

```text
每个 rank 计算 [B, V/TP] local logits
→ AllGather 成每个 rank 上的 [B, V] full logits
→ rank 0 greedy argmax
→ broadcast token ids
```

对 greedy decoding，下一步只需要一个 token id。v0.8 第一阶段验证：把完整词表 AllGather
替换为局部候选聚合，能否降低 Decode 非重叠通信并形成可观测的端到端收益。

它是一个有终止条件的研究假设。若 collective 明显缩小但原始完成吞吐和 TPOT 没有改善，
说明 full-vocab gather 不是当前端到端主瓶颈，后续转向计算、Host launch 或同步空洞。

## 2. 候选算法与等价性

每个 rank 对自己的连续 vocab shard 求：

```text
(local_max_score, local_token_id)
```

然后只 AllGather 每行两个 FP32 值，rank 0 在 TP 个候选中选全局最大值，再沿用原路径广播
token id。vocab shard 按 token id 递增排列，局部和全局 `argmax` 都选择第一个最大项，所以
分数完全相同时仍与完整 `[B,V]` 上的最小 token id tie-break 一致。Qwen3 的 token id 小于
`2^24`，因此装入 FP32 时整数值仍然精确；实现也显式拒绝超出该范围的词表。

Qwen3-32B 的 vocab size 为 151936。TP8 下每个 rank 的 BF16 local logits 每请求为：

```text
151936 / 8 × 2 bytes = 37984 bytes
```

候选输入为：

```text
2 × FP32 = 8 bytes
```

因此单次 collective 的每 rank 输入 payload 理论缩小 4748×。这只是通信量变化，不等同于
4748× 性能提升；LM Head、Transformer、collective 固定延迟和 Host 调用均未消失。

候选路径只用于整个 batch 都是 greedy 的情况。任一请求使用 temperature/top-k/top-p
sampling 时，该 batch 自动回退到完整 logits，避免改变采样分布。

## 3. 实现边界

- 基线参数：`--greedy-token-path full_gather`；
- 候选参数：`--greedy-token-path distributed_argmax`；
- 默认仍为基线，避免历史命令静默改变；
- Profiler 增加 `vocab_projection`、`vocab_full_all_gather`、
  `vocab_local_argmax`、`vocab_candidate_all_gather`、`vocab_global_argmax` 和
  `vocab_token_broadcast` 范围；
- 服务报告记录每种实际路径处理的请求行数，以及两种 collective 的理论输入 payload；
- CPU/Gloo 门禁覆盖 Prefill、Decode、多 TP degree、分数平局和 sampling fallback。

## 4. 真机 A/B 协议

第一阶段固定 TP8，只测试最直接的两个场景：

| case | workload | 原因 |
|---|---|---|
| `short_decode` | short-short/open-loop | Decode 占比最高，最容易辨认候选收益 |
| `mixed_decode` | mixed/open-loop | 检查真实排队和 SLO 下收益能否保留 |

每个 case 按 `full/candidate/candidate/full` 运行四个独立 session。每个 session 包含一次
warmup、三次 measured replay 和一次排除在性能指标外的有界 Profiler replay。ABBA 顺序
用于抵消单调机器状态漂移。

每次启动前单独采集八个 NPU 的 HBM 和 AICore：观察至少 1 秒、至少三轮查询，每个 device
至少三个不同时间戳且跨度不少于 400 ms。检查整个窗口最大值，任一 device 超过 HBM 10%
或 AICore 5% 即拒绝运行；缺设备、查询错误、采集未完成也不能当作空闲。作业期间继续以
200 ms 采集 NPU 状态。该检查用于降低复现 v0.7 设备争用污染的风险，不能识别外部进程所有者。

## 5. 决策门禁

候选路径只有同时满足下列条件才进入更大的 layout/workload 矩阵：

1. 两个 session 的 goodput 跨会话 CV 不超过 10%；
2. 原始完成吞吐至少提升 3%；
3. Profiler 非重叠通信占比至少下降 3 个百分点；
4. goodput 回退不超过 2%；
5. 请求终态、停止原因、输入长度和输出长度跨 A/B 一致。

逐 token 输出差异会完整记录。由于不同执行速度可能改变 open-loop 的 batch 边界，只要每个
session 内重复确定、跨路径计算长度相同，它作为数值警告而不直接否决性能 A/B。算法本身的
精确等价性由固定 batch 的 CPU/Gloo 测试负责。

只有证据完整且稳定时才作研究判断。跨 session CV 超限、缺失/损坏 artifact、实际路径与
逐 step 行数不符、protocol/容量不符、与冻结源 workload 哈希不符、session 时间重叠或
顺序不符，均为 `incomplete`，不能解释为优化无收益。

若任何场景出现超过 2% 的原始吞吐或 goodput 回退，状态为 `candidate_regressed`。若没有
场景达到上述收益门槛，状态为 `full_vocab_gather_not_end_to_end_bottleneck`，停止继续扩展
这条优化。

基线 goodput 或 TPOT 为零时，对应倍率为 JSON `null` / Markdown `N/A`；回退门槛直接
比较原始数值，避免除零与无穷大。基线完成吞吐为零时无法形成有效吞吐对照，判为证据不足。

## 6. Atlas A3 执行

检出 `upgrade/v0.8-decode-critical-path` 的干净提交：

```bash
export MODEL_DIR=/path/to/Qwen3-32B
export CANN_VERSION='完整版本字符串'
export INTERCONNECT_TOPOLOGY='4 physical cards, 2 logical devices per card'

bash scripts/run_v08_decode_vocab_ab.sh
```

脚本输出：

```text
runs/v08_decode_vocab_ab/decode_vocab_ab.json
runs/v08_decode_vocab_ab/RUN_LOG.md
v0.8_decode_vocab_ab_evidence.tar.gz
v0.8_decode_vocab_ab_evidence.tar.gz.sha256
```

在真机证据返回前，本阶段状态是“候选实现与 A/B 门禁完成，硬件结论待验收”。

## 7. 软件回归与恢复约定

2026-09-11 的补全增加了 rank 间首次运行前的路径/能力一致性检查（每个 engine 一次）、
只读 runner 路径配置、FP16/BF16/FP32 与 NaN/Inf/tie/大 token id 测试、固定 batch 全序列
等价性以及真正的非退化 sampling fallback。CPU/Gloo 同时验证 collective 次序和 payload。
通信量元数据明确为 `estimate`、`collective_input_per_rank_per_row`，4748× 是输入字节比，
不能写成实测通信或端到端加速比。

```bash
python tests/test_qwen3_tp.py
python tests/test_continuous_batching.py
python tests/test_serving_workloads.py
python tests/test_decode_critical_path.py
python tests/test_v08_runner.py
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_qwen3_tp.py
```

Windows 的 PyTorch 2.10 CPU wheel 在本机无法建立 Gloo transport；本地线程回归与真实
多进程回归分开记录。Linux GitHub Actions 对实际提交运行完整 CPU/Gloo 回归，具体 SHA
和 run 链接以新交接文档为准。测试中的合成测量 fixture 只验证证据门禁，不属于性能产物。

每次 session 写入带 UNIX ns 边界和退出码的 `session_status.json`。脚本无论正常结束、
preflight 拒绝、benchmark 失败还是收到中断，都尝试输出 incomplete 摘要、`EXIT_STATUS`、
`SHA256SUMS` 和 tar/sha256；原始失败退出码优先保留。已有输出拒绝覆盖，失败重跑使用新的
`V08_OUTPUT_ROOT` 和 `V08_ARCHIVE`。若归档失败，原始目录仍保留并明确报错。

当前交付是软件检查点，不打正式硬件验收 tag。根据本轮明确授权，v0.9 软件开发可以先于
v0.8 真机实验完成；v0.7/v0.7.1 冻结 tag 与历史证据保持不变。

## 8. Atlas A3 最终验收

2026-09-11 在 Atlas A3（4 张双芯卡、8 个逻辑 NPU）上以 TP8 完成了全部 8 个独立
session。运行提交为 `2a661b1372b521d04c26fefc91cfc6080c72054b`，所有 session
均正常退出，未发生 OOM；启动前检查的 HBM 最大占用为 4.0%，AICore 最大占用为 0.0%。

| case | 完成吞吐变化 | goodput 变化 | 非重叠通信占比变化 | 判定 |
|---|---:|---:|---:|---|
| `short_decode` | -1.1% | -1.1% | +8.8pp | 未达到收益门槛 |
| `mixed_decode` | +0.2% | +0.9% | +0.9pp | 未达到收益门槛 |

两组同路径 session 的 goodput CV 均低于 10%，候选路径没有超过 2% 的回退，终态、
计算长度和实际 token-selection 路径也通过核验。但它在两个场景中都没有带来至少 3%
的完成吞吐提升，非重叠通信占比也没有下降至少 3 个百分点。因此最终研究状态为
`full_vocab_gather_not_end_to_end_bottleneck`：候选 collective 的理论输入字节虽缩小
4748 倍，但 full-vocab AllGather 不是当前 Decode 端到端主瓶颈。本项目停止扩展该优化，
继续以 `full_gather` 为默认路径；`distributed_argmax` 仅保留为经过正确性验证的实验实现。

真机原始汇总中的 `complete=false` 和 `EXIT_STATUS=2` 原样保留。其唯一 incomplete 原因是
checker 错把 Python 类名 `SlotCachedTensorParallelQwen3ModelRunner` 当作报告值，而 serving
管线一直记录稳定的 `implementation_name=qwen3_tp_slot_kv_cache`。最终代码已对齐真实报告
schema，并增加反向回归用例，防止合成 fixture 再次接受类名字符串。最终提交
`8ab44345e008f3b4f90d33ff852459dd9dfff893` 的 GitHub Actions run
`34564171875` 已通过完整 CPU/Gloo 回归。原始输出、独立复核、8 个 profile manifest 与
精选 rank 产物均保存在 `v08_evidence/`，其中原始失败状态不作事后改写。
