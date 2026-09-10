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

每次启动前单独采集八个 NPU 的 HBM 和 AICore；任一 device 超过 HBM 10% 或 AICore 5%
即拒绝运行。作业期间继续以 200 ms 采集 NPU 状态。这样不会再次接受 v0.7 中被并发负载
污染的基准。

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

若任何场景出现超过 2% 的原始吞吐或 goodput 回退，状态为 `candidate_regressed`。若没有
场景达到上述收益门槛，状态为 `full_vocab_gather_not_end_to_end_bottleneck`，停止继续扩展
这条优化。

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
