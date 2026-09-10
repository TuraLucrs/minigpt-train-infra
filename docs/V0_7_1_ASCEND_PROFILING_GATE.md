# v0.7.1 Ascend Profiling Gate

## 1. 版本定位

`v0.7.1` 是 `v0.7` 的诊断增强版本，不提前实现某个 v0.8 优化。它解决路线中的循环依赖：

- v0.8 要求根据真实瓶颈选一个推理专项；
- 旧路线却把正式 Ascend Profiler 放在 v0.9；
- 没有时间线、算子、kernel 和通信证据，就无法严谨选择 v0.8。

因此本版本先建立一个短期但可冻结的 Profiling Gate。它回答“瓶颈在哪里、下一步应验证
什么”，不把阈值命中直接包装成优化结论。

## 2. 四段交付边界

### A. 服务层可解释范围

Continuous Batching scheduler 为 Decode、Prefill、control、tensor prepare、model 和 token
selection 增加 `record_function` 范围。普通 measured replay 额外记录：

- `decode_phase_ms`；
- `prefill_phase_ms`；
- `scheduler_bookkeeping_ms`；
- KV fixed-slot 的峰值容量浪费率和活动 slot 平均内部浪费率。

阶段 wall time 在 model/token selection 后显式同步设备，所以不是单纯的异步 launch 时间。

### B. Ascend 原始证据

`AscendStepProfiler` 延迟导入 `torch_npu`，只在真实 NPU 运行。一个 scheduler step 对应一个
Profiler step。正式协议固定为：

| 字段 | 值 |
|---|---:|
| skip | 8 steps |
| warmup | 2 steps |
| active | 4 steps |
| profiler level | Level1 |
| AiCore metrics | PipeUtilization |
| record shapes | true |
| profile memory / stack | false / false |
| system interconnection | true |
| ranks | 0～7 全部 |

窗口有明确上限，避免长服务 trace 失控膨胀。`profile_memory` 和 stack 在正式 gate 中关闭；若为
定位单点临时打开，结果只能算开发证据，不能替代固定协议。

每个 rank 至少要产出并通过哈希校验：

- `profiler_info*.json`；
- `operator_details.csv`；
- `kernel_details.csv`；
- `step_trace_time.csv`；
- `trace_view.json`；
- `communication.json`。

原始 `PROF_*` 数据保留在 point 目录，但 manifest 只索引可重算诊断结论的关键解析产物，避免
对巨大原始目录重复做全量哈希。

### C. 六点诊断矩阵

矩阵不是重跑 v0.7 的全部 18 点，而是从冻结结果中选三组可以归因的问题，每组只比较两个
layout，共六个独立的 8 进程作业：

| case | workload / mode | layouts | 要回答的问题 |
|---|---|---|---|
| `short_decode_replica` | `short_short` / open-loop | `2xtp4`, `4xtp2` | short request 下 4×TP2 为什么低于 2×TP4？ |
| `long_prefill_scaling` | `long_prefill_short_decode` / closed-loop | `tp8`, `4xtp2` | 长 Prefill 为什么更偏向更多 replica？ |
| `mixed_overload` | `mixed` / open-loop | `tp8`, `4xtp2` | mixed overload 的 goodput 差异来自哪里？ |

所有点继续冻结 v0.7 的 Qwen3-32B、BF16、HCCL、8 logical devices、SLO、全局 32 slots、
128 waiting queue、4096 sequence length、1 次 warmup 和 3 次 measured repeats。TP8、2×TP4、
4×TP2 只改变 TP/replica 形状，不改变全局调度容量。

### D. 证据门禁与冻结

一个 point 只有同时满足下列条件才是正式证据：

1. tracked worktree clean，完整 Git commit、模型目录与权重哈希可追踪；
2. Qwen3-32B、NPU BF16、HCCL、8 个 logical devices；
3. serving layout manifest、workload 内容/文件哈希、设备映射与 profile manifest 一致；
4. measured repeats 和额外 profile replay 的完整输出 digest 一致；
5. profile replay 明确标记为 `measurement_excluded=true`；
6. 8 个 rank 都按固定协议采集，关键 artifact 全部存在且哈希正确；
7. 同一比较中的两个 layout 使用相同 workload 文件和相同 Git commit；
8. 六个 point 全部完整。

任一条件失败仍可以输出开发摘要，但 `selection_ready=false`，不得据此冻结 v0.8 选题。

## 3. 为什么额外跑一次 replay

Profiler 会增加 CPU 调度、trace、shape 记录和落盘开销。把它直接包在三次 measured repeats
外面，会改变 TTFT、TPOT、E2E 和吞吐本身。当前实现的顺序为：

1. 一次普通 warmup；
2. 三次普通 measured replay，形成正式服务指标；
3. engine reset；
4. 复用相同确定性 admission script 做一次有界 profile replay；
5. 对比完整输出 digest；
6. profile replay 只保存 scheduler step 数、wall time 和 digest，不进入服务指标聚合。

这样服务性能和瓶颈诊断绑定到同一代码、workload 与输出，同时避免测量自污染。

## 4. 自动摘要与信号

`profile_summary.json` 汇总每个 rank 的 top operators、top kernels、通信/计算 kernel 时间，
以及 step trace 中 Computing、Communication (Not Overlapped)、Free 和通信重叠比例。

`profiling_gate.json` 再把六点合并成候选方向信号。初始阈值只用于 triage，全部连同观测值写入
报告，便于复核：

| 信号 | 初始触发条件 |
|---|---|
| Paged KV / block manager | 活动 slot 平均内部浪费 ≥ 50%，或峰值容量浪费 ≥ 25% |
| Decode communication path | short Decode 占比 ≥ 60%，且非重叠通信 ≥ 20% |
| Chunked Prefill / Prefix Cache | long 相对 short 的 Prefill 占比增加 ≥ 15 个百分点 |
| Host scheduler path | Profiler Free 占比 ≥ 15% |
| Speculative Decoding / MTP | Decode 占比高但通信不高；另需模型和 acceptance-rate 可行性门禁 |

阈值命中数不是排序分数。v0.8 只能选择一个能用“同模型、同 workload、单一改动”做 A/B 的
方向；Speculative/MTP 还必须先证明 draft/MTP 模型可用和接受率足够。

## 5. Atlas 运行

先检出待验收 commit，确认 8 个 logical devices 可见，并复用 v0.7 冻结的三个 workload
文件。正式执行：

```bash
export MODEL_DIR=/path/to/Qwen3-32B
export WORKLOAD_DIR=/path/to/v0.7/frozen-workloads
export CANN_VERSION='完整版本字符串'
export INTERCONNECT_TOPOLOGY='4 physical cards, 2 logical devices per card'

bash scripts/run_v071_ascend_profiling_gate.sh
```

可用 `PROFILE_OUTPUT_ROOT` 改输出目录，用 `MASTER_PORT_BASE` 改六个作业的起始端口。脚本拒绝
已有输出目录和 dirty tracked worktree，避免旧 trace 混入新证据。

预期目录：

```text
runs/v071_profiling_gate/
  short_decode_replica/{2xtp4,4xtp2}/
  long_prefill_scaling/{tp8,4xtp2}/
  mixed_overload/{tp8,4xtp2}/
  profiling_gate.json
```

每个 point 目录包含原有 layout reports/manifest、`profiler/rank_000`～`rank_007`、
`profile_manifest.json` 和 `profile_summary.json`。

## 6. 完成标准

软件部分必须通过 GitHub Actions 的完整 CPU/Gloo 回归及 profile manifest/解析/篡改测试。
硬件部分必须在 Atlas A3 跑完六点矩阵，`profiling_gate.json` 同时满足：

```text
complete = true
selection_ready = true
```

之后才能审阅 top operators/kernels、step fractions 和服务指标，写出一个 v0.8 研究假设、一个
对照基线、一个验收指标。未完成 Atlas 六点前，v0.7.1 只能称为“软件实现完成、硬件证据待
验收”，不能冻结版本或宣布 v0.8 方向。

## 7. 上游接口依据

- [torch_npu.profiler.profile API](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/torchnpuCustomapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md)
- [Ascend PyTorch Profiler 用户指南](https://github.com/Ascend/pytorch/blob/master/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md)
- [tensorboard_trace_handler API](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/torchnpuCustomapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-tensorboard_trace_handler.md)

本版本只依赖这些窄接口；更长时间的多机 trace、后端抽象扩展、通信微基准和跨平台对比仍
属于后续 Ascend 专项阶段。
