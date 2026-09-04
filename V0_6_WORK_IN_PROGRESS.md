# v0.6 Tensor Parallel 当前恢复点

更新时间：2026-09-04 UTC

## 基线与分支

- 当前分支：`upgrade/v0.6-qwen3-tensor-parallel`
- v0.5 基线：`3eefbc9241f127a7e5db50d36de37df3f253c4f0`
- 基线标签：`v0.5-qwen3-real-model`
- v0.5 分支与 annotated tag 已通过 GitHub Actions + 精确 Git bundle 上传并核验远端 SHA；
- v0.6 尚未创建最终 tag。真实 Gloo/HCCL 和 32B TP=2/4/8 硬件验收完成前，不得声称版本完成。

## 已实现

### 分布式运行时

- `RuntimeContext.create()` 支持并校验 `device_index`，在 process group 前绑定 local device；
- `DistributedContext` 解析 rank/local rank/world size，自动选择 Gloo/NCCL/HCCL；
- 显式 CUDA/NPU 请求禁止静默回退 CPU；
- 校验已有 process group 的 rank/world size/backend；
- 提供 barrier、sum AllReduce、last-dim AllGather、broadcast 和少量报告值 AllGather；
- `world_size=1` 保留无通信路径，`close()` 只销毁自己创建的 process group。

### Qwen3 Tensor Parallel

- query heads、KV heads、intermediate 和 vocabulary rank plan；
- TP 大于 KV-head 数时，按全局 query group 复制对应 KV head；
- vocabulary-parallel Embedding + hidden AllReduce；
- Q/K/V column parallel、O row parallel；
- gate/up column parallel、down row parallel；
- vocabulary-parallel LM head + logits AllGather；
- rank-local KV Cache；
- safetensors `get_slice()` rank-local 行/列加载，不在每个 rank 建完整 32B 参数；
- rank 0 token 选择后 broadcast，greedy/sample 都不会因 rank RNG 分叉。

### 入口、Benchmark 与证据

- `infer_qwen3_tp.py`：torchrun TP 推理入口；
- `benchmarks/infer_qwen3_tp.py`：单请求或静态 batch，记录加载时间、TTFT、TPOT、吞吐和逐 rank HBM；
- `benchmarks/benchmark_tp_collectives.py`：AllReduce/AllGather 消息大小、延迟和估算流量；
- `benchmarks/summarize_tp_scaling.py`：合并 TP=2/4/8，计算相对最小可运行基线的 speedup/efficiency；
- 正式证据要求 32B、TP≥2、accelerator、完整拓扑、完整权重 hash、干净 Git commit；
- 请求、config、权重或 commit 不同的报告不会被标成正式 Scaling。

## 当前已执行验证

运行环境：Python 3.12.13、PyTorch 2.8.0。

- 全部 v0.1～v0.5 回归测试通过；
- `python tests/test_qwen3_tp.py` 通过；
- 无 socket 的 rank-local TP=2 和 TP=4 数学仿真通过；
- TP=4 已覆盖 `world_size > KV heads` 的 KV 复制分支；
- full logits、cached Prefill、cached Decode、静态 batch greedy generation 对齐通过；
- TP Scaling 公式与不可比报告拒绝测试通过；
- TP=1 tiny CLI、TP benchmark 和 collective benchmark smoke 通过；
- compileall 与 `git diff --check` 通过。

## 当前环境无法完成的验证

设置 `MINIGPT_RUN_GLOO_TESTS=1` 后，`ProcessGroupGloo` 在 transport socket 创建阶段收到平台
`Operation not permitted`。这是当前 Codex 容器权限限制，不是数值断言失败。以下内容仍是
v0.6 最终冻结前的硬门禁：

- 允许本地 TCP 的 Linux 上 TP=2 Gloo process group 测试；
- Ascend/HCCL TP=2 smoke；
- 真实 Qwen3-32B TP=2/4/8 相同 workload 报告；
- collective 原始报告；
- TP Scaling 汇总及异常分析。

## 下一步（严格按顺序）

1. 完成当前软件候选的全量回归、文档核对和开发期 review；
2. commit 一个可供硬件拉取的 v0.6 RC 检查点，但不创建最终 v0.6 tag；
3. 立即把 RC 分支上传 GitHub，并核验远端 branch SHA；
4. 在 NPU 机器先跑 Runtime/Gloo/HCCL/tiny smoke，再跑 32B TP=2；
5. TP=2 正确后，保持同一 commit/权重/request 扩展 TP=4/8；
6. 收集 JSON 原始报告与控制台日志，运行 scaling 汇总；
7. 若发现问题，在当前 v0.6 分支修复并重复上述门禁；
8. 全部通过后才提交最终版本、创建 annotated tag、立即上传并核验；
9. 对冻结提交做独立自查，把高性价比不足纳入 v0.7，再开始 v0.7。

中断时保留全部 dirty/untracked 文件，禁止 reset、clean、checkout 回退。每个实质节点都要有
GitHub 或其他持久恢复位置；不能只依赖当前 scratch 工作区。
