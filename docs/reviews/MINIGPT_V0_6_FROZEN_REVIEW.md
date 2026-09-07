# MiniGPT-Train v0.6 冻结版本独立复核

复核日期：2026-09-07  
仓库：`TuraLucrs/minigpt-train-infra`  
结论：**通过。v0.6 保持冻结，不改写历史，可以进入 v0.7。**

## 1. 冻结对象

| 项目 | 最终值 |
|---|---|
| 分支 | `upgrade/v0.6-qwen3-tensor-parallel` |
| 源码提交 | `7998581bc61003a8cd98aef874e11d698b66e912` |
| annotated tag | `v0.6-qwen3-tensor-parallel` |
| tag 对象 | `8cd94d200dade79fb13fd6ac8c7dab40dbb1ed6b` |
| 最终 bundle SHA-256 | `924d14f17723effa2b6a925a182dba6feea66a4ca3ce80b90225dc4865231129` |

远端分支、tag 剥离后的提交、最终 bundle 中的分支 ref 和 tag ref 均指向同一源码提交。
复核没有修改 v0.6 分支、tag 或源码历史。

## 2. 新增的独立自动化复验

GitHub 主分支新增 `.github/workflows/v06-frozen-review.yml`。工作流只检出冻结 tag，
不会在 v0.6 上提交代码。它验证固定源码身份后，运行静态检查、归档检查、全部 CPU/Gloo
回归和真实 Transformers 5.x `BatchEncoding` 类型测试。

- 锁定依赖的最终工作流提交：`ebcd2dbb013ff690b28eb5148f43ee5ab7beec98`
- 最终工作流运行：`34076364408`（run #2）
- 运行时间：2026-09-07 02:27:44～02:29:08 UTC
- 结论：`success`
- 成功标记分支：`verification/v0.6-frozen-review-pass`
- 标记分支指向：`7998581bc61003a8cd98aef874e11d698b66e912`

成功标记只在所有检查完成后发布。它直接指向冻结提交，没有创建新的源码提交。

## 3. 自动化复验环境

| 组件 | 版本 |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.10.0+cpu |
| NumPy | 2.5.3 |
| safetensors | 0.8.0 |
| Transformers | 5.16.1 |
| Distributed backend | Gloo / loopback |

这些版本已经固定在工作流中，避免以后重跑时因依赖自动升级而改变证据口径。

## 4. 复验结果

### 4.1 源码与证据完整性

- 精确提交和 tag 身份检查：通过；
- `git diff --check`：通过；
- 41 个 Python 文件 AST 解析：通过；
- 12 份验收 JSON 解析：通过；
- `artifacts/v0.6_qwen3_tp_acceptance/SHA256SUMS.txt` 全项校验：通过；
- 最终 Git bundle SHA-256 和 `git bundle verify`：通过。

### 4.2 CPU/Gloo 回归

- 核心训练 smoke：通过；
- v0.3 推理与 Benchmark：通过；
- v0.4 KV Cache 与静态批处理：通过；
- Qwen3 logits 对齐、Cache、权重加载和显存规划：通过；
- Qwen3 TP=2/4 仿真：通过；
- 两进程真实 Gloo process group：通过；
- reference 与 optimized parity：通过；
- checkpoint exact resume：通过；
- TP Scaling 汇总一致性门禁：通过。

### 4.3 Transformers 5.x 兼容回归

除仓库原有的 list/dict 行为测试外，本次直接构造 Transformers 5.16.1 的真实
`BatchEncoding` 对象，令 `apply_chat_template()` 返回带 batch 维的 `input_ids` Tensor，
再调用冻结版 `Qwen3Tokenizer.encode()`。返回 token 为 `[21, 22, 23]`，测试通过。

因此，实机提交 `ea672a6f31c06fb5cad6ee474203a695b6d2bab3` 修复的
Transformers 5.x 返回类型兼容问题，已经同时具备：真实 Qwen3-32B 修复后运行证据、
仓库回归用例和独立自动化复验证据。

## 5. 既有 Ascend 实机证据结论

2026-09-04 的 Atlas A3 / HCCL / Qwen3-32B BF16 实机证据保持有效：

| TP | TTFT 中位数 | TPOT 中位数 | E2E 中位数 | 输出 tok/s | 单 rank 峰值 HBM |
|---:|---:|---:|---:|---:|---:|
| 2 | 162.1 ms | 172.1 ms | 5,494 ms | 5.824 | 31,462 MB |
| 4 | 188.9 ms | 174.2 ms | 5,602 ms | 5.712 | 15,643 MB |
| 8 | 198.0 ms | 178.1 ms | 5,713 ms | 5.601 | 8,070 MB |

它证明 Qwen3-32B 能通过 rank-local 权重加载和 HCCL TP=2/4/8 正确运行，单 rank HBM
近似按 `1 / TP` 下降，跨 TP greedy 输出一致。它不证明 batch=1 单请求加速；该负载下
通信开销抵消了计算切分收益。

本次没有重跑昂贵的 NPU/32B 测试，因为冻结对象、实机原始证据及哈希均未变化；新增缺口
只涉及 CPU/Gloo 自动化与 Transformers 返回类型兼容，已经由本次工作流直接覆盖。

## 6. 冻结 tag 内文档勘误

以下内容是冻结前写入源码的历史状态，不再代表当前事实：

| 文件 | 过期表述 | 当前事实 |
|---|---|---|
| `README.md` | 最终 v0.6 tag 等待创建 | annotated tag 已创建并核验 |
| `docs/V0_6_QWEN3_TENSOR_PARALLEL.md` | 自动回归、证据提交和冻结复核仍待完成 | 三项现已全部完成 |
| `artifacts/v0.6_qwen3_tp_acceptance/RUN_LOG.md` | `ea672a6` 尚待推送并决定 tag | 属采集时刻待办；最终分支为 `7998581`，tag 已存在 |

这些文字不在冻结 tag 上原地改写，否则会改变提交 SHA、tag 和实机证据链。后续对外判断以
本复核、`V0_6_FINAL_RELEASE_MANIFEST.md`、`POST_IMPORT_CHECKPOINT.md` 和验收目录 README
为准。

## 7. 仍然成立的范围边界

- v0.6 是分布式推理与容量扩展版本，不是单请求延迟优化版本；
- Continuous Batching、请求动态加入/退出、KV Cache 生命周期和并发吞吐属于 v0.7；
- 分布式采样、通信计算重叠、量化、Paged Attention 和生产服务化不属于 v0.6；
- HCCL timeout 配置、`barrier(device_id=...)` 提示和设备健康检查应进入后续运行规范，
  但没有导致本次已归档验收失效。

## 8. 最终决定

v0.6 的代码、真实硬件证据、恢复产物和独立自动化复验现已闭环。没有发现需要撤销 tag、
重写历史或回到 v0.6 增加功能的问题。

后续顺序：先补齐 v0.4～v0.6 学习记录，再从冻结 tag 新建 v0.7 分支，实现真实请求调度、
Continuous Batching 和 KV Cache 生命周期管理，并用并发吞吐而非 batch=1 单请求指标证明
性能收益。
