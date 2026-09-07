# MiniGPT-Train 项目总交接与续作指南

> 唯一目的：让一个新对话在只能访问 GitHub 和资料库、完全看不到旧长对话时，仍能准确恢复项目目标、版本历史、当前状态、学习进度、证据边界、后续路线和施工规则，并从正确的位置继续工作。

- 最后核对日期：2026-09-07（UTC）
- GitHub 仓库：`TuraLucrs/minigpt-train-infra`（private）
- 当前最新正式版本：`v0.6-qwen3-tensor-parallel`
- 当前冻结源码提交：`7998581bc61003a8cd98aef874e11d698b66e912`
- 当前源码分支：`upgrade/v0.6-qwen3-tensor-parallel`
- 当前进度：v0.6 已完成真实 Ascend Atlas A3 验收、远端分支/tag、双份恢复归档和冻结后独立 CPU/Gloo 自动化复核；v0.7 尚未开始。

---

## 1. 新对话先做什么

不要直接在 GitHub 默认 `main` 上续作。`main` 当前仍主要是 v0.2.1 时代的源码，并额外承载精确 Git bundle 同步桥；它不是最新项目代码。

新对话应按以下顺序恢复：

1. 先读本文件，建立项目全局状态。
2. 实时核对 GitHub：
   - 仓库：`TuraLucrs/minigpt-train-infra`
   - 分支：`upgrade/v0.6-qwen3-tensor-parallel`
   - HEAD 必须是 `7998581bc61003a8cd98aef874e11d698b66e912`
   - annotated tag：`v0.6-qwen3-tensor-parallel`
   - tag 剥离后必须同样指向 `7998581...`
3. 优先从该分支或 tag 读取源码；若 GitHub 拉取受阻，使用资料库中的最终完整 bundle：
   - `/MiniGPT-Train_版本运行档案/v0.6_Qwen3_TP_Acceptance/minigpt-train-v0.6-qwen3-tensor-parallel.bundle`
4. 阅读：
   - `docs/PROJECT_WORKING_AGREEMENT.md`
   - `docs/INFERENCE_FIRST_ROADMAP.md`
   - `docs/V0_6_QWEN3_TENSOR_PARALLEL.md`
   - `artifacts/v0.6_qwen3_tp_acceptance/README.md`
5. 阅读 `docs/reviews/MINIGPT_V0_6_FROZEN_REVIEW.md`；v0.6 冻结复核已经完成，开始 v0.7 前只需按第 10 节核对学习断点和项目入口方案。
6. 从 `7998581...` 新建 v0.7 分支；不得移动或改写现有 v0.6 tag，也不得为了头像重写已有证据链提交。

若实时 GitHub/资料库状态与本文件不一致，按第 2 节的证据优先级判断，并更新本文件，而不是静默选择一个版本。

---

## 2. 信息源与冲突处理

本项目经历过临时工作区回退、旧规划改版、实机修复和事后归档，因此同一个文件中可能保留“采集当时”的待办。按以下优先级判断当前事实：

1. GitHub 实时 branch/tag ref 和 tag 最终指向；
2. 资料库 `V0_6_FINAL_RELEASE_MANIFEST.md`、最终 bundle 及其 SHA-256；
3. 对应冻结 tag 中的版本文档和运行证据；
4. `docs/INFERENCE_FIRST_ROADMAP.md` 与 `docs/PROJECT_WORKING_AGREEMENT.md`；
5. 资料库中的规划/学习文档快照；
6. 对话记忆或旧恢复说明。

已知冲突：

| 位置 | 旧内容 | 当前解释 |
|---|---|---|
| v0.6 分支 `README.md` | 写着“最终 tag 等待创建” | tag 后来已经创建；以最终发布清单和 GitHub refs 为准 |
| `docs/V0_6_QWEN3_TENSOR_PARALLEL.md` 前部 | 写着 tag 仍待自动回归/复核 | 这是实机导入时状态；最终 tag 已在后续完成 |
| `artifacts/.../RUN_LOG.md` 末尾 | 待办是推送 `ea672a6` 和决定 tag | 采集现场历史待办；已经完成 |
| 资料库 `INFERENCE_FIRST_ROADMAP.md` | 比 v0.6 tag 内版本少数段落旧 | 以 GitHub v0.6 tag 内 `docs/INFERENCE_FIRST_ROADMAP.md` 为准 |
| 资料库 `INDUSTRIALIZATION_OPTIMIZATION_PLAN.md` 与 `(1)` | 部分 v0.4/v0.5 checkbox 仍未完成 | 都是较早快照；以 v0.6 tag 内同名文档和本文件为准 |
| 资料库 `MINIGPT_TRAIN_LEARNING_NOTES.md` 的“当前进度” | 表头仍写 v0.2.1 | 正文后来补到 v0.3 Runtime，但没有完整覆盖 v0.4～v0.6；不能把它当施工状态 |
| GitHub 默认 `main` | 仍展示训练项目/v0.2.1 内容 | 不是当前稳定源码入口 |

---

## 3. 项目的核心目的

这不是“训练出一个有用的小模型”的项目，也不是为了堆一批框架名词。它有两个互相支撑、但优先级不同的目标。

### 3.1 主目标：形成高含金量的推理 Infra 项目

最终要得到一个可运行、可解释、可复现、能在真实大模型和真实多卡硬件上验证的推理系统。它应覆盖：

- 独立推理入口与清晰的 Prefill/Decode 边界；
- KV Cache；
- 真实 Qwen3 模型与本地权重加载；
- 静态批处理、Continuous Batching 和请求生命周期；
- Tensor Parallel、多副本部署和多卡 Scaling；
- 设备/分布式抽象，兼容 CPU/Gloo、CUDA/NCCL、Ascend/HCCL；
- TTFT、TPOT、E2E、吞吐、goodput、显存、通信和 profiler 证据；
- 一项由真实瓶颈驱动、带 A/B 和消融的推理专项研究；
- 完整代码、配置、原始报告、失败记录、恢复材料和面试叙事。

职业方向固定为：推理 Infra 为主，训练 Infra 为辅。训练代码保留为知识基础和演进证据，不再占用后续主版本。

### 3.2 学习目标：真正理解底层数据流和系统边界

用户不接受只记结论或 API。核心内容应采用：

```text
先用具体数字/shape/一次真实请求举例
→ 不跳步解释数据怎样变化
→ 再抽象到通用公式和系统设计
→ 说明边界条件、失败场景和为什么这样做
```

重点不是代码数量，而是用户能回答面试追问：

- 这个优化前后的执行路径有什么变化？
- 正确性如何证明？
- 性能数字的口径是什么？
- 瓶颈如何由 profiler/指标定位？
- 为什么某种多卡方案没有加速？
- 何时应该 TP，何时应该多副本/DP？
- 结果能否迁移到 CUDA/NCCL，而不是只会某个 Ascend API？

### 3.3 作品集目标

项目必须能被面试官连续追问，而仍有真实细节可讲。高价值材料包括：

- 明确 baseline，而不是只有最终数字；
- 修改前后代码/配置和数学语义；
- 多次重复、波动、中位数/分位数；
- 真实环境、提交、权重哈希、命令和原始 JSON；
- 失败实验和优化无效的边界；
- 容量收益、延迟收益、吞吐收益分开表述；
- 能恢复的 Git 历史与版本化报告。

---

## 4. 用户情况与硬约束

### 4.1 可用资源

- 本地：Windows 11、RTX 4060、i9-14900；本地设备适合 tiny/小模型调试，不承担 Qwen3-32B BF16 正式性能结论。
- Ascend A3：曾使用 8 张双芯物理卡、16 个 logical Ascend910、每芯 64 GB HBM；v0.6 正式验收实际使用 logical device 0～7（TP=2/4/8）。
- A5 / Ascend 950DT：规划文档记录为 8 卡、每卡 96 GB HBM，可用于后续高带宽推理、KV Cache、TPOT 和最终性能验证。
- 企业机器可能无外网；模型、依赖和代码要支持离线准备，运行证据必须合规，不得保存凭据或内部敏感信息。

### 4.2 学习与沟通要求

- 以事实为准，不迎合、不揣测用户倾向；如果证据反对用户预期，要直接说明。
- 错了就明确承认并修正；若核查后原结论正确，也不能因压力随意改口。
- 核心训练/推理/显存/精度/分布式/性能代码要深入讲；低价值工程胶水可黑箱化。
- 即使跳过胶水实现，也必须说明它是什么、在哪、输入输出和用途，不能偷偷略过。
- 注释和项目文档以中文为主；token、checkpoint、Attention、forward 等术语保留英文。
- 每次只引入一个主要复杂度来源，避免同时改变模型、调度、并行和指标导致无法归因。
- 性价比优先：正确性、可恢复性、真实瓶颈、显存解锁和可测收益优先；形式化重构、界面美化和无证据微优化靠后。

### 4.3 结果表述约束

- tiny MiniGPT/tiny Qwen3 只用于白盒正确性、回归和 CPU CI，不能支撑工业性能结论。
- 正式性能结论只使用真实模型、真实硬件和完整测量协议。
- estimate 必须标成 estimate；measured peak 才能标实测峰值。
- 容量扩展、吞吐提升、单请求延迟下降是三件不同的事，禁止混用。
- 优化前必须通过数值/生成一致性门禁；失败时不能产出“有效性能提升”结论。
- benchmark 至少保存 warmup、repeats、原始 runs、中位数/分位数、环境、Git revision、命令、模型和权重 provenance。

---

## 5. 项目长期架构

```text
MiniGPT 白盒 reference ─┐
                       ├─ 通用 InferenceEngine / Model Runner 边界
Qwen3 真实模型 ────────┘
                              │
            Prefill / Decode / KV Cache / Batching
                              │
            Scheduler / Request lifecycle / KV allocator
                              │
             Single device / TP / 多副本并行
                              │
      RuntimeContext / DistributedContext / ProfilerAdapter
                 ├─ CPU / Gloo
                 ├─ CUDA / NCCL
                 └─ Ascend / HCCL
                              │
      Benchmark / ExperimentRecorder / raw reports / provenance
```

MiniGPT 长期承担：

- 可以逐层检查的 reference model；
- cached/uncached、batched/unbatched、TP/local 的正确性 oracle；
- 快速 CPU 测试、CI 和故障注入；
- 展示项目从手写训练闭环演进到推理系统的路径。

Qwen3-32B 承担：

- 正式 TTFT、TPOT、吞吐、显存和 Scaling；
- 真实 KV Cache、长上下文和不同并发负载；
- Tensor Parallel、通信和最终性能报告。

硬件差异必须收敛在窄边界。通用数学和调度逻辑中禁止散落 `.cuda()`、固定 NCCL 或厂商特有调用；但也不为了“抽象完整”重新包装全部 PyTorch。

---

## 6. 已完成版本历史

### 6.1 精确版本表

| 版本 | 正式 tag | tag 指向提交 | 结论 |
|---|---|---|---|
| v0.1 | `baseline-v0.1` | `346885825b9d342163907d581509d61ab2a37c88` | 教学单设备训练闭环已冻结 |
| v0.2 | `v0.2-native-single-device` | `66d1023a134507a5cd79067a8b9a3323f4737186` | PyTorch 原生算子升级已冻结 |
| v0.2.1 | `v0.2.1-single-device-closeout` | `14c7f13921f34dd469914a3e27bee31f7478f1ac` | 单设备训练可靠性/热路径收尾已冻结 |
| v0.2.2 | `v0.2.2-single-device-correctness` | `80a1ad32323e16aa08ea3c8a19ef6d9fb06b7b26` | resume 和指标语义修正已冻结 |
| v0.3 | `v0.3-measurable-single-device-inference` | `9da4e57a8ca9a1eb049f13f384fc5d38864eb32f` | 可测量单设备推理基线已冻结 |
| v0.4 | `v0.4-kv-cache-static-batching` | `a09e5a96879f4a1c985732d9dbefc32e2cfd5b5f` | KV Cache 与静态 Batching 已冻结 |
| v0.5 | `v0.5-qwen3-real-model` | `3eefbc9241f127a7e5db50d36de37df3f253c4f0` | Qwen3 真实模型接入已冻结 |
| v0.6 | `v0.6-qwen3-tensor-parallel` | `7998581bc61003a8cd98aef874e11d698b66e912` | TP 软件、实机验收、证据归档、最终 tag 和冻结后自动化复核已完成 |

### 6.2 v0.1：教学单设备训练闭环

实现/学习内容：

- 字符 tokenizer、训练样本 `x/y`；
- token/position embedding；
- 手写 LayerNorm、GELU、causal Attention、Cross Entropy；
- 手写 AdamW、梯度裁剪、loss scaling；
- gradient accumulation、cosine warmup LR；
- checkpoint、resume、RNG 和日志；
- 完整流水线：文本 → token → `[B,T,C]` → Attention/FFN → logits `[B,T,V]` → loss → backward → step。

价值：白盒理解和后续数值对照，不作为高性能默认实现。

### 6.3 v0.2～v0.2.2：单设备训练工业化收尾

v0.2 将已经学过且成熟的原语替换为 PyTorch 实现：

- `nn.LayerNorm`、`F.gelu`、`F.cross_entropy`；
- `torch.optim.AdamW`、原生 gradient clipping、GradScaler；
- 合并 QKV projection；
- SDPA；
- 旧 checkpoint 的模型/optimizer 状态迁移。

CPU 固定配置的一次记录中，融合 QKV 后版本相对教学基线约 `1.402x`，但这是 tiny 共享 CPU 结果，只用于回归，不外推为 GPU 收益。

v0.2.1 补齐：

- AdamW decay/no-decay 参数组；
- 降低热路径 `.item()`/同步；
- 窗口计时与 CUDA Event 边界；
- 原子 checkpoint、`latest.pt` 硬链接/复制回退；
- reference-vs-optimized 输出和梯度测试；
- 多格式 optimizer/checkpoint 恢复。

v0.2.2 修正：

- checkpoint 先加载到 CPU，避免 RNG/batcher 状态错误映射到 CUDA；
- 区分循环 step 与成功 optimizer step；
- 修正 grad norm、跳步和 validation 显存口径。

仍未形成正式结论的历史项：真实 CUDA/NPU 上的训练吞吐、fused optimizer/SDPA dispatch、Event 收益和训练峰值显存。这些不是推理主线前置门槛。

### 6.4 v0.3：可测量单设备推理基线

核心变化：

- 模型只负责 `input_ids → logits`；生成编排移入 `InferenceEngine`；
- `MiniGPTModelRunner.prefill()/decode()` 建立阶段边界；
- Decode 暂时重算完整有效上下文，作为 v0.4 的正确性 baseline；
- 新增 `RuntimeContext`，集中 device、precision/autocast、同步、计时和显存；
- 新增独立 `infer.py` 和单设备 benchmark；
- 指标包括 TTFT、TPOT、E2E、tokens/s、峰值内存和逐次原始结果；
- greedy deterministic，sampling 通过 seed 可复现。

CPU tiny 数据只证明测量链路正确。

### 6.5 v0.4：KV Cache 与静态 Batching

核心变化：

- 每层预分配 K/V Tensor，布局 `[max_batch_size, n_head, block_size, head_dim]`；
- Prefill 写历史 K/V，Decode 只处理新增 token；
- 不使用逐 token `torch.cat`，避免稳定态地址反复变化；
- 支持不同 prompt 长度、右 padding、attention mask、EOS 和 inactive row；
- 每个 batch row 使用独立 RNG；
- `recompute` 路径继续作为 oracle；
- cached/uncached logits 和 greedy token 必须一致后才能比较性能。

MiniGPT 使用 learned absolute position；窗口左滑会改变保留 token 的 position，因此当前 reference 路径需要重建 Prefill cache。这不是所有 RoPE 模型的通用规则。

### 6.6 v0.5：Qwen3 真实模型

正式目标固定为 `Qwen/Qwen3-32B` dense：

- 本地读取 Hugging Face `config.json`、tokenizer 和单/多 shard safetensors；
- meta device 建模并逐参数加载，避免额外完整权重副本；
- 实现 RMSNorm、RoPE、SwiGLU、GQA、full forward、Prefill 和单 token Decode；
- KV Cache 按 `prompt + max_new_tokens` 分配；
- Prefill 只为最后有效 hidden state 计算 LM head，避免 `[B,T,V]` 完整 logits；
- 对齐 Transformers full logits、cached Prefill 和 cached Decode；
- 读取 generation config、多 EOS、top-k/top-p；
- 明确拒绝未实现的 sliding window 和非空 `rope_scaling`；
- 官方 32B 精确参数量门禁：`32,762,123,264`。

32B BF16 权重约 62 GB；单个 64 GiB device 缺少 KV Cache、workspace 和 runtime 余量，所以 v0.5 不强行伪装成单卡 32B 性能版本，正式实测延后到 v0.6 TP。

### 6.7 v0.6：分布式推理与 Tensor Parallel

实现：

- `torchrun` 一进程一 logical device；
- CPU/Gloo、CUDA/NCCL、Ascend/HCCL 的 `DistributedContext`；
- Embedding/LM head 按词表切分；Q/K/V、gate/up 按输出列切分；O/down 按输入列切分；
- AllReduce/AllGather；
- meta + safetensors slice 的 rank-local 加载，不先构造每 rank 完整 32B 模型；
- TP rank 0 选择 token 后广播，避免随机状态分叉和 collective 死锁；
- tiny TP 数学/分片/生成测试、真实 Gloo/HCCL smoke、collective benchmark；
- 真实 Qwen3-32B TP=2/4/8 同 workload 报告；
- 环境、拓扑、完整权重哈希、逐 rank HBM、代码版本和测量协议一致性门禁。

实机首次发现 Transformers 5.x `apply_chat_template()` 返回 `BatchEncoding`，而旧路径假设直接得到 token list；修复提交为 `ea672a6f31c06fb5cad6ee474203a695b6d2bab3`，随后正式运行通过，并补了自动化回归。

---

## 7. v0.6 真实验收结论

### 7.1 环境

- 日期：2026-09-04（UTC+8）
- 机器：Ascend Atlas A3，8 物理卡 × 2 芯，16 logical devices，每芯 64 GB HBM
- 正式使用：TP=2/4/8，对应 1/2/4 张物理卡
- 模型：Qwen3-32B dense，BF16，17 个 safetensors shard
- Python 3.12.13
- PyTorch 2.10.0+cpu + torch_npu 2.10.0.post5.dev20260821
- Transformers 5.14.1
- backend：HCCL

### 7.2 门禁

- v0.1～v0.5 CPU 回归：通过；
- 真实 TP=2 Gloo process group：通过；
- 显式 NPU/BF16 Runtime smoke：通过，没有 CPU/FP32 静默回退；
- tiny Qwen3 HCCL TP=2/4/8：通过；
- HCCL AllReduce/AllGather TP=2/4/8：完成；
- Qwen3-32B TP=2 短生成：修复后通过；
- Qwen3-32B TP=2/4/8 正式报告：同 prompt、权重、commit、warmup/repeats，greedy token 完全一致；
- Scaling 可比性门禁：通过。

### 7.3 正式数字

2 次 warmup，5 次正式测量，中位数：

| TP | TTFT | TPOT | E2E | 输出吞吐 | 单 rank 峰值 HBM | 相对 TP=2 扩展效率 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 162.1 ms | 172.1 ms | 5,494 ms | 5.824 tok/s | 31,462 MB | 1.000 |
| 4 | 188.9 ms | 174.2 ms | 5,602 ms | 5.712 tok/s | 15,643 MB | 0.490 |
| 8 | 198.0 ms | 178.1 ms | 5,713 ms | 5.601 tok/s | 8,070 MB | 0.240 |

### 7.4 可以和不可以声称什么

可以声称：

- Qwen3-32B BF16 在真实 Ascend/HCCL 上通过 TP=2/4/8 推理；
- rank-local 参数和单 rank HBM 基本按 `1/TP` 下降；
- 多卡 collective、Prefill、逐 token Decode、KV Cache 和 token 广播正确；
- v0.6 实现了大模型容量扩展和多卡推理基础。

不可以声称：

- TP=8 比 TP=2 单请求更快；
- 已证明生产并发吞吐提升；
- 已达到极限 HBM 利用率；
- 单 prompt、32 token 可代表所有服务负载；
- bundle 导入 Action 成功等于项目源码 CI 已通过。

这组数据表明 batch=1 短请求的 Decode 被逐层通信主导，更多 TP 只减少每 rank 显存，没有提高单请求速度。它仍是有价值的“容量优化/分布式推理”版本，但项目要成为真正的推理性能优化项目，v0.7 必须用并发、调度和多副本部署测出系统吞吐/goodput 收益，或者诚实证明某方案无收益并解释原因。

---

## 8. GitHub 当前状态与历史特殊点

### 8.1 实时分支（2026-09-07 核对）

| 分支 | HEAD | 用途 |
|---|---|---|
| `main` | 以 GitHub 实时 HEAD 为准 | 旧 v0.2.1 源码 + 项目导航、exact bundle sync 和 v0.6 frozen-review workflow；不是最新源码入口 |
| `upgrade/native-model-primitives` | `66d1023a...` | v0.2 |
| `upgrade/v0.2-single-device-closeout` | `bd70bc8f...` | v0.2.1 等价树、用户 GitHub 作者身份版 |
| `upgrade/v0.2.2-single-device-correctness` | `01c1109c...` | v0.2.2 等价树、用户作者身份版 |
| `upgrade/v0.3-measurable-single-device-inference` | `a4c92a76...` | v0.3 等价树、用户作者身份版 |
| `upgrade/v0.4-kv-cache-static-batching` | `a09e5a96...` | v0.4 正式源码 |
| `upgrade/v0.5-qwen3-real-model` | `3eefbc92...` | v0.5 正式源码 |
| `upgrade/v0.6-qwen3-tensor-parallel` | `7998581b...` | 当前最新正式源码 |
| `codex/exact-git-sync` | `c2603a61...` | 受限的 bundle 同步桥，不是开发分支 |

v0.2.1、v0.2.2、v0.3 的 branch HEAD 与 tag commit SHA 不同，但相应 tree SHA 相同；这是早期为了让提交作者关联用户 GitHub 头像而重建作者元数据产生的并行提交链。后续 v0.4～v0.6 的正式历史沿用 tag 所在的原始提交链。不要误判为源码内容冲突，也不要再重写 v0.4～v0.6 历史，否则真实运行报告中的 commit、bundle 和权重证据链会失效。

### 8.2 GitHub 同步规则

- 已授权的正式 GitHub 连接是首选写入通道；不再依赖临时容器 SSH 私钥。
- 不保存 GitHub 密码、Cookie、session 或私钥到仓库、资料库或恢复包。
- 二进制完整历史可经 `codex/exact-git-sync`：上传 bundle + SHA-256，workflow 只允许导入 `refs/heads/upgrade/v*` 与 `refs/tags/v*`，导入后必须再次核对远端 SHA。
- “用用户账号推送”和“commit 显示用户头像”不同：前者由授权连接决定，后者由 commit author 邮箱是否绑定 GitHub 账号决定。
- 后续新提交统一使用 `TuraLucrs <Lucifer24kl@gmail.com>`；不追溯改写 v0.6 及以前的冻结历史。

### 8.3 CI 现状

仓库现有 GitHub Action 是“验证和导入 Git bundle”的同步工作流。它成功证明 bundle 完整、允许的 refs 被精确推送，不代表项目 Python 测试运行。

当前没有正式的项目源码 CI。v0.7 开始前/初期应增加一个低成本 CPU CI，至少覆盖：

- `compileall` / 静态格式检查；
- `tests/test_core.py`；
- `tests/test_reference_parity.py`；
- `tests/test_resume_consistency.py`；
- `tests/test_inference.py`；
- `tests/test_kv_cache.py`；
- `tests/test_qwen3.py`；
- `tests/test_qwen3_tp.py` 的无 socket 仿真部分；
- `tests/test_tp_scaling.py`。

NPU/HCCL 仍通过人工申请机器后的版本验收，不伪装成普通 CI。

---

## 9. 资料库与 GitHub 已有材料索引

### 9.1 GitHub：当前权威源码内文档

以下路径都应从 `upgrade/v0.6-qwen3-tensor-parallel` 或 tag `v0.6-qwen3-tensor-parallel` 读取：

| 路径 | 内容与用途 |
|---|---|
| `README.md` | 项目介绍、当前结构、运行/测试入口和阅读顺序；“等待 v0.6 tag”一句已过时 |
| `docs/PROJECT_WORKING_AGREEMENT.md` | 版本实现、验收、commit/tag、push、bundle、独立自查和中断恢复的硬规则 |
| `docs/INFERENCE_FIRST_ROADMAP.md` | 当前权威主路线：训练为基础支线，v0.3～v1.0 以推理为主 |
| `docs/INDUSTRIALIZATION_OPTIMIZATION_PLAN.md` | 大型 backlog、优先级、实验口径、后端抽象、各阶段清单和最终架构；部分 checkbox 没有随 v0.6 更新，不能单独当状态表 |
| `docs/V0_2_1_SINGLE_DEVICE_CLOSEOUT.md` | v0.2.1 优化器、热路径、checkpoint、回归和 CPU 验收 |
| `docs/V0_3_MEASURABLE_INFERENCE.md` | v0.3 的 Runtime、InferenceEngine、Benchmark、指标定义和限制 |
| `docs/V0_4_KV_CACHE_STATIC_BATCHING.md` | v0.4 KV Cache、静态 batch、窗口边界和正确性门 |
| `docs/V0_5_QWEN3_REAL_MODEL.md` | v0.5 Qwen3 结构、加载、Transformers parity、32B 容量判断和证据等级 |
| `docs/V0_6_QWEN3_TENSOR_PARALLEL.md` | v0.6 TP 切分、通信、rank-local 加载、测试、机器运行顺序和实机结果 |
| `docs/LEARNING_GUIDE.md` | 最初训练代码的阅读路线，主要服务 v0.1；不是当前 v0.7 学习顺序 |
| `docs/PLAN_REVIEW.md` | 对最初“直接上 DDP/FSDP”的计划为何需要先做单卡基础的历史评审 |
| `docs/ROADMAP_DDP_FSDP_DEEPSPEED.md` | 已废止的训练优先路线，只保留项目历史，不决定施工顺序 |
| `artifacts/v0.6_qwen3_tp_acceptance/README.md` | 可公开审阅的 v0.6 验收摘要、数字、限制和环境警告 |
| `artifacts/v0.6_qwen3_tp_acceptance/RUN_LOG.md` | 实机命令、阶段、端口/hostname 处理、故障与修复；末尾待办是历史现场状态 |
| `artifacts/v0.6_qwen3_tp_acceptance/formal_benchmarks/` | TP=2/4/8 与 Scaling 的正式 JSON |
| `artifacts/v0.6_qwen3_tp_acceptance/smoke_and_collectives/` | Runtime、Gloo/HCCL、tiny TP 和 collective 证据 |
| `artifacts/v0.6_qwen3_tp_acceptance/environment/ENVIRONMENT.md` | v0.6 硬件和软件环境快照 |
| `RECOVERY_NEXT_STEPS.md` | v0.5 冻结前恢复清单，已完成，仅作历史 |
| `V0_6_WORK_IN_PROGRESS.md` | v0.6 RC 恢复点，已被最终状态取代，仅作历史 |

### 9.2 资料库：学习与规划

| 资料库路径 | 内容与当前地位 |
|---|---|
| `/MINIGPT_TRAIN_LEARNING_NOTES.md` | 学习速查笔记。详细记录 v0.1、v0.2.1 以及 v0.3 Runtime 早期内容；“当前版本 v0.2.1”字段过时，v0.4～v0.6 尚未形成同等完整笔记 |
| `/MINIGPT_TEACHING_POLICY.md` | 讲解筛选规则：核心详讲、胶水黑箱、中文规范、笔记维护与持久化硬门禁 |
| `/INFERENCE_FIRST_ROADMAP.md` | 推理优先路线的资料库快照；比 v0.6 tag 内最新版略旧 |
| `/INDUSTRIALIZATION_OPTIMIZATION_PLAN.md` | 早期有版本历史的工业化规划快照，已被 GitHub tag 内版本取代 |
| `/INDUSTRIALIZATION_OPTIMIZATION_PLAN(1).md` | 较新但仍早于 v0.4/v0.5 完成状态的快照；只作历史对照 |
| `/MINIGPT_V0_5_FROZEN_REVIEW.md` | v0.5 从独立 bundle checkout 后进行的冻结版本审查、静态/回归核验和边界说明 |

### 9.3 资料库：版本恢复材料

| 资料库路径 | 内容与用途 |
|---|---|
| `/minigpt-train.zip`、`/minigpt-train(1).zip` | 最初 MiniGPT 教学项目副本，历史材料 |
| `/minigpt-train-infra-complete.bundle` | 较早的完整 Git 历史备份，不能替代 v0.6 最终 bundle |
| `/minigpt-train-infra-v0.2.1.zip` | v0.2.1 源码快照 |
| `/minigpt-train-v0.4-complete.bundle` | v0.4 完整 Git 恢复包 |
| `/minigpt-train-v0.4-source.tar.gz` | v0.4 源码快照 |
| `/minigpt-train-infra-v0.5-qwen3-real-model.bundle` | v0.5 完整 Git 恢复包 |
| `/minigpt-train-infra-v0.5-qwen3-real-model-source.tar.gz` | v0.5 源码快照 |
| `/minigpt-train-infra-v0.5-qwen3-real-model-MANIFEST.txt` | v0.5 文件、提交和校验说明 |
| `/minigpt-train-recovery-2026-09-03.{tar.gz,patch,status.txt}` | 工作区回退后的 v0.4/v0.5 恢复材料，历史事故证据 |
| `/minigpt-train-v0.6-wip-2026-09-03.{tar.gz,patch,status.txt}` | v0.6 开发初期 WIP 恢复点，已被最终 bundle 取代 |
| `/v0.6-rc-checkpoint-2026-09-04.md` | 上机前 RC 检查点，历史状态 |

### 9.4 资料库：v0.6 正式运行档案

目录：`/MiniGPT-Train_版本运行档案/v0.6_Qwen3_TP_Acceptance/`

核心文件：

| 文件 | 内容 |
|---|---|
| `V0_6_FINAL_RELEASE_MANIFEST.md` | 最终 branch/tag、tag 对象、bundle/source 大小、SHA-256、GitHub workflow 和验收边界；判断 v0.6 是否正式冻结的首要文档 |
| `POST_IMPORT_CHECKPOINT.md` | 实机 bundle 导入、资料归档、GitHub 同步、身份规则和后续状态 |
| `ARCHIVE_INDEX.md` | 原始 ZIP/分片 bundle 哈希、拼接命令、归档问题和证据适用范围 |
| `RUN_LOG.md` | 实机全过程和命令 |
| `01_formal_benchmarks.zip` | TP=2/4/8 正式 JSON、Scaling 和哈希 |
| `02_smoke_and_collectives.zip` | Runtime、Gloo/HCCL、tiny TP、AllReduce/AllGather |
| `03_console_logs.zip` | 控制台日志和故障现象 |
| `05_environment.zip` | 硬件、CANN、PyTorch、torch_npu、Transformers 环境 |
| `minigpt-train-v06-branch.bundle.part1/2/3` | 上机端导出的三片完整历史 |
| `minigpt-train-v06-branch.bundle` | 三片拼接后的实机代码历史，包含 `ea672a6` 修复 |
| `minigpt-train-v0.6-acceptance-7998581.bundle` | 导入验收证据与文档后的恢复包 |
| `minigpt-train-v0.6-qwen3-tensor-parallel.bundle` | 最终双-ref 完整 Git bundle，含正式分支和 annotated tag；canonical 恢复入口 |
| `minigpt-train-v0.6-qwen3-tensor-parallel-source.tar.gz` | 从最终 tag 直接导出的源码快照 |
| `MINIGPT_V0_6_FROZEN_REVIEW.md` | 2026-09-07 冻结后独立复核、锁定依赖的 CPU/Gloo CI、真实 BatchEncoding 回归和文档勘误 |
| `V0_6_EMERGENCY_CHECKPOINT_2026-09-07.md` | 复核完成后的紧急恢复状态和续作步骤 |

最终 manifest 记录：

- bundle：259,896 bytes；SHA-256 `924d14f17723effa2b6a925a182dba6feea66a4ca3ce80b90225dc4865231129`
- source tar.gz：175,480 bytes；SHA-256 `298593a1fb4e789620ea95f90507047eb60fca6d762dddb90d5d0a31253bc9ac`
- exact bundle 导入 workflow：`33857116598`，`success`

资料库根目录还存在部分 v0.6 原始 ZIP、RUN_LOG 和 bundle 分片的重复上传；它们不作为 canonical 入口，也不要在未核对哈希时删除。

---

## 10. 当前真正未完成的事项

v0.6 代码、发布和冻结后复核已经完成。开始 v0.7 前还剩学习断点与项目入口两项；它们不要求修改 v0.6 tag。

### P0-1：v0.6 冻结版本独立自查（已完成）

2026-09-07 已完成并保存 `MINIGPT_V0_6_FROZEN_REVIEW.md`：

- GitHub Actions run `34076364408`（run #2）为 `success`；
- 锁定环境为 Python 3.12.14、PyTorch 2.10.0+cpu、NumPy 2.5.3、safetensors 0.8.0、Transformers 5.16.1；
- 41 个 Python AST、12 份 JSON、验收 SHA-256、全部 CPU/tiny/Gloo 回归通过；
- 使用真实 Transformers 5.16.1 `BatchEncoding` 类型的专项回归通过；
- 成功标记分支 `verification/v0.6-frozen-review-pass` 精确指向 `7998581...`；
- 冻结 tag 未移动，旧文档中的“tag 待创建”只作为历史表述记录勘误。

### P0-2：补齐学习进度，而不是默认“版本做完=用户学完”

根据现存学习笔记，能够被文档证明已经系统学习的内容是：

- v0.1 完整训练链路；
- v0.2.1 的模型原生化、SDPA、融合 QKV、optimizer、计时、checkpoint 等；
- v0.3 `RuntimeContext` 的部分内容。

至少 v0.5～v0.6 的完整逐段讲解明确因机器窗口而推迟；v0.4 和 v0.3 其余部分也不能在无记录的情况下擅自假定已经学完。推荐从学习笔记实际断点复核，然后按真实数据流补：

1. v0.3：`InferenceEngine`、Prefill/Decode、benchmark 指标；
2. v0.4：KV Cache、不同有效长度、EOS/active row、静态 batch；
3. v0.5：Qwen3 config → 权重加载 → RMSNorm/RoPE/GQA/SwiGLU → Prefill/Decode → parity；
4. v0.6：process group → TP 分片 → rank-local loader → collective → token broadcast → benchmark/Scaling；
5. 结合 v0.6 实机结果解释为什么 TP 降显存却没有降低单请求延迟。

每讲完一个小节更新 `/MINIGPT_TRAIN_LEARNING_NOTES.md`；先修正它的“当前进度”字段，再增量写入，不应整篇重写丢失原有追问记录。

### P0-3：修复项目入口和低成本 CI（部分完成）

默认 `main` 过旧，会让新 clone/new conversation误判当前状态。不要直接 force-push 覆盖。应先设计一个可恢复方案，例如：

- 保留 `main` 和 `codex/exact-git-sync` 的同步历史；
- 创建稳定入口分支（如 `stable`）指向最新正式 tag，或在确认影响后调整默认分支；
- 在默认可见位置放置本交接文档，明确最新源码入口；
- v0.7 分支继承并扩展 CPU/tiny 项目 CI，与 bundle 同步 workflow 分开命名。

本文件和 `docs/reviews/MINIGPT_V0_6_FROZEN_REVIEW.md` 已放到 GitHub 默认分支；
`.github/workflows/v06-frozen-review.yml` 已证明冻结 tag 的 CPU/tiny/Gloo 回归可自动执行。
尚未调整默认分支或创建稳定源码入口，v0.7 开发前需先确定 `stable`/默认分支方案。

---

## 11. 后续版本规划与完成程度

版本规划的原则是：每版只新增一个主要复杂度来源；真实数据优先；若实验不支持预期，保留失败结论，不为了“好看”改口。

### v0.7：推理调度、Continuous Batching 与 KV 生命周期

#### 要解决的问题

v0.6 的 batch=1 TP=2/4/8 单请求几乎不加速。多卡真正可能产生系统收益的方式是：让每个 TP 实例同时服务多请求，或在同样 8 个 devices 上部署多个较小 TP 副本。v0.7 必须回答：

```text
8 devices 上，TP8 单实例、2×TP4、4×TP2，谁在给定延迟约束下有最高 goodput？
```

#### 实现范围

- 请求对象与 `waiting/running/finished` 状态；
- 新请求动态进入、完成请求退出；
- Continuous Batching 的逐轮调度；
- 每请求 prompt 长度、输出长度、EOS 和采样状态；
- KV Cache 的分配、复用、释放和容量门禁；
- scheduler 与 local/TP Model Runner 解耦；
- 基本 backpressure/OOM 拒绝，不依赖隐藏的无限队列；
- 第一版使用清晰可验证的连续/slot 式 KV allocator；只有碎片证据充分时才在 v0.7 直接引入 paged KV，否则留给 v0.8；
- 增加 CPU/tiny CI。

不做：Web UI、复杂 API 网关、多机容错、为了展示而做管理后台、无证据的 Paged KV 大改。

#### 正确性门

- 单请求结果与 v0.6 引擎一致；
- static batch 与 continuous batch 在确定性 greedy 条件下 token 一致；
- 动态加入/退出不会污染其他请求的 KV、position 或 RNG；
- EOS/取消/OOM/异常后 KV 必须释放；
- TP rank 的调度决定和 token 序列一致，某 rank 失败时不能让其他 rank 卡死在不同 collective；
- 长短请求混合和不同到达时刻可重复。

#### 性能验收

固定模型、硬件、精度、prompt/output 分布和测量时间，至少报告：

- request/s、input/output/total tok/s；
- goodput（满足 TTFT/TPOT SLO 的完成请求数）；
- queueing、TTFT、TPOT、E2E 的 p50/p95/p99；
- batch size 随时间变化；
- KV 已用/空闲/峰值、碎片或浪费、OOM/拒绝数；
- 设备利用率、逐 rank HBM；
- TP8 vs 2×TP4 vs 4×TP2。

建议在 A3 上以 TP2 作为 Qwen3-32B 最小可运行单元，逐步提高并发。版本完成标志不是“写出 scheduler 类”，而是得到一份可信结论：在明确延迟约束下，多副本 + Continuous Batching 相比 v0.6 单请求基线提高了系统吞吐/goodput；若没有提高，必须用队列、计算、通信或内存证据解释。

### v0.8：推理专项研究

#### 选题规则

只选一个由 v0.7 真实数据证明值得解决的问题，不提前锁题。优先级按瓶颈匹配：

| v0.7 证据 | 候选专项 |
|---|---|
| KV 碎片/混合长度浪费/OOM | Paged KV Cache 或更好的 block allocator |
| 大量重复系统前缀 | Prefix Cache |
| Decode 计算/内存带宽主导 | 量化或 Speculative Decoding |
| graph launch/shape 波动明显 | CUDA/NPU Graph 或静态执行优化 |
| TP collective/完整 vocab AllGather 主导 | distributed top-k/sample、通信/计算重叠、拓扑优化 |
| 长上下文 Prefill 主导 | chunked prefill、长上下文内存/调度优化 |

#### 到什么程度才算完成

- 明确 baseline、瓶颈指标和假设；
- 正确性/质量门禁；
- 至少一个主要方案和必要消融；
- 相同环境、相同 workload、多次重复；
- 报告收益、代价、失败区间和适用边界；
- 在真实 Qwen3/hardware 上留下原始 JSON/trace，而不是只有 tiny 数据；
- 能在面试中解释“为什么选它、为什么有效、何时无效”。

v0.8 不是把候选列表全部实现，也不是先写一堆优化再找故事。

### v0.9：Ascend 深度适配、Profiler 与真实单机 Scaling

v0.6 已经提前完成基础 `torch_npu/HCCL` 兼容和 A3 TP=2/4/8 正确性/容量实测。因此 v0.9 不能再写成“第一次让 NPU 跑起来”，其剩余价值是深度归因和专项优化。

#### 实现/实验范围

- 收敛 `torch_npu`、CANN、HCCL 和 Ascend profiler 到后端适配边界；
- 分解模型加载、Prefill、Decode、KV、GEMM、LM head 和 collective 时间；
- HCCL message-size/拓扑 microbenchmark；
- topology-aware rank placement，同卡双芯与跨物理卡对照；
- A3 的 TP/副本/Continuous Batching Scaling；
- A5 上复验高价值工作负载；
- 根据 profiler 证据选择少量 Ascend 专项算子、通信或内存优化；
- 用统一指标说明哪些结论可迁移到 CUDA/NCCL，哪些是 Ascend 特有。

#### 建议完成门槛

- A3 至少复现 2/4/8 logical devices；资源允许时补 16 logical devices；
- A5 至少完成一个与 A3 可比的正式配置，理想矩阵为 1/2/4/8；
- 至少一套可读 profiler trace 与算子/通信占比报告；
- 至少一个由 trace 驱动的 A/B 优化或一个有充分证据的“无需优化/方案无效”结论；
- 不把安装 `torch_npu`、MindSpeed 或权重转换本身当成项目原创优化。

MindSpeed 可以作为对照或后续训练/生态实验，但不能替代本项目自行实现的 Model Runner、TP、KV Cache 和调度路径，否则无法证明个人贡献。

### v1.0：完整推理 Infra 交付

v1.0 以集成、稳定、复现和作品集交付为主，不再同时引入多个大型新算法。

必须具备：

- MiniGPT reference + Qwen3 真实模型；
- Prefill/Decode、KV Cache、Static/Continuous Batching；
- 请求生命周期与 KV allocator；
- 单设备、Tensor Parallel 和多副本部署；
- CPU/Gloo、CUDA/NCCL、Ascend/HCCL 的清晰后端边界；
- 项目 CPU CI 和版本化硬件验收；
- benchmark/profiler/experiment recorder；
- 配置、命令、环境、原始数据、commit、权重哈希和恢复材料；
- 一项 v0.8 专项研究的完整 A/B/消融结论；
- 一份最终技术报告和一份简历/面试项目说明。

完成门槛：新环境从冻结 tag 或 bundle 可恢复；tiny 测试可自动运行；真实模型可按文档在设备上复现；所有性能主张都能定位到原始报告；默认项目入口不再落后多个版本；每个版本都有 GitHub ref 和资料库恢复包。

可选但不是硬门槛：薄的 CLI/服务适配、OpenAI-compatible API、FSDP/ZeRO 训练支线、多机 elastic。只有当它们直接服务调度演示或明确岗位需要时再做。

### v1.x / 可选训练支线

- DDP 只作为 rank/process group/collective 的教学实验；
- FSDP、ZeRO、activation checkpointing、分片 checkpoint 属于训练 Infra 扩展；
- 不得阻塞 v0.7～v1.0 推理主线；
- 若开展，单独做同模型/同数据的显存、吞吐、通信与恢复对照。

---

## 12. 每个版本的硬完成标准

后续所有版本执行：

```text
设计与实现
→ 开发期测试
→ 完整正确性/性能验收
→ commit + annotated tag
→ 立即 push 分支和 tag
→ 读取远端 refs，确认与本地一致
→ 生成并验证完整 Git bundle 和源码快照
→ 保存到资料库
→ 从冻结产物独立 checkout 做一次审查
→ 将高性价比问题放入下一版本
→ 按实际数据流讲解新增/改变部分并更新学习笔记
→ 用户确认理解
→ 才开始下一版本
```

未完成版本也必须：

- 保存基准 commit、tracked diff、untracked 文件和状态说明；
- 到实质可恢复节点就生成 WIP 快照并异地保存；
- 禁止 `reset --hard`、`clean`、checkout 覆盖或“从上一个版本重来”；
- 工作区、本地 commit、本地 tag 属于同一故障域，不算持久化；
- push/资料库保存失败要明确报告，不能假称完成。

版本报告至少包含：

```text
改了什么 / 数学语义是否变化
baseline 与新实现
正确性门禁
模型/权重/config/tokenizer
硬件/软件/精度/拓扑
完整命令与 Git revision
warmup/repeats/raw runs/统计方法
TTFT/TPOT/E2E/throughput/goodput/memory/communication
失败实验、限制和 fallback
是否成为默认路径
```

---

## 13. 下一次对话的建议开场指令

可以直接把下面这段发给新对话：

```text
请先从资料库读取《MINIGPT_PROJECT_HANDOFF.md》，再实时核对 GitHub 私有仓库
TuraLucrs/minigpt-train-infra 的分支、tag 和 v0.6 最终提交。不要从默认 main 续作；
当前稳定入口应为 tag v0.6-qwen3-tensor-parallel / commit 7998581。先读取
docs/reviews/MINIGPT_V0_6_FROZEN_REVIEW.md，确认冻结复核已经通过，再核对学习断点；
不得移动 v0.6 tag 或重写已有实机证据链。事实冲突时按交接文档中的证据优先级处理，
并明确报告。
```

若用户直接要求继续开发 v0.7，则新对话仍应先快速核对 P0 项是否已经被后续记录完成；完成后从 v0.6 tag 创建新分支，再进入调度设计。

---

## 14. 当前一句话状态

项目已经从“手写 MiniGPT 单卡训练教学项目”演进到“Qwen3-32B 在 Ascend Atlas A3 上完成 TP=2/4/8 的真实分布式推理与容量验证”；v0.6 已完成正式冻结和独立 CPU/Gloo 自动化复核，但它没有证明 batch=1 单请求加速。下一阶段先补齐学习断点并确定稳定源码入口，再用 v0.7 Continuous Batching、KV 生命周期和 TP/多副本部署对照，真正证明并发吞吐/goodput 的系统优化价值。
