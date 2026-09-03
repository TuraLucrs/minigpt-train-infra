# MiniGPT 推理 Infra 主线与版本路线

## 1. 项目最高原则

> 训练是前置学习阶段，用来建立模型、优化器、精度、显存、通信等基础认知；推理 Infra
> 才是项目主线、专项研究方向和最终交付成果。

此前的路线虽然口头上强调“推理为主”，但仍给 DDP、FSDP、ZeRO 等训练内容安排了多个
主版本，实际效果接近“训推并重”。这与当前目标不一致。从本文件开始，训练和推理的关系
正式调整为：

```text
训练：已经完成的知识基础 + 后续按需学习的支线
推理：版本主线 + 性能实验主体 + 专项研究 + 最终交付
```

本文件是当前版本路线的权威说明；早期以 DDP/FSDP/DeepSpeed 为主的路线文档只保留为
项目历史，不再决定施工顺序。

## 2. 训练阶段的准确位置

`v0.1～v0.2.2` 已经完成训练基础阶段，主要价值是：

- 通过 MiniGPT 理解 Transformer 的完整计算过程；
- 掌握 forward、loss、backward、optimizer 的状态边界；
- 学习 AMP、梯度累积、checkpoint 和可复现；
- 建立基本性能测量、数值正确性和回归测试意识；
- 为理解多卡通信、显存优化和模型执行打基础。

因此，这些版本是项目演进的重要起点和知识证明，但不是最终项目的主体。后续不会把
“继续补齐完整训练平台”当作推理主线的前置条件。

DDP、FSDP、ZeRO 的处理原则：

- DDP 可以作为学习多进程、rank、process group 和 collective 的短实验；
- FSDP/ZeRO 只在需要扩展训练知识面时进入独立支线；
- 训练支线不得阻塞 KV Cache、Tensor Parallel、调度器或真实模型推理；
- 专项研究不再选择训练主题，也不采用“训推结合专项研究”的表述。

## 3. DDP 与分布式推理的边界

DDP（`DistributedDataParallel`）的核心任务是在多卡训练的 backward 过程中同步梯度。
普通推理没有 loss、backward、gradient、optimizer 和参数更新，因此 DDP 本身不是关键的
推理加速技术。

推理阶段真正需要的是：

- 多进程、rank、world size 和 process group；
- NCCL/HCCL/Gloo 等通信后端；
- AllReduce、AllGather 等 collective；
- 推理 Data Parallel：多份完整模型处理不同请求；
- Tensor Parallel：一个模型或请求跨设备执行；
- 必要时的 Pipeline Parallel、Expert Parallel。

DDP 与推理共享一部分分布式基础知识，所以保留一个范围受控的 `labs/` 教学实验是有价值的；
但完整 DDP 训练系统不占用推理主版本。

## 4. MiniGPT 与真实开源模型的长期分工

接入真实模型后，不删除 MiniGPT，也不在两种模型上维护两套推理系统。通用推理引擎通过
窄 Model Runner/adapter 边界接入不同模型：

```text
                    ┌─ MiniGPT adapter：白盒参考、正确性测试、快速 CI
通用推理引擎接口 ──┤
                    └─ 真实模型 adapter：正式性能、显存、Scaling 和交付
```

### 4.1 MiniGPT 的职责

- 作为可以逐层检查的白盒 reference model；
- 作为 KV Cache、Prefill/Decode、Batching 等功能的正确性 oracle；
- 对照 cached/uncached logits 和 greedy generation 是否一致；
- 在 CPU 上执行快速单元测试、CI、故障注入和跨后端 smoke test；
- 展示项目从手写训练闭环演进到工业推理系统的完整过程。

MiniGPT 的 tiny shape 不足以代表真实 kernel、显存、通信和吞吐瓶颈，因此不能用它的性能
数字支撑工业级性能结论。

### 4.2 真实开源模型的职责

- 承担正式 TTFT、TPOT、吞吐和峰值显存测试；
- 承担真实 KV Cache、长上下文和不同 batch 的性能实验；
- 承担 Tensor Parallel、通信开销与多卡 Scaling；
- 承担最终优化对照、实验报告和面试交付。

一个典型的新功能开发流程是：先在 MiniGPT 上建立简单正确性基线，再把同一套引擎能力接入
真实模型做性能与规模验证。这样分开回答“实现是否正确”和“优化是否有实际价值”。

## 5. 后端无关的核心边界

项目主要可用算力来自 Ascend，但核心能力不能绑定单一厂商：

```text
通用模型与系统逻辑
├─ device/runtime abstraction
├─ distributed abstraction
├─ profiler abstraction
└─ benchmark/metrics
        ↓
后端适配
├─ CUDA/NCCL
├─ Ascend/HCCL
└─ CPU/Gloo（正确性测试）
```

核心业务代码避免散落 `.cuda()`、`torch.cuda.synchronize()` 和 `backend="nccl"`。
`torch_npu`、CANN 和 HCCL 应限制在 Ascend 适配层。抽象保持窄边界，只包装真正存在设备
差异的能力，不重新发明 Tensor 或 PyTorch 算子体系。

## 6. 修正后的版本路线

### 已完成：v0.1～v0.2.2——训练基础阶段

完成教学版训练闭环、单设备原生算子升级，以及正确性、恢复和指标口径收尾。

### v0.3——可测量的单设备推理基线

- 从训练入口中分离独立推理入口；
- 以 deterministic greedy decoding 建立稳定基线，同时保留可选采样；
- 明确 Prefill 与 Decode 的语义边界；
- 本版本 Decode 仍重算当前完整上下文，作为 v0.4 KV Cache 的对照基线；
- 建立最小 `RuntimeContext`，覆盖 device、precision/autocast、同步、计时和显存；
- 建立推理 Benchmark，记录 TTFT、TPOT、E2E latency、吞吐和峰值设备内存；
- 使用 warmup、多次重复、原始结果和中位数/分位数；
- 训练 Benchmark 只保留为回归依据，不再是版本主角。

版本完成标志：MiniGPT 从“训练后顺便 sample”变成具有独立入口、阶段边界和统一指标的
可测量推理 workload。

### v0.4——KV Cache 与静态 Batching

- 实现逐层 KV Cache；
- Prefill 写入历史 K/V，Decode 只处理新增 token；
- cached 与 uncached logits、greedy generation 一致；
- 支持静态 Batching、不同有效长度、mask、position 和 EOS；
- 对比 KV Cache 前后的计算量、TTFT、TPOT、吞吐和显存。

### v0.5——真实开源 decoder-only 模型

- 第一版只支持一种主流模型架构；
- 接入真实 tokenizer、config 和 safetensors；
- 处理 RoPE、RMSNorm、SwiGLU、GQA/MQA 等真实组件；
- 建立 MiniGPT reference adapter 和真实模型 adapter；
- 与可信参考实现对齐 logits、generation 和 KV Cache；
- 从这里开始，所有正式性能结论都以真实模型为准。

### v0.6——分布式推理与 Tensor Parallel

- 在进入 TP 前做范围受控的分布式基础实验；
- 建立推理 process group 和基础 collective；
- 拆分 attention/FFN 线性层；
- 验证 TP=1 与 TP>1 的数值一致性；
- 完成单卡、2/4/8 卡 Scaling；
- 记录计算、通信、TTFT、TPOT、吞吐、显存和 Scaling Efficiency。

### v0.7——推理调度与 KV Cache 生命周期

- waiting/running/finished 请求状态；
- 请求动态加入和退出；
- Continuous Batching；
- KV Cache 分配、复用和释放；
- 不同并发和请求分布下的 p50/p95/p99；
- 明确吞吐与单请求延迟之间的 trade-off。

### v0.8——推理专项研究

候选方向包括 Paged KV Cache、量化、Speculative Decoding、CUDA Graph、算子融合、
Prefix Cache、调度优化和长上下文显存优化。到该阶段根据真实 profiler 和实验瓶颈只选择
一个有证据、有对照实验的研究问题，不提前凭空锁题。

专项研究只属于推理方向。训练最多用于构造测试模型或验证数值，不构成研究主题。

### v0.9——Ascend 适配、Profiler 与真实多卡实验

- `torch_npu`、CANN、HCCL 和 Ascend profiler 限制在后端目录；
- A3 测试 2/4/8/16 个 logical devices，并同时记录物理卡数；
- A5 测试 1/2/4/8 个 devices；
- 对 Prefill/Decode、KV Cache、TP、Batching 做严格 A/B Benchmark；
- 对比通用指标下的 CUDA/NCCL 与 Ascend/HCCL 行为。

### v1.0——推理 Infra 完整交付

最终主交付包括：

- 一个可运行的推理引擎；
- 真实开源模型支持；
- Prefill/Decode 与 KV Cache；
- Batching、请求调度和 KV 生命周期；
- 单设备与 Tensor Parallel 多设备推理；
- Benchmark、Profiler、实验配置和可复现报告；
- CUDA 与 Ascend 后端适配；
- 一个有数据、有基线、有消融或 A/B 对照的推理专项研究成果。

训练代码继续保留为知识基础和项目演进证据，但不是最终简历项目的中心卖点。

## 7. 面试中的准确表述

不把项目描述成“训练了一个很强的 MiniGPT”，也不把 MiniGPT 的 tiny benchmark 当作工业
性能成果。更准确的项目叙事是：

> 先用可完全掌控的 MiniGPT 建立训练和推理参考实现，完成逐层数值验证、Prefill/Decode
> 边界和 KV Cache 前后等价性测试；随后将通用推理执行框架接入真实开源模型，在真实硬件
> 上完成单卡优化、多卡 Tensor Parallel、调度与 Scaling 实验。

版本历史用于证明对模型数据流、正确性基线和系统演进的理解；真实模型实验用于证明工程与
性能价值；最终推理引擎和专项研究才是项目主体。

## 8. 版本讲解门禁

除已明确豁免完整重讲的 `v0.2.2` 外，后续每个版本继续执行：

```text
实现 → 测试与验收 → commit/tag → 立即push分支和tag并核验远端refs
→ 生成完整bundle与源码快照并保存到资料库 → 交付完整代码
→ 冻结版本独立自查 → 按文件和实际数据流逐段讲解新增/改变部分
→ 关键疑问解决并确认理解 → 才进入下一版本
```

核心推理路径必须讲清楚“现在需要什么、为什么需要、数据经过此处怎样变化、输出代表什么”；
低价值工程胶水可以按黑箱讲，但必须说明它的存在、用途、位置、输入和输出。

若版本尚未完成就因额度、机器、网络或工具故障中断，必须保留当前半成品并立即制作异地恢复
快照，禁止 reset、clean 或回退。开发期间的检查点也要持续备份，不能等到版本完成才第一次复制。
GitHub 与账号资料库必须各保留一份；任何推送或备份失败都属于进入下一版本之前的阻塞项。
