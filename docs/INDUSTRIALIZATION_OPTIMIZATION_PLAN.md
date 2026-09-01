# MiniGPT-Train 工业化升级与性能优化清单

## 1. 文档目的

当前项目的 v0.1 是一个“可完全看懂的单卡教学实现”。它刻意手写了 tokenizer、
attention、cross entropy、AdamW、梯度裁剪和 GradScaler，用透明性换取了性能、功能和
生态兼容性。

本项目最终目标不是长期维护这些教学实现，而是：

1. 先用它们理解完整训练链路；
2. 冻结一个可运行、可解释、可回归的单卡基线；
3. 将纯教学组件替换为 PyTorch 或成熟训练生态的实现；
4. 保留训练系统中必须自己掌握的编排、状态管理、可观测性和故障恢复；
5. 在可靠的单卡基线上继续学习 DDP、FSDP、ZeRO、Profiler 和性能调优；
6. 最终形成可复现、可恢复、可扩展、可测量的工业级训练基线。

这里的“工业级”不等于简单地把几个函数换成官方 API。至少同时要求：

- 正确性：优化前后训练语义清楚，数值差异受控；
- 可复现：配置、数据、代码和随机状态可追踪；
- 可靠性：checkpoint 损坏、进程退出、机器重启后能够恢复；
- 性能：吞吐、显存、通信和 I/O 都能测量；
- 可扩展：可以从单卡演进到 DDP、FSDP 或 ZeRO；
- 可维护：尽量使用成熟组件，减少无必要的自定义底层代码。

---

## 2. 总体原则

### 2.1 先冻结教学基线，再做替换

完成当前单卡学习后，先保留一个明确版本，例如：

```text
v0.1-learning-baseline
```

它必须能够：

- 通过 `tests/test_core.py`；
- 在 CPU 或 GPU 上完成短训练；
- 保存并恢复 checkpoint；
- 解释每一步 shape 和训练状态；
- 产出一份基准吞吐、显存和 loss 记录。

工业化改造在新分支或新版本中完成，例如：

```text
v0.2-production-single-gpu
```

手写版本可以留作 reference implementation，不再作为高性能默认路径。

### 2.2 每次只替换一个组件

例如先只替换 cross entropy，验证数值一致后，再替换 AdamW。不要同时替换 attention、
optimizer、数据管线和精度策略，否则性能或 loss 变化时无法归因。

每个替换至少记录：

```text
修改内容
正确性对照
训练 loss 对照
tokens/s
峰值显存
适用硬件和精度
已知限制
```

### 2.3 区分三类改动

#### 等价实现替换

目标是不改变数学定义，例如：

- `manual_cross_entropy` → `torch.nn.functional.cross_entropy`；
- `MiniAdamW` → `torch.optim.AdamW`；
- 手写梯度裁剪 → `torch.nn.utils.clip_grad_norm_`。

这类改动适合优先完成。

#### 数值路径优化

数学目标相同，但浮点运算顺序或精度会变化，例如：

- FP32 → BF16 混合精度；
- 手写 attention → SDPA / FlashAttention；
- eager mode → `torch.compile`；
- 普通 AdamW → fused AdamW。

这类改动需要允许小范围浮点差异，并比较训练曲线而不是要求逐 bit 相等。

#### 模型定义变化

会改变模型本身，例如：

- LayerNorm → RMSNorm；
- GELU MLP → SwiGLU；
- learned position embedding → RoPE；
- 字符 tokenizer → BPE/SentencePiece。

这类改动不能叫“纯性能等价替换”，必须建立新的模型和数据基线。

### 2.4 优先级由“核心相关性和性价比”决定

这份清单是 backlog，不是要求逐项完成。后续每个候选优化先回答四个问题：

1. 它是否直接影响训练或推理的核心路径？
2. profiler、显存记录或故障记录是否证明这里存在真实瓶颈？
3. 预期收益是吞吐、延迟、显存、规模上限还是可靠性，能否量化？
4. 实现、验证和长期维护成本有多大，是否会妨碍后续 DDP/FSDP/推理学习？

执行顺序采用：

```text
正确性和恢复可靠性问题
→ 有明确证据的训练/推理热路径
→ 能解锁更大模型、batch或context的显存优化
→ 分布式扩展必需项
→ 通用工程完善
→ 与核心目标关系较弱的美化和微优化
```

#### A 级：高性价比，优先完成

- 官方 LayerNorm、cross entropy、AdamW、GradScaler、gradient clipping；
- fused/foreach optimizer 路径；
- 合并 QKV，优先使用 PyTorch SDPA；
- 删除 micro-batch 热路径 `.item()` 和无必要的逐 step CUDA 全局同步；
- 用 profiler 拆解 data/forward/backward/optimizer/communication；
- 数据预取、pinned memory 和真正的 H2D/计算重叠（数据成为瓶颈时）；
- 原子 checkpoint、准确 resume 和必要的分布式 checkpoint；
- DDP 的正确数据切分、`no_sync()` 和 rank-aware 日志；
- 推理扩展阶段的 KV cache、continuous batching 和请求调度。

#### B 级：有瓶颈证据或规模需求时实施

- `torch.compile`；
- 外部 FlashAttention 实现；
- activation checkpointing；
- mmap/streaming、多 worker 数据系统；
- 异步 checkpoint；
- FSDP/ZeRO；
- fused MLP、fused linear-cross-entropy；
- 更复杂的 tokenizer 和数据治理；
- 推理量化、speculative decoding 和 paged KV cache。

#### C 级：降低频次或暂缓

- 仅为了形式统一而迁移 Hydra、Pydantic 或其他配置框架；
- 没有影响训练的目录、命名和抽象层重构；
- 对低频启动代码做微秒级优化；
- 在没有规模需求时提前实现 tensor/pipeline/expert parallel；
- 没有 profiler 证据时堆叠第三方 fused kernel；
- 只改善展示效果、不改善定位效率的复杂日志面板。

这里的级别会随着瓶颈变化而变化。例如 mmap 对 tiny corpus 是 C 级，但当数据无法装入内存
时会直接升为 A 级；activation checkpointing 在模型能轻松放入显存时不是优先项，在 OOM
阻塞训练时则会升为 A 级。

### 2.5 硬件无关核心，厂商能力通过窄适配层接入

项目的主要可用算力来自 Ascend，但职业能力和核心代码不能被单一硬件生态绑定。总体边界为：

```text
通用模型与系统逻辑
├─ device/runtime abstraction
├─ distributed abstraction
├─ profiler abstraction
└─ benchmark / metrics / experiment recorder
        ↓
后端适配
├─ CUDA / NCCL
├─ Ascend / HCCL
└─ CPU / Gloo（正确性测试）
```

核心代码不应散落以下硬编码：

```python
tensor.cuda()
torch.cuda.synchronize()
backend = "nccl"
```

设备、混合精度、同步、内存统计、分布式 backend 和 profiler 应由配置与运行时能力决定。
Ascend 兼容性应在公共抽象建立后尽早做 smoke test，避免最后才发现算子或运行时不兼容；
但 `torch_npu`、CANN、HCCL 的专项调优、算子替换和拓扑优化保留到独立的 Ascend 阶段，
不侵入通用模型、训练循环、推理调度和指标定义。

抽象层必须保持窄边界，避免为了“跨平台”重新包装全部 PyTorch API。第一版只覆盖真正存在
后端差异的系统边界：

- `RuntimeContext`：device、autocast、显式同步、内存统计和 capability 查询；
- `DistributedContext`：backend、rank、local rank、world size、process group 和基础 collective；
- `ProfilerAdapter`：统一 start/stop/trace 导出，内部选择 PyTorch、CUDA 或 Ascend profiler；
- `ExperimentRecorder`：记录 resolved config、命令、环境、硬件、原始指标和代码版本；
- 模型数学仍使用普通 PyTorch Tensor/Module，不建立一套自定义 Tensor 或算子体系。

跨后端判断优先查询能力，例如 `supports_bf16`、`supports_sdpa`、`supports_memory_stats`，
而不是在业务代码中不断编写 `if backend == ...`。同一项优化必须使用相同的 workload、指标定义
和统计方法，厂商后端只改变执行方式，不改变实验口径。

### 2.6 当前可用硬件边界与计数口径

截至 2026-08-20，已确认的主要资源为：

| 资源 | 物理卡 | 可见芯片/逻辑设备 | 单芯 HBM | 单机聚合 HBM | 主要用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| A3 | 8 张双芯卡 | 16 | 64 GB | 1024 GB | 16 芯单机 Scaling、拓扑和通信实验 |
| A5 / Ascend 950DT | 8 张卡 | 8 | 96 GB | 768 GB | 高带宽推理、KV Cache、TPOT 和最终性能验证 |

16 张物理卡资源获取较困难，不作为项目完成条件。A3 主实验轴为 2/4/8/16 个 logical
devices，同时记录对应物理卡数；A5 主实验轴为 1/2/4/8 个 devices。所有实验必须分别记录：

```text
physical_card_count
visible_device_count
chips_per_card
world_size
hbm_per_visible_device
device_name
interconnect_topology
```

聚合 HBM 只有在 TP、FSDP、ZeRO 等切分方案中才能共同容纳模型或训练状态；DDP 仍要求每个
rank 保存完整模型，不能把单机总 HBM 当成单个进程可用内存。

### 2.7 版本完成后的讲解门禁

本项目同时承担学习和工程交付两个目标，因此不能连续堆叠版本、最后再统一补课。每个版本都
必须经过以下闭环，才能开始下一版本的优化：

```text
完成本版本实现
→ 运行正确性、恢复和性能验收
→ 冻结 commit/tag 与变更清单
→ 按文件和数据流逐段讲解本版本新增/修改的代码
→ 解决学习者提出的问题
→ 学习者确认已经理解
→ 才进入下一版本
```

讲解方式沿用 `baseline-v0.1` 的学习方法，但减少已经掌握内容的机械重复：

- 先说明本版本解决了什么问题、为什么现在需要解决；
- 给出受影响的文件地图，以及旧执行路径到新执行路径的变化；
- 主要讲新增和改变的部分，未改变的基础概念只在依赖它时简要回顾；
- 按实际执行顺序逐小段阅读代码，必要时细化到单行、单个变量、类型、shape、状态和生命周期；
- 对核心训练/推理路径使用“现在需要什么 → 为什么需要 → 怎样实现 → 输入变成了什么”的方式；
- 讲清接口背后的实际对象和语义，不能只给变量重新起名字或只复述 API 文档；
- 版本讲解未完成、关键疑问未解决时，不开始下一版本施工。

开发过程中仍可给出简短进度和局部解释，但完整代码讲解安排在该版本通过验收并冻结之后，
避免一边频繁改代码、一边讲解已经过期的实现。

---

## 3. 完成教学版单卡任务前必须保留的内容

以下组件暂时保留，直到相应原理已经学完并通过实验：

| 当前组件 | 暂时保留的原因 | 完成标志 |
| --- | --- | --- |
| `CharTokenizer` | 看清文本、token id、词表映射 | 能解释 encode/decode、词表与 checkpoint 绑定 |
| `RandomTokenBatcher` | 看清 GPT 的 x/y 右移和 batch shape | 能解释 `[B,T]`、合法采样起点和 RNG |
| `MiniLayerNorm` | 看清均值、方差、weight、bias | 能解释每个 token 沿 C 维归一化 |
| 手写 attention | 看清 Q/K/V、多头、mask、softmax | 能完整推导 `[B,H,T,D]` 和 `[B,H,T,T]` |
| `manual_cross_entropy` | 看清 logits 到 target loss | 能解释 logsumexp、gather 和平均 |
| `MiniAdamW` | 看清梯度、动量、二阶矩、参数更新 | 能解释 `m`、`v`、bias correction、weight decay |
| `SimpleGradScaler` | 看清 FP16 scaling 的时序 | 能解释 scale、unscale、inf/nan、skip step |
| 当前 training loop | 看清 forward/backward/step 的边界 | 能解释梯度累积、评估、日志、checkpoint |

完成后，它们可以从“默认实现”降级为：

```text
reference/
tests/reference/
docs/examples/
```

或者通过配置保留：

```text
implementation: reference | optimized
```

---

## 4. 第一批：单卡完成后立即替换的教学组件

这些替换主要减少 Python 循环、重复 kernel launch 和自定义状态管理，是进入分布式前的
优先工作。

| 当前实现 | 工业化默认实现 | 主要收益 | 验证重点 | 优先级 |
| --- | --- | --- | --- | --- |
| `MiniLayerNorm` | `torch.nn.LayerNorm` | 成熟内核、混合精度支持、编译器兼容 | 输出和梯度近似一致 | P0 |
| 手写 `gelu` | `torch.nn.functional.gelu` / `nn.GELU` | 成熟 kernel，便于 fusion | `approximate` 模式保持一致 | P0 |
| `manual_cross_entropy` | `torch.nn.functional.cross_entropy` | 数值稳定、成熟反向、可能融合 | loss 与梯度误差 | P0 |
| `MiniAdamW` | `torch.optim.AdamW` | 状态管理完整，支持 foreach/fused 路径 | 参数组、step 和 checkpoint | P0 |
| `clip_grad_norm` | `torch.nn.utils.clip_grad_norm_` | 支持成熟 foreach 路径和生态集成 | 裁剪前 norm 与参数梯度 | P0 |
| `SimpleGradScaler` | PyTorch `GradScaler` | 高效 inf 检测、标准 step/update 协议 | FP16 overflow 与跳步行为 | P0 |
| 手写 scheduler glue | PyTorch scheduler 或保留纯函数 | 标准 `state_dict`、生态兼容 | resume 后 LR 连续 | P1 |

### 4.1 AdamW 参数组

当前 `MiniAdamW` 对所有可训练参数使用同一个 weight decay。工业实现通常明确划分参数组：

```text
decay：Linear 等矩阵权重
no_decay：bias、LayerNorm/RMSNorm 的缩放和偏置
```

Embedding 是否 decay 需要根据具体训练方案明确决定，不能无说明地套用规则。

验收要求：

- 每个参数只出现于一个参数组；
- 所有 `requires_grad=True` 的参数都被覆盖；
- 参数组配置进入 checkpoint 和实验配置；
- resume 后 optimizer state 与参数对应关系正确。

### 4.2 GradScaler 的正确调用顺序

替换后保持以下语义：

```text
autocast forward
→ scale(loss).backward()
→ unscale_(optimizer)
→ gradient clipping
→ scaler.step(optimizer)
→ scaler.update()
```

BF16 一般不启用 loss scaling；FP16 根据硬件和数值稳定性启用。

### 4.3 Scheduler 不一定是性能热点

`cosine_lr` 每 step 只计算一次，性能成本很小。替换它的主要原因是：

- 与 optimizer 和 checkpoint 的标准状态管理集成；
- 降低边界条件错误风险；
- 更容易接入训练框架。

如果当前纯函数已经通过边界测试，也可以暂时保留。

---

## 5. 模型计算内核优化

### 5.1 合并 Q/K/V 投影

当前 attention 使用三个独立 Linear：

```text
q_proj(x)
k_proj(x)
v_proj(x)
```

可改为一次投影：

```text
qkv_proj(x) → [B,T,3C] → split(q,k,v)
```

收益：

- 减少 kernel launch；
- 减少重复读取输入 `x`；
- 更有利于编译器和 fused attention 路径。

验证：Q/K/V 切片映射必须与旧权重严格对应。

### 5.2 使用 PyTorch SDPA

把显式实现：

```text
QKᵀ
→ scale
→ causal mask
→ softmax
→ dropout
→ 乘 V
```

替换为 `scaled_dot_product_attention`。在满足硬件、dtype、shape 等条件时，PyTorch 可以
选择更高效的 attention backend，包括 Flash 或 memory-efficient 路径。

主要收益：

- 避免 Python 层拆分多个算子；
- 某些 backend 不需要显式保存完整 `[B,H,T,T]` attention 矩阵；
- 显著降低长序列 attention 的显存流量；
- 更容易使用硬件优化 kernel。

验证：

- `is_causal=True` 的语义与当前下三角 mask 一致；
- training/eval 下 dropout 行为一致；
- FP32、BF16、FP16 都做数值对照；
- 不假设所有 shape 都会自动进入 Flash backend，要记录实际 backend 和 fallback。

### 5.3 FlashAttention

如果 SDPA 没有进入期望的高性能 backend，再评估显式接入 FlashAttention。不要在
没有 profiler 证据时提前增加第三方依赖。

适用条件通常包括：

- GPU 和软件栈支持；
- FP16/BF16；
- head dimension 满足 kernel 限制；
- 序列长度足以让 attention 成为主要瓶颈。

### 5.4 删除显式 `[T,T]` causal mask buffer

当前模型注册了大小为 `[1,1,block_size,block_size]` 的 mask。使用支持 causal 标记的
SDPA/FlashAttention 后，不需要为每层保存和切片这份显式 mask。

收益在超长 context 下更明显。

### 5.5 Fused MLP 与激活函数

阶段性路线：

```text
Linear + GELU + Linear
→ 编译器融合或 fused bias-GELU
→ 若改变模型定义，再评估 SwiGLU fused MLP
```

SwiGLU 不是与 GELU 完全等价的性能替换，它会改变参数形状和模型定义，必须新建实验
基线。

### 5.6 `torch.compile`

在 eager 优化和正确性基线稳定后尝试：

- 减少 Python 调度与小 kernel；
- 融合逐元素运算；
- 捕获稳定 shape 的训练图。

前置要求：

- 先用 profiler 找到 Python/kernel launch 开销；
- 训练 shape 尽量稳定；
- 记录首次编译时间与 steady-state 吞吐；
- 检查 graph break、recompile 次数和动态 shape；
- checkpoint 必须保存原始 module 的正确 state dict。

### 5.7 Activation Checkpointing

模型增大后，可以用重计算换显存：forward 不保存部分中间激活，backward 时重新计算。

它的目标是扩大可训练模型或 batch，不是无条件加速。通常：

```text
激活显存下降
计算量和 step time 上升
```

进入 FSDP 前应至少完成一次显存/吞吐对比。

### 5.8 输出层与大词表 loss

当前产生完整 logits：

```text
[B,T,V]
```

当词表很大时，lm_head 与 cross entropy 会占据大量计算和显存。后期可以评估：

- fused linear-cross-entropy；
- chunked cross entropy；
- tensor-parallel / vocabulary-parallel lm_head 和 loss。

这不是当前 tiny 字符词表的瓶颈，等真实 BPE 词表和大模型出现后再做。

### 5.9 Tensor Core 友好形状和精度

工业 GPU 训练优先评估 BF16；FP16 使用 GradScaler。模型维度和 head dimension 尽量选择
硬件友好的倍数，但不能只为整齐而破坏模型设计。

FP32 baseline 还要明确 TF32 策略，保证性能比较时设置一致。

### 5.10 推理用 KV Cache 不进入训练主线

当前 `generate()` 每生成一个 token 都重新计算整个上下文。KV cache 能显著优化推理，
但不会直接提升预训练 forward/backward。因此它属于推理 Infra 扩展，不应阻塞当前训练
工业化和 DDP 路线。

---

## 6. 数据与 tokenizer 工业化

### 6.1 字符 tokenizer → BPE/SentencePiece/Hugging Face Tokenizers

字符 tokenizer 只保留为教学和 smoke test。真实训练使用成熟 tokenizer，要求：

- tokenizer 文件有版本和 hash；
- special token 定义固定；
- token id 与 checkpoint 严格绑定；
- 训练集、验证集的预处理配置可追踪；
- tokenizer 改变后不能直接复用旧 embedding/lm_head checkpoint。

### 6.2 离线 tokenization

当前每次启动都读取文本、训练 tokenizer、重新 encode。工业流程应拆分为：

```text
原始数据
→ 清洗与去重
→ tokenizer encode
→ 分片 token 文件 + index/metadata
→ 训练任务只读取 token shards
```

收益：

- 不在每次训练启动时重复预处理；
- 多机任务共享同一份确定数据；
- 更容易校验数据版本和恢复采样位置。

### 6.3 二进制分片、mmap 和 streaming

大语料不能一次读入内存。根据规模逐步引入：

- token dtype 选择（如合法范围内的 uint16/uint32 或训练时 long 转换）；
- binary shards；
- memory mapping；
- streaming dataset；
- shard manifest、大小、样本数和 checksum；
- 数据损坏检测和坏 shard 隔离。

### 6.4 DataLoader、多 worker 和预取

`RandomTokenBatcher` 不做多进程、预取、pin memory 或 worker 管理。工业版本可以改为
Dataset/DataLoader 或专门的 GPT data loader：

- `num_workers`；
- `persistent_workers`；
- `prefetch_factor`；
- pinned memory；
- 下一 batch 与当前 GPU 计算重叠；
- worker seed 与 epoch/rank 状态恢复。

当前 `.to(device, non_blocking=True)` 只有在源 CPU 内存满足 pinned-memory 等条件时才可能
真正异步；普通 pageable CPU Tensor 不应仅凭 `non_blocking=True` 就宣称已经实现传输重叠。

### 6.5 减少重复索引和传输

当前 `x`、`y` 分别进行高级索引和设备搬运。可评估：

- 一次取得长度 `T+1` 的连续块，再创建 x/y view；
- 在 CPU pinned buffer 中组 batch；
- 小数据集直接常驻 GPU；
- 双缓冲预取下一 batch；
- 数据拷贝使用独立 CUDA stream，并用 event 正确同步。

是否有效必须用 data time 和 GPU idle time 证明。

### 6.6 文档边界、packing 和 mask

当前把整个语料看成一条连续 token 流。真实数据需要明确：

- 文档之间是否插入 EOS；
- 是否允许 attention 跨文档；
- 短文档怎样 packing，减少 padding；
- loss mask 是否忽略 padding 或特定 token；
- train/val/test 如何按文档划分，避免数据泄漏。

### 6.7 分布式数据切分

DDP 中每个 rank 必须获得不同但整体不重不漏的数据序列，并能在 resume 后恢复到正确位置。
需要明确：

- rank、world size 与 shard 的映射；
- shuffle seed；
- epoch 或 consumed samples/tokens；
- worker RNG state；
- world size 改变后是否允许继续恢复。

---

## 7. Training Loop 性能优化

### 7.1 移除 micro-batch 内的 `.item()` 同步

当前每个 micro-batch 都执行：

```python
loss.float().item()
```

CUDA Tensor 的 `.item()` 需要把标量取回 CPU，通常会形成 CPU/GPU 同步点，破坏异步流水。

优化方向：

- 使用 `loss.detach().float()` 在设备上累计；
- 只在真正需要记录日志时取一次 `.item()`；
- 分布式时先在设备上 reduce，再一次性取回 CPU；
- 确保累计值已经 detach，避免保留整个 autograd graph。

### 7.2 不要每 step 强制 `torch.cuda.synchronize()`

当前为测量 step time，在每一步前后同步 CUDA。这使计时容易理解，但会破坏 CPU 提交与 GPU
执行的重叠。

工业计时应区分：

- steady-state GPU time：CUDA events；
- 端到端 wall time：按窗口周期性同步；
- data/CPU time：`time.perf_counter()`；
- eval/checkpoint/logging 时间：单独统计。

不要为了每一步打印精确时间而主动降低训练吞吐。

### 7.3 降低指标采集频率

以下操作不必每 step 执行：

- peak memory reset/read；
- grad norm 取回 CPU；
- CSV flush；
- 控制台打印；
- validation；
- checkpoint。

根据用途设置独立 interval，并报告指标是否包含这些额外开销。

### 7.4 梯度累积与 DDP `no_sync()`

单卡累积逻辑保留。进入 DDP 后，除最后一个 micro-batch 外使用 `no_sync()`，避免每次
backward 都做一次梯度 all-reduce：

```text
前 N-1 个 micro-batch：本地累计梯度，不通信
最后 1 个 micro-batch：backward 时同步梯度
```

全局 batch/token 数变为：

```text
micro_batch_size × sequence_length × accumulation_steps × world_size
```

### 7.5 减少重复 `zero_grad`

当前 optimizer step 前后都调用一次 `zero_grad(set_to_none=True)`。逻辑上只要保证下一轮
backward 前梯度为空即可。可以保留循环开头一次，或者循环结尾一次，但不需要两个都做。

优化幅度可能不大，主要是简化状态边界。

### 7.6 将调试检查移出热路径

shape、target 范围、`isfinite` 全量扫描等检查在 debug/smoke test 中开启，正式 benchmark
关闭。生产训练仍应保留低频健康检查和故障告警，不能完全取消数值监控。

### 7.7 评估路径

验证时：

- 使用 `model.eval()`；
- 使用 `torch.no_grad()` 或适用时 `torch.inference_mode()`；
- 固定可复现的验证数据范围；
- 避免每次验证重新构建对象；
- 分布式验证对 loss sum 和 token count 做正确 reduce；
- 用总 loss / 有效 token 数，而不是简单平均不同大小 batch 的均值。

### 7.8 稳态 benchmark 与端到端吞吐分开

至少报告两类性能：

```text
steady-state tokens/s：排除初始化、编译、eval、checkpoint
end-to-end tokens/s：包含数据、日志、eval、checkpoint 等真实成本
```

否则一个“几乎不保存 checkpoint”的实验会在表面上击败可靠的生产配置。

---

## 8. Mixed Precision 与数值稳定性

### 8.1 推荐策略

```text
FP32：正确性基线和不支持低精度的设备
BF16：支持硬件上的默认候选
FP16：需要时启用 GradScaler
```

autocast 只包围 forward 和 loss 中适合的区域。optimizer state 和关键归约根据实现保持
合适精度。

### 8.2 明确数值策略

配置和日志中至少记录：

- 参数 dtype；
- autocast dtype；
- optimizer state dtype；
- GradScaler 是否启用和当前 scale；
- TF32 设置；
- matmul precision 策略；
- deterministic 配置。

### 8.3 NaN/Inf 处理

生产策略不只是“发现后跳步”：

- 记录 step、loss scale、learning rate、grad norm；
- 保存或引用最近有效 checkpoint；
- 可选保存故障 batch 的数据标识，而不是泄露原始敏感数据；
- 设定连续跳步阈值，超过后终止并报警；
- 区分数据异常、模型溢出和通信错误。

---

## 9. Checkpoint 与故障恢复

### 9.1 完整状态

单卡 checkpoint 至少包含：

- model state；
- optimizer state；
- scaler state；
- scheduler 或当前训练进度；
- optimizer step；
- Python、PyTorch CPU、所有 CUDA RNG state；
- data sampler/batcher state；
- tokenizer、词表和 special-token 元数据；
- 数据版本、hash 或 manifest；
- 完整配置和运行时覆盖；
- 代码版本、依赖版本和硬件信息；
- checkpoint schema version。

尽量只在 optimizer-step 边界保存，避免还要恢复“梯度已经累计到第几个 micro-batch”的
中间状态。

### 9.2 原子保存

当前 `torch.save(payload, final_path)` 直接写最终文件。如果进程中途退出，可能留下损坏但
名字正常的 checkpoint。

单文件安全流程：

```text
写临时文件
→ flush/fsync（根据可靠性要求）
→ 校验写入成功
→ 同文件系统原子 rename/replace 成最终文件
→ 更新 latest 指针或 manifest
```

恢复时搜索“最新有效 checkpoint”，而不是只相信文件名最大的文件。

### 9.3 避免重复保存同一 payload

当前 numbered checkpoint 与 `latest.pt` 各保存一次完整 payload，会产生双倍序列化和 I/O。
优化方向：

- 只保存 numbered checkpoint；
- `latest` 使用小型 manifest、指针、软链接或适用环境下的硬链接；
- 对不支持链接的对象存储，维护原子更新的 metadata；
- 保留策略自动删除或归档旧 checkpoint。

### 9.4 异步与后台保存

模型增大后，同步 checkpoint 会让所有设备停顿。可评估：

- snapshot 到 CPU 后后台写盘；
- pinned CPU buffer；
- async/distributed checkpoint；
- 限制并发保存任务；
- 记录 snapshot 时间和真正落盘完成时间。

必须保证后台写入期间参数继续更新不会污染快照。

### 9.5 分片 checkpoint

进入 FSDP/ZeRO 后，不再假设 rank 0 能把全部状态收集成单文件。采用分片 checkpoint 或
`torch.distributed.checkpoint` 等成熟方案，处理：

- 每 rank/shard 写入；
- manifest；
- shard 完整性；
- world-size 变化后的 reshard；
- optimizer state 映射；
- 保存失败时全局一致性。

### 9.6 安全与兼容

- 不加载不可信来源的 pickle checkpoint；
- 需要时使用更受限的权重加载方式或 safetensors 保存纯模型权重；
- optimizer/RNG 等完整训练状态与“只发布模型权重”分开；
- schema version 做显式迁移，不能依赖字典字段永远不变；
- 加 checksum、大小和完成标记，恢复前验证。

### 9.7 断点续训验收

做两条路径：

```text
A：连续训练 N 步
B：训练 K 步 → 保存 → 重新启动 → 训练到 N 步
```

在确定性 CPU/FP32 小测试中比较：

- 下一批 token；
- learning rate；
- loss；
- model parameters；
- optimizer state；
- step 和日志序列。

在 GPU/fused kernel 无法逐 bit 确定时，使用合理误差和训练曲线一致性。

---

## 10. 日志、指标和 Profiler

### 10.1 从 CSV 升级为结构化日志后端

CSV 继续保留为轻量 fallback。工业版本可接入 TensorBoard、W&B 或内部平台，但训练代码只
依赖统一 logger interface。

日志至少记录：

- optimizer step、micro-step、consumed samples/tokens；
- train/val loss、perplexity；
- learning rate；
- grad norm；
- loss scale、skipped step；
- data/forward/backward/optimizer/communication 时间；
- steady-state 与 end-to-end tokens/s；
- allocated/reserved/peak memory；
- rank/world size；
- checkpoint 和 eval 耗时；
- 配置、代码版本和数据版本。

### 10.2 异步或批量日志

避免每条日志都：

- GPU `.item()`；
- CPU flush；
- 网络请求；
- 所有 rank 重复打印。

采用周期聚合、后台写入和有界队列。崩溃时允许损失少量非关键日志，但不能影响 checkpoint
正确性。

### 10.3 Rank-aware logging

DDP 下：

- rank 0 负责主日志和 checkpoint 元数据；
- 需要全局指标时先做 reduce；
- per-rank 性能和异常保留 rank 标签；
- 不把单 rank tokens/s 错当成全局 tokens/s。

### 10.4 计时层级

第一层：代码计时。

```text
data
H2D
forward
backward
gradient sync
optimizer
eval
checkpoint
```

第二层：PyTorch Profiler。

- CPU operator；
- CUDA kernel；
- shape；
- memory；
- stack；
- communication operator。

第三层：Nsight Systems/Compute 或硬件平台 profiler。

- CPU/GPU overlap；
- kernel gap；
- NCCL timeline；
- memory bandwidth；
- Tensor Core 利用；
- 通信与计算重叠。

没有 profiler 证据，不盲目添加 fused kernel、更多 worker 或 compile。

### 10.5 性能指标定义

统一定义：

```text
global tokens/step
= micro_batch_size × sequence_length × accumulation_steps × world_size
```

明确 tokens/s 是否包含 padding、eval、checkpoint 和失败重试。大模型阶段补充：

- samples/s；
- step time median/P95；
- Model FLOPs Utilization（MFU）；
- 通信占比；
- checkpoint 吞吐。

---

## 11. 配置与实验管理

### 11.1 当前 dataclass + JSON 的升级路径

当前配置足够单卡教学。复杂度增长后可使用 Hydra/OmegaConf、Pydantic 或内部配置系统，
但目标是解决实际问题：

- 分层配置；
- 类型和范围校验；
- 命令行覆盖；
- 训练配置快照；
- 运行时派生值；
- DDP/FSDP/DeepSpeed 后端配置；
- secret 与普通配置分离。

不要仅为了“工业感”引入复杂框架。

### 11.2 配置校验

至少校验：

- `n_embd % n_head == 0`；
- batch、sequence、accumulation、world size 合法；
- warmup 不超过总训练进度；
- precision 与硬件兼容；
- tokenizer vocab 与 model/checkpoint 一致；
- 恢复时结构性配置不能静默改变；
- 输出目录不会误覆盖其他实验。

### 11.3 实验身份

每次运行保存：

- 完整 resolved config；
- 命令行；
- git commit 和 dirty 状态；
- Python/PyTorch/CUDA/NCCL 版本；
- GPU 型号和数量；
- tokenizer/data/checkpoint hash；
- 开始时间、主机和 job id。

这样性能或精度变化才能追溯。

---

## 12. 测试体系

### 12.1 单元测试

- tokenizer encode/decode；
- batch 边界和右移；
- attention causal 性；
- loss 数值与梯度；
- optimizer 一步更新；
- scheduler 边界；
- gradient clipping；
- scaler overflow；
- config 校验；
- checkpoint schema。

### 12.2 Reference-vs-optimized 对照测试

每个教学实现替换后，保留小 shape 对照：

```text
forward output
loss
input gradient
parameter gradient
optimizer update
```

按 FP32、BF16、FP16 设置不同容差。

### 12.3 集成测试

- 1～5 step smoke train；
- eval 和 sample；
- save/load/resume；
- fresh run 输出目录保护；
- NaN/Inf 跳步；
- DDP 2-process smoke test；
- rank 0 logging/checkpoint；
- 人为终止后恢复。

### 12.4 性能回归测试

固定环境和 shape，记录：

- warmup 后 median step time；
- tokens/s；
- peak memory；
- kernel/backend；
- 编译次数；
- DDP scaling efficiency。

性能阈值不要绑定开发者个人电脑，应按硬件基线分类。

---

## 13. DDP 工业化路线

### 13.1 第一版只做标准数据并行

- `torchrun` 启动一进程一卡；
- 初始化 process group；
- 使用 `LOCAL_RANK` 绑定设备；
- 所有 rank 使用相同初始模型参数；
- 每个 rank 获取不同数据；
- DDP 包装模型；
- backward 自动 all-reduce 梯度；
- rank 0 保存主日志和 checkpoint；
- 退出时正确销毁 process group。

### 13.2 正确性重点

- 模型初始化前后的 seed 时序；
- 数据 seed 要 rank-aware；
- global batch 和 LR 规则；
- eval 指标全局 reduce；
- checkpoint 恢复后所有 rank 状态一致；
- 任何 rank 失败都不能让其他 rank 永久 hang；
- 不随意开启 `find_unused_parameters=True` 掩盖图结构问题。

### 13.3 性能重点

- gradient accumulation 使用 `no_sync()`；
- 合理 DDP bucket；
- 观察 backward 计算和 all-reduce 是否重叠；
- 避免 rank 0 数据、日志或 checkpoint 成为全局阻塞；
- 报告 scaling efficiency，而不只报告总 tokens/s；
- 先解决负载不均和 data stall，再调通信参数。

---

## 14. FSDP 与 DeepSpeed ZeRO 路线

### 14.1 何时进入 FSDP/ZeRO

满足以下条件后进入：

- DDP 正确性和 checkpoint 已稳定；
- 明确当前显存由参数、梯度、optimizer state、activation 各占多少；
- 模型或目标 batch 确实受到显存限制；
- 已有相同模型和数据的 DDP 性能基线。

### 14.2 FSDP

重点包括：

- auto-wrap policy；
- 参数 all-gather；
- 梯度 reduce-scatter；
- mixed precision policy；
- CPU offload 是否值得；
- activation checkpointing；
- sharded state dict；
- 分片 checkpoint 与 reshard；
- 通信量、显存和速度 trade-off。

### 14.3 DeepSpeed ZeRO

对比：

```text
ZeRO-1：切 optimizer state
ZeRO-2：再切 gradient
ZeRO-3：再切 parameter
```

使用同一模型、global batch、总 token 数、精度和硬件比较显存与吞吐。不要把 DeepSpeed、
FSDP 和大量其他改动同时引入，否则无法判断收益来自哪里。

### 14.4 更大规模并行

训练主线只有在数据并行和状态切分不足以容纳模型时，再引入 sequence/context、pipeline
或 expert parallel。Tensor parallel 需要区分两种用途：

- 对训练而言，它仍是有明确模型规模需求后再引入的高级并行；
- 对推理而言，它直接决定大模型权重如何跨设备放置、每层如何通信以及 TPOT/吞吐如何扩展，
  因此在阶段 F 作为推理主线学习，不等待 FSDP/ZeRO 完成。

候选并行方式包括：

- tensor parallel；
- sequence/context parallel；
- pipeline parallel；
- expert parallel。

它们不属于 MiniGPT 单卡版本的直接优化项，必须在单设备推理与分布式基础稳定后单独引入，
不能和 continuous batching、paged KV cache 等复杂机制一次性叠加。

---

## 15. 推荐执行顺序

### 阶段 A：完成并冻结教学单卡版本

- [x] 看完 `checkpoint.py`；
- [x] 看完 `config.py`；
- [x] 看完 `logging_utils.py`；
- [x] 看完 `tests/test_core.py`；
- [x] 完成连续训练与断点续训一致性实验；
- [x] 建立 CPU FP32 基准：loss、tokens/s、step time；GPU 显存基准待 GPU 环境补测；
- [x] 标记 Git 标签 `baseline-v0.1`，提交 `3468858`。

### 阶段 B：建立优化单卡版本

- [x] 替换 LayerNorm、GELU、cross entropy；
- [x] 替换 AdamW、gradient clipping、GradScaler；
- [x] 建立 optimizer 参数组，并在 state dict 中记录参数名以支持布局迁移；
- [x] 合并 QKV，并支持旧模型权重与 optimizer 动量迁移；
- [x] 替换为 SDPA；
- [x] 删除 micro-batch 热路径 `.item()`；
- [x] 用窗口/CUDA event 测量，不再每 step 全局同步；CUDA 运行效果待真实设备补测；
- [x] 去掉重复 `zero_grad`；
- [x] 原子 checkpoint，避免 numbered/latest 重复序列化完整文件；
- [x] 增加 reference-vs-optimized 输出与梯度测试；
- [ ] 重新记录正确性、吞吐和显存基线：CPU 已完成，GPU 待补测。

#### 2026-08-20 CPU 升级验收记录

固定条件：PyTorch `2.13.0+cpu`、FP32、`tiny_cpu.json`、107,008 参数、30 step；
吞吐使用排除第 1 步后的逐步 `tokens/s` 中位数。这个模型过小且共享 CPU 容易抖动，
结果只用于本机回归，不代表 GPU 收益。

| 版本 | 30-step 中位吞吐 | 相对教学基线 | step 30 train loss | best val loss |
| --- | ---: | ---: | ---: | ---: |
| 教学基线：手写模型原子组件与训练组件 | 30,744.76 tok/s | 1.000x | 2.9852 | 3.1050626 |
| 原生 LayerNorm/GELU/CE | 41,936.99 tok/s | 1.364x | 2.9852 | 3.1051 |
| 再替换 AdamW/clip/GradScaler | 35,887.48 tok/s | 1.167x | 2.9852 | 3.1051 |
| 再切换 SDPA | 36,741.51 tok/s | 1.195x | 2.9852 | 3.1051 |
| 再合并 QKV（当前版本） | 43,104.37 tok/s | 1.402x | 2.9852 | 3.1050604 |

当前版本相对教学基线的最终模型参数最大绝对差为 `2.18e-4`，best val loss 绝对差为
`2.24e-6`。差异来自官方 AdamW 的数值路径与算子浮点顺序，训练曲线保持一致。

额外验收：

- `tests/test_core.py` 通过；
- `tests/test_resume_consistency.py` 通过，连续训练与当前格式 checkpoint 恢复结果逐项完全一致；
- `baseline-v0.1` 的分离 Q/K/V 权重、causal mask、MiniAdamW 动量和 SimpleGradScaler 状态可迁移；
- SDPA 因果性测试通过：改变未来 token 不影响此前位置 logits；
- CPU 上官方 AdamW 单项没有加速，采用它是为了标准状态格式、维护性、GPU fused 路径和分布式兼容；
- GPU/BF16/FP16、Flash Attention 实际 dispatch、显存峰值仍待 GPU 环境验收。

#### 2026-08-21 v0.2.1 单设备收尾验收记录

固定 PyTorch `2.8.0`、CPU FP32、`tiny_cpu.json` 和 30 step，将
`v0.2-native-single-device` 与收尾版各独立运行三次。排除 step 1 后，三次逐 step
`tokens/s` 中位数再取中位数，旧版为 `36,461 tok/s`，收尾版为 `37,993 tok/s`。
共享 CPU 抖动较大，约 `+4.2%` 只说明未观察到明显回退，不外推为 GPU 收益。

收尾版 step 30 loss 为 `2.985147`，best val loss 为 `3.104991`；旧 v0.2 分别为
`2.985228` 和 `3.105060`。微小差异来自新参数组有意取消 bias/LayerNorm 的 weight decay。

额外验收：

- `tests/test_core.py`、`tests/test_reference_parity.py`、`tests/test_resume_consistency.py` 全部通过；
- 模拟 `torch.save` 中途失败后，上一份有效 checkpoint 保持可读，临时残片被清理；
- `latest.pt` 在当前文件系统使用 hard link，payload 只序列化一次；
- 仓库内真实 `baseline-v0.1` 与 v0.2 checkpoint 均能迁移 optimizer/QKV 状态并继续训练；
- CUDA/NPU 精度、event 计时、fused optimizer、SDPA backend 和显存仍待真实设备验收。

### 阶段 C：公共运行时、实验基础设施和最小后端兼容

- [ ] 建立窄边界的 `RuntimeContext`、`DistributedContext` 和 capability 查询；
- [ ] 建立训练/推理共用的 `ExperimentRecorder`，保存 resolved config、环境、硬件和 Git 版本；
- [ ] 训练与推理分别保留 benchmark runner，共用计时、统计和结果格式，不强行共用业务指标；
- [ ] 拆分 data/forward/backward/optimizer/checkpoint 计时；
- [ ] 统一吞吐、延迟、显存和波动的统计口径；
- [ ] 建立 CPU、CUDA（可用时）和 Ascend 的最小 smoke test；
- [ ] Ascend 阶段此时只验证设备、精度、核心算子和短训练/推理可运行，不做专项调优；
- [ ] 接入当前 workload 必需的 tokenizer/data artifact，并冻结其指纹；
- [ ] 建立正确性与性能回归测试。

### 阶段 D：单设备推理 Model Runner

#### D1：MiniGPT reference runner

- [ ] 保留“每步重算全部上下文”的 reference generation；
- [ ] 第一版使用 deterministic greedy decoding，建立稳定正确性参照；
- [ ] 明确 `prefill()` 与 `decode()` 两条执行路径；
- [ ] 实现逐层 KV Cache，并验证 cached 与 uncached logits/生成结果一致；
- [ ] 实现静态 batching、attention mask、position、EOS 和采样状态；
- [ ] 建立 TTFT、TPOT、E2E latency、input/output tokens/s、峰值显存指标；
- [ ] 扫描 batch、input length、output length，形成单设备 baseline；
- [ ] 此阶段不实现 continuous batching，避免同时引入调度和模型并行。

#### D2：一种真实开源 decoder-only 模型

- [ ] 在 MiniGPT 路径稳定后，只选择一种主流架构作为第一种工业 workload；
- [ ] 接入真实 tokenizer、config 和 safetensors 权重；
- [ ] 处理该架构实际使用的 RoPE、RMSNorm、SwiGLU、GQA/MQA 等组件；
- [ ] 建立窄 `ModelRunner` 接口，分别保留 MiniGPT reference adapter 和真实模型 adapter；
- [ ] 与可信参考实现对齐 logits、greedy generation 和 KV Cache 结果；
- [ ] MiniGPT 继续负责 CPU CI、状态检查和故障注入，真实模型负责正式性能与显存报告；
- [ ] 不在第一版追求支持大量模型架构，避免模型兼容工作淹没推理 Infra 主线。

### 阶段 E：分布式基础与 DDP 训练

- [ ] `torchrun` + 一颗可见芯片一个进程；
- [ ] backend 由配置选择 Gloo/NCCL/HCCL，不在训练代码中硬编码；
- [ ] 理解并测试 rank、local rank、world size、process group 和基础 collective；
- [ ] rank-aware data、seed、log、eval、checkpoint；
- [ ] gradient accumulation + `no_sync()`；
- [ ] 2/4 设备正确性和 scaling benchmark；
- [ ] 分布式故障与恢复测试；
- [ ] DDP 作为分布式语义训练场，完成正确实现后不继续挤占推理主线。

### 阶段 F：多设备推理与 Tensor Parallel

- [ ] 在简单同步 workload 上实现 TP，不同时引入 continuous batching；
- [ ] 拆分 attention 与 MLP 的 column/row parallel 线性层；
- [ ] 建立 TP process group、AllReduce/AllGather 等 collective；
- [ ] 明确 KV Cache 在各 TP rank 上的形状、归属和显存占用；
- [ ] 验证 TP=1 与 TP>1 的 logits/生成结果一致性；
- [ ] 记录通信时间、计算时间、TPOT、吞吐、显存和 Scaling Efficiency；
- [ ] 加入 topology-aware rank placement，区分同卡双芯与跨物理卡通信。

### 阶段 G：推理调度与 Continuous Batching

- [ ] 在稳定的 local/TP Model Runner 接口上建立请求生命周期；
- [ ] 实现 waiting/running/finished 队列与逐轮调度；
- [ ] 实现 continuous batching，并处理不同 prompt/output 长度；
- [ ] 建立 KV block 生命周期、请求结束释放和 OOM 边界；
- [ ] 根据 profiler 和碎片证据决定是否实现 paged KV cache；
- [ ] 测量不同并发和请求分布下的 TTFT、TPOT、吞吐及 p50/p95/p99；
- [ ] 记录调度策略改善吞吐时对单请求延迟造成的 trade-off。

### 阶段 H：Ascend 专项适配、Profiler 与单机 Scaling

- [ ] 将 `torch_npu`、CANN、HCCL 和 Ascend profiler 限制在后端适配目录；
- [ ] 采集实际物理卡、可见芯片、HBM、软件版本和互联拓扑；
- [ ] A3 跑 `2/4/8/16 logical devices`，同时记录 `1/2/4/8 physical cards`；
- [ ] A5 跑 `1/2/4/8 devices`；
- [ ] 用 HCCL collective microbenchmark 建立通信基线；
- [ ] 对 TP、KV Cache、Prefill/Decode、Batching 做严格 A/B benchmark；
- [ ] 根据 profiler 证据实施 Ascend 专项算子、通信或内存优化；
- [ ] 16 张物理卡/双机实验仅作为资源允许时的加分项。

### 阶段 I：训练 Infra 支线——显存与状态切分

- [ ] `torch.compile` 或对应后端图编译能力（有稳定收益时）；
- [ ] activation checkpointing；
- [ ] fused optimizer/MLP；
- [ ] FSDP；
- [ ] DeepSpeed ZeRO；
- [ ] 分片/异步 checkpoint；
- [ ] 在相同模型和数据上对比 DDP/FSDP/ZeRO 的显存、吞吐、通信与恢复；
- [ ] 保持训练支线完整，但不让它阻塞 KV Cache、TP、调度等推理主线。

### 版本里程碑与两条主线的施工顺序

“训练 Infra”和“推理 Infra”是最终交付的两个主体，但不是先完成全部训练再开始推理。按照
学习依赖和推理优先的职业方向，暂定版本顺序为：

```text
v0.2.x  优化单设备训练收尾
v0.3    公共 Runtime、指标和实验记录
v0.4    MiniGPT 单设备推理、Prefill/Decode、KV Cache
v0.5    DDP 分布式训练
v0.6    真实开源模型单设备推理
v0.7    Tensor Parallel 多设备推理
v0.8    Continuous Batching 与 KV 生命周期
v0.9    Ascend 适配、Profiler 和单机 Scaling
v1.0    FSDP/ZeRO 训练支线与完整训推交付
v1.x    训推结合的专项研究（方向到该阶段再确定）
```

该顺序允许根据真实 profiler、硬件可用性和学习反馈调整；调整时必须同步记录理由、依赖变化
和新的验收条件。专项研究当前只保留阶段位置，不提前锁定实现主题。

---

## 16. 每项优化的验收模板

每个改动都使用同一模板，避免“代码换了，看起来更工业”但没有证据：

```text
优化名称：
当前瓶颈证据：
改动前实现：
改动后实现：
数学语义是否变化：
测试环境：
模型配置：
global batch / sequence length：
精度设置：
正确性对照：
断点续训对照：
steady-state tokens/s：
end-to-end tokens/s：
peak memory：
首次编译/初始化成本：
已知限制和 fallback：
是否设为默认：
```

### 必须遵守的比较条件

- 同一 tokenizer 和数据版本；
- 同一模型定义；
- 同一 global batch；
- 同一总训练 token 数；
- 同一 LR schedule 语义；
- 同一硬件和尽量一致的软件环境；
- 有 warmup，多次重复，报告 median；
- 性能优化后检查 loss/val loss，而不只看速度。

---

## 17. 最终目标架构

```text
公共运行时与后端边界
  └─ RuntimeContext / DistributedContext / ProfilerAdapter
     ├─ CPU / Gloo
     ├─ CUDA / NCCL
     └─ Ascend / HCCL

离线数据处理
  └─ 清洗 / 去重 / tokenizer / shards / manifest

配置与启动
  └─ resolved config / torchrun / 环境与版本记录

数据管线
  └─ distributed sampler / workers / prefetch / pinned memory

模型计算
  └─ fused QKV / SDPA or Flash / fused MLP / mixed precision

训练循环
  └─ accumulation / DDP or FSDP / fused optimizer / scheduler

推理 Model Runner
  └─ prefill / decode / KV cache / static batching / TP

推理调度
  └─ request lifecycle / continuous batching / KV block lifecycle

状态管理
  └─ checkpoint / exact resume / KV state / retention

可观测性
  └─ loss / TTFT / TPOT / throughput / memory / profiler / communication

实验记录
  └─ resolved config / command / environment / hardware / raw data / Git revision

测试与回归
  └─ numerical parity / cached-vs-reference / resume / distributed smoke / performance regression
```

训练/推理 Infra 的核心不是“全部手写”，也不是“全部交给框架”，而是明确知道：

- 哪些数学和状态必须理解；
- 哪些成熟内核应该复用；
- 哪些系统边界必须由训练代码正确编排；
- 每项性能收益如何被测量和验证；
- 单设备语义如何在分布式环境中保持正确；
- 通用系统逻辑如何与 CUDA、Ascend 等厂商后端隔离。

这份清单作为后续改造 backlog。当前阶段完成阶段 B 的优化单设备收尾，随后进入阶段 C 的
公共运行时和实验基础设施，再按单设备推理、分布式基础、TP、调度、Ascend 专项验证和训练
状态切分支线逐步推进。每阶段只增加一个主要复杂度来源。
