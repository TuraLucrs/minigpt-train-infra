# 后续 DDP / FSDP / DeepSpeed 路线

> 历史说明：这是项目早期以训练为主时制定的路线，只保留用于回顾学习过程。
> 当前权威路线见 `docs/INFERENCE_FIRST_ROADMAP.md`。DDP/FSDP/ZeRO 已调整为训练学习支线，
> 不再决定主版本顺序，也不得阻塞推理主线。

当前项目是 v0.1：单卡 MiniGPT 训练闭环。

后续可以按下面顺序演进。

## v0.2：更真实的数据和 tokenizer

目标：

- 支持外部 `.txt` 数据文件。
- 支持更大的语料。
- 可选接入 BPE tokenizer。

注意：

不要太早换复杂 tokenizer。先确保你理解字符级 tokenizer 的完整流程。

## v0.3：DDP 版本

DDP = DistributedDataParallel。

你要改的点：

- 使用 `torchrun` 启动多个进程。
- 每个进程绑定一张 GPU。
- 初始化 process group。
- 根据 `LOCAL_RANK` 设置 device。
- 用 DDP 包装模型。
- 每个 rank 采样不同数据。
- 只让 rank 0 打印日志和保存 checkpoint。

启动命令大概会长这样：

```powershell
torchrun --nproc_per_node=2 train_ddp.py --config configs/ddp.json
```

你要理解的关键词：

- rank
- local_rank
- world_size
- process group
- all-reduce
- gradient synchronization
- DistributedSampler

建议第一版 DDP 只做 data parallel，不碰 FSDP。

## v0.4：FSDP 版本

FSDP = Fully Sharded Data Parallel。

DDP 每张卡都有完整模型参数、梯度和 optimizer state。FSDP 会把这些状态切分到不同卡上，从而省显存。

你要理解：

- 参数什么时候 all-gather？
- 梯度什么时候 reduce-scatter？
- optimizer state 怎么 shard？
- checkpoint 怎么保存？
- 为什么 FSDP 可能省显存但通信更多？

第一版 FSDP 目标：

- 让 MiniGPT 能被 FSDP 包起来。
- 在同一模型配置下记录显存差异。
- 写一张对比表。

对比表可以包含：

```text
mode, n_gpu, batch_size, block_size, params, peak_mem_mb, tokens_per_sec
single, 1, ...
ddp, 2, ...
fsdp, 2, ...
```

## v0.5：DeepSpeed ZeRO

DeepSpeed 重点看 ZeRO：

- ZeRO-1: shard optimizer state
- ZeRO-2: shard optimizer state + gradients
- ZeRO-3: shard optimizer state + gradients + parameters

第一版 DeepSpeed 不要追求源码理解，先做配置实验：

- 同一个模型；
- 同一个 batch / block_size；
- 对比普通 DDP、ZeRO-1、ZeRO-2、ZeRO-3；
- 记录显存和 tokens/s。

## v0.6：Profiler

训练 infra 最重要的能力之一是定位慢在哪里。

你可以把一个 step 拆成：

```text
data_time
forward_time
backward_time
optimizer_time
log_time
checkpoint_time
```

然后再看 GPU profiler。

第一阶段先不用 Nsight，直接在 Python 里用 `time.perf_counter()` 拆开就很好。

## 每一阶段的验收标准

不要用“我看过文档”当验收标准，要用“我能跑并解释”。

DDP 阶段：

- 能用 2 卡跑起来。
- 知道每个 rank 的日志为什么不同。
- 知道为什么只让 rank 0 保存 checkpoint。
- 能解释 global batch size。

FSDP 阶段：

- 能跑起来。
- 能记录 peak memory。
- 能解释为什么显存下降。
- 能解释为什么速度可能下降。

DeepSpeed 阶段：

- 能跑 ZeRO-1/2/3。
- 能写出三者切分的状态不同。
- 能记录显存和 tokens/s 对比。

Profiler 阶段：

- 能说清楚 step time 主要花在哪里。
- 能判断是数据慢、forward 慢、backward 慢，还是 optimizer 慢。

## 最重要的一句话

分布式训练不是“把单卡代码加速 N 倍”的魔法。

它是在单卡训练 step 的基础上，引入多进程、通信、状态切分、同步和故障恢复。

所以单卡训练 loop 越清楚，后面的 DDP / FSDP / DeepSpeed 越容易。
