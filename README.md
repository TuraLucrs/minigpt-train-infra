# MiniGPT-Train

一个为学习 AI 训练 infra 准备的、**从零实现核心训练链路**的小型 GPT 预训练项目。

这个项目不是为了训练出有用的大模型，而是为了让你把下面这些东西真正跑通、看懂、改得动：

- 数据加载与 tokenizer
- GPT / Transformer block
- forward、loss、backward、optimizer
- mixed precision: fp16 / bf16
- gradient accumulation
- learning rate scheduler
- checkpoint 保存与断点续训
- 日志记录：loss、tokens/s、GPU memory
- 单卡训练 baseline
- 后续 DDP / FSDP / DeepSpeed 扩展路线

## 这个项目适合你现在做吗？

适合，但要稍微改一下原计划的重心。

你现在是大二下，拿到的是训练 infra 实习 offer。第一阶段最重要的不是一口气写 DDP、FSDP、DeepSpeed，而是把**单卡训练系统的完整闭环**吃透：

```text
文本
-> tokenizer
-> token ids
-> x/y batch
-> GPT forward
-> next-token loss
-> backward
-> optimizer step
-> lr schedule
-> log
-> checkpoint
-> resume
```

DDP / FSDP / DeepSpeed 很重要，但它们应该是第二、第三阶段。否则你会在“多进程、通信、显存切分、配置框架”里迷路，还没来得及理解训练循环本身。

所以本项目的 v0.1 目标是：

```text
把单卡 GPT 预训练系统写完整、注释写明白、能跑、能断点续训、能记录指标。
```

## 当前版本与教学基线

Git 标签 `baseline-v0.1` 保存了完整教学实现，其中 LayerNorm、GELU、
cross entropy、AdamW、梯度裁剪和 loss scaling 都是手写版本。

当前工程化分支已经把学完且官方实现更成熟的部分逐步替换为 PyTorch 原生算子。

当前仍显式实现：

- 字符级 tokenizer：`src/minigpt/tokenizer.py`
- GPT 数据 batcher：`src/minigpt/data.py`
- causal multi-head self-attention：`CausalSelfAttention`
- Transformer block：`TransformerBlock`
- AdamW optimizer：`MiniAdamW`
- cosine warmup learning rate scheduler：`cosine_lr`
- gradient clipping：`clip_grad_norm`
- fp16 loss scaling：`SimpleGradScaler`
- training loop：`train.py`
- checkpoint / resume glue：`train.py` + `checkpoint.py`

已升级为 PyTorch 原生实现：

- LayerNorm：`nn.LayerNorm`
- GELU：`torch.nn.functional.gelu`
- next-token cross entropy：`torch.nn.functional.cross_entropy`

升级必须通过核心测试、精确断点续训测试以及固定配置的 loss/吞吐对照。

## 项目结构

```text
minigpt-train/
  README.md
  requirements.txt
  pyproject.toml
  train.py
  configs/
    tiny_cpu.json
    tiny_gpu.json
  data/
    tiny_corpus.txt
  docs/
    PLAN_REVIEW.md
    LEARNING_GUIDE.md
    ROADMAP_DDP_FSDP_DEEPSPEED.md
  scripts/
    run_tiny_cpu.ps1
    run_tiny_gpu.ps1
    resume_latest_cpu.ps1
  src/
    minigpt/
      tokenizer.py
      data.py
      model.py
      optim.py
      checkpoint.py
      logging_utils.py
      config.py
  tests/
    test_core.py
    test_resume_consistency.py
```

## 环境准备

建议使用 Python 3.9+。

```powershell
cd F:\ai-infra-projects\minigpt-train
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你已经有 PyTorch 环境，也可以直接运行测试。

## 先跑测试

```powershell
cd F:\ai-infra-projects\minigpt-train
python tests/test_core.py
python tests/test_resume_consistency.py
```

看到：

```text
All core smoke tests passed.
Exact resume consistency test passed.
```

第一项检查 tokenizer、model、loss、optimizer、checkpoint 的最小链路；第二项检查
连续训练和从中间 checkpoint 恢复训练能否得到完全一致的最终训练状态。

## 跑一个 CPU 小训练

```powershell
cd F:\ai-infra-projects\minigpt-train
python train.py --config configs/tiny_cpu.json --max_steps 10 --sample --overwrite
```

或者：

```powershell
.\scripts\run_tiny_cpu.ps1
```

训练输出会类似：

```text
step     1 | train loss 3.80 | lr 2.00e-04 | 12000 tok/s | gpu 0.0/0.0 MB | grad 1.23
```

输出目录：

```text
runs/tiny_cpu/
  tokenizer.json
  train_log.csv
  latest.pt
  checkpoints/
    step_000010.pt
```

## 输出目录规则

为了避免新实验和旧 checkpoint 混在一起，fresh run 如果发现 `out_dir` 里已经有
`train_log.csv`、`latest.pt`、`tokenizer.json` 或 `checkpoints/`，会直接报错。

你有三个选择：

```powershell
# 继续旧实验
python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20

# 新开一个实验目录
python train.py --config configs/tiny_cpu.json --out_dir runs/exp_lr_1e_3

# 明确覆盖旧实验产物
python train.py --config configs/tiny_cpu.json --overwrite
```

resume 时，代码会优先加载 checkpoint 所属 run 目录里的 `tokenizer.json`，并检查
`tokenizer_hash` / `vocab_size`。这可以避免 token id 映射错位。

## 跑一个 GPU 小训练

如果你有 NVIDIA GPU：

```powershell
cd F:\ai-infra-projects\minigpt-train
python train.py --config configs/tiny_gpu.json --sample --overwrite
```

默认使用 `bf16`。如果显卡不支持 bf16，可以改：

```powershell
python train.py --config configs/tiny_gpu.json --precision fp16 --overwrite
```

或者直接用 fp32：

```powershell
python train.py --config configs/tiny_gpu.json --precision fp32 --overwrite
```

## 断点续训

先跑一次训练，生成：

```text
runs/tiny_cpu/latest.pt
```

然后：

```powershell
python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20
```

注意：`--max_steps 20` 表示训练到第 20 个 optimizer step，不是再训练 20 step。

## 你应该怎么读代码？

推荐顺序：

1. `src/minigpt/tokenizer.py`
2. `src/minigpt/data.py`
3. `baseline-v0.1` 里的 `MiniLayerNorm`，再对照当前 `nn.LayerNorm`
4. `src/minigpt/model.py` 里的 `CausalSelfAttention`
5. `src/minigpt/model.py` 里的 `TransformerBlock`
6. `baseline-v0.1` 里的 `manual_cross_entropy`，再对照当前 `next_token_cross_entropy`
7. `src/minigpt/optim.py`
8. `train.py`
9. `src/minigpt/checkpoint.py`
10. `docs/ROADMAP_DDP_FSDP_DEEPSPEED.md`

不要第一天就从 `train.py` 的所有细节开始硬啃。先搞懂 token 怎么来、模型怎么 forward、loss 怎么算，再看训练循环。

## 你可以做的实验

先从很小的改动开始：

```powershell
python train.py --config configs/tiny_cpu.json --max_steps 5 --out_dir runs/exp_5_steps
python train.py --config configs/tiny_cpu.json --max_steps 50 --out_dir runs/exp_50_steps
python train.py --config configs/tiny_cpu.json --max_steps 50 --out_dir runs/exp_50_steps_lr_test
```

然后改 `configs/tiny_cpu.json`：

- `batch_size`: 8 改 16，看 tokens/s 变化
- `block_size`: 64 改 32，看训练速度和 loss 变化
- `n_layer`: 2 改 4，看参数量和速度变化
- `learning_rate`: 0.001 改 0.01，看 loss 是否不稳定
- `gradient_accumulation_steps`: 2 改 4，看等效 batch 变大后有什么变化
- `precision`: GPU 上试 `fp32` / `fp16` / `bf16`

每次只改一个参数。这样你才知道变化来自哪里。

## 日志字段怎么看？

`runs/.../train_log.csv` 里有这些字段：

- `step`: optimizer step 编号
- `split`: `train` 或 `val`
- `loss`: next-token prediction loss
- `lr`: 当前学习率
- `tokens_per_sec`: 每秒处理 token 数
- `gpu_mem_mb`: 当前 GPU 显存占用
- `gpu_peak_mb`: 这个 step 的峰值 GPU 显存
- `grad_norm`: gradient clipping 前的梯度总 norm
- `loss_scale`: fp16 时的 loss scale
- `skipped_step`: fp16 梯度出现 inf/nan 时是否跳过更新

## 为什么 tiny_corpus 这么小？

这是学习项目，不是模型效果项目。

小语料的好处：

- CPU 也能跑；
- 每一步都快；
- checkpoint 小；
- 你可以大胆改代码；
- 出错后容易定位。

等你读懂训练链路后，再换更大的文本数据。

## 后续怎么扩展？

建议路线：

```text
v0.1 单卡训练闭环，也就是当前项目
v0.2 支持更好的 tokenizer 和更大的文本数据
v0.3 改成 DDP：torchrun + 多进程 + gradient all-reduce
v0.4 加 FSDP：参数/梯度/optimizer state shard
v0.5 加 DeepSpeed：ZeRO 配置和显存对比
v0.6 加 profiler：把 step time 拆成 data/forward/backward/optimizer
```

这条路线比一开始直接冲 Megatron / DeepSpeed 源码健康很多。

## 当前版本边界

当前版本故意不做：

- BPE tokenizer
- 大规模数据 streaming
- DDP 多卡训练
- FSDP 参数切分
- DeepSpeed ZeRO
- TensorBoard / WandB
- 高性能 fused kernel
- FlashAttention

这些不是不重要，而是第一阶段先别混在一起。你先把单卡训练闭环拿下，后面加分布式才有根。
