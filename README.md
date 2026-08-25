# MiniGPT 推理 Infra

一个从透明的 MiniGPT 训练闭环起步、逐步建设真实大模型推理 Infra 的学习与工程项目。

项目最高原则：

> 训练是前置学习阶段，用来建立模型、优化器、精度、显存、通信等基础认知；推理 Infra
> 才是项目主线、专项研究方向和最终交付成果。

项目先用 tiny model 把系统链路真正跑通、看懂、改得动，再把正式性能工作迁移到真实模型：

- 数据加载与 tokenizer
- GPT / Transformer block
- forward、loss、backward、optimizer
- mixed precision: fp16 / bf16
- gradient accumulation
- learning rate scheduler
- checkpoint 保存与断点续训
- 日志记录：loss、tokens/s、GPU memory
- 单卡训练 baseline
- 独立生成入口和 deterministic greedy baseline
- Prefill/Decode 阶段边界
- TTFT、TPOT、E2E latency、吞吐和设备内存 Benchmark
- 为 KV Cache、真实模型和分布式推理准备的正确性基线

## 当前阶段

`v0.1～v0.2.2` 已经完成单设备训练基础。它们保留为知识基础和项目演进证据，但后续不再
持续扩建完整训练平台。`v0.3` 已经进入可测量的 MiniGPT 单设备推理基线。

项目第一阶段没有直接堆叠 DDP、FSDP、DeepSpeed，而是先把**单卡训练系统的完整闭环**吃透：

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

DDP 可以作为学习多进程、rank、process group 和 collective 的短实验；FSDP/ZeRO 是按需
训练支线。它们都不再阻塞 KV Cache、真实模型、Tensor Parallel 和推理调度。

已完成的 v0.1 目标是：

```text
把单卡 GPT 预训练系统写完整、注释写明白、能跑、能断点续训、能记录指标。
```

## 当前版本与教学基线

Git 标签 `baseline-v0.1` 保存了完整教学实现，其中 LayerNorm、GELU、
cross entropy、AdamW、梯度裁剪和 loss scaling 都是手写版本。

Git 标签 `v0.2-native-single-device` 保存第一批原生算子升级；`v0.2.1-single-device-closeout`
完成单设备训练的可靠性、热路径和回归测试收尾；`v0.2.2-single-device-correctness`
修复 CUDA resume 设备映射并澄清窗口指标口径。

`v0.3-measurable-single-device-inference` 将生成编排移出模型本体，增加独立 `infer.py`、
Prefill/Decode reference runner、最小 Runtime 边界和同步单请求 Benchmark。v0.3 的 Decode
仍重算完整有效上下文，不使用 KV Cache；这正是 v0.4 的对照基线。

当前工程化分支已经把学完且官方实现更成熟的部分逐步替换为 PyTorch 原生算子。

当前仍显式实现：

- 字符级 tokenizer：`src/minigpt/tokenizer.py`
- GPT 数据 batcher：`src/minigpt/data.py`
- causal multi-head self-attention：`CausalSelfAttention`
- Transformer block：`TransformerBlock`
- cosine warmup learning rate scheduler：`cosine_lr`
- training loop：`train.py`
- checkpoint / resume glue：`train.py` + `checkpoint.py`

已升级为 PyTorch 原生实现：

- LayerNorm：`nn.LayerNorm`
- GELU：`torch.nn.functional.gelu`
- next-token cross entropy：`torch.nn.functional.cross_entropy`
- Q/K/V projection：单个 `nn.Linear(n_embd, 3*n_embd)`
- causal attention kernel：`torch.nn.functional.scaled_dot_product_attention`
- optimizer：`torch.optim.AdamW`
- AdamW 参数组：矩阵/Embedding 权重 decay，bias 与 LayerNorm 参数 no-decay
- gradient clipping：`torch.nn.utils.clip_grad_norm_`
- fp16 loss scaling：`torch.amp.GradScaler`
- 训练 loss 在设备上按窗口累计，只在记录边界取回 CPU
- CUDA event/窗口计时，不再每个 optimizer step 全局同步
- 原子 checkpoint；`latest.pt` 优先硬链接 numbered checkpoint，避免重复序列化

升级必须通过核心测试、精确断点续训测试以及固定配置的 loss/吞吐对照。

## 项目结构

```text
minigpt-train/
  README.md
  requirements.txt
  pyproject.toml
  train.py
  infer.py
  benchmarks/
    infer_single_device.py
  configs/
    tiny_cpu.json
    tiny_gpu.json
  data/
    tiny_corpus.txt
  docs/
    INDUSTRIALIZATION_OPTIMIZATION_PLAN.md
    INFERENCE_FIRST_ROADMAP.md
    V0_3_MEASURABLE_INFERENCE.md
    V0_2_1_SINGLE_DEVICE_CLOSEOUT.md
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
      runtime.py
      inference.py
      benchmark.py
      experiment.py
  tests/
    test_core.py
    test_reference_parity.py
    test_resume_consistency.py
    test_inference.py
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
python tests/test_reference_parity.py
python tests/test_resume_consistency.py
python tests/test_inference.py
```

看到：

```text
All core smoke tests passed.
Reference-vs-optimized parity tests passed.
Exact resume consistency test passed.
v0.3 inference and benchmark tests passed.
```

第一项检查 tokenizer、model、optimizer 参数组、checkpoint 原子保存和旧版本迁移；
第二项对照教学公式与原生 LayerNorm、GELU、cross entropy、SDPA 的输出和梯度；
第三项检查连续训练和从中间 checkpoint 恢复能否得到完全一致的最终训练状态。
第四项检查 Prefill/Decode、greedy/sample、独立 checkpoint 加载和推理指标口径。

## 独立运行一次推理

先使用已有训练产物：

```powershell
python infer.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --strategy greedy `
  --device cpu
```

`greedy` 每次选择 logits 最大的 token，适合建立稳定正确性基线。需要观察随机采样时再使用：

```powershell
python infer.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --strategy sample `
  --temperature 0.9 `
  --top-k 20 `
  --seed 1337
```

## 运行单设备推理 Benchmark

```powershell
python benchmarks/infer_single_device.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --max-new-tokens 32 `
  --device cpu `
  --warmup 2 `
  --repeats 5 `
  --output runs/inference_benchmark.json
```

报告保留每次原始结果，并汇总 median、p50、p90、p99。v0.3 的指标定义为：

- `prefill_ms`：Prefill 开始到首 token 在 device 上计算完成；
- `TTFT`：请求开始到首 token 已经取回并 decode 为文本；
- `TPOT`：除首 token 外，后续 token 计算、取回并 decode 为文本的平均间隔；
- `E2E latency`：请求开始到完整结果 decode 完成；
- `output tokens/s`：生成 token 数除以端到端时间；
- `peak device memory`：模型已经加载后的请求测量窗口内，模型与推理共同占用的峰值设备内存。

CPU 上的 tiny 结果只用于正确性和本机回归，不代表真实模型或工业硬件性能。

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

训练基础已经讲完。学习 v0.3 新增内容时推荐顺序：

1. `src/minigpt/runtime.py`
2. `src/minigpt/inference.py` 的配置和结果对象
3. `MiniGPTModelRunner.prefill()` / `decode()`
4. `InferenceEngine.generate()`
5. `infer.py`
6. `src/minigpt/benchmark.py`
7. `benchmarks/infer_single_device.py`
8. `train.py` 中复用 Runtime/InferenceEngine 的变化
9. `tests/test_inference.py`

核心顺序是先看一次请求如何生成正确 token，再看如何测量这次请求；不要先从 CLI 参数或
checkpoint 路径寻找等工程胶水开始。

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

- `step`: 训练循环执行次数（包含因 fp16 溢出而未更新参数的尝试）
- `optimizer_step`: AdamW 实际成功完成的参数更新次数
- `split`: `train` 或 `val`
- `loss`: next-token prediction loss
- `lr`: 当前学习率
- `tokens_per_sec`: 当前纯训练计时窗口内的 wall-time tokens/s，不包含随后执行的 eval/checkpoint
- `gpu_mem_mb`: 当前 GPU 显存占用
- `gpu_peak_mb`: 当前计时窗口的峰值 GPU 显存
- `grad_norm_last`: 窗口最后一步在 gradient clipping 前的梯度总 norm
- `grad_norm_max`: 整个窗口在 gradient clipping 前的最大梯度总 norm
- `loss_scale`: fp16 时的 loss scale
- `skipped_steps`: 当前窗口因 fp16 梯度出现 inf/nan 而跳过的更新次数

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

当前权威路线（每个版本验收、冻结并讲完变化后，才进入下一版本）：

```text
baseline-v0.1     教学单设备训练闭环
v0.2.x            优化并修正单设备训练
v0.3              可测量的 MiniGPT 单设备推理基线
v0.4              KV Cache、Prefill/Decode 独立路径与静态 Batching
v0.5              真实开源模型单设备推理
v0.6              分布式基础短实验与 Tensor Parallel 推理
v0.7              Continuous Batching、调度与 KV 生命周期
v0.8              推理专项研究
v0.9              Ascend 适配、Profiler 与真实单机 Scaling
v1.0              完整推理 Infra 交付
```

MiniGPT 长期作为白盒 reference、正确性 oracle 和 CPU CI；接入真实模型后，正式性能、显存、
Scaling 和最终报告全部以真实模型为准。完整分工与验收边界见
`docs/INFERENCE_FIRST_ROADMAP.md`。

## 当前版本边界

v0.3 当前故意不做：

- BPE tokenizer
- 大规模数据 streaming
- KV Cache（v0.3 Decode 明确重算上下文）
- 静态或 Continuous Batching
- 真实开源模型
- DDP/FSDP/ZeRO 完整训练系统
- TensorBoard / WandB
- Paged KV Cache
- Tensor Parallel
- 外部 FlashAttention 或自定义 fused kernel

这些功能按推理主线逐版本进入，不能和第一版推理指标、KV Cache、真实模型、TP、调度一次性
叠加，否则结果出错或性能变化时无法归因。
