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
- 逐层预分配 KV Cache 与只处理新增 token 的 Decode
- 不同 prompt 长度、attention mask、EOS 和静态 batch
- KV Cache 与 recompute 的正确性门和性能对照
- Qwen3 tokenizer/config/safetensors 与真实模型数学
- RoPE、RMSNorm、SwiGLU、GQA 和 Qwen3 KV Cache
- Transformers logits 对齐与 Qwen3-32B HBM/TP 规划
- torchrun process group 与 CPU/Gloo、CUDA/NCCL、Ascend/HCCL 后端边界
- Qwen3 attention/MLP/vocabulary Tensor Parallel 与 rank-local safetensors 加载
- TP rank 0 token 广播、逐 rank HBM、collective 和 Scaling 报告

## 当前阶段

`v0.1～v0.2.2` 已经完成单设备训练基础。它们保留为知识基础和项目演进证据，但后续不再
持续扩建完整训练平台。`v0.5` 已将同一套生成引擎接入 Qwen3，并完成 tokenizer、config、
safetensors、full forward、KV Cache 和 Transformers 数值对齐。当前 `v0.6` 分支已完成
Tensor Parallel 软件实现和无 socket 的 TP=2/4 数学门禁，正在等待真实 Gloo/HCCL 与
Qwen3-32B TP=2/4/8 硬件验收；验收前不创建最终 v0.6 tag。正式模型固定为
`Qwen/Qwen3-32B`，tiny 模型仍只承担正确性与 CI。

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

`v0.4-kv-cache-static-batching` 增加逐层预分配 K/V、Prefill 写缓存、单 token Decode、
不同有效长度的静态 batch、EOS 停止和每请求独立采样 RNG。`recompute` 路径继续作为 oracle，
任何性能报告都必须先通过生成 token 一致性门禁。

`v0.5-qwen3-real-model` 接入本地 Hugging Face Qwen3 权重和 tokenizer，实现 RoPE、RMSNorm、
SwiGLU、GQA、full/cached forward，并建立 Transformers logits 对齐、32B 精确参数量和静态
显存门禁。32B BF16 单卡 64 GiB 没有可靠运行余量，所以正式性能从 v0.6 TP≥2 开始记录。

`v0.6` 使用一进程一 logical device 的 Tensor Parallel，按列/行切分 attention 与 MLP、按词表
切分 Embedding/LM head，并只从 safetensors 读取本 rank 参数。rank 0 统一选择并广播 token；
Benchmark 记录模型加载、逐 rank HBM、TTFT/TPOT/吞吐、collective 与相对最小可运行 TP
基线的 Scaling Efficiency。当前分支属于硬件验收候选，不把线程 collective 仿真冒充 HCCL。

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
  infer_qwen3.py
  infer_qwen3_tp.py
  benchmarks/
    infer_single_device.py
    infer_static_batch.py
    infer_qwen3_single_device.py
    infer_qwen3_static_batch.py
    check_qwen3_parity.py
    plan_qwen3_memory.py
    runtime_smoke.py
    tp_hardware_smoke.py
    infer_qwen3_tp.py
    benchmark_tp_collectives.py
    summarize_tp_scaling.py
  configs/
    tiny_cpu.json
    tiny_gpu.json
    qwen3_32b_official.json
  data/
    tiny_corpus.txt
  docs/
    INDUSTRIALIZATION_OPTIMIZATION_PLAN.md
    INFERENCE_FIRST_ROADMAP.md
    V0_3_MEASURABLE_INFERENCE.md
    V0_4_KV_CACHE_STATIC_BATCHING.md
    V0_5_QWEN3_REAL_MODEL.md
    V0_6_QWEN3_TENSOR_PARALLEL.md
    PROJECT_WORKING_AGREEMENT.md
    V0_2_1_SINGLE_DEVICE_CLOSEOUT.md
    PLAN_REVIEW.md
    LEARNING_GUIDE.md
    ROADMAP_DDP_FSDP_DEEPSPEED.md
  scripts/
    run_tiny_cpu.ps1
    run_tiny_gpu.ps1
    resume_latest_cpu.ps1
    create_tiny_qwen3_fixture.py
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
      qwen3.py
      qwen3_inference.py
      distributed.py
      qwen3_tp.py
      memory_planner.py
  tests/
    test_core.py
    test_reference_parity.py
    test_resume_consistency.py
    test_inference.py
    test_kv_cache.py
    test_qwen3.py
    test_qwen3_tp.py
    test_tp_scaling.py
```

## 环境准备

使用 Python 3.10+（与 `pyproject.toml` 一致）。

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
python tests/test_kv_cache.py
python tests/test_qwen3.py
python tests/test_qwen3_tp.py
python tests/test_tp_scaling.py
```

看到：

```text
All core smoke tests passed.
Reference-vs-optimized parity tests passed.
Exact resume consistency test passed.
v0.3 inference and benchmark tests passed.
v0.4 KV Cache and static batching tests passed.
Qwen3 parity, cache, loading and memory planning tests passed.
v0.6 Qwen3 Tensor Parallel simulation tests passed.
v0.6 TP scaling summary tests passed.
```

第一项检查 tokenizer、model、optimizer 参数组、checkpoint 原子保存和旧版本迁移；
第二项对照教学公式与原生 LayerNorm、GELU、cross entropy、SDPA 的输出和梯度；
第三项检查连续训练和从中间 checkpoint 恢复能否得到完全一致的最终训练状态。
第四项检查 Prefill/Decode、greedy/sample、独立 checkpoint 加载和推理指标口径。
第五项检查 cached/recompute logits 与生成一致性、缓存原地复用、不同长度 mask、窗口滚动、
EOS 和静态 batch Benchmark。
第六项检查 Qwen3/Transformers logits、cached Prefill/Decode、单/分片 safetensors、32B
参数量与 HBM 规划。
第七项用多个 rank-local 模型和确定性线程 collective 检查 TP=2/4 的参数分片、full/cached
logits、KV-head 复制和生成一致性；设置 `MINIGPT_RUN_GLOO_TESTS=1` 后额外执行真实 Gloo
process group。第八项检查报告可比性、speedup 与 Scaling Efficiency 公式。

## Qwen3 v0.5 快速验收

先创建离线 tiny fixture；它只用于正确性，不能作为性能数据：

```powershell
python scripts/create_tiny_qwen3_fixture.py
python benchmarks/check_qwen3_parity.py --model-dir runs/tiny_qwen3_fixture
python infer_qwen3.py `
  --model-dir runs/tiny_qwen3_fixture `
  --device cpu --precision fp32 --max-new-tokens 4 `
  --prompt "Hello world"
python benchmarks/infer_qwen3_static_batch.py `
  --model-dir runs/tiny_qwen3_fixture `
  --device cpu --precision fp32 --max-new-tokens 4 `
  --prompt "Hello world" --prompt "Qwen inference"
```

正式 32B 运行前先做容量和后端门禁：

```powershell
python benchmarks/plan_qwen3_memory.py --tp 1 2 4 8 16
python benchmarks/runtime_smoke.py --device npu --precision bf16
```

静态估算显示：32B BF16 权重约 62,488.8 MiB；加最小 KV Cache、workspace 和 runtime
reserve 后约 65,592.8 MiB，超过 64 GiB。因此 v0.5 不拿单卡 32B 冒险做正式性能结论；
v0.6 完成 TP 分片加载后，从 TP=2/4/8 开始在机器上留正式记录。详细边界见
`docs/V0_5_QWEN3_REAL_MODEL.md`。

## Qwen3 v0.6 Tensor Parallel 硬件验收

软件 smoke 可以先在 tiny fixture 上执行：

```bash
python tests/test_qwen3_tp.py
python tests/test_tp_scaling.py
python infer_qwen3_tp.py \
  --model-dir runs/tiny_qwen3_fixture \
  --device cpu --precision fp32 --max-new-tokens 4 --prompt "Hello world"
```

允许本地 TCP 的 Linux 环境还必须执行：

```bash
MINIGPT_RUN_GLOO_TESTS=1 python tests/test_qwen3_tp.py
```

Ascend 上在加载 32B 权重前，必须先让 tiny Qwen3 经过真实 HCCL、分片 loader、TP
AllReduce/AllGather、KV Cache 和 token broadcast：

```bash
python benchmarks/runtime_smoke.py --device npu --precision bf16
torchrun --standalone --nproc-per-node=2 benchmarks/tp_hardware_smoke.py \
  --device npu --backend hccl --precision bf16 \
  --output runs/tp2_hardware_smoke.json
```

显式指定 NPU/CUDA 和低精度的 smoke 在设备或精度不可用时直接失败，不会回退 CPU/FP32
后输出 `passed`。
真实 32B 使用 `torchrun` 启动 TP=2/4/8，先过对应规模的 hardware smoke 和 collective，
再跑相同 workload 的推理报告，最后合并 Scaling。完整命令、拓扑字段和证据等级见
[`docs/V0_6_QWEN3_TENSOR_PARALLEL.md`](docs/V0_6_QWEN3_TENSOR_PARALLEL.md)。

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

默认使用 `--decode-mode kv_cache`。需要运行 v0.3 对照路径时传入
`--decode-mode recompute`。

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
  --decode-mode compare `
  --warmup 2 `
  --repeats 5 `
  --output runs/inference_benchmark.json
```

`compare` 会分别运行 recompute 和 KV Cache，先验证生成 token 完全一致，再计算 TPOT 加速比
与峰值内存差。报告保留每次原始结果，并汇总 median、p50、p90、p99。指标定义为：

- `prefill_ms`：Prefill 开始到首 token 在 device 上计算完成；
- `TTFT`：请求开始到首 token 已经取回并 decode 为文本；
- `TPOT`：除首 token 外，后续 token 计算、取回并 decode 为文本的平均间隔；
- `E2E latency`：请求开始到完整结果 decode 完成；
- `output tokens/s`：生成 token 数除以端到端时间；
- `peak device memory`：模型已经加载后的请求测量窗口内，模型与推理共同占用的峰值设备内存。

CPU 上的 tiny 结果只用于正确性和本机回归，不代表真实模型或工业硬件性能。

静态 batch 使用多次 `--prompt` 指定同一批请求：

```powershell
python benchmarks/infer_static_batch.py `
  --checkpoint runs/tiny_cpu/latest.pt `
  --prompt "MiniGPT" `
  --prompt "GPT" `
  --max-new-tokens 32 `
  --device cpu
```

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

机器实验窗口结束后，学习 v0.5～v0.6 新增内容时推荐顺序：

1. `src/minigpt/runtime.py`
2. `src/minigpt/qwen3.py` 的 config 与完整 forward
3. Qwen3 attention、RoPE、GQA、Prefill/Decode cache
4. `src/minigpt/qwen3_inference.py`
5. `src/minigpt/inference.py` 的通用 `InferenceEngine`
6. `infer_qwen3.py`
7. `src/minigpt/benchmark.py`
8. `benchmarks/check_qwen3_parity.py`
9. `benchmarks/infer_qwen3_single_device.py`
10. `tests/test_qwen3.py`
11. `src/minigpt/distributed.py`
12. `src/minigpt/qwen3_tp.py`
13. `infer_qwen3_tp.py` 与 `benchmarks/infer_qwen3_tp.py`
14. `benchmarks/benchmark_tp_collectives.py` 与 `summarize_tp_scaling.py`
15. `tests/test_qwen3_tp.py` 与 `tests/test_tp_scaling.py`

核心顺序是先看真实模型怎样得到正确 logits 和 cache，再看通用生成与测量；不要先从 CLI
参数或模型目录路径等工程胶水开始。当前机器窗口优先实现与实测，完整逐段讲解暂时后移，
但每次 commit/tag 后的冻结版本自查不会省略。

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
v0.5              Qwen3 真实模型接入、KV Cache 与数值门禁
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

v0.5 当前故意不做：

- Qwen3 sliding-window attention 或 rope scaling
- 32B 单卡 64 GiB 的无余量强行加载
- Continuous Batching
- DDP/FSDP/ZeRO 完整训练系统
- TensorBoard / WandB
- Paged KV Cache
- Tensor Parallel
- 外部 FlashAttention 或自定义 fused kernel

这些功能按推理主线逐版本进入，不能与真实模型、TP、调度同时改写，否则结果出错或性能变化
时无法归因。版本施工与备份硬约束见 `docs/PROJECT_WORKING_AGREEMENT.md`。
