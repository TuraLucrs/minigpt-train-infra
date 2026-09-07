# v0.5：Qwen3 真实模型接入与数值门禁

## 1. 版本目标

v0.5 把 v0.4 的通用生成引擎从 MiniGPT 白盒模型接到 Qwen3 dense decoder。正式目标模型固定为
`Qwen/Qwen3-32B`；tiny Qwen3 只负责 CPU CI、Transformers 对齐和 CLI smoke test，任何 tiny
性能数字都不能进入正式结论。

本版本完成模型结构、权重和 tokenizer 接入，但不伪装成单卡 32B 性能版本。官方 32B BF16
权重约占 62,488.8 MiB；在 64 GiB device 上加 KV Cache、算子 workspace 和 runtime reserve
后静态估算已超过容量。正式硬件性能记录从 v0.6 的 Tensor Parallel 开始。

## 2. 已实现范围

- 读取本地 Hugging Face `config.json`、单文件或分片 safetensors；
- 使用 meta device 建模，再逐参数 CPU→目标设备复制，避免额外完整权重副本；
- Qwen3 RMSNorm、RoPE、SwiGLU、GQA 和独立 Q/K/V/O projection；
- full forward、Prefill 写 KV、单 token Decode；
- 右 padding 与左 padding 的有效 token 位置处理；
- KV Cache 按 `prompt + max_new_tokens` 分配，不默认申请完整 40,960 context；
- Prefill 在进入 LM head 前只选择每行最后有效 hidden state，避免生成完整 `[B,T,V]` logits；
- Transformers full logits、cached Prefill、cached Decode 数值门禁；
- 本地 tokenizer 与可选 chat template / thinking 参数；
- 读取官方 `generation_config.json` 的多 EOS 停止条件，并支持 top-k/top-p 采样；
- CPU/CUDA/NPU Runtime smoke 入口和最小 Ascend runtime 适配；
- 32B 参数量与按 TP rank 的静态 HBM 规划。

当前明确不支持并会报错的 config：

- `use_sliding_window=true`；
- 非空 `rope_scaling`。

## 3. 官方 32B 配置门禁

`configs/qwen3_32b_official.json` 固化当前接入目标的关键结构：64 层、hidden size 5120、
64 个 query heads、8 个 KV heads、head dim 128、vocab 151936、context 40960。特别要注意：
Q projection 宽度为 `64 × 128 = 8192`，并不等于 hidden size 5120。

解析后的精确参数量必须为：

```text
32,762,123,264
```

## 4. 验收命令

```bash
python tests/test_qwen3.py
python scripts/create_tiny_qwen3_fixture.py
python benchmarks/check_qwen3_parity.py --model-dir runs/tiny_qwen3_fixture
python infer_qwen3.py --model-dir runs/tiny_qwen3_fixture --device cpu --precision fp32
python benchmarks/infer_qwen3_single_device.py \
  --model-dir runs/tiny_qwen3_fixture --device cpu --precision fp32 \
  --max-new-tokens 4 --warmup 1 --repeats 2
python benchmarks/infer_qwen3_static_batch.py \
  --model-dir runs/tiny_qwen3_fixture --device cpu --precision fp32 \
  --prompt "Hello world" --prompt "Qwen inference" --max-new-tokens 4
python benchmarks/plan_qwen3_memory.py --tp 1 2 4 8 16
python benchmarks/runtime_smoke.py --device auto --precision bf16
```

真实权重只从显式给定的本地目录读取，代码不会在运行中静默联网下载。机器上的第一阶段任务是
先运行 `runtime_smoke.py` 和模型目录/config 检查；32B 完整权重的正式 Prefill/Decode 性能测试
等待 v0.6 TP 分片加载后进行。

正式报告会固定记录 config、tokenizer 和 generation config 哈希，并且必须传入
`--hash-weights`。未计算所有 safetensors shard SHA-256 的 32B 报告会标记为
`qwen3_32b_unhashed`；只有结构为精确 32B 且包含完整权重哈希时才标记为
`formal_qwen3_32b_hashed`。

性能对比默认使用 greedy，便于复现相同 token 序列。需要复现官方 Qwen3 采样建议时显式传入：

```text
--strategy sample --temperature 0.6 --top-k 20 --top-p 0.95
```

Qwen3 CLI 默认从模型目录的 `generation_config.json` 读取全部停止 token；重复传入
`--eos-token-id` 可显式覆盖这一集合。

## 5. 证据边界

| 结果 | 能证明 | 不能证明 |
|---|---|---|
| tiny Transformers parity | 数学结构、mask、RoPE、GQA、cache 实现正确 | 32B 性能与显存 |
| 32B config/参数量门禁 | 目标结构解析和容量计算正确 | 权重内容正确、设备能装下 |
| Runtime smoke | 后端 API、autocast、同步、Event、内存接口可执行 | 模型吞吐和 Scaling |
| 真实 32B TP Benchmark（v0.6） | 实际 TTFT/TPOT/吞吐/HBM/Scaling | 尚未执行前不得提前声称 |

## 6. v0.6 输入

v0.6 必须复用本版本的 config、权重迭代器、Qwen3 数学和通用 InferenceEngine，只新增分布式
process group、分片参数加载、列/行并行线性层、collective 和 TP 数值/性能门禁。这样性能变化
才能归因于 Tensor Parallel，而不是同时更换模型定义或生成逻辑。
