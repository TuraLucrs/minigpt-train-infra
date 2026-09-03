# v0.4：KV Cache 与静态 Batching

## 版本目标

v0.3 已把推理拆成 Prefill 和 Decode，但每个 Decode step 仍重新计算整个有效上下文。v0.4
保留这条 `recompute` 路径作为正确性 oracle，同时增加真正复用历史 K/V 的 `kv_cache`
路径，并让两条路径共享同一个 `InferenceEngine`。

## 主要实现

### 逐层预分配 KV Cache

`MiniGPTKVCache` 为每一层持有固定容量的 key/value Tensor，布局为
`[max_batch_size, n_head, block_size, head_dim]`。Prefill 把有效 token 的 K/V 写入对应位置；
Decode 每次只投影一个新增 token，然后读取当前有效长度内的历史缓存。

缓存不在每个 token 后执行 `torch.cat`，因此稳定状态下地址保持不变，避免随序列增长反复
分配和复制整段历史 K/V。

### 静态 batch 与不同有效长度

`InferenceEngine.generate_batch()` 把一组 prompt 右侧 padding 成 `[B,T]`，并传递
`attention_mask`。缓存为每行独立记录有效长度；`active_mask` 让已经遇到 EOS 的请求停止写
缓存，其余请求继续 Decode。

采样模式为每个 batch row 创建独立 RNG，种子为 `seed + row_index`，避免某行提前停止后
改变其他请求未来的随机数序列。

### MiniGPT 窗口边界

MiniGPT 使用 learned absolute position。上下文到达 `block_size` 后，若窗口向左滑动，保留
token 的 position 会整体变化，旧 K/V 已不再对应新的 position embedding，因此当前实现会
重建这个 batch 的 Prefill 缓存。

这是 MiniGPT reference model 的位置编码限制，不应外推成所有模型的 KV Cache 规则。v0.5
Qwen3 使用 RoPE，并采用不滚动、生成前容量预检的真实模型路径。

## 正确性门

`tests/test_kv_cache.py` 覆盖：

- 不同 prompt 长度下，batched Prefill 与逐行完整 forward 一致；
- cached 与 recompute 的 Prefill/Decode logits 在容差内一致；
- greedy 生成 token 完全一致；
- 正常 Decode 不改变预分配缓存地址；
- inactive row 不更新长度，并允许其他 row 继续生成；
- 窗口滚动重建后仍与 recompute 一致；
- EOS 提前停止和静态 batch 报告可重复。

Benchmark 的 `compare` 模式先检查生成 token 一致，再报告 median TPOT speedup 和峰值设备
内存差；正确性门失败时不产生有效性能结论。

## 运行入口

单请求 A/B：

```bash
python benchmarks/infer_single_device.py \
  --checkpoint runs/tiny_cpu/latest.pt \
  --prompt MiniGPT \
  --decode-mode compare
```

静态 batch：

```bash
python benchmarks/infer_static_batch.py \
  --checkpoint runs/tiny_cpu/latest.pt \
  --prompt MiniGPT \
  --prompt GPT
```

CPU tiny 数据只验证链路与回归，不用于真实模型性能结论。正式性能从真实 Qwen3 和实际设备
路径开始记录。
