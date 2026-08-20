# 学习指南

这份指南按“先读什么、跑什么、改什么”的顺序写。

## 第 0 步：确认能跑

```powershell
cd F:\ai-infra-projects\minigpt-train
python tests/test_core.py
python train.py --config configs/tiny_cpu.json --max_steps 5 --out_dir runs/guide_smoke --overwrite
```

不要跳过这一步。能跑起来，后面才有反馈。

## 第 1 步：读 tokenizer

文件：

```text
src/minigpt/tokenizer.py
```

你要看懂：

- 为什么文本不能直接喂给模型？
- `stoi` 是什么？
- `itos` 是什么？
- `encode` 输出是什么？
- `decode` 怎么还原文本？
- 为什么 tokenizer 要和 checkpoint 一起保存？

小实验：

```python
from minigpt.tokenizer import CharTokenizer

text = "abc cab"
tok = CharTokenizer.train_from_text(text)
ids = tok.encode(text)
print(ids)
print(tok.decode(ids))
```

## 第 2 步：读 data batcher

文件：

```text
src/minigpt/data.py
```

重点理解：

```text
x = [t0, t1, t2, t3]
y = [t1, t2, t3, t4]
```

GPT 不是只在最后一个位置预测，而是在每个位置都预测下一个 token。

你要能解释：

- `block_size` 是什么？
- `batch_size` 是什么？
- 为什么 y 比 x 右移一位？
- 为什么 `tokens` 必须至少有 `block_size + 1` 个？

## 第 3 步：读模型结构

文件：

```text
src/minigpt/model.py
```

推荐顺序：

1. `MiniGPTConfig`
2. `MiniLayerNorm`
3. `gelu`
4. `CausalSelfAttention`
5. `FeedForward`
6. `TransformerBlock`
7. `MiniGPT`
8. `manual_cross_entropy`

重点 shape：

```text
input_ids: [B, T]
token_emb: [B, T, C]
q/k/v: [B, n_head, T, head_dim]
attention scores: [B, n_head, T, T]
logits: [B, T, vocab_size]
targets: [B, T]
```

你可以把这些 shape 手写到纸上。Transformer 初学最容易晕的就是 shape。

## 第 4 步：读 optimizer

文件：

```text
src/minigpt/optim.py
```

先不用推导 AdamW 的全部数学细节，先知道：

- `exp_avg` 是一阶动量；
- `exp_avg_sq` 是二阶动量；
- weight decay 是让权重别无限变大；
- warmup 是训练前期慢慢提高学习率；
- cosine decay 是后期慢慢降低学习率；
- grad clipping 是防止梯度爆炸；
- fp16 scaler 是防止梯度 underflow。

## 第 5 步：读训练循环

文件：

```text
train.py
```

你要重点看这个顺序：

```text
读取配置
读取语料
训练 tokenizer
编码 tokens
切 train/val
创建 batcher
创建 model
创建 optimizer
选择 precision
训练 while loop
gradient accumulation
optimizer step
eval
log
checkpoint
```

训练 loop 里最关键的一段是：

```text
logits = model(x)
loss = manual_cross_entropy(logits, y)
loss_for_backward = loss / gradient_accumulation_steps
scaled_loss.backward()
optimizer.step()
```

你能解释这几行，就已经抓到训练系统的核心了。

## 第 6 步：做参数实验

每次只改一个参数。

实验 1：batch size

```text
batch_size: 8 -> 16
```

观察：

- tokens/s 是否变大？
- CPU/GPU memory 是否变大？
- loss 是否更平滑？

实验 2：block size

```text
block_size: 64 -> 32
```

观察：

- 单 step 是否更快？
- attention 的 `[T, T]` 成本有什么变化？

实验 3：learning rate

```text
learning_rate: 0.001 -> 0.01
```

观察：

- loss 是否变得不稳定？
- grad_norm 是否变大？

实验 4：gradient accumulation

```text
gradient_accumulation_steps: 2 -> 4
```

观察：

- tokens_per_sec 怎么算？
- optimizer step 变少了吗？
- 等效 batch size 怎么变？

## 第 7 步：练断点续训

先跑：

```powershell
python train.py --config configs/tiny_cpu.json --max_steps 10 --overwrite
```

再跑：

```powershell
python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20
```

观察：

- `start_step` 是否从 10 开始？
- `train_log.csv` 是否继续追加？
- `latest.pt` 是否更新？

## 第 8 步：准备进入分布式

完成上面步骤后，再看：

```text
docs/ROADMAP_DDP_FSDP_DEEPSPEED.md
```

不要急。单卡训练 loop 越清楚，分布式越容易学。
