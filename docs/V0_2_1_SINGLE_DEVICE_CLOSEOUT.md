# v0.2.1 单设备训练收尾记录

## 1. 版本定位

`v0.2-native-single-device` 已完成模型/训练核心原语替换。本版本不改变 GPT 主体结构，集中
收尾单设备训练的优化器语义、热路径同步、checkpoint 可靠性和回归测试。

## 2. 本版本变化

### AdamW 参数分组

- `ndim >= 2` 的 Linear/Embedding 矩阵进入 `decay` 组；
- bias、LayerNorm scale/bias 等一维参数进入 `no_decay` 组；
- 参数名写入 optimizer state dict，后续迁移不再只依赖参数顺序；
- 支持从 `baseline-v0.1` 分离 Q/K/V、自定义 optimizer，以及 v0.2 单参数组格式恢复。

### 训练热路径

- micro-batch loss 使用 `detach()` 后留在设备上累计；
- 只在日志/eval/checkpoint 边界调用一次 `.item()`；
- 去掉 optimizer step 后重复的 `zero_grad()`；
- 以窗口计算 wall-time tokens/s；CUDA 使用 event 只在窗口边界等待设备；
- eval 的多个 loss 同样先在设备上累计，再取回一次。

### Checkpoint

- `torch.save` 先写同目录临时文件，flush/fsync 成功后 `os.replace`；
- 保存失败时保留上一份有效 checkpoint，并删除临时残片；
- 每次只构造和序列化一次 payload；
- `latest.pt` 优先硬链接 numbered checkpoint，不支持硬链接时原子复制回退。

### 测试

- 参数分组覆盖、不重复和 weight decay 规则；
- v0.2 单参数组 optimizer state 到双参数组迁移；
- checkpoint 中断写入故障注入；
- `latest.pt` 原子更新；
- 手写 reference 与原生 LayerNorm/GELU/cross entropy/SDPA 的输出和梯度对照；
- 连续训练与 checkpoint resume 逐项完全一致；
- 仓库内真实 `baseline-v0.1` 和 v0.2 checkpoint 均可恢复并继续训练。

## 3. CPU 验收

环境：Python 3.12、PyTorch 2.8.0、CPU FP32、`configs/tiny_cpu.json`、30 optimizer steps。
每个版本独立运行三次；每次排除 step 1 后取逐 step tokens/s 中位数，再取三次运行的中位数。

| 版本 | 三次运行中位数 | 三次中位数的中位数 | step 30 loss | best val loss |
| --- | --- | ---: | ---: | ---: |
| `v0.2-native-single-device` | 36,461 / 39,060 / 35,705 | 36,461 tok/s | 2.985228 | 3.105060 |
| `v0.2.1` closeout | 43,045 / 36,809 / 37,993 | 37,993 tok/s | 2.985147 | 3.104991 |

共享 CPU 环境抖动明显，约 `+4.2%` 不能当作稳定性能结论；这里只确认没有观察到明显回退。
loss 的微小变化来自新版本不再对 bias/LayerNorm 参数应用 weight decay，属于有意的优化器语义变化。

## 4. 尚未完成的硬件验收

当前执行环境没有可用 CUDA/NPU，因此以下结果不能写成已经完成：

- CUDA BF16/FP16/FP32；
- CUDA event 窗口计时的实际收益；
- fused AdamW 和 SDPA 实际 backend；
- 峰值显存；
- Ascend 核心算子 smoke test。

这些测试在阶段 C 的最小后端兼容中补齐，不阻塞本版本代码冻结和讲解。

## 5. 讲解顺序

1. `src/minigpt/optim.py`：为什么分组、参数如何只出现一次、旧 optimizer state 怎样迁移；
2. `train.py`：设备侧 loss 累计、窗口边界、为什么删除第二次 `zero_grad()`；
3. `src/minigpt/logging_utils.py`：wall timer、CUDA event 和同步边界；
4. `src/minigpt/checkpoint.py`：临时文件、fsync、原子替换、硬链接；
5. `tests/test_reference_parity.py`：怎样证明“换成优化实现后仍然算的是同一件事”；
6. `tests/test_core.py` 与 `tests/test_resume_consistency.py`：迁移、故障注入和精确恢复。
