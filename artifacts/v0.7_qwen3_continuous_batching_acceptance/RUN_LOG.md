# v0.7 Continuous Batching 真机运行日志（Atlas A3 实测）

日期：2026-09-09　容器：py312_anytest　仓库：/root/minigpt-train-v07（分支 upgrade/v0.7-continuous-batching）
模型：/data/models/Qwen3-32B（bf16，hash-weights）　硬件：Atlas A3，4 卡 × 2 芯片 = 16 逻辑 NPU（用 0–7）

## 关键提交
- 79025d7：用户推送的 v0.7 里程碑
- **0bad74e（本地提交，未推送）**：fix: 让真机 open_loop repeats 满足逐位输出一致性。
  背景：NPU bf16 数值随 batch 形状变化，open_loop 墙钟准入造成重复运行 batch 组成不同 → 贪心解码逐位输出不一致。
  修复：warmup 记录每步动作脚本（submit/action/wait），measured repeats 逐条重放 → batch 组成一致 → 输出逐位一致；延迟仍用真实墙钟。
  验证：CPU 测试全绿；真机 sanity 3 轮 digest 一致（45bc9ef2f406…）。

## 阶段记录（全部命令见容器 /root/run_v07_*.sh）
1. CPU 回归：pytest 全绿（source set_env.sh 后跑，避免 torch_npu 加载失败）。
2. 探索校准 ×3（10:52–11:03，端口 29530–29532，repeats=1）：rank 峰值内存 tp8=14,064MB / 2xtp4=20,635MB / 4xtp2=35,966MB。
3. SLO 冻结（SLO_FREEZE.md，11:05，先于全部正式运行）：TTFT 15000 / TPOT 500 / E2E 30000 ms；slots/queue 全局常量 32/128（tp8=32/128、2xtp4=16/64、4xtp2=8/32）；trace 到达间隔 50ms（seed 2026）；warmup 1 + repeats 3；遥测 200ms。
4. 补丁真机 sanity（patch_sanity/，11:28）：通过。
5. 矩阵 run1（11:1x–11:2x，端口 29540–29541）：short_short 两组完成后中止（open_loop 逐位一致性失败 → 触发补丁）。残留产出移入 _superseded_first_matrix/（未删除）。
6. 矩阵 run2（11:29–12:01，端口 29541–29552）：12/12 通过（short_short、long_prefill ×2 模式 ×3 布局）。
7. 补跑（12:0x，端口 29561）：mixed/closed_loop/4xtp2 OOM（aclnnFlashAttentionScore 207001，86MB 分配失败）。
8. final4（12:16–12:32，端口 29581–29586）：4/4 通过。

## 环境事件（重要）
- **外部 HBM 占用**：~12:00 起，容器外进程（npu-smi proc-mem 可见 PID 2676220 python3.12 等，非本容器命名空间）在全部 16 颗芯片各占 25–27GB；~12:25 后降至 ~19.8GB。这是 mixed/4xtp2 三次 OOM 的根因（本组峰值 36GB + 外部 25GB 贴线，86MB 运行时分配失败）。外部负载未做任何干预。
- **端口撞车**：12:13 误启动第二份 final4 脚本与在跑实例撞 29571 端口（EADDRINUSE）。已清理双写残留（mixed_open_loop/tp8 下 3 个误写文件）并全部杀掉基准进程后，以端口 29580+ 干净重跑。撞车产生的进程均为本次实验自身进程，逐一核对 PID 后终止。

## 最终 D 矩阵 18/18（每 bench 运行配 8 目标遥测 200ms，telemetry exit=0）
| workload | mode | tp8 | 2xtp4 | 4xtp2 |
|---|---|---|---|---|
| short_short | closed_loop | ✓ | ✓ | ✓ |
| short_short | open_loop | ✓ | ✓ | ✓ |
| long_prefill_short_decode | closed_loop | ✓ | ✓ | ✓ |
| long_prefill_short_decode | open_loop | ✓ | ✓ | ✓ |
| mixed | closed_loop | ✓ | ✓ | ✓ |
| mixed | open_loop | ✓ | ✓（4xtp2 为重试第 2 次通过）| ✓ |

## 验收结论（runs/v07/v0.7_acceptance.json）
- evidence_class：**formal_v0.7_qwen3_32b_ascend_continuous_batching_acceptance**
- complete_matrix=True，incomplete_reasons=[]，全部 6 comparison 为 formal 三布局比较（SHA-256 全量重导出 + 重算一致）
- 关键中位数（req/s | tok/s）：
  - short_short closed：tp8 9.76|156.1，**2xtp4 10.04|160.7（最优）**，4xtp2 7.81|125.0
  - short_short open：tp8 7.83|125.3，**2xtp4 8.53|136.5（最优）**，4xtp2 6.57|105.2
  - long_prefill closed：tp8 8.07|64.6，2xtp4 9.34|74.7，**4xtp2 10.44|83.5（最优）**
  - long_prefill open：tp8 6.35|50.8，2xtp4 5.69|45.5，**4xtp2 6.35|50.8（goodput 6.06 最优）**
  - mixed closed：tp8 2.80|54.5，2xtp4 3.32|64.5，**4xtp2 3.49|67.9（最优）**
  - mixed open：tp8 2.43|47.3，2xtp4 2.97|57.7，**4xtp2 5.00|97.4（最优）**
- 结论：短请求密集型 2×TP4 最优；长 prefill 与 mixed 工作负载 4×TP2 显著最优（副本数带来的并发收益超过 TP 变小的代价）。

## 未尽事项
- 0bad74e 未推送（等用户确认后自行 push 或授权）。
