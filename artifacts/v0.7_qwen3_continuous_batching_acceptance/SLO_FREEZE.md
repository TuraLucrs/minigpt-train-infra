# v0.7 D 阶段参数冻结记录（exploratory 校准，2026-09-09）

来源（全部 exploratory，不与正式数据混合）：
- runs/v07/calibration/tp8_mixed/replica-00.json   （config_sha256=97e295b6…，rank 峰值 14064.5MB）
- runs/v07/calibration/2xtp4_mixed/replica-00.json （rank 峰值 20635.2MB）
- runs/v07/calibration/4xtp2_mixed/replica-00.json （rank 峰值 35966.4MB）
- 校准 workload：mixed（trace sha256=0024963d8a4697a7ad3039a2560beb2be8efae07140b77a147f83a9c42a0f50a）
  closed_loop，clients=64，warmup=1，repeats=1；三个布局均无 OOM。

## 冻结值（对全部 18 次正式运行统一）

| 参数 | 冻结值 | 依据 |
|---|---|---|
| max_seq_len | 4096 | trace 最长 prompt 1848 字符 + max_new_tokens ≤32，远低于上限；三布局均不 OOM |
| closed-loop clients | 64 | 与冻结 trace 的 64 请求一致 |
| open-loop arrival | 50ms | 已固化在冻结 trace（arrival_time_ms 0..3150，seed=2026），open_loop 直接复用同源 trace |
| slots/queue tp8 | 32 / 128 | 文档 §12 表 |
| slots/queue 2xtp4 | 16 / 64 | 文档 §12 表（全局 32/128 不变） |
| slots/queue 4xtp2 | 8 / 32 | 文档 §12 表（全局 32/128 不变） |
| warmup / repeats | 1 / 3 | 文档 §12 模板 |
| TTFT SLO | 15000 ms | 覆盖三布局实测最大值 11477.1ms（tp8）+31% 余量 |
| TPOT SLO | 500 ms | 覆盖三布局实测最大值 414.6ms（tp8）+21% 余量 |
| E2E SLO | 30000 ms | 覆盖三布局实测最大值 16918.1ms（tp8）+77% 余量 |
| 遥测 | 200ms 间隔，targets 0=0:0 … 7=3:1 | 文档 §12；≥2 样本/卡/轮 |

冻结时间：2026-09-09 11:2x（UTC+8），先冻结后运行 D 矩阵；正式运行中不再改动。
冻结时校准结论（exploratory）：tp8 混合负载吞吐最高（73.6 tok/s / 3.78 req/s），
副本增加改善单请求延迟（TPOT p50 312→267→222ms）但降低聚合吞吐（73.6→45.6→27.5 tok/s）。
