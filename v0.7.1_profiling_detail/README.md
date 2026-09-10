# v0.7.1 Profiling Detail Package（per-rank 原始工件层）

这是 `v0.7.1_ascend_profiling_gate_evidence.tar.gz`（汇总层，752KB）的配套细节包：
六点 × 8 rank × 11 项 profiler 工件（共 528 个文件），逐件 gzip。

## 收集与校验方式

- 以各点 profile_manifest.json 的 artifacts 清单为准（manifest 已随汇总包推送，
  内含每工件 raw sha256 + size_bytes）；收集时逐文件复核 raw sha256，**528/528 全部一致**。
- 每个文件压缩后为 <原名>.gz；`detail_index.json` 记录 kind / 原始路径 /
  raw_sha256 / raw_size / gz 文件名 / gz_sha256 / gz_size。
- SHA256SUMS 覆盖本包全部 .gz 与 index（不含自身）。

## 工件种类（每 rank 11 项）

trace_view.json（Chrome/Perfetto 时间线，解压后可直接拖入 perfetto.dev）、
kernel_details.csv、operator_details.csv、step_trace_time.csv、communication.json、
communication_matrix.csv、api_statistic.csv、op_statistic.csv、hccs.csv、pcie.csv、
profiler_info_N.json。

## 用法

解压单文件：`gzip -dk <file>.gz`；核对：`sha256sum -c SHA256SUMS`；
将 .gz 文件名/哈希对回 detail_index.json 可溯源到 profile_manifest 的 raw 哈希。

## 有意排除（未进包，留于运行机）

- `PROF_*/`（CANN 原始设备 dump，2.81GB）：解析过程的输入中间格式，
  信息已被上述解析后工件完全覆盖；
- `FRAMEWORK/`（Python 调用栈，0.95GB）：host 侧深度取证备用，需要时另行提供；
- r1 失败现场与 OOM 日志（与验收无关，按验收纪律不入包）。
