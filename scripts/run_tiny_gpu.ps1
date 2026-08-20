Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 在有 NVIDIA GPU 的机器上跑一个稍大一点的训练。
# 如果你的显卡不支持 bf16，可以临时加 --precision fp16 或 --precision fp32。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python train.py --config configs/tiny_gpu.json --sample --overwrite
