Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 在项目根目录运行一个很小的 CPU 训练。
# 这个脚本适合第一次验证环境，不需要 GPU。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python train.py --config configs/tiny_cpu.json --max_steps 10 --sample --overwrite
