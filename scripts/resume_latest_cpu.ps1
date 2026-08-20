Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 演示断点续训：先确保 scripts/run_tiny_cpu.ps1 已经生成 runs/tiny_cpu/latest.pt。
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python train.py --config configs/tiny_cpu.json --resume runs/tiny_cpu/latest.pt --max_steps 20
