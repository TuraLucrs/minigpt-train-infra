"""Numerical parity checks for teaching/reference and optimized primitives.

Run:
    python tests/test_reference_parity.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.model import CausalSelfAttention, MiniGPTConfig, next_token_cross_entropy  # noqa: E402


def assert_close(left: torch.Tensor, right: torch.Tensor, label: str, atol: float = 1e-6) -> None:
    if not torch.allclose(left, right, atol=atol, rtol=1e-5):
        difference = (left - right).abs().max().item()
        raise AssertionError(f"{label}: maximum absolute difference is {difference}")


def reference_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized * weight + bias


def reference_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def reference_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_flat = logits.reshape(-1, logits.shape[-1]).float()
    targets_flat = targets.reshape(-1)
    log_normalizer = torch.logsumexp(logits_flat, dim=-1)
    target_logits = logits_flat.gather(1, targets_flat[:, None]).squeeze(1)
    return (log_normalizer - target_logits).mean()


def reference_attention(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, channels = x.shape
    q, k, v = F.linear(x, module.qkv_proj.weight, module.qkv_proj.bias).split(channels, dim=-1)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, sequence_length, module.n_head, module.head_dim).transpose(1, 2)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    scores = q @ k.transpose(-2, -1) / math.sqrt(module.head_dim)
    causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    y = probabilities @ v
    y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
    return F.linear(y, module.out_proj.weight, module.out_proj.bias)


def compare_outputs_and_gradients(
    optimized_output: torch.Tensor,
    reference_output: torch.Tensor,
    optimized_inputs: tuple[torch.Tensor, ...],
    reference_inputs: tuple[torch.Tensor, ...],
    label: str,
    atol: float = 1e-6,
) -> None:
    assert_close(optimized_output, reference_output, f"{label} output", atol=atol)
    optimized_gradients = torch.autograd.grad(optimized_output.sum(), optimized_inputs)
    reference_gradients = torch.autograd.grad(reference_output.sum(), reference_inputs)
    for index, (optimized_gradient, reference_gradient) in enumerate(
        zip(optimized_gradients, reference_gradients)
    ):
        assert_close(
            optimized_gradient,
            reference_gradient,
            f"{label} gradient {index}",
            atol=atol,
        )


def main() -> None:
    torch.manual_seed(2026)

    layer_norm = torch.nn.LayerNorm(8)
    layer_norm_x_optimized = torch.randn(2, 4, 8, requires_grad=True)
    layer_norm_x_reference = layer_norm_x_optimized.detach().clone().requires_grad_(True)
    compare_outputs_and_gradients(
        layer_norm(layer_norm_x_optimized),
        reference_layer_norm(
            layer_norm_x_reference,
            layer_norm.weight,
            layer_norm.bias,
            layer_norm.eps,
        ),
        (layer_norm_x_optimized, layer_norm.weight, layer_norm.bias),
        (layer_norm_x_reference, layer_norm.weight, layer_norm.bias),
        "LayerNorm",
    )

    gelu_x_optimized = torch.linspace(-4.0, 4.0, 33, requires_grad=True)
    gelu_x_reference = gelu_x_optimized.detach().clone().requires_grad_(True)
    compare_outputs_and_gradients(
        F.gelu(gelu_x_optimized, approximate="tanh"),
        reference_gelu_tanh(gelu_x_reference),
        (gelu_x_optimized,),
        (gelu_x_reference,),
        "GELU",
    )

    logits_optimized = torch.randn(2, 5, 11, requires_grad=True)
    logits_reference = logits_optimized.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 11, (2, 5))
    compare_outputs_and_gradients(
        next_token_cross_entropy(logits_optimized, targets),
        reference_cross_entropy(logits_reference, targets),
        (logits_optimized,),
        (logits_reference,),
        "cross entropy",
    )

    config = MiniGPTConfig(
        vocab_size=17,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config).eval()
    attention_x_optimized = torch.randn(2, 6, 16, requires_grad=True)
    attention_x_reference = attention_x_optimized.detach().clone().requires_grad_(True)
    parameters = tuple(attention.parameters())
    compare_outputs_and_gradients(
        attention(attention_x_optimized),
        reference_attention(attention, attention_x_reference),
        (attention_x_optimized, *parameters),
        (attention_x_reference, *parameters),
        "fused-QKV SDPA attention",
        atol=2e-6,
    )

    print("Reference-vs-optimized parity tests passed.")


if __name__ == "__main__":
    main()
