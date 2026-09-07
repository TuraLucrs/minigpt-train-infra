"""MiniGPT-Train最小smoke test。

这个文件不追求覆盖所有训练行为，只检查最核心的部件能不能协同工作：
- tokenizer 能 encode/decode
- model forward shape 正确
- next-token loss 能 backward
- PyTorch AdamW 能更新参数
- 教学版 optimizer/scaler checkpoint 能迁移
- checkpoint 能保存和加载

运行：
    python tests/test_core.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    update_latest_checkpoint,
)
from minigpt.data import RandomTokenBatcher  # noqa: E402
from minigpt.model import MiniGPT, MiniGPTConfig, migrate_model_state_dict, next_token_cross_entropy  # noqa: E402
from minigpt.optim import build_adamw_param_groups, load_grad_scaler_state, load_optimizer_state  # noqa: E402
from minigpt.tokenizer import CharTokenizer  # noqa: E402
from train import checkpoint_run_dir, find_resume_tokenizer_path, tokenizer_fingerprint, validate_resume_metadata  # noqa: E402


def main() -> None:
    text = "hello minigpt\nhello training\n"
    tokenizer = CharTokenizer.train_from_text(text)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer_groups = build_adamw_param_groups(model, weight_decay=0.01)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    assert [group["group_name"] for group in optimizer.param_groups] == ["decay", "no_decay"]
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    grouped_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len({id(parameter) for parameter in grouped_parameters}) == len(grouped_parameters)
    assert {id(parameter) for parameter in grouped_parameters} == {id(parameter) for parameter in trainable_parameters}
    assert all(parameter.ndim >= 2 for parameter in optimizer.param_groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in optimizer.param_groups[1]["params"])
    assert optimizer.state_dict()["param_groups"][0]["param_names"]

    x = torch.tensor([ids[:8], ids[1:9]], dtype=torch.long)
    y = torch.tensor([ids[1:9], ids[2:10]], dtype=torch.long)

    logits = model(x)
    assert logits.shape == (2, 8, tokenizer.vocab_size)

    # 因果 Attention：改变未来 token 不得影响更早位置的 logits。
    model.eval()
    causal_a = x[:1].clone()
    causal_b = causal_a.clone()
    causal_b[:, 4:] = torch.flip(causal_b[:, 4:], dims=(1,))
    logits_a = model(causal_a)
    logits_b = model(causal_b)
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)
    model.train()

    loss = next_token_cross_entropy(logits, y)
    expected_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert loss.ndim == 0

    try:
        next_token_cross_entropy(logits, torch.full_like(y, tokenizer.vocab_size), debug_checks=True)
    except ValueError:
        pass
    else:
        raise AssertionError("debug_checks should reject token ids outside the vocabulary")

    loss.backward()

    # block_size+1 个 token 恰好足够组成一个 GPT 样本。
    exact_batcher = RandomTokenBatcher(
        tokens=torch.arange(9),
        batch_size=4,
        block_size=8,
        device=torch.device("cpu"),
        seed=123,
    )
    bx, by = exact_batcher.get_batch()
    assert bx.shape == (4, 8)
    assert by.shape == (4, 8)
    assert torch.equal(by, bx + 1)

    # 共有10个token且block_size为8时，起点0和1都合法；旧版边界错误会漏掉起点1。
    boundary_batcher = RandomTokenBatcher(
        tokens=torch.arange(10),
        batch_size=256,
        block_size=8,
        device=torch.device("cpu"),
        seed=123,
    )
    bx, by = boundary_batcher.get_batch()
    assert torch.equal(by, bx + 1)
    assert torch.any(bx[:, 0] == 1)

    before = model.token_embedding.weight.detach().clone()
    optimizer.step()
    optimizer.zero_grad()
    after = model.token_embedding.weight.detach().clone()
    assert not torch.equal(before, after)

    # v0.2按模型注册顺序只保存一个AdamW参数组；恢复到新的decay/no_decay布局时，
    # 必须按参数名保留动量。
    old_native_optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)
    old_native_optimizer.zero_grad(set_to_none=True)
    old_native_loss = next_token_cross_entropy(model(x), y)
    old_native_loss.backward()
    old_native_optimizer.step()
    old_native_state = old_native_optimizer.state_dict()

    grouped_optimizer = torch.optim.AdamW(build_adamw_param_groups(model, 0.01), lr=9e-4)
    load_optimizer_state(grouped_optimizer, old_native_state, model)
    assert len(grouped_optimizer.state) == len(old_native_optimizer.state)
    assert torch.equal(
        grouped_optimizer.state[model.token_embedding.weight]["exp_avg"],
        old_native_optimizer.state[model.token_embedding.weight]["exp_avg"],
    )
    assert grouped_optimizer.param_groups[0]["weight_decay"] == 0.02
    assert grouped_optimizer.param_groups[1]["weight_decay"] == 0.0

    # checkpoint迁移：baseline-v0.1按参数逐项保存状态，而不是原生optimizer参数ID/组。
    migrated_optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4)
    parameters = [parameter for group in migrated_optimizer.param_groups for parameter in group["params"]]
    legacy_optimizer_state = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "step_num": 3,
        "state": [
            {"exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.ones_like(parameter)}
            for parameter in parameters
        ],
    }
    load_optimizer_state(migrated_optimizer, legacy_optimizer_state, model)
    assert migrated_optimizer.state[parameters[0]]["step"].item() == 3
    assert migrated_optimizer.param_groups[0]["betas"] == (0.9, 0.95)

    migrated_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0)
    load_grad_scaler_state(
        migrated_scaler,
        {
            "enabled": True,
            "scale": 4096.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "growth_tracker": 7,
        },
    )
    assert migrated_scaler.get_scale() == 4096.0

    # 原生SDPA删除mask buffer，融合QKV替代三个Linear；重建旧布局并验证无损严格加载。
    legacy_model_state = dict(model.state_dict())
    for block_index in range(config.n_layer):
        prefix = f"blocks.{block_index}.attn."
        qkv_weight = legacy_model_state.pop(prefix + "qkv_proj.weight")
        qkv_bias = legacy_model_state.pop(prefix + "qkv_proj.bias")
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        for projection, weight, bias in (
            ("q_proj", q_weight, q_bias),
            ("k_proj", k_weight, k_bias),
            ("v_proj", v_weight, v_bias),
        ):
            legacy_model_state[prefix + projection + ".weight"] = weight
            legacy_model_state[prefix + projection + ".bias"] = bias
        legacy_model_state[prefix + "causal_mask"] = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    migrated_model_state = migrate_model_state_dict(legacy_model_state)
    assert "blocks.0.attn.causal_mask" not in migrated_model_state
    assert torch.equal(
        migrated_model_state["blocks.0.attn.qkv_proj.weight"],
        model.state_dict()["blocks.0.attn.qkv_proj.weight"],
    )
    model.load_state_dict(migrated_model_state, strict=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=3,
            config={"test": True},
            best_val_loss=1.23,
        )
        save_checkpoint(ckpt_path, payload)
        loaded = load_checkpoint(ckpt_path, map_location="cpu")
        assert loaded["step"] == 3
        assert loaded["config"]["test"] is True
        assert "model_state" in loaded

        # 替换失败时必须保留原有效checkpoint，并清理不完整的临时文件。
        original_torch_save = torch.save

        def fail_after_partial_write(_payload, file, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            file.write(b"partial checkpoint")
            file.flush()
            raise RuntimeError("simulated checkpoint interruption")

        torch.save = fail_after_partial_write  # type: ignore[assignment]
        try:
            try:
                save_checkpoint(ckpt_path, {"step": 999})
            except RuntimeError:
                pass
            else:
                raise AssertionError("simulated checkpoint failure should propagate")
        finally:
            torch.save = original_torch_save  # type: ignore[assignment]

        loaded_after_failure = load_checkpoint(ckpt_path, map_location="cpu")
        assert loaded_after_failure["step"] == 3
        assert not list(ckpt_path.parent.glob(f".{ckpt_path.name}.*.tmp"))

        second_path = Path(tmpdir) / "checkpoints" / "step_000004.pt"
        latest_path = Path(tmpdir) / "latest.pt"
        save_checkpoint(second_path, {"step": 4})
        method = update_latest_checkpoint(latest_path, second_path)
        assert method in {"hardlink", "copy"}
        assert load_checkpoint(latest_path, map_location="cpu")["step"] == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        tokenizer_path = run_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)

        latest_path = run_dir / "latest.pt"
        step_path = checkpoints_dir / "step_000003.pt"
        latest_path.write_bytes(b"not a real checkpoint")
        step_path.write_bytes(b"not a real checkpoint")

        checkpoint = {
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_hash": tokenizer_fingerprint(tokenizer),
            "data_hash": "abc",
            "config": {"runtime": {}},
        }
        assert checkpoint_run_dir(latest_path) == run_dir
        assert checkpoint_run_dir(step_path) == run_dir
        assert find_resume_tokenizer_path(step_path, checkpoint, Path(tmpdir) / "new_run") == tokenizer_path
        validate_resume_metadata(checkpoint, tokenizer, tokenizer_fingerprint(tokenizer), "abc")

        bad_checkpoint = dict(checkpoint)
        bad_checkpoint["tokenizer_hash"] = "definitely-wrong"
        try:
            validate_resume_metadata(bad_checkpoint, tokenizer, tokenizer_fingerprint(tokenizer), "abc")
        except ValueError:
            pass
        else:
            raise AssertionError("resume metadata validation should reject tokenizer hash mismatches")

    print("All core smoke tests passed.")


if __name__ == "__main__":
    main()
