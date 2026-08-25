"""当前训练循环仍需要的 optimizer 辅助函数。

AdamW、gradient clipping 和 fp16 scaling 已改用 PyTorch 维护的实现；
原始教学实现保存在 Git 标签 ``baseline-v0.1``。
"""

from __future__ import annotations

import math
from typing import Any

import torch


def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """把可训练参数分成应用和不应用 weight decay 的 AdamW 参数组。

    矩阵参数（Linear/Embedding weight）应用 weight decay；一维参数（bias 和
    LayerNorm scale/bias）不应用。参数名会作为 checkpoint 元数据写入参数组，
    后续布局迁移无需猜测 optimizer 状态属于哪个模型参数。
    """

    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise ValueError(f"Model parameter appears more than once: {name}")
        seen.add(parameter_id)

        if parameter.ndim >= 2:
            decay.append(parameter)
            decay_names.append(name)
        else:
            no_decay.append(parameter)
            no_decay_names.append(name)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise ValueError("Optimizer parameter grouping did not cover every trainable model parameter exactly once")

    return [
        {
            "params": decay,
            "param_names": decay_names,
            "group_name": "decay",
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "param_names": no_decay_names,
            "group_name": "no_decay",
            "weight_decay": 0.0,
        },
    ]


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """把调度器算出的学习率写入所有 optimizer 参数组。"""

    for group in optimizer.param_groups:
        group["lr"] = lr


def _optimizer_named_parameters(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise ValueError("Optimizer contains a parameter that is not present in the model")
            result.append((name, parameter))
    return result


def _separate_qkv_parameter_names(current_names: list[str]) -> list[str]:
    """把当前融合QKV参数名展开成融合前的参数顺序。"""

    legacy_names = []
    for name in current_names:
        if name.endswith(".attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            for projection in ("q_proj", "k_proj", "v_proj"):
                legacy_names.extend([prefix + projection + ".weight", prefix + projection + ".bias"])
        elif name.endswith(".attn.qkv_proj.bias"):
            continue
        else:
            legacy_names.append(name)
    return legacy_names


def _merge_qkv_optimizer_states(name: str, states_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    suffix = "qkv_proj.weight" if name.endswith("qkv_proj.weight") else "qkv_proj.bias"
    prefix = name[: -len(suffix)]
    old_suffix = "weight" if suffix.endswith("weight") else "bias"
    sources = [states_by_name[prefix + projection + "." + old_suffix] for projection in ("q_proj", "k_proj", "v_proj")]
    if not all(sources):
        return {}

    merged: dict[str, Any] = {}
    for key in sources[0]:
        values = [state[key] for state in sources]
        first = values[0]
        if isinstance(first, torch.Tensor) and first.ndim > 0:
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = first
    return merged


def _move_optimizer_state(state: dict[str, Any], parameter: torch.nn.Parameter) -> dict[str, Any]:
    moved = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif key == "step":
            moved[key] = value.to(device=parameter.device, dtype=torch.float32)
        else:
            moved[key] = value.to(device=parameter.device, dtype=parameter.dtype)
    return moved


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
    model: torch.nn.Module,
) -> None:
    """加载原生状态，并迁移教学版或分离QKV checkpoint。"""

    named_parameters = _optimizer_named_parameters(optimizer, model)
    current_names = [name for name, _ in named_parameters]
    current_count = len(current_names)

    if "param_groups" in payload:
        saved_groups = payload["param_groups"]
        current_groups = optimizer.state_dict()["param_groups"]
        saved_ids = [parameter_id for group in saved_groups for parameter_id in group["params"]]
        same_group_sizes = len(saved_groups) == len(current_groups) and all(
            len(saved_group["params"]) == len(current_group["params"])
            for saved_group, current_group in zip(saved_groups, current_groups)
        )
        saved_names = [name for group in saved_groups for name in group.get("param_names", [])]
        current_group_names = [name for group in current_groups for name in group.get("param_names", [])]
        names_match = bool(saved_names) and saved_names == current_group_names

        # 当前版本的原生checkpoint带有param_names；若optimizer仍为单参数组且
        # 参数数量未变，旧单组checkpoint也可以直接加载。
        if len(saved_ids) == current_count and same_group_sizes and (
            names_match or (len(saved_groups) == 1 and len(current_groups) == 1)
        ):
            # 保留当前设备选择的执行路径选项。例如，CPU checkpoint中的
            # fused=False不应在恢复到CUDA时关闭当前可用的fused AdamW。
            runtime_options = [
                {
                    key: group[key]
                    for key in ("fused", "foreach", "capturable", "differentiable")
                    if key in group
                }
                for group in optimizer.param_groups
            ]
            optimizer.load_state_dict(payload)
            for group, options in zip(optimizer.param_groups, runtime_options):
                group.update(options)
            return

        if saved_names:
            if len(saved_names) != len(saved_ids):
                raise ValueError("Optimizer checkpoint param_names do not match saved parameter IDs")
            source_names = saved_names
        else:
            # v0.2使用model.parameters()注册顺序；新版会把矩阵参数放在一维参数前，
            # 因此这里必须按模型顺序而不是当前optimizer顺序解释旧状态。
            model_order_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            separate_qkv_names = _separate_qkv_parameter_names(model_order_names)
            if len(saved_ids) == len(model_order_names):
                source_names = model_order_names
            elif len(saved_ids) == len(separate_qkv_names):
                source_names = separate_qkv_names
            else:
                raise ValueError("Optimizer state does not match current or separate-QKV model parameters")

        states_by_name = {
            name: payload["state"].get(parameter_id, {})
            for name, parameter_id in zip(source_names, saved_ids)
        }
        hyperparameters = saved_groups[0]
        saved_group_by_name = {
            group["group_name"]: group
            for group in saved_groups
            if isinstance(group.get("group_name"), str)
        }
    else:
        old_states = payload.get("state")
        if not isinstance(old_states, list):
            raise ValueError("Unsupported optimizer checkpoint format")
        model_order_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        source_names = (
            model_order_names
            if len(old_states) == len(model_order_names)
            else _separate_qkv_parameter_names(model_order_names)
        )
        if len(old_states) != len(source_names):
            raise ValueError("Optimizer state does not match current or separate-QKV model parameters")
        step = torch.tensor(float(payload["step_num"]), dtype=torch.float32)
        states_by_name = {
            name: {"step": step, **state}
            for name, state in zip(source_names, old_states)
        }
        hyperparameters = {
            "lr": float(payload["lr"]),
            "betas": (float(payload["beta1"]), float(payload["beta2"])),
            "eps": float(payload["eps"]),
            "weight_decay": float(payload["weight_decay"]),
        }
        saved_group_by_name = {}

    for group in optimizer.param_groups:
        for key in ("lr", "betas", "eps", "amsgrad", "maximize"):
            if key in hyperparameters:
                group[key] = hyperparameters[key]
        group_name = group.get("group_name")
        if group_name in saved_group_by_name:
            group["weight_decay"] = saved_group_by_name[group_name]["weight_decay"]
        elif group_name == "no_decay":
            group["weight_decay"] = 0.0
        elif "weight_decay" in hyperparameters:
            group["weight_decay"] = hyperparameters["weight_decay"]

    optimizer.state.clear()
    for name, parameter in named_parameters:
        if name in states_by_name:
            state = states_by_name[name]
        elif name.endswith(("qkv_proj.weight", "qkv_proj.bias")):
            state = _merge_qkv_optimizer_states(name, states_by_name)
        else:
            state = {}
        if state:
            optimizer.state[parameter] = _move_optimizer_state(state, parameter)


def load_grad_scaler_state(scaler: torch.amp.GradScaler, payload: dict[str, Any]) -> None:
    """加载原生GradScaler状态，或迁移教学版scaler格式。"""

    if not scaler.is_enabled() or not payload:
        return
    if "_growth_tracker" in payload:
        scaler.load_state_dict(payload)
        return
    if not payload.get("enabled", False):
        return

    scaler.load_state_dict(
        {
            "scale": float(payload["scale"]),
            "growth_factor": float(payload["growth_factor"]),
            "backoff_factor": float(payload["backoff_factor"]),
            "growth_interval": int(payload["growth_interval"]),
            "_growth_tracker": int(payload["growth_tracker"]),
        }
    )


def cosine_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """先线性warmup，再用cosine从``base_lr``衰减到``min_lr``。"""

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)
