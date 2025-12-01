import re

import torch
from torch import nn


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()

        self.base_layer = base_layer

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, device=self.base_layer.weight.device,
                                dtype=self.base_layer.weight.dtype)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False, device=self.base_layer.weight.device,
                                dtype=self.base_layer.weight.dtype)
        nn.init.zeros_(self.lora_B.weight)

        self.scale_factor = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result_base = self.base_layer(x)
        result_additional = self.scale_factor * self.lora_B(self.lora_A(x))
        return result_base + result_additional


def apply_lora_custom_inplace_(
        master_module: nn.Module,
        target_pattern: re.Pattern,
        rank: int,
        alpha: int,
        modules_full_train: list[str]
):
    for param in master_module.parameters():
        param.requires_grad = False

    for submod_name, submod in master_module.named_modules():
        submod_name: str

        if not isinstance(submod, nn.Linear):
            continue
        if not target_pattern.fullmatch(submod_name):
            continue
        if '.' in submod_name:
            parent_mod_name, child_mod_name = submod_name.rsplit('.', 1)
            parent_mod = master_module.get_submodule(parent_mod_name)
        else:
            child_mod_name = submod_name
            parent_mod = master_module
        parent_mod.register_module(child_mod_name, LoraLinear(
            base_layer=submod,
            rank=rank,
            alpha=alpha
        ))

    for module_full_train_name in modules_full_train:
        for param in master_module.get_submodule(module_full_train_name).parameters():
            param.requires_grad = True


@torch.no_grad()
def merge_lora_custom_inplace_(
        master_module: nn.Module
):
    for submod_name, submod in master_module.named_modules():
        if not isinstance(submod, LoraLinear):
            continue

        if '.' in submod_name:
            parent_mod_name, child_mod_name = submod_name.rsplit('.', 1)
            parent_mod = master_module.get_submodule(parent_mod_name)
        else:
            parent_mod = master_module
            child_mod_name = submod_name

        lora_linear = submod
        new_linear = submod.base_layer
        new_linear.weight += lora_linear.scale_factor * (lora_linear.lora_B.weight @ lora_linear.lora_A.weight)

        parent_mod.register_module(child_mod_name, new_linear)

