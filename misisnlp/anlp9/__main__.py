import re
from pathlib import Path
from typing import Any

import click
import torch
import torchmetrics
from peft import get_peft_model, LoraConfig
from torch import nn
from torchvision.datasets import FakeData
from transformers import AutoModelForCausalLM, AutoTokenizer

from misisnlp.anlp9.config import LLMSFTConfig
from misisnlp.anlp9.custom_lora import apply_lora_custom_inplace_, merge_lora_custom_inplace_
from misisnlp.anlp9.fake_dataset import FakeDataset, FakeCollator
from misisnlp.trainer.trainer import Trainable, Trainer


# LORA rank 8: 9.5 it/s, 53% GPU mem
# LORA rank 512: 7.6 it/s, 77% GPU mem


def _print_printable_parameters(model: nn.Module):
    trainable_params = []
    total_params_numel = 0
    trainable_params_numel = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param_name)
            trainable_params_numel += param.numel()
        total_params_numel += param.numel()
    print(f'Trainable Parameter List: {trainable_params}')
    print(f'Total Parameter Count: {total_params_numel / 1000 / 1000:.2f}M')
    print(f'Trainable Parameter Count: {trainable_params_numel / 1000 / 1000:.2f}M')
    print(f'Trainable Parameter %: {trainable_params_numel / total_params_numel * 100:.2f}%')


class SftTrainable(Trainable):
    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        model_outs = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"]
        )
        loss = model_outs.loss

        return loss, {
            'loss': loss
        }


    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric()
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])


@click.command()
@click.option('--config-path', type=Path, default='./config/anlp9/lora-qwen.json')
def main(config_path: Path):
    config = LLMSFTConfig.model_validate_json(config_path.read_text(encoding='utf-8'))

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16,
        device_map='cuda'
    )
    if config.use_custom_lora:
        apply_lora_custom_inplace_(
            model,
            target_pattern=re.compile(config.lora.modules_apply_re),
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            modules_full_train=config.lora.modules_full_train
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora.rank,
                target_modules=config.lora.modules_apply_re,
                lora_alpha=config.lora.alpha,
                lora_dropout=0.0,
                bias='lora_only',
                modules_to_save=config.lora.modules_full_train
            )
        )
    print('== LoRA Applied ==')
    _print_printable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_dataset = FakeDataset(tokenizer)
    trainer = Trainer(config.trainer, model, SftTrainable(), FakeCollator())
    trainer.train(train_dataset, None)

    print('== LoRA Merged ==')
    if config.use_custom_lora:
        merge_lora_custom_inplace_(model)
    else:
        model = model.merge_and_unload()
    model.save_pretrained(str(config.export_path))
    _print_printable_parameters(model)


if __name__ == '__main__':
    main()
