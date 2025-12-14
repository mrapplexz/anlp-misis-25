from pathlib import Path
from typing import Any

import click
import torch
import torchmetrics
from peft import get_peft_model, LoraConfig
from torch import nn
from torchmetrics import MeanMetric
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from misisnlp.anlp11.data_loading import TorchPreferenceDataset, TorchPreferenceCollator
from misisnlp.anlp12.config import RewardTrainConfig
from misisnlp.anlp12.prompt_loading import load_anthropic_hh_rlhf_prompts
from misisnlp.trainer.trainer import Trainable, Trainer


class RewardTrainable(Trainable):
    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        reward_chosen = model(
            input_ids=model_inputs['chosen']['input_ids'],
            attention_mask=model_inputs['chosen']['attention_mask'],
        ).logits
        reward_rejected = model(
            input_ids=model_inputs['rejected']['input_ids'],
            attention_mask=model_inputs['rejected']['attention_mask'],
        ).logits
        loss_value = -F.logsigmoid(reward_chosen - reward_rejected)
        return loss_value, {
            'loss': loss_value,
            'reward_chosen': reward_chosen,
            'reward_rejected': reward_rejected,
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy': MeanMetric(),
            'reward_chosen': MeanMetric(),
            'reward_rejected': MeanMetric()
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['reward_chosen'].update(model_outputs['reward_chosen'])
        metrics['reward_rejected'].update(model_outputs['reward_rejected'])
        metrics['accuracy'].update(model_outputs['reward_chosen'] > model_outputs['reward_rejected'],
                                   torch.ones_like(model_outputs['reward_chosen']))


@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    config = RewardTrainConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = load_anthropic_hh_rlhf_prompts(
        tokenizer=tokenizer,
        num_proc=config.dataset.num_proc,
        shuffle_seed=config.dataset.shuffle_seed,
        use_samples=config.dataset.use_samples,
        load_prompts_only=False
    )
    dataset = TorchPreferenceDataset(dataset)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16,
        device_map='cuda',
        num_labels=1
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora.rank,
            target_modules=config.lora.modules_apply_re,
            lora_alpha=config.lora.alpha,
            lora_dropout=0.0,
            modules_to_save=config.lora.modules_full_train
        )
    )
    trainer = Trainer(config.trainer, model, RewardTrainable(), TorchPreferenceCollator())
    trainer.train(dataset, None)
    model = model.merge_and_unload()
    model.save_pretrained(str(config.export_path))


if __name__ == "__main__":
    main()
