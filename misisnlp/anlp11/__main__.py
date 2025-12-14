from pathlib import Path

import click
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from misisnlp.anlp11.config import LLMTrainingConfig
from misisnlp.anlp11.data_loading import load_anthropic_hh_rlhf, TorchPreferenceCollator, TorchPreferenceDataset
from misisnlp.anlp11.trainables import DpoTrainable, build_preference_trainable
from misisnlp.trainer.trainer import Trainer


@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    config = LLMTrainingConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = load_anthropic_hh_rlhf(
        tokenizer=tokenizer,
        num_proc=config.dataset.num_proc,
        shuffle_seed=config.dataset.shuffle_seed,
        use_samples=config.dataset.use_samples
    )
    dataset = TorchPreferenceDataset(dataset)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16,
        device_map='cuda'
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
    trainer = Trainer(config.trainer, model, build_preference_trainable(config.algorithm), TorchPreferenceCollator())
    trainer.train(dataset, None)
    model = model.merge_and_unload()
    model.save_pretrained(str(config.export_path))


if __name__ == "__main__":
    main()
