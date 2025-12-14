from pathlib import Path

import click
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from misisnlp.anlp11.data_loading import TorchPreferenceDataset, TorchPreferenceCollator
from misisnlp.anlp11.trainables import build_preference_trainable
from misisnlp.anlp12.answer_generator import AnswerGenerator
from misisnlp.anlp12.config import IterativeTrainConfig
from misisnlp.anlp12.prompt_loading import load_anthropic_hh_rlhf_prompts
from misisnlp.anlp12.rejector import AnswerRejector
from misisnlp.anlp12.reward_assessor import RewardAssessor
from misisnlp.trainer.trainer import Trainer


@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    config = IterativeTrainConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        device_map='cuda',
        dtype=torch.bfloat16,
        num_labels=1
    ).eval()
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16,
        device_map='cuda'
    )
    answer_generator = AnswerGenerator(
        model,
        device='cuda',
        config=config.answer_sampling,
        pad_token_id=tokenizer.pad_token_id
    )
    reward_assessor = RewardAssessor(
        reward_model,
        device='cuda'
    )
    rejector = AnswerRejector()
    for iter_i in range(config.num_iterations):
        model.eval()
        print(f'Starting iteration {iter_i}')
        print('Sampling prompts')
        prompts = load_anthropic_hh_rlhf_prompts(
            tokenizer=tokenizer,
            num_proc=config.dataset.num_proc,
            # use different seeds for each iteration to ensure different sampling
            shuffle_seed=config.dataset.shuffle_seed + iter_i,
            use_samples=config.dataset.use_samples,
            load_prompts_only=True
        )
        print('Generating answers')
        answers = answer_generator.generate(prompts['prefix_tokens'][:])
        print('Assessing answers')
        answer_scores = reward_assessor.assess(answers)
        best_worst = rejector.select(prompts['prefix_tokens'][:], answers, answer_scores)
        model.train()
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

        config.trainer.experiment_name = f'{config.trainer.experiment_name}-iter-{iter_i}'
        trainer = Trainer(config.trainer, model, build_preference_trainable(config.algorithm), TorchPreferenceCollator())
        trainer.train(TorchPreferenceDataset(best_worst), None)

        model = model.merge_and_unload()
        model.save_pretrained(str(config.export_path))


if __name__ == "__main__":
    main()
