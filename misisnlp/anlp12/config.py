from pathlib import Path

from pydantic import BaseModel

from misisnlp.anlp11.config import LoraConfigPydantic, AnyPOConfig
from misisnlp.trainer.config import TrainerConfig


class DatasetConfig(BaseModel):
    num_proc: int
    shuffle_seed: int
    use_samples: int


class RewardTrainConfig(BaseModel):
    base_model: str
    dataset: DatasetConfig
    export_path: Path

    trainer: TrainerConfig
    lora: LoraConfigPydantic


class AnswerSamplingConfig(BaseModel):
    num_answers: int
    temperature: float
    top_p: float
    max_new_tokens: int


class IterativeTrainConfig(BaseModel):
    base_model: str
    dataset: DatasetConfig
    answer_sampling: AnswerSamplingConfig
    reward_model_path: Path

    num_iterations: int
    trainer: TrainerConfig
    lora: LoraConfigPydantic
    algorithm: AnyPOConfig

    export_path: Path
