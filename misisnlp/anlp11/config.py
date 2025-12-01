from pathlib import Path
from typing import Literal, Annotated

from pydantic import BaseModel, Field

from misisnlp.trainer.config import TrainerConfig


class LoraConfigPydantic(BaseModel):
    modules_apply_re: str
    modules_full_train: list[str]
    rank: int
    alpha: int


class DpoConfig(BaseModel):
    algorithm: Literal['dpo'] = 'dpo'

    beta: float


class OrpoConfig(BaseModel):
    algorithm: Literal['orpo'] = 'orpo'

    po_loss_weight: float


class SimpoConfig(BaseModel):
    algorithm: Literal['simpo'] = 'simpo'

    beta: float
    margin: float


AnyPOConfig = Annotated[DpoConfig | OrpoConfig | SimpoConfig, Field(discriminator='algorithm')]


class DatasetConfig(BaseModel):
    num_proc: int
    shuffle_seed: int
    use_samples: int


class LLMTrainingConfig(BaseModel):
    base_model: str
    dataset: DatasetConfig
    export_path: Path

    trainer: TrainerConfig
    lora: LoraConfigPydantic
    algorithm: AnyPOConfig
