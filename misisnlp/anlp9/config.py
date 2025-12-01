import re
import typing
from pathlib import Path

from pydantic import BaseModel

from misisnlp.trainer.config import TrainerConfig


class LoraConfigPyd(BaseModel):
    modules_apply_re: str
    modules_full_train: list[str]
    rank: int
    alpha: int


class LLMSFTConfig(BaseModel):
    base_model: str
    trainer: TrainerConfig
    lora: LoraConfigPyd
    export_path: Path
    use_custom_lora: bool

