from pathlib import Path

from pydantic import BaseModel

from misisnlp.anlp4.bert.data import BertPreTrainingDataConfig
from misisnlp.anlp4.blocks.config import TransformerConfig
from misisnlp.trainer.config import TrainerConfig


class BertPreTrainingConfig(BaseModel):
    architecture: TransformerConfig
    tokenizer_path: Path
    data: BertPreTrainingDataConfig
    trainer: TrainerConfig
