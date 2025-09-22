from pydantic import BaseModel

from misisnlp.anlp3.model import TextModelConfig
from misisnlp.trainer.config import TrainerConfig


class TextClassificationPipelineConfig(BaseModel):
    data_path: str
    text_model: TextModelConfig
    trainer: TrainerConfig
