from enum import StrEnum

import torch
from pydantic import BaseModel
from torch import nn
from transformers import AutoModel


class TaskType(StrEnum):
    classify = 'classify'
    span_extraction = 'span_extraction'


class TextModelConfig(BaseModel):
    base_model_name: str
    num_classes: int


class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, task: TaskType):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.head = nn.Linear(hidden_size, num_classes)
        self._task = task

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._task == TaskType.classify:
            x = hidden_states[:, 0]  # get [CLS] token embedding
        else:
            x = hidden_states
        x = self.dense(x)
        x = self.activation(x)
        x = self.head(x)
        return x


class TextClassificationModel(nn.Module):
    def __init__(self, config: TextModelConfig, task: TaskType):
        super().__init__()

        self._base_model = AutoModel.from_pretrained(config.base_model_name)
        self._cls_head = BertClassificationHead(self._base_model.config.hidden_size, config.num_classes, task)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_states = self._base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self._cls_head(last_hidden_states)
        return logits
