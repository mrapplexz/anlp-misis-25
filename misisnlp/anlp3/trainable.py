from typing import Any

import torch
import torchmetrics
from torch import nn

from misisnlp.anlp3.model import TextModelConfig
from misisnlp.trainer.trainer import Trainable


class TextClassificationTrainable(Trainable):
    def __init__(self, config: TextModelConfig):
        self._config = config
        self._loss = nn.CrossEntropyLoss()

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        logits = model(model_inputs['input_ids'], model_inputs['attention_mask'])
        loss = self._loss(logits, model_inputs['target'])
        return loss, {
            'loss': loss,
            'predict': logits.argmax(dim=-1),
            'target': model_inputs['target']
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric(),
            'accuracy': torchmetrics.Accuracy('multiclass', num_classes=self._config.num_classes)
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['accuracy'].update(model_outputs['predict'], model_outputs['target'])


class SpanExtractionTrainable(Trainable):
    def __init__(self, config: TextModelConfig):
        self._config = config
        self._loss = nn.CrossEntropyLoss()

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        logits = model(model_inputs['input_ids'], model_inputs['attention_mask'])  # [batch; num_tokens; 2]
        logits_start = logits[:, :, 0]
        logits_end = logits[:, :, 1]
        start_ce = self._loss(logits_start, model_inputs['start_token_index'])
        end_ce = self._loss(logits_end, model_inputs['end_token_index'])
        loss = start_ce + end_ce

        start_match = logits_start.argmax(dim=-1) == model_inputs['start_token_index']
        end_match = logits_end.argmax(dim=-1) == model_inputs['end_token_index']
        both_match = start_match & end_match
        return loss, {
            'loss': loss,
            'match': both_match
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric(),
            'match': torchmetrics.MeanMetric()
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['match'].update(model_outputs['match'])
