from typing import Any

import torch
import torchmetrics
from torch import nn

from misisnlp.trainer.trainer import Trainable


class BertTrainable(Trainable):
    def __init__(self):
        self._mlm_loss = nn.CrossEntropyLoss(size_average=None, ignore_index=-100, reduction='mean')
        self._nsp_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        logits, nsp_logit = model(
            input_ids=model_inputs["input_ids"],
            segment_ids=model_inputs["segment_ids"],
            attention_mask=model_inputs["attention_mask"]
        )

        loss_mlm = self._mlm_loss(logits.reshape(-1, logits.shape[-1]), model_inputs["mlm_labels"].reshape(-1))
        loss_nsp = self._nsp_loss(nsp_logit.reshape(-1), model_inputs["nsp_target"].to(nsp_logit.dtype))
        loss = loss_mlm + loss_nsp
        return loss, {
            'loss': loss,
            'loss_mlm': loss_mlm,
            'loss_nsp': loss_nsp,
        }


    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric(),
            'loss_mlm': torchmetrics.MeanMetric(),
            'loss_nsp': torchmetrics.MeanMetric(),
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['loss_mlm'].update(model_outputs['loss_mlm'])
        metrics['loss_nsp'].update(model_outputs['loss_nsp'])
