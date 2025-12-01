from typing import Any

import torch
import torchmetrics
from torch import nn

import torch.nn.functional as F
from torchmetrics import MeanMetric

from misisnlp.anlp11.config import DpoConfig, AnyPOConfig, OrpoConfig, SimpoConfig
from misisnlp.trainer.trainer import Trainable


def compute_logp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits.float()

    # note: this implementation is inefficient since we compute logits and CE even for tokens that are
    # unused for calculating answer logPs

    shift_labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = shift_labels[..., 1:].contiguous()
    ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction='none')
    ce = ce.view(input_ids.shape[0], input_ids.shape[1])
    ce = ce.sum(dim=1)  # sum by tokens, not by batches
    logps = -ce
    return logps


def build_trainable(config: AnyPOConfig):
    match config:
        case DpoConfig():
            return DpoTrainable(config)
        case OrpoConfig():
            return OrpoTrainable(config)
        case SimpoConfig():
            return SimpoTrainable(config)
        case _:
            raise ValueError()



class DpoTrainable(Trainable):
    def __init__(self, config: DpoConfig):
        self.config = config

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        beta = self.config.beta

        with torch.no_grad():
            with model.disable_adapter():
                # since we can calculate cross-entropy loss by HF model (model(...).loss), we can infer logps from it by
                # doing logps=-CE
                logps_ref_chosen = compute_logp(
                    model,
                    input_ids=model_inputs['chosen']['input_ids'],
                    attention_mask=model_inputs['chosen']['attention_mask'],
                    labels=model_inputs['chosen']['labels'],
                )
                logps_ref_rejected = compute_logp(
                    model,
                    input_ids=model_inputs['rejected']['input_ids'],
                    attention_mask=model_inputs['rejected']['attention_mask'],
                    labels=model_inputs['rejected']['labels'],
                )

        logps_policy_chosen = compute_logp(
            model,
            input_ids=model_inputs['chosen']['input_ids'],
            attention_mask=model_inputs['chosen']['attention_mask'],
            labels=model_inputs['chosen']['labels'],
        )
        logps_policy_rejected = compute_logp(
            model,
            input_ids=model_inputs['rejected']['input_ids'],
            attention_mask=model_inputs['rejected']['attention_mask'],
            labels=model_inputs['rejected']['labels'],
        )

        chosen_reward = beta * (logps_policy_chosen - logps_ref_chosen)
        rejected_reward = beta * (logps_policy_rejected - logps_ref_rejected)

        reward_diff = chosen_reward - rejected_reward

        result = -F.logsigmoid(reward_diff)

        loss = result.mean()

        return loss, {
            'loss': loss,
            'chosen_reward': chosen_reward,
            'rejected_reward': rejected_reward,
            'reward_diff': reward_diff
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': MeanMetric(),
            'chosen_reward': MeanMetric(),
            'rejected_reward': MeanMetric(),
            'reward_diff': MeanMetric(),
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['chosen_reward'].update(model_outputs['chosen_reward'])
        metrics['rejected_reward'].update(model_outputs['rejected_reward'])
        metrics['reward_diff'].update(model_outputs['reward_diff'])


class OrpoTrainable(Trainable):
    def __init__(self, config: OrpoConfig):
        self.config = config

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        po_weight = self.config.po_loss_weight

        answer_token_count_chosen = (model_inputs['chosen']['labels'] != -100).sum()
        answer_token_count_rejected = (model_inputs['rejected']['labels'] != -100).sum()

        logps_chosen = compute_logp(
            model,
            input_ids=model_inputs['chosen']['input_ids'],
            attention_mask=model_inputs['chosen']['attention_mask'],
            labels=model_inputs['chosen']['labels'],
        )
        logps_rejected = compute_logp(
            model,
            input_ids=model_inputs['rejected']['input_ids'],
            attention_mask=model_inputs['rejected']['attention_mask'],
            labels=model_inputs['rejected']['labels'],
        )

        logps_chosen_norm = logps_chosen / answer_token_count_chosen
        logps_rejected_norm = logps_rejected / answer_token_count_rejected

        logodds_chosen = logps_chosen_norm - torch.log1p(-torch.exp(logps_chosen_norm))
        logodds_rejected = logps_rejected_norm - torch.log1p(-torch.exp(logps_rejected_norm))

        reward_diff = logodds_chosen - logodds_rejected

        po_part = -F.logsigmoid(reward_diff)
        sft_part = -logps_chosen_norm

        loss = po_weight * po_part + sft_part

        loss = loss.mean()

        return loss, {
            'loss': loss,
            'chosen_reward': logodds_chosen,
            'rejected_reward': logodds_rejected,
            'reward_diff': reward_diff
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': MeanMetric(),
            'chosen_reward': MeanMetric(),
            'rejected_reward': MeanMetric(),
            'reward_diff': MeanMetric(),
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['chosen_reward'].update(model_outputs['chosen_reward'])
        metrics['rejected_reward'].update(model_outputs['rejected_reward'])
        metrics['reward_diff'].update(model_outputs['reward_diff'])


class SimpoTrainable(Trainable):
    def __init__(self, config: SimpoConfig):
        self.config = config

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        beta = self.config.beta
        margin = self.config.margin

        answer_token_count_chosen = (model_inputs['chosen']['labels'] != -100).sum()
        answer_token_count_rejected = (model_inputs['rejected']['labels'] != -100).sum()

        logps_chosen = compute_logp(
            model,
            input_ids=model_inputs['chosen']['input_ids'],
            attention_mask=model_inputs['chosen']['attention_mask'],
            labels=model_inputs['chosen']['labels'],
        )
        logps_rejected = compute_logp(
            model,
            input_ids=model_inputs['rejected']['input_ids'],
            attention_mask=model_inputs['rejected']['attention_mask'],
            labels=model_inputs['rejected']['labels'],
        )

        logps_chosen_norm = logps_chosen / answer_token_count_chosen
        logps_rejected_norm = logps_rejected / answer_token_count_rejected

        chosen_reward = beta * logps_chosen_norm
        rejected_reward = beta * logps_rejected_norm

        reward_diff = chosen_reward - rejected_reward - margin

        loss = -F.logsigmoid(reward_diff)

        loss = loss.mean()

        return loss, {
            'loss': loss,
            'chosen_reward': chosen_reward,
            'rejected_reward': rejected_reward,
            'reward_diff': reward_diff
        }

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': MeanMetric(),
            'chosen_reward': MeanMetric(),
            'rejected_reward': MeanMetric(),
            'reward_diff': MeanMetric(),
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['chosen_reward'].update(model_outputs['chosen_reward'])
        metrics['rejected_reward'].update(model_outputs['rejected_reward'])
        metrics['reward_diff'].update(model_outputs['reward_diff'])
