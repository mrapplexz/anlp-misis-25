import torch


class RewardAssessor:
    def __init__(self, model, device):
        self._model = model
        self._device = device

    @torch.inference_mode
    def assess(self, groups: list[list[torch.Tensor]]) -> list[list[float]]:
        # todo you may add batching and dataloading here
        groups_rewards = []
        for group in groups:
            group_rewards = []
            for answer in group:
                logits = self._model(answer[None, :].to(self._device)).logits
                group_rewards.append(logits.item())
            groups_rewards.append(group_rewards)
        return groups_rewards
