import torch


class AnswerRejector:
    def __init__(self):
        pass

    def select(
            self,
            prompts: list[torch.Tensor],
            groups: list[list[torch.Tensor]],
            scores_groups: list[list[float]]
    ):
        pairs = []
        for prompt, group, scores_group in zip(prompts, groups, scores_groups):
            sorted_group = sorted(list(zip(group, scores_group)), key=lambda x: x[1])
            best = sorted_group[-1][0]
            worst = sorted_group[0][0]
            pairs.append(
                {
                    'prefix_tokens': prompt,
                    'chosen_tokens': best[len(prompt):].tolist(),
                    'rejected_tokens': worst[len(prompt):].tolist(),
                }
            )
        return pairs
