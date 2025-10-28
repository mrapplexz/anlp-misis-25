import torch
from torch import nn

import torch.nn.functional as F

from misisnlp.anlp56.blocks.ffn import SwiGLU


class MoEBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_experts: int,
            top_k: int
    ):
        super().__init__()
        self._experts = nn.ModuleList([SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        ) for _ in range(num_experts)])
        self._router = nn.Linear(hidden_size, num_experts, bias=False)
        self._top_k = top_k
        self._num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hs = hidden_states.shape[2]

        hidden_states = hidden_states.reshape(batch_size * seq_len, hs)

        routes = self._router(hidden_states)
        route_scores, route_top = torch.topk(routes, k=self._top_k, dim=-1)
        route_scores = F.softmax(route_scores, dim=-1)
        route_map = torch.zeros(
            (batch_size * seq_len, self._num_experts),
            dtype=torch.bool,
            device=hidden_states.device
        ).scatter(-1, route_top, 1)
        route_scores = torch.zeros(
            (batch_size * seq_len, self._num_experts),
            dtype=route_scores.dtype,
            device=route_scores.device
        ).scatter(-1, route_top, route_scores)

        hidden_states_result = torch.zeros_like(hidden_states)

        for expert_i, expert in enumerate(self._experts):
            route_map_for_this_expert = route_map[:, expert_i]
            hidden_states_for_this_expert = hidden_states[route_map_for_this_expert]
            hidden_states_for_this_expert = expert(hidden_states_for_this_expert)
            scores_for_this_expert = route_scores[route_map_for_this_expert, expert_i]
            scaled_for_this_expert = scores_for_this_expert[:, None] * hidden_states_for_this_expert
            hidden_states_result[route_map_for_this_expert] += scaled_for_this_expert

        return hidden_states_result.reshape(batch_size, seq_len, hs)



if __name__ == '__main__':
    l = MoEBlock(512, 1024, 16, top_k=4)

    ret = l(torch.randn((3, 122, 512)))
