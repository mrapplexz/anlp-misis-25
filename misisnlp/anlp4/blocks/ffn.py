import torch
from torch import nn

from misisnlp.anlp4.blocks.config import TransformerConfig


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._linear_in = nn.Linear(config.hidden_size, config.feedforward_hidden_size)
        self._act = nn.GELU()
        self._linear_out = nn.Linear(config.feedforward_hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._linear_in(hidden_states)
        x = self._act(x)
        x = self._linear_out(x)
        return x
