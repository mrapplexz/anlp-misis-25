import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self._up_proj = nn.Linear(hidden_size, intermediate_size)
        self._gate_proj = nn.Linear(hidden_size, intermediate_size)
        self._down_proj = nn.Linear(intermediate_size, hidden_size)
        self._act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._down_proj(self._act(self._gate_proj(hidden_states)) * self._up_proj(hidden_states))
        return x


if __name__ == '__main__':
    l = SwiGLU(512, 1024)

    ret = l(torch.randn((3, 122, 512)))
