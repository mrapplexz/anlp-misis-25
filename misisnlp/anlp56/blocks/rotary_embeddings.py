import torch
from torch import nn

from misisnlp.anlp56.blocks.config import ROPEConfig


def _compute_rope_vecs(
        max_positions: int,
        rope_base: float,
        d: int
) -> tuple[torch.Tensor, torch.Tensor]:
    thetas = torch.tensor([rope_base ** (-2 * (i - 1) / d) for i in range(1, d // 2 + 1)])
    thetas = thetas.repeat_interleave(2)
    thetas = thetas[None, :]  # 1, d
    positions = torch.arange(1, max_positions + 1)
    positions = positions[:, None]
    rope_args = thetas * positions
    rope_args_cos = torch.cos(rope_args)
    rope_args_sin = torch.sin(rope_args)
    return rope_args_cos, rope_args_sin


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            config: ROPEConfig
    ):
        super().__init__()
        args_first, args_second = _compute_rope_vecs(
            max_positions=config.max_positions,
            rope_base=config.base,
            d=hidden_size
        )
        self._args_first = nn.Buffer(args_first, persistent=False)
        self._args_second = nn.Buffer(args_second, persistent=False)

    def forward(self, inputs: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:  # [b; seq; d] -> [b; seq; d]
        if position_ids is None:
            args_cos = self._args_first[None, :inputs.shape[1], :]
        else:
            args_cos = self._args_first[position_ids, :]
        inputs_cos = inputs
        cos_part = inputs_cos * args_cos

        if position_ids is None:
            args_sin = self._args_second[None, :inputs.shape[1], :]
        else:
            args_sin = self._args_second[position_ids, :]
        inputs_sin = inputs.reshape(inputs.shape[0], inputs.shape[1], -1, 2)
        inputs_sin = torch.stack((-inputs_sin[:, :, :, 1], inputs_sin[:, :, :, 0]), dim=-1)
        inputs_sin = inputs_sin.reshape(inputs_sin.shape[0], inputs_sin.shape[1], -1)
        sin_part = inputs_sin * args_sin

        return cos_part + sin_part
