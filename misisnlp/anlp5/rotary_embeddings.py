import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


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
            rope_base: float,
            max_positions: int,
            hidden_size: int
    ):
        super().__init__()
        args_first, args_second = _compute_rope_vecs(
            max_positions=max_positions,
            rope_base=rope_base,
            d=hidden_size
        )
        self._args_first = nn.Buffer(args_first, persistent=False)
        self._args_second = nn.Buffer(args_second, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # [b; seq; d] -> [b; seq; d]
        args_cos = self._args_first[None, :inputs.shape[1], :]
        inputs_cos = inputs
        cos_part = inputs_cos * args_cos

        args_sin = self._args_second[None, :inputs.shape[1], :]
        inputs_sin = inputs.reshape(inputs.shape[0], inputs.shape[1], -1, 2)
        inputs_sin = torch.stack((-inputs_sin[:, :, :, 1], inputs_sin[:, :, :, 0]), dim=-1)
        inputs_sin = inputs_sin.reshape(inputs_sin.shape[0], inputs_sin.shape[1], -1)
        sin_part = inputs_sin * args_sin

        return cos_part + sin_part


if __name__ == '__main__':
    rope_module = RotaryPositionalEmbeddings(rope_base=10_000, max_positions=8000, hidden_size=128)
    rope_module_lucidrains = RotaryEmbedding(dim = 128, theta=10_000, cache_max_seq_len=8000)
    test_vec = torch.ones((4, 3123, 128))
    test_vec_result = rope_module(test_vec)
    print(test_vec_result)
