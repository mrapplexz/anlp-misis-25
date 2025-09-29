import torch
from torch import nn

from misisnlp.anlp4.blocks.config import TransformerConfig


def sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # query: [batch size; seq len query; hidden size]
    # keys or values: [batch size; seq len kv; hidden size]
    attention_values = torch.bmm(query, key.transpose(-1, -2))  # [batch size; seq len query; seq len key]
    scale_factor = query.shape[-1] ** 0.5
    attention_values = attention_values / scale_factor
    attention_values = attention_values + attention_mask
    attention_scores = torch.softmax(attention_values, dim=-1)  # [batch size; seq len query; seq len key]
    result = attention_scores @ value  # [batch size; seq len query; hidden size]
    return result


def _split_for_head(state: torch.Tensor, num_heads: int) -> torch.Tensor:
    # state: [batch size; seq len; hidden_size]
    assert state.shape[-1] % num_heads == 0
    # [batch size; seq len; num heads; hidden size // num_heads]
    x = state.reshape(state.shape[0], state.shape[1], num_heads, state.shape[2] // num_heads)
    # [batch size; num heads; seq len; hidden size // num_heads]
    x = x.permute(0, 2, 1, 3)
    # [batch size * num_heads; seq len; hidden size // num_heads]
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    return x


def _concat_from_head(state: torch.Tensor, num_heads: int) -> torch.Tensor:
    assert state.shape[0] % num_heads == 0
    # [batch size * num_heads; seq len; hidden size // num_heads]
    x = state
    # [batch size; num heads; seq len; hidden size // num_heads]
    x = x.reshape(x.shape[0] // num_heads, num_heads, x.shape[1], x.shape[2])
    # [batch size; seq len; num heads; hidden size // num_heads]
    x = x.permute(0, 2, 1, 3)
    # state: [batch size; seq len; hidden_size]
    x = x.reshape(x.shape[0], x.shape[1], x.shape[3] * num_heads)
    return x


def _repeat_attention_mask_for_heads(mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    return mask.repeat_interleave(num_heads, dim=0)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self._key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self._value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self._out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self._num_heads = config.attention_heads

    def get_dtype(self) -> torch.dtype:
        return self._query_proj.weight.dtype


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # q or k or v: [batch size; some seq len; hidden size]
        query = self._query_proj(query)
        key = self._key_proj(key)
        value = self._value_proj(value)
        query = _split_for_head(query, self._num_heads)
        key = _split_for_head(key, self._num_heads)
        value = _split_for_head(value, self._num_heads)
        attention_mask = _repeat_attention_mask_for_heads(attention_mask, self._num_heads)
        result = sdpa(query, key, value, attention_mask)
        result = _concat_from_head(result, self._num_heads)
        result = self._out_proj(result)
        return result
