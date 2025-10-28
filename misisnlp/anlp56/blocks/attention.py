import warnings

import torch
from torch import nn

import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, BlockMask, flex_attention, create_block_mask

from misisnlp.anlp56.att_mask import construct_mask_for_decoder
from misisnlp.anlp56.blocks.config import GroupedQueryAttentionConfig, MultiLatentAttentionConfig, ROPEConfig, \
    SlidingWindowAttentionConfig, MultiHeadAttentionConfig
from misisnlp.anlp56.blocks.rotary_embeddings import RotaryPositionalEmbeddings


def sdpa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    # query: [batch size; seq len query; hidden size]
    # keys or values: [batch size; seq len kv; hidden size]
    attention_values = torch.bmm(query, key.transpose(-1, -2))  # [batch size; seq len query; seq len key]
    scale_factor = query.shape[-1] ** 0.5
    attention_values = attention_values / scale_factor
    attention_values = attention_values + attention_mask
    attention_scores = torch.softmax(attention_values, dim=-1)  # [batch size; seq len query; seq len key]
    result = attention_scores @ value  # [batch size; seq len query; hidden size]
    return result


def _build_block_mask_for_causal_sliding_window_attention(
        window_size: int,
        batch_size: int,
        hidden_size: int,
        seq_len: int,
        device: str
) -> BlockMask:
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def sliding_window_mask(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    total_mask = and_masks(causal_mask, sliding_window_mask)

    total_mask = create_block_mask(total_mask, B=batch_size, H=hidden_size, Q_LEN=seq_len, KV_LEN=seq_len,
                                   device=device)

    return total_mask


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


def _split_for_head_4d(state: torch.Tensor, num_heads: int) -> torch.Tensor:
    # state: [batch size; seq len; hidden_size]
    assert state.shape[-1] % num_heads == 0
    # [batch size; seq len; num heads; hidden size // num_heads]
    x = state.reshape(state.shape[0], state.shape[1], num_heads, state.shape[2] // num_heads)
    # [batch size; num heads; seq len; hidden size // num_heads]
    x = x.permute(0, 2, 1, 3)
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


def _concat_from_head_4d(state: torch.Tensor, num_heads: int) -> torch.Tensor:
    # [batch size; num heads; seq len; hidden size // num_heads]
    x = state
    # [batch size; seq len; num heads; hidden size // num_heads]
    x = x.permute(0, 2, 1, 3)
    # state: [batch size; seq len; hidden_size]
    x = x.reshape(x.shape[0], x.shape[1], x.shape[3] * num_heads)
    return x


def _repeat_batch_for_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    return tensor.repeat_interleave(num_heads, dim=0)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, config: GroupedQueryAttentionConfig):
        super().__init__()

        assert hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

        projection_size = hidden_size // config.num_attention_heads

        self._query_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._key_proj = nn.Linear(hidden_size, projection_size * config.num_key_value_heads)
        self._value_proj = nn.Linear(hidden_size, projection_size * config.num_key_value_heads)
        self._out_proj = nn.Linear(projection_size * config.num_attention_heads, hidden_size)

        self._rope = RotaryPositionalEmbeddings(hidden_size, config.rope)

        self._num_attention_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._normalize_qk = config.enable_qk_norm

    def get_dtype(self) -> torch.dtype:
        return self._query_proj.weight.dtype

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # q or k or v: [batch size; some seq len; hidden size]
        query = self._query_proj(hidden_state)
        key = self._key_proj(hidden_state)
        value = self._value_proj(hidden_state)

        query = self._rope(query)
        key = self._rope(key)

        query = _split_for_head(query, self._num_attention_heads)
        key = _split_for_head(key, self._num_kv_heads)
        value = _split_for_head(value, self._num_kv_heads)

        key = _repeat_batch_for_heads(key, self._num_attention_heads // self._num_kv_heads)
        value = _repeat_batch_for_heads(value, self._num_attention_heads // self._num_kv_heads)

        attention_mask = _repeat_batch_for_heads(attention_mask, self._num_attention_heads)

        if self._normalize_qk:
            # originally it was LayerNorm in https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf
            query = F.rms_norm(query, normalized_shape=(query.shape[-1],))
            key = F.rms_norm(key, normalized_shape=(key.shape[-1],))

        result = sdpa(query, key, value, attention_mask)

        result = _concat_from_head(result, self._num_attention_heads)

        result = self._out_proj(result)
        return result


class SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_size: int, config: SlidingWindowAttentionConfig):
        super().__init__()

        assert hidden_size % config.num_attention_heads == 0

        projection_size = hidden_size // config.num_attention_heads

        self._query_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._key_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._value_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._out_proj = nn.Linear(projection_size * config.num_attention_heads, hidden_size)

        self._rope = RotaryPositionalEmbeddings(hidden_size, config.rope)

        self._num_attention_heads = config.num_attention_heads
        self._window_size = config.window_size

    def get_dtype(self) -> torch.dtype:
        return self._query_proj.weight.dtype

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is not None:
            warnings.warn('sliding window attention does not support providing attention mask explicitly')
        # q or k or v: [batch size; some seq len; hidden size]
        query = self._query_proj(hidden_state)
        key = self._key_proj(hidden_state)
        value = self._value_proj(hidden_state)

        query = self._rope(query)
        key = self._rope(key)

        query = _split_for_head_4d(query, self._num_attention_heads)
        key = _split_for_head_4d(key, self._num_attention_heads)
        value = _split_for_head_4d(value, self._num_attention_heads)

        block_mask = _build_block_mask_for_causal_sliding_window_attention(
            window_size=self._window_size,
            batch_size=query.shape[0],
            hidden_size=query.shape[-1],
            seq_len=query.shape[-2],
            device=query.device
        )
        result = flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask
        )

        result = _concat_from_head_4d(result, self._num_attention_heads)

        result = self._out_proj(result)
        return result


class MultiLatentAttention(nn.Module):
    def __init__(self, hidden_size: int, config: MultiLatentAttentionConfig):
        super().__init__()

        assert hidden_size % config.num_attention_heads == 0
        assert config.rope_hidden_size % config.num_attention_heads == 0

        self._query_proj_low_rank = nn.Linear(hidden_size, config.query_low_rank)
        self._query_proj_up_rank_rope = nn.Linear(config.query_low_rank, config.rope_hidden_size)
        self._query_proj_up_rank_content = nn.Linear(config.query_low_rank, hidden_size)

        self._key_value_proj_low_rank = nn.Linear(hidden_size, config.key_value_low_rank)
        self._key_proj_up_rank_rope = nn.Linear(hidden_size, config.rope_hidden_size // config.num_attention_heads)
        self._key_proj_up_rank_content = nn.Linear(config.key_value_low_rank, hidden_size)
        self._value_proj_up_rank = nn.Linear(config.key_value_low_rank, hidden_size)
        self._out_proj = nn.Linear(hidden_size, hidden_size)

        self._rope_query = RotaryPositionalEmbeddings(config.rope_hidden_size, config.rope)
        self._rope_key = RotaryPositionalEmbeddings(config.rope_hidden_size // config.num_attention_heads, config.rope)

        self._num_heads = config.num_attention_heads
        self._normalize_qk = config.enable_qk_norm

    def get_dtype(self) -> torch.dtype:
        return self._query_proj.weight.dtype

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # q or k or v: [batch size; some seq len; hidden size]
        c_query = self._query_proj_low_rank(hidden_state)
        query_content = self._query_proj_up_rank_content(c_query)
        query_content = query_content.reshape((query_content.shape[0], query_content.shape[1], self._num_heads, -1))

        query_rope = self._query_proj_up_rank_rope(c_query)
        query_rope = self._rope_query(query_rope)
        query_rope = query_rope.reshape((query_rope.shape[0], query_rope.shape[1], self._num_heads, -1))

        query = torch.cat((query_content, query_rope), dim=-1)
        query = query.reshape((query.shape[0], query.shape[1], -1))

        # this one tensor is getting KV-cached \/
        c_keyvalue = self._key_value_proj_low_rank(hidden_state)
        # /\

        key_content = self._key_proj_up_rank_content(c_keyvalue)
        key_content = key_content.reshape((key_content.shape[0], key_content.shape[1], self._num_heads, -1))

        key_rope = self._rope_key(self._key_proj_up_rank_rope(hidden_state))
        key_rope = key_rope[:, :, None, :].repeat(1, 1, self._num_heads, 1)

        key = torch.cat((key_content, key_rope), dim=-1)
        key = key.reshape((key.shape[0], key.shape[1], -1))

        value = self._value_proj_up_rank(c_keyvalue)

        query = _split_for_head(query, self._num_heads)
        key = _split_for_head(key, self._num_heads)
        value = _split_for_head(value, self._num_heads)
        attention_mask = _repeat_batch_for_heads(attention_mask, self._num_heads)

        if self._normalize_qk:
            # originally it was LayerNorm in https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf
            query = F.rms_norm(query, normalized_shape=(query.shape[-1],))
            key = F.rms_norm(key, normalized_shape=(key.shape[-1],))

        result = sdpa(query, key, value, attention_mask)
        result = _concat_from_head(result, self._num_heads)
        result = self._out_proj(result)
        return result


class MultiHeadAttentionWithKV(nn.Module):
    def __init__(self, hidden_size: int, config: MultiHeadAttentionConfig):
        super().__init__()

        assert hidden_size % config.num_attention_heads == 0

        projection_size = hidden_size // config.num_attention_heads

        self._query_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._key_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._value_proj = nn.Linear(hidden_size, projection_size * config.num_attention_heads)
        self._out_proj = nn.Linear(projection_size * config.num_attention_heads, hidden_size)

        self._rope = RotaryPositionalEmbeddings(hidden_size, config.rope)

        self._num_attention_heads = config.num_attention_heads

    def get_dtype(self) -> torch.dtype:
        return self._query_proj.weight.dtype

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor,
            past_key: torch.Tensor,
            past_value: torch.Tensor,
            position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # q or k or v: [batch size; some seq len; hidden size]
        query = self._query_proj(hidden_state)
        key = self._key_proj(hidden_state)
        value = self._value_proj(hidden_state)

        query = self._rope(query, position_ids)
        key = self._rope(key, position_ids)

        query = _split_for_head(query, self._num_attention_heads)
        key = _split_for_head_4d(key, self._num_attention_heads)
        value = _split_for_head_4d(value, self._num_attention_heads)

        if past_key is not None and past_value is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
            attention_mask = torch.cat(
                (torch.zeros(attention_mask.shape[0], attention_mask.shape[1], past_key.shape[2]), attention_mask),
                dim=2
            )

        new_past_key = key
        new_past_value = value

        key = key.reshape(key.shape[0] * key.shape[1], key.shape[2], key.shape[3])
        value = value.reshape(value.shape[0] * value.shape[1], value.shape[2], value.shape[3])

        attention_mask = _repeat_batch_for_heads(attention_mask, self._num_attention_heads)

        result = sdpa(query, key, value, attention_mask)

        result = _concat_from_head(result, self._num_attention_heads)

        result = self._out_proj(result)
        return result, new_past_key, new_past_value


def main():
    with torch.inference_mode():
        torch.random.manual_seed(42)
        cfg = MultiHeadAttentionConfig(
            num_attention_heads=8,
            rope=ROPEConfig(base=10_000.0, max_positions=1024)
        )
        att = MultiHeadAttentionWithKV(hidden_size=512, config=cfg)

        input_hidden_state = torch.randn((1, 1, 512))

        single_token_mask = construct_mask_for_decoder(torch.ones((1, 1), dtype=torch.long), target_dtype=torch.float32)

        past_k, past_v = None, None

        all_hidden_states_pre = []
        all_hidden_states_post = []

        new_hidden_state = input_hidden_state
        for i in range(200):
            all_hidden_states_pre.append(new_hidden_state)
            new_hidden_state, past_k, past_v = att(
                new_hidden_state,
                single_token_mask,
                past_key=past_k,
                past_value=past_v,
                position_ids=torch.tensor([[i]], dtype=torch.long)
            )
            all_hidden_states_post.append(new_hidden_state)
            # do something with updated HS here
        all_hidden_states_pre = torch.cat(all_hidden_states_pre, dim=1)
        all_hidden_states_post = torch.cat(all_hidden_states_post, dim=1)

        new_hidden_state_if_it_was_rerun_from_scratch = att(
            all_hidden_states_pre,
            construct_mask_for_decoder(torch.ones((1, 200), dtype=torch.long), target_dtype=torch.float32),
            past_key=None,
            past_value=None,
            position_ids=None
        )[0]

        assert torch.allclose(new_hidden_state_if_it_was_rerun_from_scratch, all_hidden_states_post, atol=1e-10, rtol=1e-2)




if __name__ == '__main__':
    main()
