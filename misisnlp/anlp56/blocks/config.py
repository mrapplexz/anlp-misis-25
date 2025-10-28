from pydantic import BaseModel


class ROPEConfig(BaseModel):
    max_positions: int
    base: float


class SlidingWindowAttentionConfig(BaseModel):
    num_attention_heads: int
    window_size: int

    rope: ROPEConfig



class GroupedQueryAttentionConfig(BaseModel):
    num_attention_heads: int
    num_key_value_heads: int

    enable_qk_norm: bool
    rope: ROPEConfig


class MultiHeadAttentionConfig(BaseModel):
    num_attention_heads: int

    rope: ROPEConfig


class MultiLatentAttentionConfig(BaseModel):
    num_attention_heads: int

    query_low_rank: int
    key_value_low_rank: int

    enable_qk_norm: bool
    rope_hidden_size: int
    rope: ROPEConfig
