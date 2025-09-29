from pydantic import BaseModel


class TransformerConfig(BaseModel):
    vocab_size: int
    segment_vocab_size: int
    hidden_size: int = 128
    feedforward_hidden_size: int = 128 * 4
    max_positions: int = 512

    attention_heads: int = 4
    num_layers: int = 2

    dropout_proba: float = 0.1
