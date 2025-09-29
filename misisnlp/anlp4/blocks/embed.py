import torch
from bitsandbytes.nn import StableEmbedding
from torch import nn

from misisnlp.anlp4.blocks.config import TransformerConfig


class BertEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.token_embedding = StableEmbedding(config.vocab_size, config.hidden_size)

        self._segment_embedding = StableEmbedding(config.segment_vocab_size, config.hidden_size)

        self._position_embedding = StableEmbedding(config.max_positions, config.hidden_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        return (self.token_embedding(input_ids) +
                self._position_embedding(position_ids)[None, :, :] +
                self._segment_embedding(segment_ids))
