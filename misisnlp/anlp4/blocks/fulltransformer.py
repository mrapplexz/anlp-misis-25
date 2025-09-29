import torch
from torch import nn

from misisnlp.anlp4.blocks.attention import MultiHeadAttention
from misisnlp.anlp4.blocks.config import TransformerConfig
from misisnlp.anlp4.blocks.embed import BertEmbeddings
from misisnlp.anlp4.blocks.ffn import FeedForward


class SubLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._layer_norm = nn.LayerNorm(config.hidden_size)
        self._dropout = nn.Dropout(config.dropout_proba)

    def forward(self, x: torch.Tensor, result: torch.Tensor):
        return self._layer_norm(x + self._dropout(result))


class SelfAttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._attention = MultiHeadAttention(config)
        self._sub = SubLayer(config)

    def get_dtype(self) -> torch.dtype:
        return self._attention.get_dtype()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._sub(hidden_states, self._attention(hidden_states, hidden_states, hidden_states, attention_mask))


class FeedForwardBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._ffn = FeedForward(config)
        self._sub = SubLayer(config)

    def forward(self, hidden_states: torch.Tensor):
        return self._sub(hidden_states, self._ffn(hidden_states))


class EncoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._self_attention = SelfAttentionBlock(config)
        self._feed_forward = FeedForwardBlock(config)

    def get_dtype(self) -> torch.dtype:
        return self._self_attention.get_dtype()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self._self_attention(hidden_states, attention_mask)
        hidden_states = self._feed_forward(hidden_states)
        return hidden_states


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._self_attention = SelfAttentionBlock(config)
        self._feed_forward = FeedForwardBlock(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask_for_self: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._self_attention(hidden_states, attention_mask_for_self)
        hidden_states = self._feed_forward(hidden_states)
        return hidden_states


def construct_mask_for_encoder(attention_mask: torch.Tensor, query_size: int | None,
                               target_dtype: torch.dtype) -> torch.Tensor:
    # masks PAD tokens only
    # attention_mask: [batch size; seq len]
    # [1 1 1 1 0 0]
    # one indicates that it is a regular token, zero indicates that it is a PAD token

    if query_size is None:
        query_size = attention_mask.shape[1]

    mask = torch.zeros(
        (attention_mask.shape[0], query_size, attention_mask.shape[1]),
        device=attention_mask.device,
        dtype=target_dtype
    )
    attention_mask_selector = attention_mask.unsqueeze(1).repeat(1, query_size, 1) == 0
    mask[attention_mask_selector] = torch.finfo(target_dtype).min
    return mask


class Bert(nn.Module):  # bert-like
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embeddings = BertEmbeddings(config)
        self._encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            segment_ids: torch.Tensor
    ):
        attention_mask = construct_mask_for_encoder(attention_mask, query_size=None,
                                                    target_dtype=self.embeddings.token_embedding.weight.dtype)
        encoder_state = self.embeddings(input_ids, segment_ids=segment_ids)
        for encoder_layer in self._encoder_layers:
            encoder_state = encoder_layer(encoder_state, attention_mask)
        return encoder_state


class BertForPreTraining(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._bert = Bert(config)

        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._nsp_head = nn.Linear(config.hidden_size, 1)
        self._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                segment_ids: torch.Tensor):
        hidden_states = self._bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        lm_logits = self._lm_head(hidden_states)
        nsp_logit = self._nsp_head(hidden_states[:, 0])

        return lm_logits, nsp_logit

    def _tie_weights(self):
        self._lm_head.weight = self._bert.embeddings.token_embedding.weight


class DecoderOnlyTransformer(nn.Module):  # gpt-like
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._embeddings = BertEmbeddings(config)
        self._decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # self._lm_head.weight = self._embeddings._token_embedding.weight  # tie weights (optionally)

    def forward(
            self,
            input_ids_decoder: torch.Tensor,
            attention_mask_decoder_self: torch.Tensor
    ):
        decoder_state = self._embeddings(input_ids_decoder)
        for decoder_layer in self._decoder_layers:
            decoder_state = decoder_layer(
                hidden_states=decoder_state,
                hidden_states_encoder=None,
                attention_mask_for_self=attention_mask_decoder_self,
                attention_mask_for_enc_decoder=None
            )

        return self._lm_head(decoder_state)
