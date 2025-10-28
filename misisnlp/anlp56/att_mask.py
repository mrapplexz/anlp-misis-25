import torch


def construct_mask_for_decoder(attention_mask: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(
        (attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[1]),
        device=attention_mask.device,
        dtype=target_dtype) * torch.finfo(target_dtype).min
    mask = torch.triu(mask, diagonal=1)
    return mask
