import re

import click
import torch.nn.functional
from torch import nn
from transformers import AutoModelForCausalLM

from misisnlp.anlp9.custom_lora import apply_lora_custom_inplace_, merge_lora_custom_inplace_


class TestLora(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Linear(128, 256)
        self.b = nn.Linear(256, 128)

    def forward(self, x):
        x = self.a(x)
        x = torch.nn.functional.gelu(x)
        x = self.b(x)
        return x


@click.command()
@torch.no_grad()
def main():
    model = TestLora()
    torch.manual_seed(42)
    test_tensor = torch.randn((15, 128))
    orig_out = model(test_tensor)
    apply_lora_custom_inplace_(
        master_module=model,
        target_pattern=re.compile('.*a|b.*'),
        rank=8,
        alpha=1,
        modules_full_train=[]
    )
    lora_out = model(test_tensor)
    assert torch.allclose(lora_out, orig_out)

    model.a.lora_B.weight += 0.1
    model.b.lora_B.weight += 0.1

    lora_out_trained = model(test_tensor)
    assert not torch.allclose(lora_out_trained, lora_out)

    merge_lora_custom_inplace_(model)

    lora_out_trained_merged = model(test_tensor)
    assert torch.allclose(lora_out_trained, lora_out_trained_merged, atol=1e-5)

    print(123)


if __name__ == '__main__':
    main()