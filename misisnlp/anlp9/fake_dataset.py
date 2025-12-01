import random

import torch
from torch import Tensor
from transformers import Qwen2Tokenizer, AutoTokenizer


import torch.nn.functional as F


def create_fake_dataset_message() -> list[dict]:
    random.seed(42)
    outputs = []
    for _ in range(10000):
        a = random.randint(0, 500_000_000)
        b = random.randint(0, 500_000_000)
        result = a + b
        outputs.append([
            {
                "role": "user",
                "content": f'{a} + {b} = ?'
            },
            {
                "role": "assistant",
                "content": f"Result of your expression is {a} + {b} = {result}. Have a nice day!"
            }
        ])
    return outputs


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Qwen2Tokenizer):
        self._tokenizer = tokenizer
        self._data = create_fake_dataset_message()  # possible memory leak

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        tokens_no_assistant = self._tokenizer.apply_chat_template(
            conversation=self._data[item][:-1],
            add_generation_prompt=True,
        )

        tokens_with_assistant = self._tokenizer.apply_chat_template(
            conversation=self._data[item],
            add_generation_prompt=False
        )

        compute_loss_mask = torch.tensor(([0] * len(tokens_no_assistant) +
                                          [1] * (len(tokens_with_assistant) - len(tokens_no_assistant))),
                                         dtype=torch.long)

        input_ids = torch.tensor(tokens_with_assistant, dtype=torch.long)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[compute_loss_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class FakeCollator:
    def _stack_pad_tensors(self, items: list[Tensor], pad_with: int) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=pad_with) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items], pad_with=0),
            'attention_mask': self._stack_pad_tensors([x['attention_mask'] for x in items], pad_with=0),
            'labels': self._stack_pad_tensors([x['labels'] for x in items], pad_with=-100),
        }


if __name__ == '__main__':
    ds = FakeDataset(AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507'))
    ret = FakeCollator()([ds[0], ds[1]])
    print(ret)