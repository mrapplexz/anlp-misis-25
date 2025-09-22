import warnings
from pathlib import Path

import datasets
import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

from misisnlp.anlp2.loader import SportDatasetContainer


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, texts: list[str], targets: list[int]):
        self._texts = texts
        self._targets = targets
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx: int):
        item = self._texts[idx]
        target = self._targets[idx]
        encoding = self._tokenizer(item, max_length=512).encodings[0]
        return {
            'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
            'attention_mask': torch.tensor(encoding.attention_mask, dtype=torch.long),
            'target': torch.scalar_tensor(target, dtype=torch.long)
        }


class SpanExtractionDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, dataset: datasets.Dataset):
        self._tokenizer = tokenizer
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int):
        item = self._dataset[idx]
        encoding = self._tokenizer(item['context'], item['question'], max_length=512).encodings[0]
        if len(item['answers']['answer_start']) > 0:
            answer_start_char = item['answers']['answer_start'][0]
            answer_end_char = answer_start_char + len(item['answers']['text'][0]) - 1
            answer_start_token = encoding.char_to_token(answer_start_char)
            answer_end_token = encoding.char_to_token(answer_end_char)
        else:
            answer_start_token = 0
            answer_end_token = 0

        if answer_start_token is None or answer_end_token is None:
            answer_start_token = 0
            answer_end_token = 0
            warnings.warn(f'None start/end token ID: {encoding}')

        return {
            'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
            'attention_mask': torch.tensor(encoding.attention_mask, dtype=torch.long),
            'start_token_index': torch.scalar_tensor(answer_start_token, dtype=torch.long),
            'end_token_index': torch.scalar_tensor(answer_end_token, dtype=torch.long),
        }


class TextClassificationCollator:
    def _stack_pad_tensors(self, items: list[Tensor]) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=0) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items]),
            'attention_mask': self._stack_pad_tensors([x['attention_mask'] for x in items]),
            'target': torch.stack([x['target'] for x in items])
        }


class SpanExtractionCollator:
    def _stack_pad_tensors(self, items: list[Tensor]) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=0) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items]),
            'attention_mask': self._stack_pad_tensors([x['attention_mask'] for x in items]),
            'start_token_index': torch.stack([x['start_token_index'] for x in items]),
            'end_token_index': torch.stack([x['end_token_index'] for x in items]),
        }


if __name__ == '__main__':
    data = datasets.load_dataset('rajpurkar/squad_v2')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
    data = SpanExtractionDataset(tokenizer, data['train'])
    item = data[0]
    print(123)