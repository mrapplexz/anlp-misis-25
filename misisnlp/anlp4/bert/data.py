import random

import datasets
import tokenizers
import torch
from pydantic import BaseModel
from torch import Tensor

import torch.nn.functional as F

from misisnlp.anlp4.data import load_wikipedia


class BertPreTrainingDataConfig(BaseModel):
    max_text_length: int
    min_segment_length: int
    filtering_num_proc: int


class BertPreTrainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: datasets.Dataset,
            tokenizer: tokenizers.Tokenizer,
            config: BertPreTrainingDataConfig
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._tokenizer.enable_truncation(max_length=config.max_text_length)
        self._config = config

        self._vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

        self._cls_token = self._tokenizer.token_to_id('[CLS]')
        self._sep_token = self._tokenizer.token_to_id('[SEP]')
        self._mask_token = self._tokenizer.token_to_id('[MASK]')

    def __len__(self):
        return len(self._dataset)

    def _sample_other_text(self, this_text: int) -> int:
        while True:
            sampled = random.randint(0, len(self._dataset) - 1)
            if sampled != this_text:
                return sampled

    def _sample_span_pairs(self, item: int):
        text_current = self._dataset[item]["text"]
        text_current = self._tokenizer.encode(text_current)

        if random.random() <= 0.5:  # use the same text
            left_segment_length = random.randint(
                self._config.min_segment_length,
                len(text_current.ids) - self._config.min_segment_length
            )
            right_segment_length = len(text_current.ids) - left_segment_length
            left_segment = text_current.ids[:left_segment_length]
            right_segment = text_current.ids[left_segment_length:]
            nsp_target = 1
        else:  # use different texts
            text_other = self._dataset[self._sample_other_text(item)]["text"]
            text_other = self._tokenizer.encode(text_other)
            left_segment_length = random.randint(
                self._config.min_segment_length,
                len(text_current.ids) - self._config.min_segment_length
            )
            # todo sample not from only start of the text
            right_segment_length = min(random.randint(
                self._config.min_segment_length,
                len(text_other.ids) - self._config.min_segment_length
            ), self._config.max_text_length - left_segment_length)
            left_segment = text_current.ids[:left_segment_length]
            right_segment = text_other.ids[:right_segment_length]
            nsp_target = 0

        input_ids = ([self._cls_token] +
                     left_segment +
                     [self._sep_token] +
                     right_segment +
                     [self._sep_token])  # 515 tokens at most

        special_token_mask = ([1] + [0] * left_segment_length + [1] + [0] * right_segment_length + [1])

        segment_ids = ([0] * (left_segment_length + 2) +  # 2 is [CLS] and leftmost [SEP]
                       [1] * (right_segment_length + 1))  # 1 is rightmost [SEP]
        attention_mask = [1] * len(input_ids)

        return {
            'nsp_target': nsp_target,
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask,
            'special_token_mask': special_token_mask,
        }

    def _mask_tokens(
            self, tokens: torch.Tensor, special_token_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        tokens_should_train = (torch.rand(tokens.shape) <= 0.15) & (special_token_mask != 1)
        tokens_train_type = torch.rand(tokens.shape)
        tokens_should_mask = tokens_should_train & (tokens_train_type <= 0.8)
        tokens_should_random = tokens_should_train & (tokens_train_type > 0.8) & (tokens_train_type <= 0.9)
        tokens_should_retain = tokens_should_train & (tokens_train_type > 0.9)

        inputs = tokens.clone()
        inputs[tokens_should_mask] = self._mask_token
        inputs[tokens_should_random] = torch.randint(low=0, high=self._vocab_size,
                                                     size=(tokens_should_random.sum().item(),))

        labels = tokens.clone()
        labels[~tokens_should_train] = -100

        return {
            'input_ids': inputs,
            'mlm_labels': labels
        }

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        encoding = self._sample_span_pairs(item)
        encoding = {
            'nsp_target': torch.scalar_tensor(encoding['nsp_target'], dtype=torch.long),
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'segment_ids': torch.tensor(encoding['segment_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'special_token_mask': torch.tensor(encoding['special_token_mask'], dtype=torch.long),
        }
        encoding.update(
            self._mask_tokens(encoding['input_ids'], encoding['special_token_mask'])
        )
        return encoding



class BertPreTrainingCollator:
    def _stack_pad_tensors(self, items: list[Tensor], pad_with: int) -> Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=pad_with) for x in items]
        return torch.stack(items)

    def __call__(self, items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items], pad_with=0),
            'attention_mask': self._stack_pad_tensors([x['attention_mask'] for x in items], pad_with=0),
            'segment_ids': self._stack_pad_tensors([x['segment_ids'] for x in items], pad_with=0),
            'mlm_labels': self._stack_pad_tensors([x['mlm_labels'] for x in items], pad_with=-100),
            'nsp_target': torch.stack([x['nsp_target'] for x in items], dim=0)
        }



if __name__ == '__main__':
    dataset = BertPreTrainingDataset(
        load_wikipedia()["train"],
        tokenizer=tokenizers.Tokenizer.from_file('/home/me/projects/tochka/misis-nlp-25/data/anlp4/tokenizer.json'),
        config=BertPreTrainingDataConfig()
    )
    batch = BertPreTrainingCollator()([dataset[0], dataset[12312]])
    print(batch)