import copy
import re

import datasets
import torch.utils.data
from datasets import load_dataset
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, AutoTokenizer

_RE_REPLICA = re.compile(r'''\n\n(Human|Assistant): (.*?)(?=\n\n(?:Human|Assistant): |$)''', re.DOTALL)


def parse_replicas(data_string: str) -> list[dict]:
    replicas = []
    for match in _RE_REPLICA.finditer(data_string):
        replicas.append({
            "role": "user" if match.group(1) == "Human" else "assistant",
            "content": match.group(2)
        })
    return replicas


def load_dataset_item(data: dict, tokenizer: PreTrainedTokenizerBase):
    chosen_replicas = parse_replicas(data['chosen'])
    rejected_replicas = parse_replicas(data['rejected'])

    # extract
    prefix = chosen_replicas[:-1]
    chosen_answer = chosen_replicas[-1]['content']
    rejected_answer = rejected_replicas[-1]['content']

    # apply templates
    prefix = tokenizer.apply_chat_template(
        prefix,
        add_generation_prompt=True,
        tokenize=False
    )
    chosen_answer = f'{chosen_answer}{tokenizer.eos_token}'
    rejected_answer = f'{rejected_answer}{tokenizer.eos_token}'


    # tokenize
    prefix = tokenizer(prefix, add_special_tokens=False)['input_ids']
    chosen_answer = tokenizer(chosen_answer, add_special_tokens=False)['input_ids']
    rejected_answer = tokenizer(rejected_answer, add_special_tokens=False)['input_ids']


    return {
        'prefix_tokens': prefix,
        'chosen_tokens': chosen_answer,
        'rejected_tokens': rejected_answer
    }


def load_anthropic_hh_rlhf(
        tokenizer: PreTrainedTokenizerBase,
        num_proc: int,
        shuffle_seed: int,
        use_samples: int
):
    data = load_dataset('Anthropic/hh-rlhf')['train']
    data = data.shuffle(seed=shuffle_seed).take(use_samples)
    data = data.map(load_dataset_item, fn_kwargs={'tokenizer': tokenizer}, num_proc=num_proc)
    return data


class TorchPreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        chosen_input_ids = torch.tensor(item['prefix_tokens'] + item['chosen_tokens'], dtype=torch.long)
        chosen_att_mask = torch.ones_like(chosen_input_ids)
        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:len(item['prefix_tokens'])] = -100  # ignore prefix tokens while computing proba

        rejected_input_ids = torch.tensor(item['prefix_tokens'] + item['rejected_tokens'], dtype=torch.long)
        rejected_att_mask = torch.ones_like(rejected_input_ids)
        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:len(item['rejected_tokens'])] = -100

        return {
            'chosen': {
                'input_ids': chosen_input_ids,
                'attention_mask': chosen_att_mask,
                'labels': chosen_labels,
            },
            'rejected': {
                'input_ids': rejected_input_ids,
                'attention_mask': rejected_att_mask,
                'labels': rejected_labels,
            }
        }


class TorchPreferenceCollator:
    def _stack_pad_tensors(self, items: list[torch.Tensor], pad_with: int) -> torch.Tensor:
        max_len = max(len(x) for x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode='constant', value=pad_with) for x in items]
        return torch.stack(items)

    def __call__(self, batches):
        return {
            'chosen': {
                'input_ids': self._stack_pad_tensors([x['chosen']['input_ids'] for x in batches], pad_with=0),
                'attention_mask': self._stack_pad_tensors([x['chosen']['attention_mask'] for x in batches], pad_with=0),
                'labels': self._stack_pad_tensors([x['chosen']['labels'] for x in batches], pad_with=-100)
            },
            'rejected': {
                'input_ids': self._stack_pad_tensors([x['rejected']['input_ids'] for x in batches], pad_with=0),
                'attention_mask': self._stack_pad_tensors([x['rejected']['attention_mask'] for x in batches], pad_with=0),
                'labels': self._stack_pad_tensors([x['rejected']['labels'] for x in batches], pad_with=-100)
            }
        }


if __name__ == '__main__':
    dat = load_anthropic_hh_rlhf(AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'), 16, 42, 1000)
    dat = TorchPreferenceDataset(dat)
    print(TorchPreferenceCollator()([dat[0], dat[1]]))
