import re

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

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
        tokenize=False,
        enable_thinking=False
    )
    chosen_answer = f'{chosen_answer}{tokenizer.eos_token}'
    rejected_answer = f'{rejected_answer}{tokenizer.eos_token}'

    # tokenize
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)['input_ids']
    chosen_tokens = tokenizer(chosen_answer, add_special_tokens=False)['input_ids']
    rejected_tokens = tokenizer(rejected_answer, add_special_tokens=False)['input_ids']

    return {
        'prefix_tokens': prefix_tokens,
        'chosen_tokens': chosen_tokens,
        'rejected_tokens': rejected_tokens,
    }


def load_anthropic_hh_rlhf_prompts(
        tokenizer: PreTrainedTokenizerBase,
        num_proc: int,
        shuffle_seed: int,
        use_samples: int,
        load_prompts_only: bool
):
    data = load_dataset('Anthropic/hh-rlhf')['train']
    data = data.shuffle(seed=shuffle_seed).take(use_samples)
    data = data.map(load_dataset_item, fn_kwargs={'tokenizer': tokenizer}, num_proc=num_proc,
                    remove_columns=['chosen', 'rejected'])
    if load_prompts_only:
        data = data.select_columns(['prefix_tokens'])
    return data
