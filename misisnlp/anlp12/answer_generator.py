import datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from misisnlp.anlp12.config import AnswerSamplingConfig
from misisnlp.anlp12.prompt_loading import load_anthropic_hh_rlhf_prompts


class AnswerGenerator:
    def __init__(
            self,
            model,
            device,
            config: AnswerSamplingConfig,
            pad_token_id: int
    ):
        self._model = model
        self._device = device
        self._num_answers = config.num_answers
        self._temperature = config.temperature
        self._top_p = config.top_p
        self._max_new_tokens = config.max_new_tokens
        self._pad_token_id = pad_token_id

    @torch.inference_mode
    def generate(self, prompts: list[list[int]]) -> list[list[torch.Tensor]]:
        self._model.eval()
        groups = []
        for prompt in tqdm(prompts):
            group = []
            prompt = torch.tensor(prompt, dtype=torch.long, device=self._device)[None, :].repeat(self._num_answers, 1)
            answers = self._model.generate(
                prompt,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=True,
                max_new_tokens=self._max_new_tokens
            )
            for answer in answers:
                group.append(answer[answer != self._pad_token_id].cpu())
            groups.append(group)
        return groups
