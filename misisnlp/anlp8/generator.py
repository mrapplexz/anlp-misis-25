import importlib.resources

from openai import OpenAI
from pydantic import BaseModel


class GenerateProblemEnvironmentDto(BaseModel):
    actors: list[str]
    subjects: list[str]


class GenerateProblemConditionsDto(BaseModel):
    conditions: list[str]


class GenerateProblemSolutionDto(BaseModel):
    plan: str
    steps: list[str]
    result: str


class SyntheticGenerator:
    def __init__(self, llm_api_url: str, llm_api_key: str):
        self._client = OpenAI(
            base_url=llm_api_url,
            api_key=llm_api_key,
        )

    def generate(self, question: str) -> str:
        environment = self._client.chat.completions.parse(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": (importlib.resources.files('misisnlp.anlp8') / 'prompt_generate_characters.txt').read_text()
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            response_format=GenerateProblemEnvironmentDto
        ).choices[0].message.parsed

        conditions = self._client.chat.completions.parse(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": (importlib.resources.files(
                        'misisnlp.anlp8') / 'prompt_generate_conditions.txt').read_text()
                },
                {
                    "role": "user",
                    "content": environment.model_dump_json(indent=4)
                }
            ],
            response_format=GenerateProblemConditionsDto
        ).choices[0].message.parsed

        new_question = self._client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": (importlib.resources.files(
                        'misisnlp.anlp8') / 'prompt_generate_task.txt').read_text()
                },
                {
                    "role": "user",
                    "content": conditions.model_dump_json(indent=4)
                }
            ]
        ).choices[0].message.content

        new_question_answer = self._client.chat.completions.parse(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": (importlib.resources.files(
                        'misisnlp.anlp8') / 'prompt_generate_solution.txt').read_text()
                },
                {
                    "role": "user",
                    "content": new_question
                }
            ],
            response_format=GenerateProblemSolutionDto
        ).choices[0].message.parsed



        steps_joined = '\n'.join(new_question_answer.steps)
        return [
            {
                "role": "system",
                "content": "Solve a math problem given to you, generate a step-by-step solution."
            },
            {
                "role": "user",
                "content": new_question
            },
            {
                "role": "assistant",
                "content": f"{steps_joined}\nResult: {new_question_answer.result}"
            }
        ]
