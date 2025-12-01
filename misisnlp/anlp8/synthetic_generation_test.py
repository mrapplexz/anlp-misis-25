import datasets
from datasets import Dataset
from tqdm import tqdm

from misisnlp.anlp8.generator import SyntheticGenerator


def main():
    data = datasets.load_dataset('openai/gsm8k', 'main')['train']
    generator = SyntheticGenerator('http://localhost:8000/v1', '123')
    result_samples = []
    for item in tqdm(data.take(10)):
        result_samples.append(generator.generate(item["question"]))
    Dataset.from_dict({'dialogues': result_samples}).save_to_disk('my_math_problems')


if __name__ == '__main__':
    main()
