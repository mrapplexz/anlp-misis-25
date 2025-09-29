import statistics
from pathlib import Path

import click
import tokenizers

from misisnlp.anlp4.data import load_wikipedia


@click.command()
@click.option('--load-from', type=Path, default="data/anlp4/tokenizer.json")
def main(load_from: Path):
    dataset = load_wikipedia()["test"].select(range(10))

    tokenizer = tokenizers.Tokenizer.from_file(str(load_from))

    compressions = []

    for item in dataset:
        text = item["text"]
        encoding = tokenizer.encode(text)

        text_retok = tokenizer.decode(encoding.ids)
        print(f'TEXT: {text}\nTOKENS (char): {" ".join(encoding.tokens)}\nTEXT (retokenized): {text_retok}')
        compressions.append(len(text) / len(encoding.ids))

    print(f"Chars per Token: {statistics.mean(compressions)}")




if __name__ == '__main__':
    main()
