import string
from pathlib import Path

import click
import tokenizers
from tokenizers import models, pre_tokenizers, normalizers, decoders
from tokenizers.trainers import BpeTrainer

from misisnlp.anlp4.data import load_wikipedia

_ALPHABET_RU = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
_ALPHABET_EN = string.ascii_letters
_ALPHABET_DIGITS = string.digits
_ALPHABET_PUNCT = string.punctuation

_ALPHABET = list(_ALPHABET_RU + _ALPHABET_EN + _ALPHABET_DIGITS + _ALPHABET_PUNCT)


@click.command()
@click.option('--save-to', type=Path, default="data/anlp4/tokenizer.json")
def main(save_to: Path):
    dataset = load_wikipedia()["train"]

    tokenizer = tokenizers.Tokenizer(
        models.BPE(
            unk_token="[UNK]",
            continuing_subword_prefix='##',
            fuse_unk=True
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.normalizer = normalizers.BertNormalizer(
        lowercase=False,
        strip_accents=True,
        handle_chinese_chars=True,
        clean_text=True,
    )
    tokenizer.decoder = decoders.WordPiece(
        prefix='##',
        cleanup=True
    )


    trainer = BpeTrainer(
        vocab_size=30_000,
        show_progress=True,
        special_tokens=["[CLS]", "[SEP]", "[MASK]", "[UNK]"],
        limit_alphabet=300,
        initial_alphabet=_ALPHABET,
        continuing_subword_prefix='##',
        max_token_length=16
    )

    tokenizer.train_from_iterator(dataset['text'], trainer)
    tokenizer.save(str(save_to), pretty=True)




if __name__ == '__main__':
    main()
