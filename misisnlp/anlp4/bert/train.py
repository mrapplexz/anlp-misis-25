from pathlib import Path

import click
import datasets
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from misisnlp.anlp4.bert.config import BertPreTrainingConfig
from misisnlp.anlp4.bert.data import BertPreTrainingCollator, BertPreTrainingDataset, BertPreTrainingDataConfig
from misisnlp.anlp4.bert.trainable import BertTrainable
from misisnlp.anlp4.blocks.fulltransformer import BertForPreTraining
from misisnlp.anlp4.data import load_wikipedia
from misisnlp.trainer.trainer import Trainer

def _filter_short_texts(data: datasets.Dataset, tokenizer: Tokenizer, config: BertPreTrainingDataConfig) -> datasets.Dataset:
    return data.filter(
        lambda x, tkn: len(tkn.encode(x['text']).ids) >= config.min_segment_length * 2,
        fn_kwargs={'tkn': tokenizer},
        num_proc=config.filtering_num_proc
    )



@click.command()
@click.option('--config-path', type=Path, default='./config/anlp4/bert-pretrain.json')
def main(config_path: Path):
    config = BertPreTrainingConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    model = BertForPreTraining(config.architecture)
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    trainable = BertTrainable()
    collator = BertPreTrainingCollator()
    wikipedia = load_wikipedia()
    wikipedia['train'] = _filter_short_texts(wikipedia['train'], tokenizer, config.data)
    wikipedia['test'] = _filter_short_texts(wikipedia['test'], tokenizer, config.data)
    dataset_train = BertPreTrainingDataset(wikipedia['train'], tokenizer, config.data)
    dataset_test = BertPreTrainingDataset(wikipedia['test'], tokenizer, config.data)
    trainer = Trainer(config.trainer, model, trainable, collator)
    trainer.train(dataset_train, dataset_test)


if __name__ == "__main__":
    main()