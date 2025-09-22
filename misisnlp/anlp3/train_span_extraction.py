from pathlib import Path

import click
import datasets
from transformers import AutoTokenizer

from misisnlp.anlp2.loader import SportDatasetContainer
from misisnlp.anlp3.config import TextClassificationPipelineConfig
from misisnlp.anlp3.data import TextClassificationCollator, TextClassificationDataset, SpanExtractionCollator, \
    SpanExtractionDataset
from misisnlp.anlp3.model import TextClassificationModel, TaskType
from misisnlp.anlp3.trainable import TextClassificationTrainable, SpanExtractionTrainable
from misisnlp.trainer.trainer import Trainer


@click.command()
@click.option('--config-path', type=Path, default='./config/anlp3/span_extraction/bert.json')
def main(config_path: Path):
    config = TextClassificationPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    model = TextClassificationModel(config.text_model, TaskType.span_extraction)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model.base_model_name)
    trainable = SpanExtractionTrainable(config.text_model)
    collator = SpanExtractionCollator()
    dataset = datasets.load_dataset(config.data_path)
    dataset_train = SpanExtractionDataset(tokenizer, dataset['train'])
    dataset_test = SpanExtractionDataset(tokenizer, dataset['validation'])
    trainer = Trainer(config.trainer, model, trainable, collator)
    trainer.train(dataset_train, dataset_test)


if __name__ == "__main__":
    main()