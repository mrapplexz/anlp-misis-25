from pathlib import Path

import click
from transformers import AutoTokenizer

from misisnlp.anlp2.loader import SportDatasetContainer
from misisnlp.anlp3.config import TextClassificationPipelineConfig
from misisnlp.anlp3.data import TextClassificationCollator, TextClassificationDataset
from misisnlp.anlp3.model import TextClassificationModel, TaskType
from misisnlp.anlp3.trainable import TextClassificationTrainable
from misisnlp.trainer.trainer import Trainer


@click.command()
@click.option('--config-path', type=Path, default='./config/anlp3/classification/roberta.json')
def main(config_path: Path):
    config = TextClassificationPipelineConfig.model_validate_json(config_path.read_text(encoding='utf-8'))
    model = TextClassificationModel(config.text_model, TaskType.classify)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model.base_model_name)
    trainable = TextClassificationTrainable(config.text_model)
    collator = TextClassificationCollator()
    dataset = SportDatasetContainer(Path(config.data_path))
    dataset.load()
    dataset_train = TextClassificationDataset(tokenizer, dataset.train_texts, dataset.train_target)
    dataset_test = TextClassificationDataset(tokenizer, dataset.test_texts, dataset.test_target)
    trainer = Trainer(config.trainer, model, trainable, collator)
    trainer.train(dataset_train, dataset_test)


if __name__ == "__main__":
    main()