from pathlib import Path

import click
from tqdm import tqdm

from misisnlp.anlp2.loader import SportDatasetContainer
from misisnlp.anlp2.processor import ClassicDataProcessingConfig, ClassicDataProcessor


def _data_to_fasttext(tokens: list[str], target: int) -> str:
    return f'__label__{target} {" ".join(tokens)}\n'


@click.command()
@click.option("--data-path", type=Path, default='/home/me/projects/tochka/misis-nlp-25/data/train.csv')
@click.option('--processing-config-path', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/config/anlp2/data_processing/preproc_stem.json')
@click.option('--save-dir', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/data/fasttext-train')
def main(data_path: Path, processing_config_path: Path, save_dir: Path):
    data = SportDatasetContainer(data_path)
    data.load()
    preproc_config = ClassicDataProcessingConfig.model_validate_json(processing_config_path.read_text(encoding='utf-8'))
    print(preproc_config)
    processor = ClassicDataProcessor(
        preproc_config
    )
    train_data = [processor.process(x) for x in tqdm(data.train_texts)]
    test_data = [processor.process(x) for x in tqdm(data.test_texts)]
    train_rows = [_data_to_fasttext(x, y) for x, y in zip(train_data, data.train_target)]
    test_rows = [_data_to_fasttext(x, y) for x, y in zip(test_data, data.test_target)]
    save_dir.mkdir(exist_ok=True, parents=True)
    with (save_dir / 'train.txt').open('w', encoding='utf-8') as f:
        f.writelines(train_rows)
    with (save_dir / 'test.txt').open('w', encoding='utf-8') as f:
        f.writelines(test_rows)


if __name__ == '__main__':
    main()
