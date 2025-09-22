from pathlib import Path

import click
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from misisnlp.anlp2.loader import SportDatasetContainer
from misisnlp.anlp2.processor import ClassicDataProcessingConfig, ClassicDataProcessor


@click.command()
@click.option("--data-path", type=Path, default='/home/me/projects/tochka/misis-nlp-25/data/train.csv')
@click.option('--pred-path', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/data/fasttext-train/pred.txt')
def main(data_path: Path, pred_path: Path):
    data = SportDatasetContainer(data_path)
    data.load()
    eval_tgt = data.test_target
    eval_pred = [int(x[len('__label__'):]) for x in pred_path.read_text(encoding='utf-8').split()]
    print(accuracy_score(eval_tgt, eval_pred))


if __name__ == '__main__':
    main()
