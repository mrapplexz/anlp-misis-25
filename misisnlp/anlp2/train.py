from pathlib import Path

import click
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from misisnlp.anlp2.loader import SportDatasetContainer
from misisnlp.anlp2.models import model_from_config, ClassicModelConfig
from misisnlp.anlp2.processor import ClassicDataProcessor, ClassicDataProcessingConfig
from misisnlp.anlp2.vectorizers import vectorizer_from_config, ClassicVectorizerConfig


@click.command()
@click.option('--data-path', type=Path, default='/home/me/projects/tochka/misis-nlp-25/data/train.csv')
@click.option('--processing-config-path', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/config/anlp2/data_processing/preproc_stem.json')
@click.option('--model-config-path', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/config/anlp2/model/logreg.json')
@click.option('--vectorizer-config-path', type=Path,
              default='/home/me/projects/tochka/misis-nlp-25/config/anlp2/vectorizer/fasttext.json')
def main(data_path: Path, processing_config_path: Path, model_config_path: Path, vectorizer_config_path: Path):
    data = SportDatasetContainer(data_path)
    data.load()
    preproc_config = ClassicDataProcessingConfig.model_validate_json(processing_config_path.read_text(encoding='utf-8'))
    model_config = ClassicModelConfig.model_validate_json(model_config_path.read_text(encoding='utf-8'))
    vectorizer_config = ClassicVectorizerConfig.model_validate_json(vectorizer_config_path.read_text(encoding='utf-8'))
    print(preproc_config)
    processor = ClassicDataProcessor(
        preproc_config
    )
    train_data = [processor.process(x) for x in tqdm(data.train_texts)]
    test_data = [processor.process(x) for x in tqdm(data.test_texts)]

    vectorizer = vectorizer_from_config(vectorizer_config)
    train_vecs = vectorizer.fit_transform(train_data)
    print('Train vecs shape', train_vecs.shape)
    test_vecs = vectorizer.transform(test_data)

    model = model_from_config(model_config)
    model.train(train_vecs, data.train_target)
    model_pred = model.eval(test_vecs)
    print('Accuracy: ', accuracy_score(data.test_target, model_pred))

    # bow + catboost (1000 iters)
    # 0.7454827 - no preprocessing
    # 0.7918172 - preproc and stemming
    # 0.7934951 - preproc and lemmatization

    # bow + logreg (100iters)
    # 0.8168559628291172 - no preprocessing
    # 0.8470573051109964 - preproc and stemming
    # 0.8501548786783686 - preproc and lemmatization

    # tfidf + logreg (100iters)
    # 0.859189468249871 - preproc and stemming

    # tfidf (bigrams) + logreg (100iters)
    # 0.8630614352090862 - preproc and stemming

    # tfidf (trigrams) + logreg (100iters)
    # 0.8625451729478575 - preproc and stemming

    # fasttext (from scratch)
    # 0.7271553949406299 - no preproc
    # 0.8335054207537429 - preproc and stemming

    # fasttext (pretrained Russian) + logreg (100iters)
    # 0.6937274135260713 - no preproc
    # 0.6601703665462054 - preproc and stemming


if __name__ == '__main__':
    main()
