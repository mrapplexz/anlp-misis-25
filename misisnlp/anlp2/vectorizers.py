import abc
from enum import StrEnum

import fasttext
import numpy as np
from catboost import CatBoostClassifier
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class VectorizerKind(StrEnum):
    bow = "bow"
    tfidf = "tfidf"
    fasttext = "fasttext"


class ClassicVectorizerConfig(BaseModel):
    vectorizer_kind: VectorizerKind
    ngram_low: int = 1
    ngram_high: int = 1
    max_features: int = 50000


class MyFastText:
    def __init__(self, model_path: str):
        self._model = fasttext.load_model(model_path)

    def fit(self, x):
        pass

    def transform(self, X: list[list[str]]):
        return np.asarray([self._model.get_sentence_vector(" ".join(x)) for x in tqdm(X)])

    def fit_transform(self, x):
        return self.transform(x)


def vectorizer_from_config(config: ClassicVectorizerConfig):
    match config.vectorizer_kind:
        case VectorizerKind.bow:
            return CountVectorizer(
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
                max_features=config.max_features,
                ngram_range=(config.ngram_low, config.ngram_high)
            )
        case VectorizerKind.tfidf:
            return TfidfVectorizer(
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
                max_features=config.max_features,
                ngram_range=(config.ngram_low, config.ngram_high)
            )
        case VectorizerKind.fasttext:
            return MyFastText("model/cc.ru.300.bin")
