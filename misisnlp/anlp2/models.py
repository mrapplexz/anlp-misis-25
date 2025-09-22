import abc
from enum import StrEnum

import numpy as np
from catboost import CatBoostClassifier
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression


class ModelKind(StrEnum):
    catboost = "catboost"
    logreg = "logreg"


class ClassicModelConfig(BaseModel):
    model_kind: ModelKind


class BaseClassicModel(abc.ABC):
    @abc.abstractmethod
    def train(self, train_vecs: np.ndarray, train_target: np.ndarray):
        ...

    @abc.abstractmethod
    def eval(self, test_vecs: np.ndarray) -> np.ndarray:
        ...


class CatBoostModel(BaseClassicModel):
    def __init__(self):
        self._model = None

    def train(self, train_vecs: np.ndarray, train_target: np.ndarray):
        model = CatBoostClassifier(loss_function='MultiClass', random_seed=42, task_type='GPU', eval_metric='Accuracy')
        model.fit(X=train_vecs, y=train_target, verbose=True)
        self._model = model

    def eval(self, test_vecs: np.ndarray) -> np.ndarray:
        return self._model.predict(test_vecs)


class LogRegModel(BaseClassicModel):
    def __init__(self):
        self._model = None

    def train(self, train_vecs: np.ndarray, train_target: np.ndarray):
        model = LogisticRegression(verbose=1)
        model.fit(X=train_vecs, y=train_target)
        self._model = model

    def eval(self, test_vecs: np.ndarray) -> np.ndarray:
        return self._model.predict(test_vecs)


def model_from_config(config: ClassicModelConfig):
    match config.model_kind:
        case ModelKind.catboost:
            return CatBoostModel()
        case ModelKind.logreg:
            return LogRegModel()
