from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SportDatasetContainer:
    def __init__(self, path_to_data: Path, test_size: float = 0.2, seed: int = 42):
        self._path_to_data = path_to_data
        self._test_size = test_size
        self._seed = seed

        self._train = []
        self._train_tgt = []
        self._test = []
        self._test_tgt = []
        self._num_classes = None

    def load(self):
        encoder_tgt = LabelEncoder()
        df = pd.read_csv(self._path_to_data, delimiter=',')
        df["category"] = encoder_tgt.fit_transform(df["category"])
        self._num_classes = df["category"].nunique()
        df_train, df_test = train_test_split(df, test_size=self._test_size, random_state=self._seed,
                                             stratify=df["category"])
        self._train = df_train['text'].tolist()
        self._test = df_test['text'].tolist()
        self._train_tgt = df_train['category'].tolist()
        self._test_tgt = df_test['category'].tolist()

    @property
    def train_texts(self) -> list[str]:
        return self._train

    @property
    def train_target(self) -> list[int]:
        return self._train_tgt

    @property
    def test_texts(self) -> list[str]:
        return self._test

    @property
    def test_target(self) -> list[int]:
        return self._test_tgt

    @property
    def num_classes(self) -> int:
        return self._num_classes
