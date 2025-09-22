import re
from enum import StrEnum
from unittest import case

import nltk
from pydantic import BaseModel
from pymorphy3 import MorphAnalyzer


class TokenizeMethod(StrEnum):
    nltk = 'nltk'
    split = 'split'


class NormalizationMethod(StrEnum):
    stem = 'stem'
    lemmatize = 'lemmatize'


class ClassicDataProcessingConfig(BaseModel):
    lowercase: bool
    tokenize_method: TokenizeMethod
    remove_non_alphanum: bool
    remove_stopwords: bool
    normalize_method: NormalizationMethod | None



_RE_CHECK_TOKEN = re.compile(r'^\w+$')


class ClassicDataProcessor:
    def __init__(self, config: ClassicDataProcessingConfig):
        self._config = config
        self._stopwords = set(nltk.corpus.stopwords.words('russian'))
        self._stemmer = nltk.stem.SnowballStemmer('russian')
        self._morph = MorphAnalyzer()

    def _tokenize(self, text: str) -> list[str]:
        match self._config.tokenize_method:
            case TokenizeMethod.split:
                return text.split()
            case TokenizeMethod.nltk:
                return nltk.word_tokenize(text, language="russian")

    def _normalize(self, word: str) -> str:
        match self._config.normalize_method:
            case NormalizationMethod.stem:
                return self._stemmer.stem(word)
            case NormalizationMethod.lemmatize:
                return self._morph.parse(word)[0].normal_form
            case _:
                raise NotImplementedError()

    def process(self, text: str) -> list[str]:
        if self._config.lowercase:
            text = text.lower()

        text = self._tokenize(text)

        if self._config.remove_non_alphanum:
            text = [x for x in text if _RE_CHECK_TOKEN.fullmatch(x)]

        if self._config.remove_stopwords:
            text = [x for x in text if x not in self._stopwords]

        if self._config.normalize_method:
            text = [self._normalize(x) for x in text]

        return text