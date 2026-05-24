import math
from abc import ABC, abstractmethod

class Ranker(ABC):
    """
    Abstract Base Class for all ranking functions.
    Ensures that any new ranking method implemented will follow the same structure.
    """
    @abstractmethod
    def score(self, tf: int, df: int, doc_len: int) -> float:
        pass


class TFIDFRanker(Ranker):
    def __init__(self, n_docs: int):
        self.n_docs = n_docs

    def score(self, tf: int, df: int, doc_len: int = 0) -> float:
        if tf <= 0:
            return 0.0
        # stable, positive IDF
        idf = math.log((self.n_docs + 1) / (df + 1)) + 1.0
        return tf * idf


class BM25Ranker(Ranker):
    def __init__(self, n_docs: int, avgdl: float, k1: float = 1.2, b: float = 0.75):
        self.n_docs = n_docs
        self.avgdl = max(avgdl, 1e-9)
        self.k1 = k1
        self.b = b

    def score(self, tf: int, df: int, doc_len: int) -> float:
        if tf <= 0:
            return 0.0

        idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
        norm = (1.0 - self.b) + self.b * (doc_len / self.avgdl)

        return idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)


class CoordinateMatchRanker(Ranker):
    def score(self, tf: int, df: int, doc_len: int) -> float:
        return 1.0 if tf > 0 else 0.0
    
    