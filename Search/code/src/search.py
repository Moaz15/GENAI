import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

from tokenizer import Tokenizer
from inverted_index import InvertedIndex, Posting
from rankers import TFIDFRanker, BM25Ranker, CoordinateMatchRanker, Ranker


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SearchResult:
    doc_id: str
    score: float


class Searcher:

    def __init__(
        self,
        index: InvertedIndex,
        doc_len: Dict[str, int],
        n_docs: int,
        avgdl: float,
        tokenizer: Tokenizer,
    ):
        self.index = index
        self.doc_len = doc_len
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.tokenizer = tokenizer

    @classmethod
    def from_dir(
        cls,
        index_dir: str,
        prefix: str,
        tokenizer: Tokenizer,
    ) -> "Searcher":
        index_path = os.path.join(index_dir, f"{prefix}_inverted_index.json")
        doclen_path = os.path.join(index_dir, f"{prefix}_doc_len.json")
        stats_path = os.path.join(index_dir, f"{prefix}_collection_stats.json")

        idx = InvertedIndex.load(index_path)
        doc_len = load_json(doclen_path)
        stats = load_json(stats_path)

        n_docs = int(stats["N"])
        avgdl = float(stats["avgdl"])

        doc_len = {str(k): int(v) for k, v in doc_len.items()}

        return cls(
            index=idx,
            doc_len=doc_len,
            n_docs=n_docs,
            avgdl=avgdl,
            tokenizer=tokenizer,
        )

    def _make_ranker(self, name: str) -> Ranker:
        name = name.lower().strip()
        if name in ("tfidf", "tf-idf"):
            return TFIDFRanker(n_docs=self.n_docs)
        if name in ("bm25",):
            return BM25Ranker(n_docs=self.n_docs, avgdl=self.avgdl)
        if name in ("coord", "coordinate", "boolean"):
            return CoordinateMatchRanker()
        raise ValueError(f"Unknown ranker: {name}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        ranker_name: str = "bm25",
    ) -> List[SearchResult]:
        ranker = self._make_ranker(ranker_name)

        q_terms = self.tokenizer.tokenize(query)
        if not q_terms:
            return []

        scores: Dict[str, float] = {}

        for term in q_terms:
            plist: List[Posting] = self.index.postings.get(term, [])
            if not plist:
                continue

            df = int(self.index.df.get(term, 0))
            if df <= 0:
                continue

            for p in plist:
                dl = self.doc_len.get(p.doc_id, 0)
                scores[p.doc_id] = scores.get(p.doc_id, 0.0) + ranker.score(
                    tf=int(p.tf),
                    df=df,
                    doc_len=int(dl),
                )
                
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]


def load_stopwords(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Simple lexical search over inverted index.")
    parser.add_argument("--index_dir", required=True, help="Directory containing saved index files")
    parser.add_argument("--prefix", required=True, help="Prefix used when saving index files")
    parser.add_argument("--ranker", default="bm25", choices=["bm25", "tfidf", "coord"], help="Ranking function")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--stopwords", default=None)
    parser.add_argument("--stemming", action="store_true")
    parser.add_argument("--query", default=None, help="If provided, run once and exit")

    args = parser.parse_args()

    stop_words = load_stopwords(args.stopwords)
    tokenizer = Tokenizer(stop_words=stop_words, use_stemming=args.stemming)

    searcher = Searcher.from_dir(
        index_dir=args.index_dir,
        prefix=args.prefix,
        tokenizer=tokenizer,
    )

    if args.query is not None:
        results = searcher.search(args.query, top_k=args.top_k, ranker_name=args.ranker)
        for r in results:
            print(f"{r.doc_id}\t{r.score:.6f}")
        return

if __name__ == "__main__":
    main()