import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Iterable, Any, Optional, Tuple

from tokenizer import Tokenizer


@dataclass
class Posting:
    doc_id: str
    tf: int


class InvertedIndex:
    """
    Inverted index:
      postings[term] -> list of Posting(doc_id, tf)
      df[term]       -> document frequency - How many unique documents contain a term.
      cf[term]       -> collection frequency - The total number of times a term appears across the entire corpus.
    """
    def __init__(self):
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.df: Dict[str, int] = {}
        self.cf: Dict[str, int] = {}

    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        if not tokens:
            return
        tf_counter = Counter(tokens)
        for term, tf in tf_counter.items():
            self.postings[term].append(Posting(doc_id=doc_id, tf=tf))

    def finalize(self, sort_postings: bool = True) -> None:
        df: Dict[str, int] = {}
        cf: Dict[str, int] = {}

        for term, plist in self.postings.items():
            if sort_postings:
                # For reproducibility; for large-scale ,store numeric doc_ids and sort ints
                plist.sort(key=lambda p: p.doc_id)

            df[term] = len(plist)
            cf[term] = sum(p.tf for p in plist)

        self.df = df
        self.cf = cf

    def to_jsonable(self) -> Dict[str, Any]:
        # JSON-friendly representation
        return {
            "postings": {
                term: [[p.doc_id, p.tf] for p in plist]
                for term, plist in self.postings.items()
            },
            "df": self.df,
            "cf": self.cf,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_jsonable(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        idx = cls()
        postings_obj = obj.get("postings", {})
        for term, plist in postings_obj.items():
            idx.postings[term] = [Posting(doc_id=d, tf=tf) for d, tf in plist]

        idx.df = obj.get("df", {})
        idx.cf = obj.get("cf", {})
        return idx


def iter_jsonl(path: str) -> Iterable[dict]:
    """Stream JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_stopwords(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def build_inverted_index(
    corpus_path: str,
    tokenizer: Tokenizer,
    include_title: bool = True,
    include_text: bool = True,
    progress_every: int = 2000
) -> Tuple[InvertedIndex, Dict[str, Any]]:
    """
    Builds an inverted index from JSONL corpus.

    Returns:
      - idx: InvertedIndex
      - meta: basic metadata/stats (useful for sanity checking)
    """
    idx = InvertedIndex()

    n_docs = 0
    total_tokens = 0

    for doc in iter_jsonl(corpus_path):
        doc_id = str(doc["_id"]) if "_id" in doc else str(n_docs)

        title = doc.get("title", "") if include_title else ""
        text = doc.get("text", "") if include_text else ""

        content = (title + " " + text).strip()
        tokens = tokenizer.tokenize(content)

        idx.add_document(doc_id, tokens)

        n_docs += 1
        total_tokens += len(tokens)

        if progress_every and n_docs % progress_every == 0:
            print(f"Indexed {n_docs} docs...")

    idx.finalize()

    vocab_size = len(idx.postings)
    avg_tokens_per_doc = (total_tokens / n_docs) if n_docs else 0.0

    meta = {
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "vocab_size": vocab_size,
    }
    return idx, meta


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="Build an inverted index from a JSONL corpus.")
    parser.add_argument(
        "--corpus",
        default=os.path.join(project_root, "data", "scifact", "corpus.jsonl"),
        help="Path to corpus.jsonl"
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(project_root, "data", "index"),
        help="Output directory"
    )
    parser.add_argument("--prefix", default="scifact", help="Prefix for saved files")
    parser.add_argument(
        "--stopwords",
        default=os.path.join(project_root, "data", "stopwords.txt"),
        help="Path to stopwords.txt"
    )
    parser.add_argument("--stemming", action="store_true", help="Enable Porter stemming")
    parser.add_argument("--include_title", action="store_true", default=True, help="Include document title in index")
    parser.add_argument("--no_title", action="store_false", dest="include_title", help="Exclude document title from index")
    parser.add_argument("--include_text", action="store_true", default=True, help="Include document text in index")
    parser.add_argument("--no_text", action="store_false", dest="include_text", help="Exclude document text from index")
    parser.add_argument("--progress_every", type=int, default=2000, help="Print progress every N documents")
    args = parser.parse_args()

    stop_words = load_stopwords(args.stopwords)
    tokenizer = Tokenizer(stop_words=stop_words, use_stemming=args.stemming)

    os.makedirs(args.out_dir, exist_ok=True)

    idx, meta = build_inverted_index(
        corpus_path=args.corpus,
        tokenizer=tokenizer,
        include_title=args.include_title,
        include_text=args.include_text,
        progress_every=args.progress_every
    )

    index_path = os.path.join(args.out_dir, f"{args.prefix}_inverted_index.json")
    meta_path = os.path.join(args.out_dir, f"{args.prefix}_lexicon_stats.json")

    idx.save(index_path)
    save_json(meta, meta_path)

    # Sanity prints
    print(f"Docs: {meta['n_docs']}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Avg tokens/doc: {meta['avg_tokens_per_doc']:.2f}")
    print(f"Saved: {index_path}")
    print(f"Saved: {meta_path}")

    # Optional: show some very frequent terms (by CF)
    top_terms = sorted(idx.cf.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top terms by CF (most frequent):")
    for t, cf in top_terms:
        print(f"  {t:20s} cf={cf} df={idx.df.get(t, 0)}")


if __name__ == "__main__":
    main()
