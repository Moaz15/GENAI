import argparse
import json
import os
from typing import Dict, Optional, Tuple, List

from tokenizer import Tokenizer


class Indexer:
    def __init__(self, corpus_path: str, tokenizer: Tokenizer):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer

    def iter_jsonl(self):
        """Stream JSONL file line by line."""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    #doesn’t load the entire corpus into memory
                    yield json.loads(line)

    def process_document(self, doc: dict) -> Tuple[str, int]:
        """Process a single document and return (doc_id, doc_length)."""
        doc_id = doc["_id"]
        title = doc.get("title", "")
        text = doc.get("text", "")

        content = (title + " " + text).strip()
        tokens = self.tokenizer.tokenize(content)

        return doc_id, len(tokens)

    def build_doc_stats(self) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Compute document lengths and collection stats."""
        doc_len = {}
        total_len = 0
        N = 0

        for doc in self.iter_jsonl():
            doc_id, dl = self.process_document(doc)

            doc_len[doc_id] = dl
            total_len += dl
            N += 1

        avgdl = (total_len / N) if N else 0.0
        stats = {"N": float(N), "avgdl": float(avgdl)}

        return doc_len, stats


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_stopwords(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out_dir", default="index")
    parser.add_argument("--prefix", default="scifact")
    parser.add_argument("--stopwords", default=None)
    parser.add_argument("--stemming", action="store_true")

    args = parser.parse_args()

    stop_words = load_stopwords(args.stopwords)
    tokenizer = Tokenizer(stop_words=stop_words, use_stemming=args.stemming)

    indexer = Indexer(args.corpus, tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)

    doc_len, stats = indexer.build_doc_stats()

    doc_len_path = os.path.join(args.out_dir, f"{args.prefix}_doc_len.json")
    stats_path = os.path.join(args.out_dir, f"{args.prefix}_collection_stats.json")

    save_json(doc_len, doc_len_path)
    save_json(stats, stats_path)

    # Debug / sanity checks
    print(f"N={int(stats['N'])}, avgdl={stats['avgdl']:.2f}")

    items = sorted(doc_len.items(), key=lambda x: x[1])
    print("Shortest docs:", items[:5])
    print("Longest docs:", items[-5:])

    print(f"Saved: {doc_len_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()