import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

from tokenizer import Tokenizer
from search import Searcher


def load_stopwords(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_scifact_queries(path: str) -> List[Tuple[str, str]]:
    """
    SciFact queries.jsonl looks like:
      {"_id": "0", "text": "...", "metadata": {...}}
    Returns: list of (qid, query_text)
    """
    out = []
    for obj in iter_jsonl(path):
        qid = str(obj["_id"])
        query = str(obj["text"])
        out.append((qid, query))
    return out


def load_scifact_qrels_tsv(path: str) -> Dict[str, Dict[str, float]]:
    """
    SciFact qrels TSV has header:
      query-id  corpus-id  score

    Returns:
      qrels[qid][doc_id] = rel
    """
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                # skip header
                first = False
                if "query-id" in line and "corpus-id" in line:
                    continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad qrels line: {line}")

            qid, corpus_id, score = parts
            qrels[str(qid)][str(corpus_id)] = float(score)

    return qrels


def dcg_at_k(rels: List[float], k: int) -> float:
    rels = rels[:k]
    s = 0.0
    for i, rel in enumerate(rels, start=1):
        s += (2**rel - 1.0) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked: List[str], qrel: Dict[str, float], k: int) -> float:
    rels = [qrel.get(d, 0.0) for d in ranked[:k]]
    dcg = dcg_at_k(rels, k)

    ideal = sorted(qrel.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(
    ranked: List[str],
    qrel: Dict[str, float],
    k: int,
    rel_threshold: float = 1.0,
) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0.0) >= rel_threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(
    ranked: List[str],
    qrel: Dict[str, float],
    k: int,
    rel_threshold: float = 1.0,
) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if qrel.get(d, 0.0) >= rel_threshold)
    return hits / len(top)


def recall_at_k(
    ranked: List[str],
    qrel: Dict[str, float],
    k: int,
    rel_threshold: float = 1.0,
) -> float:
    relevant = {d for d, r in qrel.items() if r >= rel_threshold}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument(
        "--queries", required=True, help="data/scifact/queries.jsonl"
    )
    ap.add_argument(
        "--qrels",
        required=True,
        help="data/scifact/qrels/test.tsv or train.tsv",
    )
    ap.add_argument(
        "--ranker", default="bm25", choices=["bm25", "tfidf", "coord"]
    )
    ap.add_argument("--top_k", type=int, default=100)

    ap.add_argument("--stopwords", default="data/stopwords.txt")
    ap.add_argument("--stemming", action="store_true")

    ap.add_argument("--ndcg_k", type=int, default=10)
    ap.add_argument("--mrr_k", type=int, default=10)
    ap.add_argument("--p_k", type=int, default=10)
    ap.add_argument("--r_k", type=int, default=100)
    ap.add_argument("--rel_threshold", type=float, default=1.0)

    ap.add_argument("--show_worst", type=int, default=10)
    args = ap.parse_args()

    tokenizer = Tokenizer(
        stop_words=load_stopwords(args.stopwords), use_stemming=args.stemming
    )

    searcher = Searcher.from_dir(
        index_dir=args.index_dir,
        prefix=args.prefix,
        tokenizer=tokenizer,
    )

    queries = load_scifact_queries(args.queries)
    qrels = load_scifact_qrels_tsv(args.qrels)

    rows = []
    nds, mrrs, ps, rs = [], [], [], []

    for qid, text in queries:
        qrel = qrels.get(qid)
        if not qrel:
            continue

        results = searcher.search(
            text, top_k=args.top_k, ranker_name=args.ranker
        )
        ranked = [r.doc_id for r in results]

        nd = ndcg_at_k(ranked, qrel, args.ndcg_k)
        mr = mrr_at_k(
            ranked, qrel, args.mrr_k, rel_threshold=args.rel_threshold
        )
        pk = precision_at_k(
            ranked, qrel, args.p_k, rel_threshold=args.rel_threshold
        )
        rk = recall_at_k(
            ranked, qrel, args.r_k, rel_threshold=args.rel_threshold
        )

        nds.append(nd)
        mrrs.append(mr)
        ps.append(pk)
        rs.append(rk)
        rows.append((qid, text, nd, mr, pk, rk))

    if not rows:
        print(
            "No queries evaluated. Check qid alignment between queries and qrels."
        )
        return

    def avg(xs):
        return sum(xs) / len(xs)

    print(f"Evaluated queries: {len(rows)}")
    print(f"Ranker: {args.ranker}")
    print(f"NDCG@{args.ndcg_k}: {avg(nds):.4f}")
    print(f"MRR@{args.mrr_k}:  {avg(mrrs):.4f}")
    print(f"P@{args.p_k}:     {avg(ps):.4f}")
    print(f"R@{args.r_k}:     {avg(rs):.4f}")

    rows.sort(key=lambda x: x[2])  # sort by ndcg ascending
    print("\nWorst queries by NDCG:")
    for qid, text, nd, mr, pk, rk in rows[: args.show_worst]:
        print(
            f"- qid={qid}  ndcg={nd:.4f}  mrr={mr:.4f}  p@{args.p_k}={pk:.4f}  r@{args.r_k}={rk:.4f} :: {text}"
        )


if __name__ == "__main__":
    main()
