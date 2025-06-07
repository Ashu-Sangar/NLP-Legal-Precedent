# bert_eval.py  (or any file name you like)
import json
import time          # ← optional, but you imported it
import numpy as np
from tqdm import tqdm

def evaluate_collection(
    retrieval,
    k_values=(5, 10, 20, 50),
    max_queries=None,          # None → evaluate every case
    min_gold=1,                # skip queries with <min_gold relevant docs
    verbose=True,
    outfile="colbert_metrics.json"
):
    # --- running sums -------------------------------------------------------
    agg = {f"precision@{k}": 0.0 for k in k_values}
    agg.update({f"recall@{k}": 0.0 for k in k_values})
    agg["mrr@50"] = agg["map"] = 0.0
    counted = 0

    # --- iterate over cases -------------------------------------------------
    dataset_slice = retrieval.dataset.cases[:max_queries] if max_queries else retrieval.dataset.cases
    pbar = tqdm(dataset_slice, disable=not verbose, desc="Evaluating collection")

    for qid, _ in pbar:
        metrics = retrieval.evaluate(qid, k_values=k_values)
        if not metrics or len(metrics) < 1:
            continue                    # skip: no gold labels or evaluation failed
        if len(metrics.get("gold", [])) < min_gold:   # OPTIONAL: depends on your evaluate()
            continue

        # accumulate
        for k, v in metrics.items():
            if k in agg:          # precision@5, recall@20, etc.
                agg[k] += v
            elif k == "mrr":
                agg["mrr@50"] += v
            elif k in ("map", "average_precisions"):
                agg["map"] += v
        counted += 1

    if counted == 0:
        raise RuntimeError("No queries with gold citations were evaluated!")

    # --- average ------------------------------------------------------------
    averaged = {k: v / counted for k, v in agg.items()}
    result = {"metrics": averaged, "total_cases": counted}

    # --- write JSON ---------------------------------------------------------
    if outfile:
        with open(outfile, "w") as fh:
            json.dump(result, fh, indent=2)
        if verbose:
            print(f"📄  Saved aggregate metrics → {outfile}")

    return result

# ---------------------------------------------------------------------------
# CLI wrapper so you can run:  python bert_eval.py --data-dir ... --device cuda
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from colbert_retrieval import load_retrieval   # convenience loader you added

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--max-queries", type=int, help="limit for quick smoke-tests")
    ap.add_argument("--outfile", default="colbert_metrics.json")
    args = ap.parse_args()

    retr = load_retrieval(
        data_dir=args.data_dir,
        device=args.device
    )
    summary = evaluate_collection(
        retr,
        max_queries=args.max_queries,
        outfile=args.outfile
    )
    print(json.dumps(summary, indent=2))