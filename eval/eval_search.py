import sys
sys.path.append("src")

import csv
import pandas as pd
from filter_search import FilterMontessoriSearchEngine

QUERY_FILE = "eval/eval_queries.txt"
EVAL_FILE = "eval/eval_queries.csv"
TOP_K = 5


engine = FilterMontessoriSearchEngine()

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = [q.strip() for q in f if q.strip()]

with open(EVAL_FILE, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["query", "doc_id", "relevant"])

    for query in queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        results = engine.search(query, k=TOP_K)

        for i, r in enumerate(results, 1):
            print(f"\nResult {i}")
            print(f"Doc ID: {r['doc_id']}")
            print(f"Source: {r['source_title']}")
            print(f"Evidence: {r['evidence_type']}")
            print(f"Approach: {r['approach']}")
            print(f"Domain: {r['domain']}")
            print(f"Text: {r['raw_text'][:400]}...")

            label = input("Relevant? [y/n]: ").strip().lower()
            relevant = 1 if label == "y" else 0

            writer.writerow([query, r["doc_id"], relevant])

print(f"\nSaved evaluation file to {EVAL_FILE}")