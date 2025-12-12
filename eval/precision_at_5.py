import pandas as pd

EVAL_FILE = "eval/eval_queries.csv"

df = pd.read_csv(EVAL_FILE)

def precision_at_k(df, k=5):
    results = []

    for query in df["query"].unique():
        q_df = df[df["query"] == query]

        retrieved_k = q_df.head(k)
        relevant_retrieved = retrieved_k["relevant"].sum()
        total_relevant = q_df["relevant"].sum()

        precision = relevant_retrieved / k

        results.append({
            "query": query,
            "precision@k": precision,
            "relevant_retrieved": relevant_retrieved,
            "total_relevant": total_relevant
        })

    return pd.DataFrame(results)


metrics = precision_at_k(df, k=5)

print("\n=== Per-Query Results ===")
print(metrics.to_string(index=False))

print("\n=== Macro Averages ===")
print(f"Mean Precision@5: {metrics['precision@k'].mean():.3f}")