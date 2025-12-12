import pickle
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
from idx_tfidf import preprocess_text


VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MATRIX_FILE = "models/tfidf_matrix.pkl"
CORPUS_FILE = "models/corpus_processed.pkl"


class BasicMontessoriSearchEngine:
    """
    BasicMontessoriSearchEngine:
    Class to perform a basic query returning top-k ranked documents
    based on fitted TF-IDF.

    No filtering for certain vocabulary (as per the metadata) is applied.
    """
    def __init__(self):
        print("Loading TF-IDF index...")
        self.vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))
        self.tfidf_matrix = pickle.load(open(MATRIX_FILE, "rb"))
        self.corpus = pickle.load(open(CORPUS_FILE, "rb"))

        print(f"Loaded {len(self.corpus)} documents.")
        print("Search engine ready.\n")

    # TO DO FOR ADVANCED MODEL: filters for approach="Montessori", domain="Cognitive", evidence_type="Study"
    def search(self, query, k=5):
        processed = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed])

        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        valid_indices = np.arange(len(scores))

        # ranking
        filtered_scores = scores[valid_indices]
        top_k_idx = np.argsort(filtered_scores)[::-1][:k]
        actual_indices = valid_indices[top_k_idx]

        results = []
        for idx in actual_indices:
            row = self.corpus.iloc[idx]
            results.append({
                "score": float(scores[idx]),
                "doc_id": row["doc_id"],
                "text": row["text"],
                "raw_text": row["raw_text"],
                "approach": row["approach"],
                "domain": row["domain"],
                "evidence_type": row["evidence_type"],
                "source_title": row["source_title"],
                "source_type": row["source_type"],
                "paragraph_index": row["paragraph_index"]
            })

        return results
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Query needed! Run:\n  python basic_search.py 'your query here.'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n=== QUERY: {query} ===")
    print("\nSearching for relevant texts...")

    engine = BasicMontessoriSearchEngine()
    results = engine.search(query, k=5)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {r['score']:.4f}")
        print(f"Source: {r['source_title']}")
        print(f"Evidence Type: {r['evidence_type']}")
        if r["approach"] is not None and str(r["approach"]).lower() != "nan":
            print(f"Approach: {r['approach']}")
        if r["domain"] is not None and str(r["domain"]).lower() != "nan":
            print(f"Domain: {r['domain']}")
        print(f"Text: {r['raw_text'][:min(500, len(r['raw_text']))]}...")
    print("\n=== END SEARCH ===\n")