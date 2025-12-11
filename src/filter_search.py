import pickle
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
from idx_tfidf import preprocess_text


VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MATRIX_FILE = "models/tfidf_matrix.pkl"
CORPUS_FILE = "models/corpus_processed.pkl"


class FilterMontessoriSearchEngine:
    def __init__(self):
        print("Loading TF-IDF index...")
        self.vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))
        self.tfidf_matrix = pickle.load(open(MATRIX_FILE, "rb"))
        self.corpus = pickle.load(open(CORPUS_FILE, "rb"))

        print(f"Loaded {len(self.corpus)} documents.")
        print("Search engine ready.\n")

    def infer_filters(self, query):
        ### FILTERABLES ARE HARD CODED ... COULD BE IMPROVED UPON ###
        q = query.lower()

        # APPROACH
        if "montessori" in q and "traditional" not in q:
            approach = "Montessori"
        elif "traditional" in q and "montessori" not in q:
            approach = "Traditional"
        else:
            approach = None

        # EVIDENCE TYPE
        if any(w in q for w in ["study", "studies", "research", "citation"]):
            evidence = "Study"
        elif any(w in q for w in ["material", "pink tower", "metal insets", "binomial cube"]):
            evidence = "Material"
        else:
            evidence = None

        # DOMAIN
        domain = []
        if any(w in q for w in ["cognitive", "attention", "memory", "executive"]):
            domain.append("Cognitive")
        if any(w in q for w in ["behavior", "self-regulation", "motivation"]):
            domain.append("Behavioral/Cognitive")
            domain.append("Behavioral")

        if len(domain) == 0:
            domain = None

        return approach, evidence, domain


    ## improved search with filters :)
    def search(self, query, k=5):
        approach_f, evidence_f, domain_f = self.infer_filters(query)

        print(f"\t** attempting filters: approach={approach_f}, evidence_type={evidence_f}, domain={domain_f} **")

        processed = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        valid_indices = np.arange(len(scores))

        # APPLY FILTERS
        if approach_f:
            valid_indices = valid_indices[self.corpus["approach"].str.lower() == approach_f.lower()]
        
        if evidence_f:
            valid_indices = valid_indices[self.corpus["evidence_type"].str.lower() == evidence_f.lower()]

        if domain_f:
            domain_f_lower = [d.lower() for d in domain_f]
            mask = self.corpus["domain"].str.lower().isin(domain_f_lower)
            valid_indices = valid_indices[mask.values]

        # if not enough docs, revert to whole corpus
        if len(valid_indices) < k:
            print(f"\t** filters invalid! reverting to approach={approach_f}**")
            valid_indices = np.arange(len(scores))
            # only filter on approach instead
            if approach_f:
                valid_indices = valid_indices[self.corpus["approach"].str.lower() == approach_f.lower()]
        else:
            print("\t** filters valid! proceeding... **")

        # top-k among valid indices
        top_k = valid_indices[np.argsort(scores[valid_indices])[::-1][:k]]

        results = []
        for idx in top_k:
            row = self.corpus.iloc[idx]
            results.append({
                "score": float(scores[idx]),
                "doc_id": row["doc_id"],
                "text": row["text"],
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

    engine = FilterMontessoriSearchEngine()
    results = engine.search(query, k=5)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {r['score']:.4f}")
        print(f"Source: {r['source_title']} ({r['source_type']})")
        print(f"Evidence Type: {r['evidence_type']}")
        print(f"Approach: {r['approach']}")
        print(f"Domain: {r['domain']}")
        print(f"Text: {r['text'][:min(500, len(r['text']))]}...")
    print("\n=== END SEARCH ===\n")