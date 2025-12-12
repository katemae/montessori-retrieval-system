import pickle
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
from idx_tfidf import preprocess_text


VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MATRIX_FILE = "models/tfidf_matrix.pkl"
CORPUS_FILE = "models/corpus_processed.pkl"

MONTESSORI_MATERIALS = {
    "pink tower", "metal inset", "stamp game", "movable alphabet",
    "spindle box", "binomial cube", "trinomial cube",
    "knobbed cylinder", "golden bead", "sandpaper letter",
    "red rod", "long rod", "brown stair", "color tablet",
    "sound cylinder", "rough and smooth board", "geometry cabinet",
    "botany cabinet", "fraction inset"
}

DOMAIN_FILTER_MAP = {
    "Cognitive": {
        "keywords": [
            "cognitive", "attention", "memory", "executive",
            "problem solving", "thinking", "concentration"
        ],
        "implies": [
            "Cognitive",
            "Behavioral/Cognitive",
            "Academic/Cognitive",
            "Behavioral", # pretty hand in hand/similar
        ],
    },

    "Behavioral": {
        "keywords": [
            "behavior", "misbehavior", "discipline", "self-regulation",
            "motivation", "reward", "punishment", "control"
        ],
        "implies": [
            "Behavioral",
            "Behavioral/Cognitive",
            "Behavioral/Social",
            "Cognitive", # pretty hand in hand/similar
        ],
    },

    "Social": {
        "keywords": [
            "social", "peer", "collaboration", "interaction",
            "community", "group", "age"
        ],
        "implies": [
            "Social",
            "Behavioral/Social",
            "Cognitive/Social",
        ],
    },

    "Academic": {
        "keywords": [
            "academic", "reading", "math", "literacy",
            "numeracy", "achievement", "school performance"
        ],
        "implies": [
            "Academic",
            "Academic/Cognitive",
            "Academic/Behavioral",
        ],
    },

    "Environment": {
        "keywords": [
            "environment", "classroom", "noise", "space",
            "materials", "prepared environment"
        ],
        "implies": [
            "Environment",
            "Cognitive/Environment",
            "Academic/Environment",
        ],
    },
}

def infer_domains(query):
    q = query.lower()
    domains = set()

    for group in DOMAIN_FILTER_MAP.values():
        if any(kw in q for kw in group["keywords"]):
            domains.update(group["implies"])

    return list(domains) if domains else None


def infer_evidence_type(query):
    q = query.lower()

    if any(mat in q for mat in MONTESSORI_MATERIALS):
        return "Material"
    
    if any(w in q for w in ["study", "studies", "research", "citation"]):
        return "Study"

    return None


class FilterMontessoriSearchEngine:
    
    def __init__(self):
        # print("Loading TF-IDF index...")
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

        evidence = infer_evidence_type(query)
        domain = infer_domains(query)

        return approach, evidence, domain


    ## improved search with filters :)
    def search(self, query, k=5):
        approach_f, evidence_f, domain_f = self.infer_filters(query)
        processed = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        valid_indices = np.arange(len(scores)) # for filtering; default is ALL

        if any(x is not None for x in (approach_f, evidence_f, domain_f)):

            filters = {
                "approach": approach_f,
                "evidence_type": evidence_f,
                "domain": domain_f
            }
            active_filters = {k: v for k, v in filters.items() if v is not None}
            filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items())
            apply = input(
                f"\n\t** Recommended filters detected\n"
                f"\t{filter_str}\n"
                f"\tApply these filters? [y/N]: "
            )

            while apply.lower() not in ["y", "n"]:
                print("\n\tSorry, I didn't quite catch that.")
                apply = input(
                    f"\n\t** Recommended filters detected\n"
                    f"\t{filter_str}\n"
                    f"\tApply these filters? [y/N]: "
                )

            if apply.lower().strip() == "y":

                # APPLY FILTERS
                if approach_f:
                    mask = (self.corpus["approach"].str.lower() == approach_f.lower()).values
                    valid_indices = valid_indices[mask[valid_indices]]

                if evidence_f:
                    mask = (self.corpus["evidence_type"].str.lower() == evidence_f.lower()).values
                    valid_indices = valid_indices[mask[valid_indices]]

                if domain_f:
                    domain_f_lower = [d.lower() for d in domain_f]
                    mask = self.corpus["domain"].str.lower().isin(domain_f_lower).values
                    valid_indices = valid_indices[mask[valid_indices]]


                # if not enough docs, revert to whole corpus
                if len(valid_indices) < k:
                    print(f"\t** Filters invalid! reverting to approach filter only... ")
                    valid_indices = np.arange(len(scores))
                    # only filter on approach instead
                    if approach_f:
                        valid_indices = valid_indices[self.corpus["approach"].str.lower() == approach_f.lower()]
                else:
                    print("\t** Filters valid! proceeding...")

        # top-k among valid indices
        top_k = valid_indices[np.argsort(scores[valid_indices])[::-1][:k]]

        results = []
        for idx in top_k:
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
    print("-=+ MONTOSSEORI EVIDENCE RETRIEVAL SYSTEM +=-")
    engine = FilterMontessoriSearchEngine()
    
    while True:
        
        query = input("Enter your query. Type 'q' or 'quit' to exit:\n")
        if query == "q" or query == "quit":
            break
        if len(query) < 1:
            continue
        
        results = engine.search(query, k=5)
        print(f"\n=== QUERY: {query} ===")
        # print("\nSearching for relevant texts...")

        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {r['score']:.4f}")
            print(f"Source: {r['source_title']}")
            print(f"Evidence Type: {r['evidence_type']}")
            if r["approach"] is not None and str(r["approach"]).lower() != "nan":
                print(f"Approach: {r['approach']}")
            if r["domain"] is not None and str(r["domain"]).lower() != "nan":
                print(f"Domain: {r['domain']}")
            print(f"Text: {r['raw_text'][:min(500, len(r['text']))]}...")
        print(f"\n=== END SEARCH ON: {query} ===\n")

    print("\033c", end="")
    print("Thanks for stopping by! Goodbye 👋")
    print()
    exit()