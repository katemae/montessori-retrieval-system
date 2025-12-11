import pandas as pd
import re
import os

EXCERPT_FILE = "metadata/all_excerpts.csv"
PASSAGE_FILE = "data/corpus.csv"
OUTPUT_FILE = "data/full_corpus.csv"


# ----------------------------------------------
# HELPERS
# ----------------------------------------------

MONTESSORI_MATERIALS = [
    "pink tower", "metal insets", "stamp game", "moveable alphabet",
    "spindle box", "binomial cube", "trinomial cube", "knobbed cylinders",
    "golden beads", "sandpaper letters", "red rods", "long rods",
    "brown stair", "color tablets", "montessori material", "montessori materials"
]

def contains_material(text):
    """
    Check if a passage contains a Montessori material name.
    """
    text = text.lower()
    return any(mat in text for mat in MONTESSORI_MATERIALS)


def detect_citation(text):
    """
    Detect study-like evidence using citation patterns.
    """
    citation_patterns = [
        r"\([A-Za-z]+\,\s*\d{4}\)",   # (Name, 2001)
        r"\(\d{4}\)",                 # (2001)
        r"\[[0-9]+\]",                # [12]
        r"\(see.*?\)",                # (see Chapter 2)
        r"\(p{1,2}\.\s*\d+\)"         # (p. 45) or (pp. 21)
    ]
    return any(re.search(p, text) for p in citation_patterns)


# ----------------------------------------------
# EXCERPTS
# ----------------------------------------------

def load_excerpts():
    df = pd.read_csv(EXCERPT_FILE)

    df = df.rename(columns={
        "Evidence": "text",
        "Approach": "approach",
        "Domain": "domain",
        "Type of Evidence (Example, Material, Study)": "evidence_type",
        "Chapter Name": "source_title",
    })

    df["source_type"] = "excerpt"
    df["source_file"] = "excerpt_dataset"
    df["paragraph_index"] = None
    df["doc_id"] = ["excerpt_" + str(i) for i in range(len(df))]

    # only select columns
    return df[[
        "doc_id",
        "text",
        "approach",
        "domain",
        "evidence_type",
        "source_type",
        "source_title",
        "source_file",
        "paragraph_index"
    ]]


# ----------------------------------------------
# PASSAGES
# ----------------------------------------------

def load_passages():
    df = pd.read_csv(PASSAGE_FILE)

    approaches = []
    evidence_types = []
    domains = []
    source_titles = []

    for i, row in df.iterrows():
        filename = row["source_file"]
        text = row["text"]
        source_title = row["source_title"]
        source_titles.append(source_title)

        # === APPROACH ===
        if filename.startswith("cleaned_ch"):
            if "traditional" in text.lower():
                approaches.append("Traditional")
            else:
                approaches.append("Montessori")
        else:
            approaches.append(None)

        # === EVIDENCE ===
        if filename.startswith("research-paper"):
            evidence_types.append("Study")
        
        elif contains_material(text):
            evidence_types.append("Material")
        elif detect_citation(text):
            evidence_types.append("Study")
        else:
            evidence_types.append("Example")

        # === DOMAIN ===
        if filename.startswith("research-paper"):
            domains.append("Cognitive")
        elif evidence_types[-1] == "Study": # if curr is study
            domains.append("Behavioral/Cognitive")
        else:
            domains.append(None)

    df["approach"] = approaches
    df["evidence_type"] = evidence_types
    df["domain"] = domains
    df["source_title"] = source_titles

    return df[[
        "doc_id",
        "text",
        "approach",
        "domain",
        "evidence_type",
        "source_type",
        "source_title",
        "source_file",
        "paragraph_index"
    ]]


# ----------------------------------------------
# MERGE ALL CORPORA
# ----------------------------------------------

def build_full_corpus():
    excerpts = load_excerpts()
    passages = load_passages()

    corpus = pd.concat([excerpts, passages], ignore_index=True)

    os.makedirs("data", exist_ok=True)
    corpus.to_csv(OUTPUT_FILE, index=False)
    total = len(corpus)

    print(f"Saved merged corpus to {OUTPUT_FILE}")
    print(f"Total docs: {total}")
    

    print("\n=========================================")
    print("Breakdown by source_type (%):")
    print(corpus["source_type"].value_counts() / total * 100)

    print("\n=========================================")
    print("Breakdown by evidence_type (%):")
    print(corpus["evidence_type"].value_counts(dropna=False) / total * 100)

    print("\n=========================================")
    print("Breakdown by domain (%):")
    print(corpus["domain"].value_counts(dropna=False) / total * 100)


if __name__ == "__main__":
    build_full_corpus()