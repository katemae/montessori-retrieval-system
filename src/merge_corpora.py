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
    "pink tower", "metal inset", "stamp game", "moveable alphabet",
    "spindle box", "binomial cube", "trinomial cube", "knobbed cylinder",
    "golden bead", "sandpaper letter", "red rod", "long rod",
    "brown stair", "color tablet", "sound cylinder", "rough and smooth board",
    "smooth board", "rough board", "wooden cylinder", "geometry cabinet",
    "glass bead", "fraction inset", "inset", "botany cabinet",
    "divergent and convergent lines"
]

def contains_material(text):
    """
    checks if a passage contains a Montessori material.
    """
    text = text.lower()
    return any(mat in text for mat in MONTESSORI_MATERIALS)


def detect_citation(text):
    """
    detects citations to categorize studies.

    eg:
    (Author, 1999)
    (Author & Author, 1972)
    (Author et al., 1996)
    (Author, 1948/1976)
    (Author, 1969, p. 2)
    (Author, 1980; Other & Author, 1992)
    """

    citation_pattern = r"""
    \(                                  # opening parenthesis
    [^()]*?                             # any text, notably author name or text/study name
    \b\d{4}(?:/\d{4})?\b                # year or year/year
    [^()]*?                             # any text to capture page numbers or other authors
    \)                                  # closing parenthesis
    """

    return re.search(citation_pattern, text, re.VERBOSE) is not None

def build_indexed_text(row):
    """
    combine metadata + evidence into a single text.
    """
    parts = []

    for label, col in [
        ("Comparison", "Comparison"),
        ("Category", "Category"),
        ("Concept", "Title"),
        ("Approach", "Approach"),
        ("Domain", "Domain"),
        ("Evidence Type", "Type of Evidence (Example, Material, Study)")
    ]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{label}: {row[col]}")

    parts.append(f"Excerpt: {row['Evidence']}")
    return "\n".join(parts)

def build_passage_indexed_text(row):
    """
    combine newly found metadata + evidence into a single text.
    """
    parts = []

    if pd.notna(row.get("approach")):
        parts.append(f"Approach: {row['approach']}")

    if pd.notna(row.get("domain")):
        parts.append(f"Domain: {row['domain']}")

    if pd.notna(row.get("evidence_type")):
        parts.append(f"Evidence Type: {row['evidence_type']}")

    if pd.notna(row.get("source_title")):
        parts.append(f"Source: {row['source_title']}")

    parts.append(f"Excerpt: {row['raw_text']}")

    return "\n".join(parts)


# ----------------------------------------------
# EXCERPTS
# ----------------------------------------------

def load_excerpts():
    df = pd.read_csv(EXCERPT_FILE)

    # separate the raw text/evidence for displaying
    df["raw_text"] = df["Evidence"]

    # text enriched w/ metadata
    df["text"] = df.apply(build_indexed_text, axis=1)

    df = df.rename(columns={
        "Approach": "approach",
        "Domain": "domain",
        "Type of Evidence (Example, Material, Study)": "evidence_type",
        "Chapter Name": "source_title",
    })

    df["source_type"] = "excerpt"
    df["source_file"] = "excerpt_dataset"
    df["paragraph_index"] = None
    df["doc_id"] = [f"excerpt_{i}" for i in range(len(df))]

    # only select columns
    return df[[
        "doc_id",
        "text",
        "raw_text",    # for display only
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

    # separate the raw para/evidence for displaying
    df["raw_text"] = df["text"]

    # enriched text, if applicable
    df["text"] = df.apply(build_passage_indexed_text, axis=1)


    return df[[
        "doc_id",
        "text",
        "raw_text",    # for display only
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