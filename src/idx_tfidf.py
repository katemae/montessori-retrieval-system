import pandas as pd
import pickle
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer


CORPUS_FILE = "data/full_corpus.csv"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MATRIX_FILE = "models/tfidf_matrix.pkl"
PROCESSED_CORPUS_FILE = "models/corpus_processed.pkl"


# ----------------------------------------------
# preproc: lowercase, remove punctuation
# ----------------------------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------
# build TF-IDF index
# ----------------------------------------------
def build_tfidf_index():
    df = pd.read_csv(CORPUS_FILE)
    df["processed_text"] = df["text"].apply(preprocess_text)

    # processed text for search.py convenience
    os.makedirs("models", exist_ok=True)
    pickle.dump(df, open(PROCESSED_CORPUS_FILE, "wb"))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams and bigrams
        max_df=0.9,
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

    print("Saving vectorizer + TF-IDF matrix...")
    pickle.dump(vectorizer, open(VECTORIZER_FILE, "wb"))
    pickle.dump(tfidf_matrix, open(MATRIX_FILE, "wb"))

    print("\n-=+ TF-IDF Index Built Successfully +=-")
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    print()


if __name__ == "__main__":
    build_tfidf_index()
