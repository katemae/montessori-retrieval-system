import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, stopwords="english"):
        self.vectorizer = TfidfVectorizer(stop_words=stopwords)
        self.X = None        # TF-IDF matrix
        self.df = None       # retrieval dataframe

    def fit(self, df, text_col="Evidence"):
        """
        Fit TF-IDF vectorizer on the Evidence column.
        """
        self.df = df.reset_index(drop=True)
        corpus = df[text_col].astype(str).unique().tolist()

        self.X = self.vectorizer.fit_transform(corpus)

        print(f"[TF-IDF] Fitted on {len(corpus)} documents.")
        print(f"[TF-IDF] Matrix shape: {self.X.shape}")  # (num_docs, vocab_size)

    def search(self, query, top_k=5):
        """
        Extract top_k most relevant excerpts for the query.
        
        Returns
        -------
        results_df: pandas DataFrame of top_k results
        scores: numpy array of score ratings
        """
        if self.X is None:
            raise ValueError("You must fit() the retriever before searching.")
        
        if "study" in query:
            corpus = self.df[self.df["EvidenceType"].str.upper() == "Study"].astype(str).unique().tolist()
            self.X = self.vectorizer.fit_transform(corpus)

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.X).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]


        results = self.df.iloc[top_idx].copy()
        results["score"] = scores[top_idx]

        return results, scores[top_idx]
