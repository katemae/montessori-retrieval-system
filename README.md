# Montessori Evidence Retrieval System

This project aims to develop an information retrieval and text-mining tool that indexes and analyzes excerpts from research comparing Montessori and Traditional education methods. The tool will allow users to search Montessori-related excerpts by key phrases (for example, "environment," "self-regulation," or "material") and view the most relevant evidences comparing Montessori and Traditional education across domains such as cognitive, behavioral, and academic development.

Educational research on Montessori methods and pedagogy is abundant, but it can be difficult for parents and educators to access and interpret for their own understanding. Much of the evidence exists in academic books, research papers, and studies that aren’t easy to parse through while balancing teaching or parenting. This project aims to develop an information retrieval and text-mining tool that indexes and analyzes excerpts from research comparing Montessori and Traditional education methods. The tool will allow users to search excerpts by key phrases (for example, "environment," "self-regulation," or "material") and view the most relevant evidences comparing Montessori and Traditional education across domains such as cognitive, behavioral, and academic development.

This system will utilize a (proprietary) dataset of Montessori and Traditional education excerpts that have already been tagged by domain and educational approach (Traditional vs. Montessori). The focus is on contextual retrieval and interpretability, making it easier to see not only which results are relevant, but also how they contrast across educational philosophies. I will vectorize the evidence text using TF-IDF, implement a ranked retrieval function to return top-k matching excerpts, and, if time allows, integrate topic clustering or summary generation to highlight broader themes across the corpus.

Effectiveness will be demonstrated through qualitative evaluation of example queries and top-ranked results. If time permits, I plan to compute simple precision and recall metrics to further validate performance.

As a solo project, I will be responsible for all aspects of implementation, testing, and documentation of the system. This tool addresses that challenge by centralizing key evidence into one searchable system and presenting it in a digestible format. It aims to help parents, educators, and researchers quickly locate and compare findings that support different educational methodologies, reducing the need to manually read through all the complex research literature to decide what's best for their child(ren).

---

# Methodology

The dataset consists of two parts: raw `.txt` files of Montessori texts, research papers, studies, and more diving into scienfitic evidences, school anecdotes, Montessori materials, and examples of the benefits and differences between Montessori and Traditional education. Alongside this is a `.csv` file of `metadata`, where data analysts at [Y Montessori](https://ymontessori.com/data) have broken up the book _Montessori: The Science Behind the Genius_ into different educational categories and domains for comparison. 

The system first segments the raw educational `.txt` files into paragraph-level passages. These passages, along with the excerpts from the metadata file, are then vectorized using TF-IDF to construct a shared feature space. User queries are processed using the same preprocessing and vectorization pipeline and are ranked against the corpus using cosine similarity.

In addition to basic retrieval, the system performs lightweight text mining by inferring structured metadata directly from text. This includes identifying the instructional approach (Montessori or Traditional), evidence type (Study, Material, or Example), and relevant educational domains (e.g., Cognitive, Behavioral, Academic, Social, Environment). These inferred attributes enable optional, query-driven filtering to refine retrieval results.

Evaluation is performed using Precision@5 on a small set of manually labeled queries. Recall@5 was initially computed but resulted in a value of 1.00 due to relevance judgments being collected only for the top retrieved results. As a result, Precision@5 is reported as the primary evaluation metric.

---

# Using the Search Engine

**(1) Create the conda environment**

```
conda env create -f requirements.yml
```



**(2) Activate the environment**

```
conda activate montessori-retrieval
```
If you used a custom name, replace `montessori-retrieval` accordingly.



**(3) Build the corpus and TF-IDF index (optional)**

The `models/` directory is already populated.
Additionally, because the raw text files are proprietary and therefore hidden,
you must skip the first two steps:

a. First two steps (must be skipped unless you have access to raw data):
```
python src/build_corpus.py 
python src/merge_corpora.py
```

b. Runnable step, optional:
```
python src/idx_tfidf.py
```

This will generate:
- `data/full_corpus.csv`
- `models/tfidf_vectorizer.pkl`
- `models/tfidf_matrix.pkl`
- `models/corpus_processed.pkl`



**(4) Run the interactive search system/engine**

```
python src/filter_search.py
```

From here:

a. Enter a Montessori-related query and number of documents (optional) when prompted.

b. Accept or decline recommended filters, if applicable. (Not all queries will have one!)

c. Type q or quit to exit, or enter another query to continue.

**(5) Optionally, run an evaluation of `Precision@5` on 5 select queries.**

```
python eval/precision_at_5.py
```
