import os
import glob
import pandas as pd
import re

TEXT_DIR = "text-data/"
CORP_FILE = "data/corpus.csv"


def clean_text(text):
    """
    Clean up excess empty lines and whitespace
    -----
    :param text: .txt content
    """
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_passages(text):
    """
    Split text/passages by paragraph
    -----
    :param text: .txt content
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def clean_title(filename):
    """
    Clean up my filenames into a more readable title.
    -----
    :param filename: original name of file
    -----
    Example:
    cleaned_ch01-montessori-science-behind-genius.txt
        RETURNS "Montessori Science Behind Genius (Chapter 1)"
    """

    base = os.path.basename(filename).replace(".txt", "")
    # chapter from Dr. Lillard
    if base.startswith("cleaned_ch"):
        match = re.match(r"cleaned_ch(\d+)-(.*)", base)
        if match:
            chapter_num = match.group(1)
            title_raw = match.group(2).replace("-", " ").title()
            return f"{title_raw} (Chapter {chapter_num})"
    
    # research paper
    title = base.replace("-", " ").title()
    return title


def build_passage_corpus():
    records = []
    filepaths = glob.glob(os.path.join(TEXT_DIR, "*.txt"))
    
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        cleaned = clean_text(raw_text)
        passages = split_into_passages(cleaned)
        source_title = clean_title(filepath)

        for i, passage in enumerate(passages):
            records.append({
                "doc_id": f"{os.path.basename(filepath)}_p{i}",
                "source_file": os.path.basename(filepath),
                "source_title": source_title,
                "paragraph_index": i,
                "text": passage,
                "source_type": "full_text_passage"
            })

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CORP_FILE, index=False)
    print(f"Saved {len(df)} passages to {CORP_FILE}")


if __name__ == "__main__":

    TEXT_DIR = "text-data/"
    filepaths = glob.glob(os.path.join(TEXT_DIR, "*.txt"))

    for filepath in filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                f.read()
            print(f"OK: {filepath}")
        except Exception as e:
            print(f"ERROR in file: {filepath}")
            print(f"  {type(e).__name__}: {e}")

    build_passage_corpus()