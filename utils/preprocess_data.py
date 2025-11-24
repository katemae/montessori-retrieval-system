import pandas as pd

def load_corpus(path="text-data/all_excerpts.csv"):
    # simplify column names
    COL_MAP = {
        "Type of Evidence (Example, Material, Study)": "EvidenceType",
        "Page Link": "PageLink",
        "Page Number": "PageNumber"
    }

    # read in csv
    df = pd.read_csv(path)
    df = df.rename(columns=COL_MAP)

    # properly convert columns datatypes
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    
    # drop empty/"nan" Evidence rows (no text/doc to work with)
    ### technically speaking, no evidence rows should be empty, but ###
    ### adding this just as a precaution                            ###
    df = df[df["Evidence"].str.len() > 0]
    df = df[df["Evidence"] != "nan"]

    
    if "SORTING" in df.columns:
        df["SORTING"] = pd.to_numeric(df["SORTING"], errors="coerce")
        df = df.sort_values("SORTING").reset_index(drop=True)

    return df