import pandas as pd
import os

def combine_data(path="text-data/"):
    csv_files = [path + f for f in os.listdir(path) if f.endswith(".csv")]
    print(f"== {len(csv_files)}/20 FILES FOUND ==")

    comparisons = []
    for file in csv_files:
        df = pd.read_csv(file)
        comparisons.append(df)

    combined = pd.concat(comparisons, ignore_index=True)

    return combined

if __name__ == "__main__":
    combined_df = combine_data()
    combined_df.to_csv("../text-data/all_excerpts.csv", index=False)
    print(f"\nDATA FILE STATS:\n   SHAPE: {combined_df.shape}")
    print(f"   CATEGORIES: {len(combined_df['Category'].unique())}\n")
    print("==       DONE!       ==")