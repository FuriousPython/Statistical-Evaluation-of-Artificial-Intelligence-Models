import pandas as pd
from pathlib import Path


# CONFIG

folder1 = Path("folder1")   # First subfolder
folder2 = Path("folder2")   # Second subfolder

output_file = "master_merged.csv"

merge_columns = [
    "Language",
    "Prompt Variant",
    "Profession",
    "Gender"
]

# MERGE Pair

all_merged = []

folder1_files = {f.name: f for f in folder1.glob("*.csv")}
folder2_files = {f.name: f for f in folder2.glob("*.csv")}

common_files = sorted(set(folder1_files.keys()) & set(folder2_files.keys()))

print(f"Found {len(common_files)} matching files.")

for filename in common_files:

    print(f"Processing: {filename}")

    df1 = pd.read_csv(folder1_files[filename])
    df2 = pd.read_csv(folder2_files[filename])

    # Verify merge columns exist
    missing1 = [c for c in merge_columns if c not in df1.columns]
    missing2 = [c for c in merge_columns if c not in df2.columns]

    if missing1:
        print(f"  Skipped - missing in folder1: {missing1}")
        continue

    if missing2:
        print(f"  Skipped - missing in folder2: {missing2}")
        continue

    merged = pd.merge(
        df1,
        df2,
        on=merge_columns,
        how="outer",
        suffixes=("_folder1", "_folder2")
    )

    merged["source_file"] = filename

    all_merged.append(merged)

# Master CSV

if all_merged:

    master_df = pd.concat(all_merged, ignore_index=True)

    master_df.to_csv(output_file, index=False)

    print(f"\nDone!")
    print(f"Rows: {len(master_df):,}")
    print(f"Saved to: {output_file}")

else:
    print("No files were successfully merged.")