import pandas as pd
from pathlib import Path




base_dir = Path(__file__).parent.resolve()

folder1 = (
    base_dir
    / "FULL_TEST"
    / "french_gender_bias_results"
    / "per_profession_raw_prompts"
)

folder2 = (
    base_dir
    / "pilot_tests_french"
    / "french_gender_bias_results"
    / "per_profession_raw_prompts"
)


all_rows = []

for file1 in folder1.glob("*.csv"):

    file2 = folder2 / file1.name

    if not file2.exists():
        continue

    df1 = pd.read_csv(file1)

    cols = [
        "Language",
        "Prompt Variant",
        "Profession",
        "Gender"
    ]

    df1 = df1[cols].copy()
    df1["source_file"] = file1.name

    all_rows.append(df1)

master = pd.concat(all_rows, ignore_index=True)

master.to_csv("master_merged.csv", index=False)