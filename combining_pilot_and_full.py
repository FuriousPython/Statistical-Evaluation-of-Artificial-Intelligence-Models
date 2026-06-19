import pandas as pd
from pathlib import Path




base_dir = Path(__file__).parent.resolve()

folder1 = (
    base_dir
    / "FULL_TEST"
    / "spanish_gender_bias_results_limited"
    / "per_profession_raw_prompts"
)

folder2 = (
    base_dir
    / "pilot_tests_spanish"
    / "spanish_gender_bias_results_limited"
    / "per_profession_raw_prompts"
)

cols = [
    "Language",
    "Prompt Variant",
    "Profession",
    "Gender"
]

all_rows = []

for file1 in folder1.glob("*.csv"):

    file2 = folder2 / file1.name

    if not file2.exists():
        print(f"Missing match: {file1.name}")
        continue

    df1 = pd.read_csv(file1)[cols]
    df2 = pd.read_csv(file2)[cols]

    combined = pd.concat([df1, df2], ignore_index=True)

    all_rows.append(combined)

master = pd.concat(all_rows, ignore_index=True)

master.to_csv("master_merged_sp_limited.csv", index=False)

print(f"Rows written: {len(master):,}")