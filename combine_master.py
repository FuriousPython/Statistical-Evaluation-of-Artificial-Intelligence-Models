import pandas as pd
from pathlib import Path

# Master files to combine
files = [
    "master_merged_fr_limited.csv",
    "master_merged_fr_standard.csv",
    "master_merged_sp_limited.csv",
    "master_merged_sp_standard.csv"
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

summary = (
    df.groupby(
        ["Language", "Prompt Variant", "Profession", "Gender"]
    )
    .size()
    .unstack(fill_value=0)
)

# Ensure all gender columns exist
for gender in ["M", "F", "N", "U"]:
    if gender not in summary.columns:
        summary[gender] = 0

summary = summary[["M", "F", "N", "U"]]
summary["Total"] = summary.sum(axis=1)
summary = summary.reset_index()

summary.to_csv("gender_counts_master.csv", index=False)

print(summary.head())
print(f"\nSaved {len(summary)} rows to gender_counts_master.csv")