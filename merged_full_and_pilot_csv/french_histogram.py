import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Current working directory:")
print(os.getcwd())

print("\nFiles in directory:")
print(os.listdir())

# ==========================
# Load data
# ==========================
csv_file = r"merged_full_and_pilot_csv/gender_counts_master.csv"
df = pd.read_csv(csv_file)

# Keep only French rows
df = df[df["Language"] == "French"].copy()

# Profession order exactly as it appears in the file
profession_order = (
    df["Profession"]
    .drop_duplicates()
    .tolist()
)

# Split datasets
standard = df[df["Prompt Variant"] == "standard"].copy()
limited = df[df["Prompt Variant"] == "limited"].copy()

# Force same profession order
standard["Profession"] = pd.Categorical(
    standard["Profession"],
    categories=profession_order,
    ordered=True
)

limited["Profession"] = pd.Categorical(
    limited["Profession"],
    categories=profession_order,
    ordered=True
)

standard = standard.sort_values("Profession")
limited = limited.sort_values("Profession")

profession_names = profession_order

# x-axis positions (1-22)
x = np.arange(1, len(profession_names) + 1)

# ==========================
# Plot
# ==========================
width = 0.25

fig, ax = plt.subplots(figsize=(16, 8))

# Male
ax.bar(
    x - width,
    M_std,
    width,
    color="darkblue",
    label="Male (French-standard)"
)

ax.bar(
    x - width,
    M_lim,
    width,
    bottom=M_std,
    color="blue",
    label="Male (French-limited)"
)

# Female
ax.bar(
    x,
    F_std,
    width,
    color="red",
    label="Female (French-standard)"
)

ax.bar(
    x,
    F_lim,
    width,
    bottom=F_std,
    color="orange",
    label="Female (French-limited)"
)

# Neutral
ax.bar(
    x + width,
    N_std,
    width,
    color="black",
    label="Neutral (French-standard)"
)

ax.bar(
    x + width,
    N_lim,
    width,
    bottom=N_std,
    color="grey",
    label="Neutral (French-limited)"
)

# ==========================
# Formatting
# ==========================
ax.set_xlabel("Profession")
ax.set_ylabel("Count")
ax.set_title("French Standard vs French Limited")


ax.set_xticks(x)
ax.set_xticklabels(
    profession_names,
    rotation=45,
    ha="right"
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# ==========================
# Profession index reference
# ==========================
print("\nProfession numbering:\n")
for i, profession in enumerate(profession_names, start=1):
    print(f"{i:2d}: {profession}")