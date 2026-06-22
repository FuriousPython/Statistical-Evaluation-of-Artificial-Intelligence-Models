import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load data
# ==========================
csv_file = "gender_counts_master.csv"
df = pd.read_csv(csv_file)

# ==========================
# Profession order
# ==========================
profession_order = (
    df["Profession"]
    .drop_duplicates()
    .tolist()
)

# ==========================
# Create datasets
# ==========================
fr_standard = df[
    (df["Language"] == "French") &
    (df["Prompt Variant"] == "standard")
].copy()

fr_limited = df[
    (df["Language"] == "French") &
    (df["Prompt Variant"] == "limited")
].copy()

sp_standard = df[
    (df["Language"] == "Spanish") &
    (df["Prompt Variant"] == "standard")
].copy()

sp_limited = df[
    (df["Language"] == "Spanish") &
    (df["Prompt Variant"] == "limited")
].copy()

# ==========================
# Apply profession ordering
# ==========================
datasets = [
    fr_standard,
    fr_limited,
    sp_standard,
    sp_limited
]

for dataset in datasets:
    dataset["Profession"] = pd.Categorical(
        dataset["Profession"],
        categories=profession_order,
        ordered=True
    )
    dataset.sort_values("Profession", inplace=True)

profession_names = profession_order

# ==========================
# Extract counts
# ==========================
def get_counts(df_subset):
    return (
        df_subset["M"].values,
        df_subset["F"].values,
        df_subset["N"].values
    )

M_fr_std, F_fr_std, N_fr_std = get_counts(fr_standard)
M_fr_lim, F_fr_lim, N_fr_lim = get_counts(fr_limited)

M_sp_std, F_sp_std, N_sp_std = get_counts(sp_standard)
M_sp_lim, F_sp_lim, N_sp_lim = get_counts(sp_limited)

# ==========================
# Plotting function
# ==========================
def plot_comparison(
    M_a, F_a, N_a,
    M_b, F_b, N_b,
    profession_names,
    title,
    label_a,
    label_b,
    colors
):

    x = np.arange(1, len(profession_names) + 1)
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 8))

    # Male
    ax.bar(
        x - width,
        M_a,
        width,
        color=colors["male_a"],
        label=f"Male ({label_a})"
    )

    ax.bar(
        x - width,
        M_b,
        width,
        bottom=M_a,
        color=colors["male_b"],
        label=f"Male ({label_b})"
    )

    # Female
    ax.bar(
        x,
        F_a,
        width,
        color=colors["female_a"],
        label=f"Female ({label_a})"
    )

    ax.bar(
        x,
        F_b,
        width,
        bottom=F_a,
        color=colors["female_b"],
        label=f"Female ({label_b})"
    )

    # Neutral
    ax.bar(
        x + width,
        N_a,
        width,
        color=colors["neutral_a"],
        label=f"Neutral ({label_a})"
    )

    ax.bar(
        x + width,
        N_b,
        width,
        bottom=N_a,
        color=colors["neutral_b"],
        label=f"Neutral ({label_b})"
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Profession")
    ax.set_ylabel("Count")

    ax.set_xticks(x)
    ax.set_xticklabels(
        profession_names,
        rotation=45,
        ha="right"
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(title.replace(":", "").replace(" ", "_") + ".png",
            dpi=300,
            bbox_inches="tight")
    plt.close()

# ==========================
# Colors
# ==========================
colors = {
    "male_a": "darkblue",
    "male_b": "blue",
    "female_a": "darkred",
    "female_b": "orange",
    "neutral_a": "black",
    "neutral_b": "grey"
}

# ==========================
# Plot 1
# French: Standard vs Limited
# ==========================
plot_comparison(
    M_fr_std, F_fr_std, N_fr_std,
    M_fr_lim, F_fr_lim, N_fr_lim,
    profession_names,
    "French: Standard vs Limited",
    "Standard",
    "Limited",
    colors
)

# ==========================
# Plot 2
# Spanish: Standard vs Limited
# ==========================
plot_comparison(
    M_sp_std, F_sp_std, N_sp_std,
    M_sp_lim, F_sp_lim, N_sp_lim,
    profession_names,
    "Spanish: Standard vs Limited",
    "Standard",
    "Limited",
    colors
)

# ==========================
# Plot 3
# Standard: French vs Spanish
# ==========================
plot_comparison(
    M_fr_std, F_fr_std, N_fr_std,
    M_sp_std, F_sp_std, N_sp_std,
    profession_names,
    "Standard: French vs Spanish",
    "French",
    "Spanish",
    colors
)

# ==========================
# Plot 4
# Limited: French vs Spanish
# ==========================
plot_comparison(
    M_fr_lim, F_fr_lim, N_fr_lim,
    M_sp_lim, F_sp_lim, N_sp_lim,
    profession_names,
    "Limited: French vs Spanish",
    "French",
    "Spanish",
    colors
)

# ==========================
# Profession index reference
# ==========================
print("\nProfession numbering:\n")

for i, profession in enumerate(profession_names, start=1):
    print(f"{i:2d}: {profession}")