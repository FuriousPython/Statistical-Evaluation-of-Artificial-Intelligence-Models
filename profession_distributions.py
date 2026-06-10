import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


dir_path = os.getcwd()
raw_datasets_path = f"{dir_path}/raw_datasets"
datasets_path = f"{dir_path}/datasets"
distribution_plots_path = f"{dir_path}/distribution_plots"

os.makedirs(datasets_path, exist_ok=True)
os.makedirs(distribution_plots_path, exist_ok=True)

print()
print(dir_path)
print()

germany_datapath = f"{datasets_path}/raw_germany_gender_occupation_2023.csv"
france_datapath = f"{datasets_path}/raw_france_gender_occupation_2023.csv"
spain_datapath = f"{datasets_path}/raw_spain_gender_occupation_2023.csv"


columns_to_drop = [
    "STRUCTURE",
    "STRUCTURE_ID",
    "STRUCTURE_NAME",
    "freq",
    "Time frequency",
    "sex",
    "Age class",
    "Unit of measure",
    "Geopolitical entity (reporting)",
    "Time",
    "Observation value",
    "Observation status (Flag) V2 structure",
    "CONF_STATUS",
    "Confidentiality status (flag)"
]

isco_codes_to_remove = [
    "NRP",
    "OC0",
    "OC1",
    "OC2",
    "OC3",
    "OC4",
    "OC5",
    "OC6",
    "OC7",
    "OC8",
    "OC9"
]


def load_and_filter_dataset(datapath):
    df = pd.read_csv(datapath)

    df = df.drop(columns=columns_to_drop)

    df = df[~df["OBS_FLAG"].isin(["du", "u"])]

    df = df[~df["isco08"].isin(isco_codes_to_remove)]

    return df


def create_gender_distribution(df):
    occupation_col = "International Standard Classification of Occupations 2008 (ISCO-08)"

    df_small = df[["isco08", occupation_col, "Sex", "OBS_VALUE"]].copy()

    pivot_df = df_small.pivot_table(
        index=["isco08", occupation_col],
        columns="Sex",
        values="OBS_VALUE",
        aggfunc="sum"
    ).reset_index()

    pivot_df["female"] = (pivot_df["Females"] / pivot_df["Total"]) * 100
    pivot_df["male"] = (pivot_df["Males"] / pivot_df["Total"]) * 100

    result = pd.DataFrame({
        "occupation": pivot_df[occupation_col],
        "female": pivot_df["female"].round(2),
        "male": pivot_df["male"].round(2),
        "total_obs": pivot_df["Total"]
    })

    result = result.dropna()
    result = result.reset_index(drop=True)

    return result


def sort_by_female_percentage(df):
    return df.sort_values(
        by="female",
        ascending=False
    ).reset_index(drop=True)


def add_dominance_category(df):
    categorized_df = df.copy()

    bins = [0, 20, 40, 60, 80, 100]

    labels = [
        "strongly male dominated",
        "male dominated",
        "evenly dominated",
        "female dominated",
        "strongly female dominated"
    ]

    categorized_df["dominance_category"] = pd.cut(
        categorized_df["female"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return categorized_df


def occupations_by_category_text(df, country_name):
    text = ""
    text += f"--- {country_name} occupations by dominance category ---\n"

    categories = [
        "strongly female dominated",
        "female dominated",
        "evenly dominated",
        "male dominated",
        "strongly male dominated"
    ]

    for category in categories:
        occupations = df[df["dominance_category"] == category]["occupation"].tolist()

        text += "\n"
        text += f"{category.upper()} ({len(occupations)} occupations)\n"

        for occupation in occupations:
            text += f"- {occupation}\n"

    text += "\n"

    return text


def plot_category_distribution(df, country_name, output_path):
    category_order = [
        "strongly male dominated",
        "male dominated",
        "evenly dominated",
        "female dominated",
        "strongly female dominated"
    ]

    counts = df["dominance_category"].value_counts().reindex(category_order)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Occupation gender dominance categories in {country_name}")
    plt.xlabel("Dominance category")
    plt.ylabel("Number of occupations")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    filename = f"{country_name.lower()}_occupation_dominance_distribution.png"
    filepath = f"{output_path}/{filename}"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {filepath}")


# --- Load and filter datasets ---
germany_df = load_and_filter_dataset(germany_datapath)
france_df = load_and_filter_dataset(france_datapath)
spain_df = load_and_filter_dataset(spain_datapath)


# --- Create nation gender distribution per profession ---
germany_gender_distribution = create_gender_distribution(germany_df)
france_gender_distribution = create_gender_distribution(france_df)
spain_gender_distribution = create_gender_distribution(spain_df)


# --- Sort by female percentage, highest first ---
germany_gender_distribution = sort_by_female_percentage(germany_gender_distribution)
france_gender_distribution = sort_by_female_percentage(france_gender_distribution)
spain_gender_distribution = sort_by_female_percentage(spain_gender_distribution)


# --- Save clean datasets in datasets/ ---
germany_gender_distribution.to_csv(
    f"{datasets_path}/germany_gender_distribution_2023.csv",
    index=False
)

france_gender_distribution.to_csv(
    f"{datasets_path}/france_gender_distribution_2023.csv",
    index=False
)

spain_gender_distribution.to_csv(
    f"{datasets_path}/spain_gender_distribution_2023.csv",
    index=False
)


# --- Print preview results ---
print()
print("Germany")
print(germany_gender_distribution.head())

print()
print("France")
print(france_gender_distribution.head())

print()
print("Spain")
print(spain_gender_distribution.head())

print()
print(germany_gender_distribution.shape)
print(france_gender_distribution.shape)
print(spain_gender_distribution.shape)


# --- Create categorized datasets without overwriting original ones ---
germany_categorized = add_dominance_category(germany_gender_distribution)
france_categorized = add_dominance_category(france_gender_distribution)
spain_categorized = add_dominance_category(spain_gender_distribution)


# --- Save categorized datasets in datasets/ ---
germany_categorized.to_csv(
    f"{datasets_path}/germany_gender_distribution_categorized_2023.csv",
    index=False
)

france_categorized.to_csv(
    f"{datasets_path}/france_gender_distribution_categorized_2023.csv",
    index=False
)

spain_categorized.to_csv(
    f"{datasets_path}/spain_gender_distribution_categorized_2023.csv",
    index=False
)


# --- Create text output ---
occupation_text = ""
occupation_text += occupations_by_category_text(germany_categorized, "Germany")
occupation_text += occupations_by_category_text(france_categorized, "France")
occupation_text += occupations_by_category_text(spain_categorized, "Spain")


# --- Print occupation domination text ---
print()
print(occupation_text)


# --- Save occupation domination text in distribution_plots/ ---
occupation_text_path = f"{distribution_plots_path}/occupation_domination.txt"

with open(occupation_text_path, "w", encoding="utf-8") as file:
    file.write(occupation_text)

print(f"Saved text file: {occupation_text_path}")


# --- Save plots in distribution_plots/ ---
plot_category_distribution(germany_categorized, "Germany", distribution_plots_path)
plot_category_distribution(france_categorized, "France", distribution_plots_path)
plot_category_distribution(spain_categorized, "Spain", distribution_plots_path)