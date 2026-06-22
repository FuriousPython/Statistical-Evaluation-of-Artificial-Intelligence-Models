import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


MODEL_FILES = {
    ("Spanish", "standard"): "master_merged_sp_standard - Kopi.csv",
    ("Spanish", "limited"): "master_merged_sp_limited - Kopi.csv",
    ("French", "standard"): "master_merged_fr_standard - Kopi.csv",
    ("French", "limited"): "master_merged_fr_limited - Kopi.csv",
}

REFERENCE_FILES = {
    "Spanish": "spain_gender_distribution_categorized_2023 - Kopi.csv",
    "French": "france_gender_distribution_categorized_2023 - Kopi.csv",
}


# Main profession-to-reference mapping.
#
# Note:
# Electrician is difficult. Spain has "Electrical and electronic trades workers",
# which is closer than "Science and engineering associate professionals".
# France does not contain that exact category in the provided reference file,
# so French electrician remains mapped approximately to
# "Science and engineering associate professionals".
REFERENCE_MAPPING = {
    "Cleaning assistant": {
        "occupation": "Cleaners and helpers",
        "quality": "Close",
    },
    "Director": {
        "occupation": "Chief executives, senior officials and legislators",
        "quality": "Close",
    },
    "Doctor": {
        "occupation": "Health professionals",
        "quality": "Close",
    },
    "Electrician": {
        "occupation": "Science and engineering associate professionals",
        "quality": "Approximate",
    },
    "Engineer": {
        "occupation": "Science and engineering professionals",
        "quality": "Exact/close",
    },
    "Firefighter": {
        "occupation": "Protective services workers",
        "quality": "Close",
    },
    "Flight attendant": {
        "occupation": "Personal service workers",
        "quality": "Approximate",
    },
    "Lawyer": {
        "occupation": "Legal, social and cultural professionals",
        "quality": "Close",
    },
    "Metal worker": {
        "occupation": "Metal, machinery and related trades workers",
        "quality": "Close",
    },
    "Nurse": {
        "occupation": "Health associate professionals",
        "quality": "Close",
    },
    "Personal care worker": {
        "occupation": "Personal care workers",
        "quality": "Exact",
    },
    "Pilot": {
        "occupation": "Science and engineering associate professionals",
        "quality": "Approximate",
    },
    "Police Officer": {
        "occupation": "Protective services workers",
        "quality": "Close",
    },
    "Politician": {
        "occupation": "Chief executives, senior officials and legislators",
        "quality": "Close",
    },
    "Programmer": {
        "occupation": "Information and communications technology professionals",
        "quality": "Close",
    },
    "Receptionist": {
        "occupation": "Customer services clerks",
        "quality": "Close",
    },
    "Secretary": {
        "occupation": "General and keyboard clerks",
        "quality": "Close",
    },
    "Social worker": {
        "occupation": "Legal, social, cultural and related associate professionals",
        "quality": "Close",
    },
    "Soldier": {
        "occupation": "Armed forces occupations, other ranks",
        "quality": "Close",
    },
    "Surgeon": {
        "occupation": "Health professionals",
        "quality": "Close",
    },
    "Teacher": {
        "occupation": "Teaching professionals",
        "quality": "Close",
    },
    "Therapist": {
        "occupation": "Health professionals",
        "quality": "Approximate",
    },
}


# Country-specific overrides.
#
# This makes Spain-electrician more exact because the Spain reference file contains
# "Electrical and electronic trades workers".
COUNTRY_SPECIFIC_OVERRIDES = {
    ("Spanish", "Electrician"): {
        "occupation": "Electrical and electronic trades workers",
        "quality": "Approximate",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kendall tau-b analysis between model gender outputs and reference gender distributions."
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Folder containing the four model CSV files and two reference CSV files. "
            "If omitted, the script uses the folder where this .py file is saved."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="kendall_tau_outputs",
        help=(
            "Folder where output CSV files will be written. If this is a relative path, "
            "it will be created inside the input folder."
        ),
    )

    parser.add_argument(
        "--disable-country-overrides",
        action="store_true",
        help="Use only the base mapping table, without country-specific occupation overrides.",
    )

    return parser.parse_args()


def script_folder():
    """Return the folder where this Python file is saved."""
    return Path(__file__).resolve().parent


def normalize_filename(name):
    """
    Normalize filenames so small differences are easier to catch.
    This helps with OneDrive / Windows spacing or case issues.
    """
    return " ".join(name.strip().lower().split())


def resolve_input_dir(input_dir_arg):
    """
    VSCode-friendly input folder resolver.

    If you click 'Run Python File' in VSCode, the current working directory can be
    different from the script folder. Therefore, by default this script uses the
    folder where the .py file is saved.
    """
    if input_dir_arg is None:
        return script_folder()

    return Path(input_dir_arg).expanduser().resolve()


def resolve_output_dir(out_dir_arg, input_dir):
    """Create output folder inside input_dir unless an absolute path is provided."""
    out_dir = Path(out_dir_arg).expanduser()

    if not out_dir.is_absolute():
        out_dir = input_dir / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()


def available_csvs(input_dir):
    """Return all CSV filenames in the input folder."""
    return sorted(p.name for p in input_dir.glob("*.csv"))


def find_required_file(input_dir, expected_filename):
    """
    Find a required CSV file.

    First tries the exact filename. If that fails, it tries a normalized filename
    comparison so that tiny spacing/case differences produce a helpful match.
    """
    exact_path = input_dir / expected_filename

    if exact_path.is_file():
        return exact_path

    expected_normalized = normalize_filename(expected_filename)

    for candidate in input_dir.glob("*.csv"):
        if normalize_filename(candidate.name) == expected_normalized:
            print(f"Using CSV with slightly different formatting: {candidate.name}")
            return candidate

    csv_list = available_csvs(input_dir)
    csv_text = "\n".join(f"  - {name!r}" for name in csv_list) if csv_list else "  No CSV files found."

    raise FileNotFoundError(
        "\nMissing required CSV file.\n"
        f"Expected: {expected_filename!r}\n"
        f"Looked in: {str(input_dir)!r}\n\n"
        "CSV files Python can see in that folder:\n"
        f"{csv_text}\n\n"
        "Fix: put the missing CSV in this folder, or rename the file so it matches the expected name."
    )

def dominance_score(percent_female):
    """
    Convert percentage female into ordered dominance category.

    1 = strongly male-dominated: [0, 20)
    2 = male-dominated:          [20, 40)
    3 = evenly distributed:      [40, 60)
    4 = female-dominated:        [60, 80)
    5 = strongly female-dominated: [80, 100]
    """
    if pd.isna(percent_female):
        return np.nan

    if percent_female < 20:
        return 1
    if percent_female < 40:
        return 2
    if percent_female < 60:
        return 3
    if percent_female < 80:
        return 4

    return 5


def dominance_label(score):
    labels = {
        1: "strongly male-dominated",
        2: "male-dominated",
        3: "evenly distributed",
        4: "female-dominated",
        5: "strongly female-dominated",
    }
    return labels.get(score, np.nan)


def load_model_outputs(input_dir):
    dfs = []

    for (language, prompt), filename in MODEL_FILES.items():
        path = find_required_file(input_dir, filename)

        df = pd.read_csv(path)

        required_columns = {"Language", "Prompt Variant", "Profession", "Gender"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")

        df["Language"] = df["Language"].astype(str).str.strip()
        df["Prompt Variant"] = df["Prompt Variant"].astype(str).str.strip().str.lower()
        df["Profession"] = df["Profession"].astype(str).str.strip()
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()

        expected_prompt = prompt.lower()

        bad_language = df[df["Language"].str.lower() != language.lower()]
        bad_prompt = df[df["Prompt Variant"].str.lower() != expected_prompt]

        if len(bad_language) > 0:
            raise ValueError(
                f"{filename} contains unexpected Language values: "
                f"{sorted(df['Language'].unique())}"
            )

        if len(bad_prompt) > 0:
            raise ValueError(
                f"{filename} contains unexpected Prompt Variant values: "
                f"{sorted(df['Prompt Variant'].unique())}"
            )

        invalid_gender = sorted(set(df["Gender"]) - {"M", "F", "N", "U"})

        if invalid_gender:
            raise ValueError(
                f"{filename} contains unexpected Gender values: {invalid_gender}"
            )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_reference_data(input_dir):
    reference = {}

    for language, filename in REFERENCE_FILES.items():
        path = find_required_file(input_dir, filename)

        df = pd.read_csv(path)

        required_columns = {"occupation", "female", "male", "total_obs", "dominance_category"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")

        df["occupation"] = df["occupation"].astype(str).str.strip()
        df["female"] = pd.to_numeric(df["female"], errors="raise")
        df["male"] = pd.to_numeric(df["male"], errors="raise")

        reference[language] = df.set_index("occupation")

    return reference


def get_mapping(language, profession, use_country_overrides=True):
    if profession not in REFERENCE_MAPPING:
        raise KeyError(f"No reference mapping found for profession: {profession}")

    mapping = dict(REFERENCE_MAPPING[profession])

    if use_country_overrides:
        override = COUNTRY_SPECIFIC_OVERRIDES.get((language, profession))
        if override is not None:
            mapping.update(override)

    return mapping


def count_model_outputs(model_df):
    counts = (
        model_df
        .groupby(["Language", "Prompt Variant", "Profession", "Gender"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for category in ["M", "F", "N", "U"]:
        if category not in counts.columns:
            counts[category] = 0

    counts["valid_n"] = counts["M"] + counts["F"] + counts["N"]
    counts["gendered_n"] = counts["M"] + counts["F"]

    counts["model_female_gendered_pct"] = np.where(
        counts["gendered_n"] > 0,
        100 * counts["F"] / counts["gendered_n"],
        np.nan,
    )

    counts["model_male_gendered_pct"] = np.where(
        counts["gendered_n"] > 0,
        100 * counts["M"] / counts["gendered_n"],
        np.nan,
    )

    counts["model_neutral_rate"] = np.where(
        counts["valid_n"] > 0,
        counts["N"] / counts["valid_n"],
        np.nan,
    )

    counts["u_rate"] = np.where(
        (counts["valid_n"] + counts["U"]) > 0,
        counts["U"] / (counts["valid_n"] + counts["U"]),
        np.nan,
    )

    return counts


def attach_reference_info(counts, reference, use_country_overrides=True):
    rows = []

    for _, row in counts.iterrows():
        language = row["Language"]
        profession = row["Profession"]

        mapping = get_mapping(
            language=language,
            profession=profession,
            use_country_overrides=use_country_overrides,
        )

        occupation = mapping["occupation"]
        quality = mapping["quality"]

        ref_table = reference[language]

        if occupation not in ref_table.index:
            raise KeyError(
                f"Reference occupation '{occupation}' for profession '{profession}' "
                f"was not found in the {language} reference file."
            )

        ref_row = ref_table.loc[occupation]

        rows.append(
            {
                "reference_occupation": occupation,
                "mapping_quality": quality,
                "reference_female_pct": ref_row["female"],
                "reference_male_pct": ref_row["male"],
                "reference_dominance_category_original": ref_row["dominance_category"],
            }
        )

    ref_info = pd.DataFrame(rows, index=counts.index)
    out = pd.concat([counts.reset_index(drop=True), ref_info.reset_index(drop=True)], axis=1)

    out["reference_category_score"] = out["reference_female_pct"].apply(dominance_score)
    out["model_category_score"] = out["model_female_gendered_pct"].apply(dominance_score)

    out["reference_category"] = out["reference_category_score"].apply(dominance_label)
    out["model_category"] = out["model_category_score"].apply(dominance_label)

    return out


def kendall_result(df, analysis_name):
    valid = df.dropna(
        subset=[
            "reference_category_score",
            "model_category_score",
            "reference_female_pct",
            "model_female_gendered_pct",
        ]
    )

    if len(valid) < 3:
        return {
            "analysis": analysis_name,
            "n": len(valid),
            "kendall_tau_category": np.nan,
            "p_value_category": np.nan,
            "kendall_tau_continuous": np.nan,
            "p_value_continuous": np.nan,
            "mean_abs_category_difference": np.nan,
            "exact_category_match_rate": np.nan,
        }

    tau_cat, p_cat = kendalltau(
        valid["reference_category_score"],
        valid["model_category_score"],
        variant="b",
    )

    tau_cont, p_cont = kendalltau(
        valid["reference_female_pct"],
        valid["model_female_gendered_pct"],
        variant="b",
    )

    category_diff = (
        valid["model_category_score"]
        -
        valid["reference_category_score"]
    )

    return {
        "analysis": analysis_name,
        "n": len(valid),
        "kendall_tau_category": tau_cat,
        "p_value_category": p_cat,
        "kendall_tau_continuous": tau_cont,
        "p_value_continuous": p_cont,
        "mean_abs_category_difference": category_diff.abs().mean(),
        "exact_category_match_rate": (category_diff == 0).mean(),
    }


def aggregate_by_language(condition_df):
    """
    Aggregate standard and limited prompts within each language-profession pair.
    This gives one model estimate per country/language and profession.
    """
    grouped = (
        condition_df
        .groupby(["Language", "Profession", "reference_occupation", "mapping_quality"], as_index=False)
        [["M", "F", "N", "U"]]
        .sum()
    )

    grouped["Prompt Variant"] = "aggregated"

    grouped["valid_n"] = grouped["M"] + grouped["F"] + grouped["N"]
    grouped["gendered_n"] = grouped["M"] + grouped["F"]

    grouped["model_female_gendered_pct"] = np.where(
        grouped["gendered_n"] > 0,
        100 * grouped["F"] / grouped["gendered_n"],
        np.nan,
    )

    grouped["model_male_gendered_pct"] = np.where(
        grouped["gendered_n"] > 0,
        100 * grouped["M"] / grouped["gendered_n"],
        np.nan,
    )

    grouped["model_neutral_rate"] = np.where(
        grouped["valid_n"] > 0,
        grouped["N"] / grouped["valid_n"],
        np.nan,
    )

    grouped["u_rate"] = np.where(
        (grouped["valid_n"] + grouped["U"]) > 0,
        grouped["U"] / (grouped["valid_n"] + grouped["U"]),
        np.nan,
    )

    # Reference values are constant within Language-Profession, so take first from original.
    ref_cols = [
        "Language",
        "Profession",
        "reference_female_pct",
        "reference_male_pct",
        "reference_dominance_category_original",
        "reference_category_score",
        "reference_category",
    ]

    ref_lookup = (
        condition_df[ref_cols]
        .drop_duplicates(subset=["Language", "Profession"])
    )

    grouped = grouped.merge(ref_lookup, on=["Language", "Profession"], how="left")

    grouped["model_category_score"] = grouped["model_female_gendered_pct"].apply(dominance_score)
    grouped["model_category"] = grouped["model_category_score"].apply(dominance_label)

    return grouped


def make_analysis_summary(condition_df, language_agg_df):
    rows = []

    # Condition-specific analyses: Spanish standard, Spanish limited, French standard, French limited.
    for (language, prompt), sub in condition_df.groupby(["Language", "Prompt Variant"]):
        rows.append(kendall_result(sub, f"{language} {prompt}"))

        sub_no_approx = sub[sub["mapping_quality"] != "Approximate"]
        rows.append(kendall_result(sub_no_approx, f"{language} {prompt}, excluding approximate mappings"))

    # Language-aggregated analyses: Spanish aggregated, French aggregated.
    for language, sub in language_agg_df.groupby("Language"):
        rows.append(kendall_result(sub, f"{language} aggregated across prompts"))

        sub_no_approx = sub[sub["mapping_quality"] != "Approximate"]
        rows.append(kendall_result(sub_no_approx, f"{language} aggregated across prompts, excluding approximate mappings"))

    # Combined language-specific rows.
    # This uses 44 rows: 22 Spanish profession rows and 22 French profession rows.
    # It is useful descriptively, but professions appear twice, so report cautiously.
    rows.append(kendall_result(language_agg_df, "Combined language-specific rows"))

    combined_no_approx = language_agg_df[language_agg_df["mapping_quality"] != "Approximate"]
    rows.append(kendall_result(combined_no_approx, "Combined language-specific rows, excluding approximate mappings"))

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    out_dir = resolve_output_dir(args.out_dir, input_dir)

    print()
    print("Kendall tau reference-alignment analysis")
    print(f"Script folder: {script_folder()}")
    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {out_dir}")
    print()
    print("CSV files found in input folder:")
    for csv_name in available_csvs(input_dir):
        print(f"  - {csv_name}")
    print()

    use_country_overrides = not args.disable_country_overrides

    model_df = load_model_outputs(input_dir)
    reference = load_reference_data(input_dir)

    counts = count_model_outputs(model_df)
    condition_comparison = attach_reference_info(
        counts=counts,
        reference=reference,
        use_country_overrides=use_country_overrides,
    )

    language_aggregated = aggregate_by_language(condition_comparison)

    summary = make_analysis_summary(
        condition_df=condition_comparison,
        language_agg_df=language_aggregated,
    )

    # Add category difference columns for interpretation.
    condition_comparison["category_difference_model_minus_reference"] = (
        condition_comparison["model_category_score"]
        -
        condition_comparison["reference_category_score"]
    )

    language_aggregated["category_difference_model_minus_reference"] = (
        language_aggregated["model_category_score"]
        -
        language_aggregated["reference_category_score"]
    )

    condition_path = out_dir / "kendall_condition_level_comparison.csv"
    language_path = out_dir / "kendall_language_aggregated_comparison.csv"
    summary_path = out_dir / "kendall_tau_summary.csv"

    condition_comparison.to_csv(condition_path, index=False, encoding="utf-8")
    language_aggregated.to_csv(language_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print()
    print("Kendall tau reference-alignment analysis complete.")
    print(f"Country-specific mapping overrides used: {use_country_overrides}")
    print()
    print(f"Condition-level comparison written to: {condition_path}")
    print(f"Language-aggregated comparison written to: {language_path}")
    print(f"Kendall tau summary written to: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()