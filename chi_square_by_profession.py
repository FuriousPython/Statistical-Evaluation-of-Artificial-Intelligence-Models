import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


DEFAULT_PROJECT_DIR = (
    r"C:\Users\pjalt\OneDrive\Skrivebord\DTU-projects (mult-repos)"
    r"\Statistical-Evaluation-of-Artificial-Intelligence-Models"
)

DEFAULT_INPUT = os.path.join(
    DEFAULT_PROJECT_DIR,
    "merged_full_and_pilot_csv",
    "gender_counts_master.csv",
)

DEFAULT_OUT_SUMMARY = os.path.join(
    DEFAULT_PROJECT_DIR,
    "merged_full_and_pilot_csv",
    "chi_square_by_profession.csv",
)

DEFAULT_OUT_RESIDUALS = os.path.join(
    DEFAULT_PROJECT_DIR,
    "merged_full_and_pilot_csv",
    "chi_square_residuals_by_profession.csv",
)


CONDITION_ORDER = [
    ("Spanish", "standard"),
    ("Spanish", "limited"),
    ("French", "standard"),
    ("French", "limited"),
]


def normalize_colname(name):
    return str(name).strip().lower().replace("_", " ")


def find_column(df, possible_names):
    normalized = {normalize_colname(col): col for col in df.columns}

    for name in possible_names:
        key = normalize_colname(name)
        if key in normalized:
            return normalized[key]

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run profession-level chi-square tests across language-prompt conditions."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Input CSV containing Language, Prompt Variant, Profession, M, F, N, U, Total.",
    )

    parser.add_argument(
        "--out-summary",
        default=DEFAULT_OUT_SUMMARY,
        help="Output CSV for profession-level chi-square results.",
    )

    parser.add_argument(
        "--out-residuals",
        default=DEFAULT_OUT_RESIDUALS,
        help="Output CSV for standardized residuals by profession, condition, and response category.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Overall alpha before Bonferroni correction. Default: 0.05.",
    )

    parser.add_argument(
        "--n-tests",
        type=int,
        default=22,
        help="Number of profession-level chi-square tests. Default: 22.",
    )

    return parser.parse_args()


def cramers_v(chi2, n, rows, cols):
    if n <= 0:
        return np.nan

    denom = n * min(rows - 1, cols - 1)

    if denom <= 0:
        return np.nan

    return np.sqrt(chi2 / denom)


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)

    col_language = find_column(df, ["Language"])
    col_prompt = find_column(df, ["Prompt Variant", "Prompt", "Prompt variant"])
    col_profession = find_column(df, ["Profession"])
    col_m = find_column(df, ["M", "Male"])
    col_f = find_column(df, ["F", "Female"])
    col_n = find_column(df, ["N", "Neutral"])

    required = {
        "Language": col_language,
        "Prompt Variant": col_prompt,
        "Profession": col_profession,
        "M": col_m,
        "F": col_f,
        "N": col_n,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    for col in [col_m, col_f, col_n]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    df[col_language] = df[col_language].astype(str).str.strip()
    df[col_prompt] = df[col_prompt].astype(str).str.strip().str.lower()
    df[col_profession] = df[col_profession].astype(str).str.strip()

    alpha_bonferroni = args.alpha / args.n_tests

    summary_rows = []
    residual_rows = []

    professions = sorted(df[col_profession].unique())

    for profession in professions:
        prof_df = df[df[col_profession] == profession].copy()

        observed_rows = []
        condition_labels = []
        missing_conditions = []

        for language, prompt in CONDITION_ORDER:
            match = prof_df[
                (prof_df[col_language].str.lower() == language.lower())
                & (prof_df[col_prompt].str.lower() == prompt.lower())
            ]

            label = f"{language}_{prompt}"

            if match.empty:
                missing_conditions.append(label)
                continue

            if len(match) > 1:
                raise ValueError(
                    f"Duplicate rows found for profession={profession}, "
                    f"language={language}, prompt={prompt}"
                )

            row = match.iloc[0]
            observed_rows.append([row[col_m], row[col_f], row[col_n]])
            condition_labels.append(label)

        observed = np.array(observed_rows, dtype=float)

        result = {
            "Profession": profession,
            "n_conditions": len(condition_labels),
            "total_valid_n": int(observed.sum()) if observed.size > 0 else 0,
            "alpha_bonferroni": alpha_bonferroni,
            "missing_conditions": "; ".join(missing_conditions),
            "test_performed": False,
            "reason_not_performed": "",
            "chi2": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "significant_bonferroni": False,
            "cramers_v": np.nan,
            "min_expected": np.nan,
            "low_expected_warning": False,
            "max_abs_standardized_residual": np.nan,
            "largest_residual_condition": "",
            "largest_residual_category": "",
        }

        if observed.shape[0] < 2:
            result["reason_not_performed"] = "Fewer than two conditions available."
            summary_rows.append(result)
            continue

        # Remove response categories with zero total across all conditions.
        # This avoids invalid expected counts caused by entirely empty columns.
        category_names = np.array(["M", "F", "N"])
        col_totals = observed.sum(axis=0)
        keep_cols = col_totals > 0

        observed_reduced = observed[:, keep_cols]
        kept_categories = category_names[keep_cols]

        if observed_reduced.shape[1] < 2:
            result["reason_not_performed"] = (
                "Only one response category observed across all conditions; "
                "no distributional variation to test."
            )
            summary_rows.append(result)
            continue

        try:
            chi2, p_value, dof, expected = chi2_contingency(
                observed_reduced,
                correction=False,
            )
        except ValueError as e:
            result["reason_not_performed"] = str(e)
            summary_rows.append(result)
            continue

        standardized_residuals = (observed_reduced - expected) / np.sqrt(expected)

        max_idx = np.unravel_index(
            np.nanargmax(np.abs(standardized_residuals)),
            standardized_residuals.shape,
        )

        max_abs_resid = abs(standardized_residuals[max_idx])
        max_condition = condition_labels[max_idx[0]]
        max_category = kept_categories[max_idx[1]]

        result["test_performed"] = True
        result["reason_not_performed"] = ""
        result["chi2"] = chi2
        result["df"] = dof
        result["p_value"] = p_value
        result["significant_bonferroni"] = p_value < alpha_bonferroni
        result["cramers_v"] = cramers_v(
            chi2=chi2,
            n=observed_reduced.sum(),
            rows=observed_reduced.shape[0],
            cols=observed_reduced.shape[1],
        )
        result["min_expected"] = expected.min()
        result["low_expected_warning"] = bool((expected < 5).any())
        result["max_abs_standardized_residual"] = max_abs_resid
        result["largest_residual_condition"] = max_condition
        result["largest_residual_category"] = max_category

        summary_rows.append(result)

        for i, condition in enumerate(condition_labels):
            for k, category in enumerate(kept_categories):
                residual_rows.append(
                    {
                        "Profession": profession,
                        "Condition": condition,
                        "Category": category,
                        "Observed": observed_reduced[i, k],
                        "Expected": expected[i, k],
                        "Standardized residual": standardized_residuals[i, k],
                        "Abs standardized residual": abs(standardized_residuals[i, k]),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    residuals_df = pd.DataFrame(residual_rows)

    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_residuals), exist_ok=True)

    summary_df.to_csv(args.out_summary, index=False, encoding="utf-8")
    residuals_df.to_csv(args.out_residuals, index=False, encoding="utf-8")

    n_performed = int(summary_df["test_performed"].sum())
    n_significant = int(summary_df["significant_bonferroni"].sum())
    n_low_expected = int(summary_df["low_expected_warning"].sum())

    print()
    print(f"Input file: {args.input}")
    print(f"Professions found: {len(professions)}")
    print(f"Chi-square tests performed: {n_performed}")
    print(f"Bonferroni alpha: {alpha_bonferroni:.6f}")
    print(f"Significant after Bonferroni: {n_significant}")
    print(f"Tests with at least one expected count below 5: {n_low_expected}")
    print()
    print(f"Summary output written to: {args.out_summary}")
    print(f"Residuals output written to: {args.out_residuals}")
    print()

    if n_low_expected > 0:
        print(
            "Warning: Some tests had expected counts below 5. "
            "Interpret those chi-square p-values cautiously."
        )


if __name__ == "__main__":
    main()