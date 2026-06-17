import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

ALPHA = 0.05
POWER = 0.80
DELTA = 0.20
BONFERRONI = True

# Select ONE language at a time:
#   "spanish"
#   "german"
#   "french"
SELECTED_LANGUAGE = "french"

# Prompt variants:
#   french_gender_bias_results
#   french_gender_bias_results_limited
#   french_gender_bias_results_one_word
PROMPT_SUFFIXES = [
    "",           # regular prompt
    "_limited",  # limited prompt
    "_one_word", # one-word prompt
]

# Your project directory.
# This automatically becomes:
# C:\Users\pjalt\OneDrive\Skrivebord\DTU-projects (mult-repos)\Statistical-Evaluation-of-Artificial-Intelligence-Models
DIR_PATH = Path(os.getcwd())

OUTPUT_DIR = DIR_PATH / "sample_size_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep valid gender labels and map:
        M =  1
        F = -1
        N =  0

    U is removed because it is unclassifiable / invalid.
    """
    df = df.copy()

    if "Gender" not in df.columns:
        raise ValueError("CSV does not contain a 'Gender' column.")

    df = df.loc[df["Gender"] != "U"].reset_index(drop=True)

    df["Gender"] = df["Gender"].map({
        "M": 1,
        "F": -1,
        "N": 0,
    })

    df = df.dropna(subset=["Gender"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# SAMPLE SIZE CALCULATION
# ---------------------------------------------------------------------

def sample_size(
    df: pd.DataFrame,
    alpha: float = ALPHA,
    power: float = POWER,
    delta: float = DELTA,
    n_professions: int = 1,
    bonferroni: bool = BONFERRONI,
) -> dict:
    """
    Calculate required sample size for one profession in one prompt variant.

    Bias score:
        Y =  1 for male-coded
        Y = -1 for female-coded
        Y =  0 for neutral / both / no gender

    Estimand:
        theta = E[Y] = p_M - p_F

    Variance:
        sigma^2 = p_M + p_F - theta^2
    """
    n_pilot = len(df)

    if n_pilot == 0:
        raise ValueError("No valid pilot observations after filtering U.")

    y = df["Gender"]

    p_male = np.mean(y == 1)
    p_female = np.mean(y == -1)
    p_neutral = np.mean(y == 0)

    theta_hat = p_male - p_female
    sigma2_hat = p_male + p_female - theta_hat ** 2

    if bonferroni:
        alpha_star = alpha / n_professions
    else:
        alpha_star = alpha

    z_alpha = norm.ppf(1 - alpha_star / 2)
    z_beta = norm.ppf(power)

    n_required = ((z_alpha + z_beta) ** 2 * sigma2_hat) / (delta ** 2)
    n_required_ceil = int(np.ceil(n_required))

    return {
        "n_pilot": n_pilot,
        "p_male": p_male,
        "p_female": p_female,
        "p_neutral": p_neutral,
        "theta_hat": theta_hat,
        "sigma2_hat": sigma2_hat,
        "alpha_used": alpha_star,
        "power": power,
        "delta": delta,
        "n_required": n_required,
        "n_required_ceil": n_required_ceil,
    }


# ---------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------

def get_prompt_name_from_suffix(suffix: str) -> str:
    if suffix == "":
        return "regular"

    return suffix.strip("_")


def get_results_folder(language: str, suffix: str) -> Path:
    """
    Returns the actual folder containing per-profession CSV files.

    Example:
        french_gender_bias_results/per_profession
        french_gender_bias_results_limited/per_profession
        french_gender_bias_results_one_word/per_profession
    """
    return DIR_PATH / f"{language}_gender_bias_results{suffix}" / "per_profession"


def is_profession_csv(file_path: Path, language: str) -> bool:
    """
    Only include per-profession CSV files.

    Include:
        french_nurse.csv
        french_doctor.csv

    Exclude anything else.
    """
    if file_path.suffix.lower() != ".csv":
        return False

    name = file_path.name.lower()

    if not name.startswith(f"{language.lower()}_"):
        return False

    excluded_words = [
        "raw",
        "summary",
        "sample_size",
        "results",
        "percentages",
        "generations",
    ]

    if any(word in name for word in excluded_words):
        return False

    return True


def get_profession_name_from_file(file_path: Path, language: str) -> str:
    """
    Convert:
        french_personal_care_worker.csv

    into:
        Personal Care Worker
    """
    stem = file_path.stem

    prefix = f"{language}_"
    if stem.lower().startswith(prefix.lower()):
        stem = stem[len(prefix):]

    return stem.replace("_", " ").title()


# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------

def analyse_prompt_folder(
    folder_path: Path,
    language: str,
    prompt_name: str,
) -> pd.DataFrame:
    """
    Analyse all profession CSVs inside one per_profession folder.

    The upper bound is calculated separately for this prompt variant.
    """
    if not folder_path.exists():
        print("\n-----------------------------")
        print(f"Skipping missing folder: {folder_path}")
        print("-----------------------------")
        return pd.DataFrame()

    profession_files = [
        file
        for file in sorted(folder_path.iterdir())
        if file.is_file() and is_profession_csv(file, language)
    ]

    if not profession_files:
        print("\n-----------------------------")
        print(f"Skipping folder with no profession CSVs: {folder_path}")
        print("-----------------------------")
        return pd.DataFrame()

    n_professions = len(profession_files)

    print("\n=============================")
    print(f"Language: {language}")
    print(f"Prompt:   {prompt_name}")
    print(f"Folder:   {folder_path}")
    print(f"Profession CSVs found: {n_professions}")
    print("=============================")

    results = []

    for file_path in profession_files:
        dataset = file_path.name
        profession = get_profession_name_from_file(file_path, language)

        try:
            df = pd.read_csv(file_path)
            df = normalise(df)

            stats = sample_size(
                df,
                alpha=ALPHA,
                power=POWER,
                delta=DELTA,
                n_professions=n_professions,
                bonferroni=BONFERRONI,
            )

        except Exception as e:
            print(f"\n{dataset}")
            print(f"ERROR: {e}")

            stats = {
                "n_pilot": 0,
                "p_male": np.nan,
                "p_female": np.nan,
                "p_neutral": np.nan,
                "theta_hat": np.nan,
                "sigma2_hat": np.nan,
                "alpha_used": np.nan,
                "power": POWER,
                "delta": DELTA,
                "n_required": np.nan,
                "n_required_ceil": np.nan,
                "error": str(e),
            }

        stats["language"] = language
        stats["prompt"] = prompt_name
        stats["folder"] = str(folder_path)
        stats["dataset"] = dataset
        stats["profession"] = profession
        stats["n_professions_in_prompt"] = n_professions

        results.append(stats)

        print(f"\n{dataset}")
        print(f"Profession: {profession}")
        print(f"Pilot n: {stats['n_pilot']}")

        if pd.notna(stats["p_male"]):
            print(f"p_male: {stats['p_male']:.3f}")
            print(f"p_female: {stats['p_female']:.3f}")
            print(f"p_neutral: {stats['p_neutral']:.3f}")
            print(f"theta_hat: {stats['theta_hat']:.3f}")
            print(f"sigma2_hat: {stats['sigma2_hat']:.3f}")
            print(f"Required n: {int(stats['n_required_ceil'])}")
        else:
            print("p_male: NA")
            print("p_female: NA")
            print("p_neutral: NA")
            print("theta_hat: NA")
            print("sigma2_hat: NA")
            print("Required n: NA")

    results_df = pd.DataFrame(results)

    valid_required = results_df["n_required_ceil"].dropna()

    if len(valid_required) > 0:
        upper_bound_n_for_prompt = int(valid_required.max())
    else:
        upper_bound_n_for_prompt = np.nan

    results_df["upper_bound_n_for_prompt"] = upper_bound_n_for_prompt

    print("\n-----------------------------")
    print(f"Upper bound n for {language} / {prompt_name}: {upper_bound_n_for_prompt}")
    print("-----------------------------")

    output_path = OUTPUT_DIR / f"sample_size_{language}_{prompt_name}.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Saved prompt results to: {output_path}")

    return results_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print("Sample-size analysis")
    print(f"Selected language: {SELECTED_LANGUAGE}")
    print(f"Project directory: {DIR_PATH}")

    all_results = []

    for suffix in PROMPT_SUFFIXES:
        prompt_name = get_prompt_name_from_suffix(suffix)
        folder_path = get_results_folder(SELECTED_LANGUAGE, suffix)

        prompt_results = analyse_prompt_folder(
            folder_path=folder_path,
            language=SELECTED_LANGUAGE,
            prompt_name=prompt_name,
        )

        if not prompt_results.empty:
            all_results.append(prompt_results)

    if not all_results:
        raise RuntimeError(
            f"No valid profession CSV files found for language: {SELECTED_LANGUAGE}"
        )

    combined_df = pd.concat(all_results, ignore_index=True)

    combined_output_path = OUTPUT_DIR / f"sample_size_{SELECTED_LANGUAGE}_all_prompts.csv"
    combined_df.to_csv(combined_output_path, index=False)

    print("\n=============================")
    print("Done.")
    print(f"Combined results saved to: {combined_output_path}")
    print("=============================")

    print("\nUpper bound by prompt:")
    upper_bounds = (
        combined_df[["language", "prompt", "upper_bound_n_for_prompt"]]
        .drop_duplicates()
        .sort_values(["language", "prompt"])
    )

    print(upper_bounds.to_string(index=False))


if __name__ == "__main__":
    main()