import numpy as np
import pandas as pd
import os
from scipy.stats import norm

dir_path = os.getcwd()
path = f"{dir_path}/pilot_tests_spanish/spanish_gender_bias_results/per_profession_raw_prompts"


# -----------------------------
# Settings
# -----------------------------

ALPHA = 0.05 # type-1 error (false positives)
POWER = 0.80
BETA = 1 - POWER # type-2 error (failing to detect a real effect.)

N_PROFESSIONS = 16
BONFERRONI = True

# Smallest meaningful bias you want to detect.
# Example: 0.20 means a 20 percentage-point male-vs-female imbalance.
DELTA = 0.20


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep valid gender labels and map:
    M = 1
    F = -1
    N = 0

    U is removed because it is unclassifiable / invalid.
    """
    df = df.copy()

    df = df.loc[df["Gender"] != "U"].reset_index(drop=True)

    df["Gender"] = df["Gender"].map({
        "M": 1,
        "F": -1,
        "N": 0
    })

    # Remove rows where mapping failed, e.g. unexpected labels
    df = df.dropna(subset=["Gender"]).reset_index(drop=True)

    return df


def sample_size(df: pd.DataFrame,
                alpha: float = ALPHA,
                power: float = POWER,
                delta: float = DELTA,
                n_professions: int = N_PROFESSIONS,
                bonferroni: bool = BONFERRONI) -> dict:
    """
    Calculate required sample size for one profession, one model.

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
        raise ValueError("No valid pilot observations after filtering.")

    y = df["Gender"]

    p_male = np.mean(y == 1)
    p_female = np.mean(y == -1)
    p_neutral = np.mean(y == 0)

    theta_hat = p_male - p_female

    sigma2_hat = p_male + p_female - theta_hat**2

    if bonferroni: # avoiding type-1 errors
        alpha_star = alpha / n_professions
    else:
        alpha_star = alpha

    z_alpha = norm.ppf(1 - alpha_star / 2)
    z_beta = norm.ppf(power)

    n_required = ((z_alpha + z_beta) ** 2 * sigma2_hat) / (delta ** 2)

    # Upper integer for this profession
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
        "n_required_ceil": n_required_ceil
    }


# -----------------------------
# Iterate through all pilot tests
# -----------------------------

results = []

for dataset in os.listdir(path):
    if not dataset.endswith(".csv"):
        continue

    file_path = os.path.join(path, dataset)

    df = pd.read_csv(file_path)
    df = normalise(df)

    stats = sample_size(df)

    stats["dataset"] = dataset
    results.append(stats)

    print(f"\n{dataset}")
    print(f"Pilot n: {stats['n_pilot']}")
    print(f"p_male: {stats['p_male']:.3f}")
    print(f"p_female: {stats['p_female']:.3f}")
    print(f"p_neutral: {stats['p_neutral']:.3f}")
    print(f"theta_hat: {stats['theta_hat']:.3f}")
    print(f"sigma2_hat: {stats['sigma2_hat']:.3f}")
    print(f"Required n: {stats['n_required_ceil']}")


# -----------------------------
# Save results
# -----------------------------

results_df = pd.DataFrame(results)

upper_bound_n = int(results_df["n_required_ceil"].max())

print("\n-----------------------------")
print(f"Upper bound n across professions: {upper_bound_n}")
print("-----------------------------")

results_df["upper_bound_n"] = upper_bound_n

output_path = os.path.join(dir_path, "sample_size_results.csv")

