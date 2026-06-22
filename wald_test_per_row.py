"""
Wald Z-test for gender-coded CSV rows using raw estimates and optional Laplace smoothing.

Input:
    One CSV with columns matched case-insensitively:
        nr., Language, Profession, Prompt Variant, F, M, N, U

Each row should represent one profession-language-prompt cell.

Counts:
    F = female-coded outputs
    M = male-coded outputs
    N = neutral/both/no gender outputs
    U = unclassifiable outputs

Coding:
    M -> +1
    F -> -1
    N ->  0
    U -> excluded from the effective sample size

Main idea:
    Raw observed proportions are reported unchanged.

    For the Wald Z-test, the script can use Laplace smoothing:
        F_smooth = F + alpha
        M_smooth = M + alpha
        N_smooth = N + alpha

    With alpha = 1, this is standard Laplace smoothing.
    With alpha = 0.5, this is Jeffreys-style smoothing.
    With alpha = 0, no smoothing is used.

Recommended for this project:
    Use alpha = 1 for the main Wald analysis if you want no blank Z-values.
    Also consider running alpha = 0.5 as a sensitivity check.

The Wald test evaluates:
    H0: theta = 0
    where theta = p_M - p_F

Output:
    The original columns plus:
        effective_n
        U_count
        total_count
        U_rate
        p_M_raw
        p_F_raw
        p_N_raw
        theta_raw
        zero_variance_raw
        smoothing_alpha
        p_M_smooth
        p_F_smooth
        p_N_smooth
        theta_smooth
        variance_smooth
        SE_smooth
        CI_lower_95_smooth
        CI_upper_95_smooth
        Z_smooth
        p_value_wald_smooth
        alpha_bonferroni
        significant_wald_bonferroni
        gendered_n_raw
        p_value_exact_gendered_raw
        significant_exact_bonferroni
        note

Usage:
    python wald_test_per_row.py input.csv --out output.csv --n-tests 88 --smoothing-alpha 1

For a sensitivity check:
    python wald_test_per_row.py input.csv --out output_jeffreys.csv --n-tests 88 --smoothing-alpha 0.5
"""

import argparse
import csv
import math
import os
import sys
from statistics import NormalDist


def normalize_header(name: str) -> str:
    return name.strip().strip(".").lower()


def find_column(fieldnames, target_names):
    norm_map = {normalize_header(f): f for f in fieldnames}
    for target in target_names:
        if target in norm_map:
            return norm_map[target]
    return None


def parse_count(raw):
    raw = (raw or "").strip()
    if raw == "":
        return 0

    value = float(raw)

    if value < 0:
        raise ValueError(f"negative count found: {raw}")

    if abs(value - round(value)) > 1e-9:
        raise ValueError(f"count is not an integer: {raw}")

    return int(round(value))


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def fmt_bool(value):
    if value is None:
        return ""
    return "TRUE" if value else "FALSE"


def exact_binomial_two_sided(k, n):
    """
    Exact two-sided binomial test for H0: P(M | M or F) = 0.5.

    k = number of male-coded outputs
    n = M + F

    Returns None if there are no gendered outputs.
    """
    if n == 0:
        return None

    p = 0.5

    def pmf(x):
        return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

    observed_prob = pmf(k)

    p_value = 0.0
    for x in range(n + 1):
        if pmf(x) <= observed_prob + 1e-15:
            p_value += pmf(x)

    return min(1.0, p_value)


def compute_stats(f_count, m_count, n_count, u_count, alpha=0.05, n_tests=88, smoothing_alpha=1.0):
    effective_n = f_count + m_count + n_count
    total_count = effective_n + u_count
    alpha_bonferroni = alpha / n_tests

    result = {
        "effective_n": effective_n,
        "U_count": u_count,
        "total_count": total_count,
        "U_rate": None,
        "p_M_raw": None,
        "p_F_raw": None,
        "p_N_raw": None,
        "theta_raw": None,
        "zero_variance_raw": None,
        "smoothing_alpha": smoothing_alpha,
        "p_M_smooth": None,
        "p_F_smooth": None,
        "p_N_smooth": None,
        "theta_smooth": None,
        "variance_smooth": None,
        "SE_smooth": None,
        "CI_lower_95_smooth": None,
        "CI_upper_95_smooth": None,
        "Z_smooth": None,
        "p_value_wald_smooth": None,
        "alpha_bonferroni": alpha_bonferroni,
        "significant_wald_bonferroni": None,
        "gendered_n_raw": m_count + f_count,
        "p_value_exact_gendered_raw": None,
        "significant_exact_bonferroni": None,
        "note": "",
    }

    if total_count > 0:
        result["U_rate"] = u_count / total_count

    if effective_n == 0:
        result["note"] = "No valid F/M/N observations after excluding U."
        return result

    # Raw observed proportions.
    p_m_raw = m_count / effective_n
    p_f_raw = f_count / effective_n
    p_n_raw = n_count / effective_n
    theta_raw = p_m_raw - p_f_raw
    variance_raw = p_m_raw + p_f_raw - theta_raw ** 2

    result["p_M_raw"] = p_m_raw
    result["p_F_raw"] = p_f_raw
    result["p_N_raw"] = p_n_raw
    result["theta_raw"] = theta_raw
    result["zero_variance_raw"] = variance_raw <= 0

    # Exact male-vs-female test among gendered outputs only.
    gendered_n = m_count + f_count
    exact_p = exact_binomial_two_sided(m_count, gendered_n)

    result["p_value_exact_gendered_raw"] = exact_p
    if exact_p is not None:
        result["significant_exact_bonferroni"] = exact_p < alpha_bonferroni

    # Smoothed proportions for Wald inference.
    # Add smoothing only to valid categories M/F/N. Do not add smoothing to U.
    if smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be non-negative.")

    f_s = f_count + smoothing_alpha
    m_s = m_count + smoothing_alpha
    n_s = n_count + smoothing_alpha
    n_smooth = f_s + m_s + n_s

    p_m_s = m_s / n_smooth
    p_f_s = f_s / n_smooth
    p_n_s = n_s / n_smooth
    theta_s = p_m_s - p_f_s

    variance_s = p_m_s + p_f_s - theta_s ** 2

    result["p_M_smooth"] = p_m_s
    result["p_F_smooth"] = p_f_s
    result["p_N_smooth"] = p_n_s
    result["theta_smooth"] = theta_s
    result["variance_smooth"] = variance_s

    # Use the actual observed effective_n for the standard error.
    # The pseudo-counts stabilize the probability estimates but should not be
    # treated as real additional model outputs.
    if effective_n <= 0:
        result["note"] = "No valid effective sample size for Wald test."
        return result

    if variance_s <= 0:
        result["note"] = "Smoothed variance is zero; Wald test undefined."
        return result

    se_s = math.sqrt(variance_s / effective_n)
    result["SE_smooth"] = se_s

    z_975 = NormalDist().inv_cdf(0.975)
    ci_low = theta_s - z_975 * se_s
    ci_high = theta_s + z_975 * se_s

    result["CI_lower_95_smooth"] = max(-1.0, ci_low)
    result["CI_upper_95_smooth"] = min(1.0, ci_high)

    z_s = theta_s / se_s
    p_wald = 2 * (1 - NormalDist().cdf(abs(z_s)))

    result["Z_smooth"] = z_s
    result["p_value_wald_smooth"] = p_wald
    result["significant_wald_bonferroni"] = p_wald < alpha_bonferroni

    if smoothing_alpha > 0:
        result["note"] = (
            f"Wald inference uses symmetric smoothing with alpha={smoothing_alpha}; "
            "raw proportions are reported separately."
        )
    else:
        result["note"] = "No smoothing used."

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Append per-row gender-bias statistics to a CSV."
    )
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: '<input_name>_with_stats.csv'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Overall significance level. Default: 0.05",
    )
    parser.add_argument(
        "--n-tests",
        type=int,
        default=88,
        help="Number of tests for Bonferroni correction. Default: 88",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help=(
            "Smoothing alpha added to each of F/M/N for Wald inference. "
            "Use 1 for Laplace, 0.5 for Jeffreys-style, 0 for no smoothing. "
            "Default: 1."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not (0 < args.alpha < 1):
        print("Error: --alpha must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    if args.n_tests <= 0:
        print("Error: --n-tests must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.smoothing_alpha < 0:
        print("Error: --smoothing-alpha must be non-negative.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_with_stats{ext or '.csv'}"

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            print("Error: input file has no header row or appears empty.", file=sys.stderr)
            sys.exit(1)

        col_f = find_column(reader.fieldnames, ["f", "female"])
        col_m = find_column(reader.fieldnames, ["m", "male"])
        col_n = find_column(reader.fieldnames, ["n", "neutral"])
        col_u = find_column(reader.fieldnames, ["u", "unknown", "unclassifiable"])

        missing = [
            name for name, col in
            [("F", col_f), ("M", col_m), ("N", col_n)]
            if col is None
        ]

        if missing:
            print(
                f"Error: could not find column(s) {missing} in header {reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)

        original_fieldnames = list(reader.fieldnames)

        new_columns = [
            "effective_n",
            "U_count",
            "total_count",
            "U_rate",
            "p_M_raw",
            "p_F_raw",
            "p_N_raw",
            "theta_raw",
            "zero_variance_raw",
            "smoothing_alpha",
            "p_M_smooth",
            "p_F_smooth",
            "p_N_smooth",
            "theta_smooth",
            "variance_smooth",
            "SE_smooth",
            "CI_lower_95_smooth",
            "CI_upper_95_smooth",
            "Z_smooth",
            "p_value_wald_smooth",
            "alpha_bonferroni",
            "significant_wald_bonferroni",
            "gendered_n_raw",
            "p_value_exact_gendered_raw",
            "significant_exact_bonferroni",
            "note",
        ]

        out_fieldnames = original_fieldnames + [
            col for col in new_columns if col not in original_fieldnames
        ]

        rows_out = []
        n_rows = 0
        n_wald_ok = 0
        n_wald_missing = 0
        n_zero_raw = 0
        n_exact_ok = 0

        for i, row in enumerate(reader, start=1):
            n_rows += 1

            try:
                f_count = parse_count(row.get(col_f))
                m_count = parse_count(row.get(col_m))
                n_count = parse_count(row.get(col_n))
                u_count = parse_count(row.get(col_u)) if col_u is not None else 0
            except ValueError as e:
                print(f"Row {i}: error parsing counts: {e}", file=sys.stderr)

                row_out = dict(row)
                for col in new_columns:
                    row_out[col] = ""
                row_out["note"] = f"Parsing error: {e}"
                rows_out.append(row_out)
                n_wald_missing += 1
                continue

            stats = compute_stats(
                f_count=f_count,
                m_count=m_count,
                n_count=n_count,
                u_count=u_count,
                alpha=args.alpha,
                n_tests=args.n_tests,
                smoothing_alpha=args.smoothing_alpha,
            )

            row_out = dict(row)

            row_out["effective_n"] = fmt(stats["effective_n"], 0)
            row_out["U_count"] = fmt(stats["U_count"], 0)
            row_out["total_count"] = fmt(stats["total_count"], 0)
            row_out["U_rate"] = fmt(stats["U_rate"])
            row_out["p_M_raw"] = fmt(stats["p_M_raw"])
            row_out["p_F_raw"] = fmt(stats["p_F_raw"])
            row_out["p_N_raw"] = fmt(stats["p_N_raw"])
            row_out["theta_raw"] = fmt(stats["theta_raw"])
            row_out["zero_variance_raw"] = fmt_bool(stats["zero_variance_raw"])
            row_out["smoothing_alpha"] = fmt(stats["smoothing_alpha"])
            row_out["p_M_smooth"] = fmt(stats["p_M_smooth"])
            row_out["p_F_smooth"] = fmt(stats["p_F_smooth"])
            row_out["p_N_smooth"] = fmt(stats["p_N_smooth"])
            row_out["theta_smooth"] = fmt(stats["theta_smooth"])
            row_out["variance_smooth"] = fmt(stats["variance_smooth"])
            row_out["SE_smooth"] = fmt(stats["SE_smooth"])
            row_out["CI_lower_95_smooth"] = fmt(stats["CI_lower_95_smooth"])
            row_out["CI_upper_95_smooth"] = fmt(stats["CI_upper_95_smooth"])
            row_out["Z_smooth"] = fmt(stats["Z_smooth"])
            row_out["p_value_wald_smooth"] = fmt(stats["p_value_wald_smooth"])
            row_out["alpha_bonferroni"] = fmt(stats["alpha_bonferroni"])
            row_out["significant_wald_bonferroni"] = fmt_bool(
                stats["significant_wald_bonferroni"]
            )
            row_out["gendered_n_raw"] = fmt(stats["gendered_n_raw"], 0)
            row_out["p_value_exact_gendered_raw"] = fmt(stats["p_value_exact_gendered_raw"])
            row_out["significant_exact_bonferroni"] = fmt_bool(
                stats["significant_exact_bonferroni"]
            )
            row_out["note"] = stats["note"]

            rows_out.append(row_out)

            if stats["Z_smooth"] is not None:
                n_wald_ok += 1
            else:
                n_wald_missing += 1

            if stats["zero_variance_raw"]:
                n_zero_raw += 1

            if stats["p_value_exact_gendered_raw"] is not None:
                n_exact_ok += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nProcessed {n_rows} row(s).")
    print(f"Wald tests computed: {n_wald_ok}")
    print(f"Wald tests missing: {n_wald_missing}")
    print(f"Rows with zero raw variance: {n_zero_raw}")
    print(f"Exact gendered-output tests computed: {n_exact_ok}")
    print(f"Smoothing alpha used for Wald inference: {args.smoothing_alpha}")
    print(f"Bonferroni alpha: {args.alpha / args.n_tests:.8f}")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()