"""
Wald Z-test for a gender-coded CSV file, computed PER ROW.

Input: one CSV with columns (names matched case-insensitively, whitespace-trimmed):
    nr., Language, Profession, Prompt Variant, F, M, N, U

Each row represents one index (e.g. one Language/Profession/Prompt Variant
combo) with F/M/N/U as COUNTS out of ~245 total tests for that row.

Coding:
    M (male)    -> +1
    F (female)  -> -1
    N (neutral) ->  0
    U (unknown) -> dropped entirely (not counted in n, not counted in sum)

For each row:
    n      = F + M + N   (U excluded)
    mean   = (M*1 + F*-1 + N*0) / n  =  (M - F) / n
    var    = sample variance (ddof=1) of the underlying +1/-1/0 values,
             derived analytically from the F/M/N proportions:
                 p_M = M/n, p_F = F/n, p_N = N/n
                 E[X]    = p_M - p_F                (since N contributes 0)
                 E[X^2]  = p_M + p_F                (N^2 term is 0)
                 pop_var = E[X^2] - E[X]^2
                 sample_var = pop_var * n / (n-1)    (unbiased estimate)
    SE     = sqrt(sample_var / n)
    Z      = mean / SE

CONTINUITY CORRECTION (zero-variance rows) --- J:
    If a row has zero variance in the raw counts -- e.g. all 245 observations
    fell in one category (245 M / 0 F / 0 N) -- the standard error is 0 and
    Z would be mathematically undefined (division by zero). To still produce
    a finite, very large Z for these rows, a small continuity correction of
    0.5 is applied ONLY to such rows:
        - all-M row:  M -= 0.5, F += 0.5
        - all-F row:  F -= 0.5, M += 0.5
        - all-N row:  N -= 0.5, M += 0.25, F += 0.25  (symmetric, gives Z=0)
    Rows that already have nonzero variance are left completely untouched.
    Rows where this correction was applied are flagged in the terminal output
    (stderr) so you can see exactly which rows were affected.

Output: the ORIGINAL file, with every original column preserved exactly,
plus one new column "Z" appended at the end. "Z" is only left blank when
n=0 or n=1 (genuinely no data to test), which should be rare to nonexistent
given your counts sum to ~245 per row.

Usage:
    python wald_test_per_row.py input.csv --out output.csv

If --out is omitted, writes to "<input_name>_with_z.csv" alongside the input.
"""

import argparse
import csv
import math
import os
import sys


def normalize_header(name: str) -> str:
    return name.strip().strip(".").lower()


def find_column(fieldnames, target_names):
    norm_map = {normalize_header(f): f for f in fieldnames}
    for target in target_names:
        if target in norm_map:
            return norm_map[target]
    return None


def parse_num(raw):
    raw = (raw or "").strip()
    if raw == "":
        return 0.0
    return float(raw)


CONTINUITY_CORRECTION = 0.5  # nudge applied only to zero-variance rows


def compute_row_z(f_count, m_count, n_count):
    """
    Returns (z, mean, variance, se, note, corrected) for one row.
    z/mean/variance/se are None if undefined; note explains why.
    corrected=True if a continuity correction was applied because the
    raw counts gave zero variance (e.g. all observations in one category).
    """
    n = f_count + m_count + n_count

    if n == 0:
        return None, None, None, None, "n=0 (no F/M/N after excluding U)", False

    if n <= 1:
        p_m = m_count / n
        p_f = f_count / n
        mean = p_m - p_f
        return None, mean, None, None, "n<=1: sample variance undefined", False

    p_m = m_count / n
    p_f = f_count / n
    mean = p_m - p_f
    e_x2 = p_m + p_f
    pop_var = e_x2 - mean ** 2
    sample_var = pop_var * n / (n - 1)

    corrected = False
    if sample_var <= 0:
        # Zero variance: every observation fell in one category (e.g. all M,
        # all F, or all N). Apply a small continuity correction by nudging
        # the dominant category down by 0.5 and the next-largest category
        # (whichever was at 0) up by 0.5, so there's a sliver of spread to
        # compute a finite (very large) Z from, instead of an undefined one.
        corrected = True
        f_c, m_c, n_c = f_count, m_count, n_count

        if m_c == n:  # all male
            m_c -= CONTINUITY_CORRECTION
            f_c += CONTINUITY_CORRECTION
        elif f_c == n:  # all female
            f_c -= CONTINUITY_CORRECTION
            m_c += CONTINUITY_CORRECTION
        elif n_c == n:  # all neutral -> no inherent direction; split evenly
            n_c -= CONTINUITY_CORRECTION
            m_c += CONTINUITY_CORRECTION / 2
            f_c += CONTINUITY_CORRECTION / 2
        # (n_c == n case still yields mean 0 by symmetry, but now has
        # nonzero variance so Z = 0 / SE = 0.0 instead of blank.)

        p_m = m_c / n
        p_f = f_c / n
        mean = p_m - p_f
        e_x2 = p_m + p_f
        pop_var = e_x2 - mean ** 2
        sample_var = pop_var * n / (n - 1)

    se = math.sqrt(sample_var / n)
    z = mean / se
    note = "continuity correction applied (zero variance in raw counts)" if corrected else ""
    return z, mean, sample_var, se, note, corrected


def main():
    parser = argparse.ArgumentParser(
        description="Append a per-row Wald Z column to a gender-coded CSV (M=+1, F=-1, N=0, U excluded)."
    )
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: '<input_name>_with_z.csv')",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_with_z{ext or '.csv'}"

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Error: input file has no header row / appears empty", file=sys.stderr)
            sys.exit(1)

        col_f = find_column(reader.fieldnames, ["f", "female"])
        col_m = find_column(reader.fieldnames, ["m", "male"])
        col_n = find_column(reader.fieldnames, ["n", "neutral"])

        missing = [name for name, col in
                   [("F", col_f), ("M", col_m), ("N", col_n)] if col is None]
        if missing:
            print(
                f"Error: could not find column(s) {missing} in header {reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)

        original_fieldnames = list(reader.fieldnames)
        out_fieldnames = original_fieldnames + ["Z"]

        rows_out = []
        n_ok = 0
        n_blank = 0
        n_corrected = 0

        for i, row in enumerate(reader, start=1):
            try:
                f_count = parse_num(row.get(col_f))
                m_count = parse_num(row.get(col_m))
                n_count = parse_num(row.get(col_n))
            except ValueError as e:
                print(f"Row {i}: ERROR parsing F/M/N as numbers - {e}", file=sys.stderr)
                row["Z"] = ""
                rows_out.append(row)
                n_blank += 1
                continue

            z, mean, var, se, note, corrected = compute_row_z(f_count, m_count, n_count)

            row_out = dict(row)  # preserve all original columns/values as-is
            row_out["Z"] = f"{z:.6f}" if z is not None else ""
            rows_out.append(row_out)

            if z is not None:
                n_ok += 1
                if corrected:
                    n_corrected += 1
                    print(f"Row {i}: {note} -> Z={z:.4f}", file=sys.stderr)
            else:
                n_blank += 1
                print(f"Row {i}: Z left blank - {note}", file=sys.stderr)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nProcessed {len(rows_out)} row(s): {n_ok} with Z computed "
          f"({n_corrected} via continuity correction), {n_blank} left blank.")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()