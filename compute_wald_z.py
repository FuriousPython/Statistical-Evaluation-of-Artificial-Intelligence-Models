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
 
Output: the ORIGINAL file, with every original column preserved exactly,
plus one new column "Z" appended at the end. If Z can't be computed for a
row (n=0, n=1, or zero variance e.g. all-N row), "Z" is left blank.
 
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
 
 
def compute_row_z(f_count, m_count, n_count):
    """
    Returns (z, mean, variance, se, note) for one row.
    z/mean/variance/se are None if undefined; note explains why.
    """
    n = f_count + m_count + n_count
 
    if n == 0:
        return None, None, None, None, "n=0 (no F/M/N after excluding U)"
 
    p_m = m_count / n
    p_f = f_count / n
 
    mean = p_m - p_f
    e_x2 = p_m + p_f
    pop_var = e_x2 - mean ** 2
 
    if n <= 1:
        return None, mean, None, None, "n<=1: sample variance undefined"
 
    sample_var = pop_var * n / (n - 1)
 
    if sample_var <= 0:
        return None, mean, sample_var, None, "variance is 0 (no spread) - Z undefined"
 
    se = math.sqrt(sample_var / n)
    z = mean / se
    return z, mean, sample_var, se, ""
 
 
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
 
            z, mean, var, se, note = compute_row_z(f_count, m_count, n_count)
 
            row_out = dict(row)  # preserve all original columns/values as-is
            row_out["Z"] = f"{z:.6f}" if z is not None else ""
            rows_out.append(row_out)
 
            if z is not None:
                n_ok += 1
            else:
                n_blank += 1
                print(f"Row {i}: Z left blank - {note}", file=sys.stderr)
 
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
 
    print(f"\nProcessed {len(rows_out)} row(s): {n_ok} with Z computed, {n_blank} left blank.")
    print(f"Output written to: {out_path}")
 
 
if __name__ == "__main__":
    main()