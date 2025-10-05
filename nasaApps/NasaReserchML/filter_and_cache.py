#!/usr/bin/env python3
"""
Filter a CSV, save results to output, and cache to a .pkl.

Examples:
  # Equality filter
  python filter_and_cache.py data.csv --equals status=active --out filtered.csv

  # Contains (substring) filter (case-insensitive)
  python filter_and_cache.py data.csv --contains name=val --out filtered.json

  # Numeric range filter
  python filter_and_cache.py data.csv --ge score=70 --lt score=90 --out midrange.csv

  # Pandas query string (power users)
  python filter_and_cache.py data.csv --query "age >= 18 and country == 'US'"

  # Select columns
  python filter_and_cache.py data.csv --equals role=engineer --columns name,email,team
"""

import argparse
import os
import sys
import pickle
from datetime import datetime

import pandas as pd


def parse_kv_list(items):
    """Parse ['col=val','a=b'] -> [('col','val'),('a','b')]."""
    pairs = []
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Invalid filter '{item}'. Use key=value.")
        k, v = item.split('=', 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"Invalid filter '{item}': empty key.")
        pairs.append((k, v))
    return pairs


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df

    # Equals
    for col, val in parse_kv_list(args.equals):
        if col not in out.columns:
            raise KeyError(f"--equals: column '{col}' not found.")
        out = out[out[col].astype(str) == str(val)]

    # Contains (case-insensitive substring)
    for col, val in parse_kv_list(args.contains):
        if col not in out.columns:
            raise KeyError(f"--contains: column '{col}' not found.")
        out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]

    # Greater-equal / Greater-than
    for col, val in parse_kv_list(args.ge):
        if col not in out.columns:
            raise KeyError(f"--ge: column '{col}' not found.")
        out = out[pd.to_numeric(out[col], errors='coerce') >= float(val)]

    for col, val in parse_kv_list(args.gt):
        if col not in out.columns:
            raise KeyError(f"--gt: column '{col}' not found.")
        out = out[pd.to_numeric(out[col], errors='coerce') > float(val)]

    # Less-equal / Less-than
    for col, val in parse_kv_list(args.le):
        if col not in out.columns:
            raise KeyError(f"--le: column '{col}' not found.")
        out = out[pd.to_numeric(out[col], errors='coerce') <= float(val)]

    for col, val in parse_kv_list(args.lt):
        if col not in out.columns:
            raise KeyError(f"--lt: column '{col}' not found.")
        out = out[pd.to_numeric(out[col], errors='coerce') < float(val)]

    # Pandas query (advanced; runs last)
    if args.query:
        try:
            out = out.query(args.query, engine="python")
        except Exception as e:
            raise ValueError(f"--query error: {e}")

    # Column selection (after filtering)
    if args.columns:
        cols = [c.strip() for c in args.columns.split(',') if c.strip()]
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise KeyError(f"--columns missing: {missing}")
        out = out[cols]

    return out


def write_output(df: pd.DataFrame, out_path: str):
    if out_path is None:
        return
    ext = os.path.splitext(out_path)[1].lower()
    if ext in ('.csv', ''):
        df.to_csv(out_path, index=False)
    elif ext in ('.json',):
        df.to_json(out_path, orient='records', lines=False)
    else:
        raise ValueError(f"Unsupported output extension '{ext}'. Use .csv or .json.")


def write_cache(df: pd.DataFrame, cache_path: str, src_csv: str, args: argparse.Namespace):
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_csv": os.path.abspath(src_csv),
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "columns": list(df.columns),
        "filters": {
            "equals": args.equals,
            "contains": args.contains,
            "ge": args.ge,
            "gt": args.gt,
            "le": args.le,
            "lt": args.lt,
            "query": args.query,
            "columns_select": args.columns,
        },
        # Keep the actual filtered data
        "dataframe": df
    }
    # Protocol 4 for broad compatibility (Python 3.4+)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)


def main():
    p = argparse.ArgumentParser(description="Filter CSV -> cache.pkl + output.")
    p.add_argument("csv", help="Input CSV file path.")
    p.add_argument("--out", help="Output file (.csv or .json). If omitted, no file is written.", default=None)
    p.add_argument("--cache", help="Path to cache pickle.", default="cache.pkl")

    # Filters
    p.add_argument("--equals", nargs="*", default=[], help="Exact match: col=val (repeatable).")
    p.add_argument("--contains", nargs="*", default=[], help="Substring (case-insensitive): col=substr (repeatable).")
    p.add_argument("--ge", nargs="*", default=[], help="Numeric >= : col=value (repeatable).")
    p.add_argument("--gt", nargs="*", default=[], help="Numeric >  : col=value (repeatable).")
    p.add_argument("--le", nargs="*", default=[], help="Numeric <= : col=value (repeatable).")
    p.add_argument("--lt", nargs="*", default=[], help="Numeric <  : col=value (repeatable).")
    p.add_argument("--query", help="Pandas query string, e.g., \"age >= 18 and country == 'US'\".")
    p.add_argument("--columns", help="Comma-separated columns to keep after filtering.")

    # CSV read options
    p.add_argument("--sep", default=",", help="CSV delimiter (default ',').")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8).")
    p.add_argument("--na", nargs="*", default=None, help="Strings to treat as NA.")

    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: input CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv, sep=args.sep, encoding=args.encoding, na_values=args.na)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        filtered = apply_filters(df, args)
    except Exception as e:
        print(f"Filter error: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        write_output(filtered, args.out)
    except Exception as e:
        print(f"Output error: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        write_cache(filtered, args.cache, args.csv, args)
    except Exception as e:
        print(f"Cache write error: {e}", file=sys.stderr)
        sys.exit(4)

    # Final summary
    print(f"Rows in: {df.shape[0]}, rows out: {filtered.shape[0]}")
    if args.out:
        print(f"Wrote output -> {os.path.abspath(args.out)}")
    print(f"Wrote cache  -> {os.path.abspath(args.cache)}")


if __name__ == "__main__":
    main()
