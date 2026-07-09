import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Print basic dataset statistics including multi-function/multi-file fractions.")
    p.add_argument("--input", "-i", required=True, help="Path to JSONL dataset (e.g., FuncFileRepo-MF.jsonl)")
    args = p.parse_args()

    df = pd.read_json(args.input, lines=True)
    n = df.shape[0]
    mv = int((df["vulnerable"] == True).sum())
    mnv = int((df["vulnerable"] == False).sum())
    mf = float(df.get("is_multi_function", False).mean()) if "is_multi_function" in df.columns else 0.0
    mfi = float(df.get("is_multi_file", False).mean()) if "is_multi_file" in df.columns else 0.0
    langs = df["language"].value_counts().to_dict()
    print({
        "samples": n,
        "vulnerable": mv,
        "non_vulnerable": mnv,
        "multi_function_frac": mf,
        "multi_file_frac": mfi,
        "language_dist": langs,
    })


if __name__ == "__main__":
    main()
