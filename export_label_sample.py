"""
Export a sample of logs with weak labels for manual review and correction.

Usage:
    python export_label_sample.py processed_dataset.csv sample_for_review.csv --n 100
"""
import sys
import pandas as pd
import random

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_label_sample.py processed_dataset.csv sample_for_review.csv [--n N]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n = 100
    if len(sys.argv) == 5 and sys.argv[3] == "--n":
        n = int(sys.argv[4])
    df = pd.read_csv(input_path)
    if len(df) > n:
        df_sample = df.sample(n, random_state=42)
    else:
        df_sample = df
    df_sample.to_csv(output_path, index=False)
    print(f"Exported {len(df_sample)} rows to {output_path} for manual review.")
