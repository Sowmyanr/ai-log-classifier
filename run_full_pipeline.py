"""
Orchestrator script: Ingest, preprocess, retrain, and save model artifacts in one go.

Usage:
    python run_full_pipeline.py

- Scans data/raw/ for all .log files
- Preprocesses and combines into data/processed/all_logs_processed.csv
- Retrains model and saves artifacts
"""

import os
import glob
import pandas as pd
import subprocess
import sys
from pathlib import Path
from app.model import preprocessing

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/all_logs_processed.csv")
ARTIFACT_DIR = Path("app/model")

# 1. Ingest and preprocess all raw logs
def ingest_and_preprocess():
    log_files = glob.glob(str(RAW_DIR / "*.log"))
    all_lines = []
    for file in log_files:
        with open(file, encoding="utf-8", errors="ignore") as f:
            all_lines.extend(f.readlines())
    df = preprocessing.build_training_dataframe(all_lines)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Preprocessed {len(df)} logs from {len(log_files)} files → {PROCESSED_PATH}")
    return PROCESSED_PATH

# 2. Retrain model
def retrain_model(processed_path):
    cmd = [
        sys.executable, "app/model/train.py",
        "--dataset", str(processed_path),
        "--artifact-dir", str(ARTIFACT_DIR),
        "--ngram-min", "1",
        "--ngram-max", "3",
        "--min-df", "1",
        "--max-df", "0.9",
        "--classifier", "rf"
    ]
    print("Retraining model with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    processed_path = ingest_and_preprocess()
    retrain_model(processed_path)
    print("Pipeline complete. Artifacts updated.")
