"""Train TF-IDF + Logistic Regression for infrastructure log classification.

Step 3 of the project pipeline:
- Load processed dataset
- Train model
- Evaluate model
- Save artifacts for inference/API
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET_PATH = Path("data/processed/hdfs_2k_weak_labeled.csv")
DEFAULT_ARTIFACT_DIR = Path("app/model")
VECTORIZER_FILE = "vectorizer.pkl"
CLASSIFIER_FILE = "classifier.pkl"
METRICS_FILE = "metrics.json"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load processed dataset and validate required columns."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required_columns = {"clean_log", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["clean_log", "label"]).copy()
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    return df


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int):
    """Split data with stratification when feasible."""
    y = df["label"]
    min_class_count = y.value_counts().min()
    stratify_target = y if min_class_count >= 2 else None

    return train_test_split(
        df["clean_log"],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def train_model(x_train, y_train, ngram_range=(1,2), min_df=2, max_df=0.95, classifier_type="logreg") -> tuple[TfidfVectorizer, object, dict]:
    """Train vectorizer and classifier with tuning options."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)

    if classifier_type == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif classifier_type == "rf":
        classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    classifier.fit(x_train_vec, y_train)

    # Cross-validation
    cv_scores = cross_val_score(classifier, x_train_vec, y_train, cv=3, scoring="f1_weighted")
    cv_metrics = {"cv_f1_weighted_mean": float(cv_scores.mean()), "cv_f1_weighted_std": float(cv_scores.std())}

    return vectorizer, classifier, cv_metrics


def evaluate_model(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    x_test,
    y_test,
) -> Dict[str, object]:
    """Compute evaluation metrics and report."""
    x_test_vec = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_vec)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "class_distribution": y_test.value_counts().to_dict(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def save_artifacts(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    metrics: Dict[str, object],
    artifact_dir: Path,
) -> None:
    """Persist model artifacts and metrics to disk."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, artifact_dir / VECTORIZER_FILE)
    joblib.dump(classifier, artifact_dir / CLASSIFIER_FILE)

    metrics_path = artifact_dir / METRICS_FILE
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")



def run_training(
    dataset_path: Path,
    artifact_dir: Path,
    test_size: float,
    random_state: int,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    classifier_type="logreg",
) -> Dict[str, object]:
    """Execute full training workflow and return metrics."""
    df = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df, test_size, random_state)
    vectorizer, classifier, cv_metrics = train_model(x_train, y_train, ngram_range, min_df, max_df, classifier_type)
    metrics = evaluate_model(vectorizer, classifier, x_test, y_test)
    metrics.update(cv_metrics)
    save_artifacts(vectorizer, classifier, metrics, artifact_dir)
    return metrics



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train log classifier model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to processed CSV with clean_log and label columns.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where vectorizer and classifier artifacts are saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset used for evaluation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic split/training.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        choices=["logreg", "rf"],
        help="Classifier type: logreg (LogisticRegression) or rf (RandomForest)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    metrics = run_training(
        dataset_path=args.dataset,
        artifact_dir=args.artifact_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        classifier_type=args.classifier,
    )

    print("Training completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"CV Weighted F1 (mean ± std): {metrics['cv_f1_weighted_mean']:.4f} ± {metrics['cv_f1_weighted_std']:.4f}")
    print(f"Artifacts saved in: {args.artifact_dir}")


if __name__ == "__main__":
    main()
