"""Inference utilities for infrastructure log classification.

Step 4 of the project pipeline:
- Load saved model artifacts
- Preprocess input logs
- Return prediction label and confidence
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib

from app.model.preprocessing import clean_log_text


DEFAULT_ARTIFACT_DIR = Path("app/model")
VECTORIZER_PATH = DEFAULT_ARTIFACT_DIR / "vectorizer.pkl"
CLASSIFIER_PATH = DEFAULT_ARTIFACT_DIR / "classifier.pkl"


@lru_cache(maxsize=1)
def load_artifacts(
    vectorizer_path: Path = VECTORIZER_PATH,
    classifier_path: Path = CLASSIFIER_PATH,
):
    """Load and cache vectorizer and classifier artifacts."""
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found: {vectorizer_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier artifact not found: {classifier_path}")

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    return vectorizer, classifier


def predict_log(log_text: str) -> Dict[str, object]:
    """Predict failure category and confidence for one log message."""
    if not isinstance(log_text, str) or not log_text.strip():
        raise ValueError("log_text must be a non-empty string")

    vectorizer, classifier = load_artifacts()

    clean_text = clean_log_text(log_text)
    features = vectorizer.transform([clean_text])

    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    confidence = float(probabilities.max())

    return {
        "raw_log": log_text,
        "clean_log": clean_text,
        "prediction": prediction,
        "confidence": confidence,
    }


def predict_logs(log_texts: List[str]) -> List[Dict[str, object]]:
    """Predict failure categories for a list of logs."""
    if not isinstance(log_texts, list) or not log_texts:
        raise ValueError("log_texts must be a non-empty list")

    results = []
    for log_text in log_texts:
        results.append(predict_log(log_text))
    return results
