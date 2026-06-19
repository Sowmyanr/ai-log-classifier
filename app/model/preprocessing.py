"""Preprocessing and weak-label generation for HDFS infrastructure logs.

Step 1 (labels): define a repeatable, keyword-based label strategy so the
project can bootstrap supervised training when hand-labeled data is absent.

Step 2 (preprocessing): clean and normalize raw log lines into model-ready text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


# Regexes compiled once for performance and consistency.
IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
BLOCK_ID_PATTERN = re.compile(r"\bblk_[\-\d]+\b")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
MULTISPACE_PATTERN = re.compile(r"\s+")


# Initial taxonomy for weak supervision.
# You can refine this once you introduce manually verified labels.
LABEL_KEYWORDS: Dict[str, List[str]] = {
	"Network Failure": [
		"failed to connect",
		"connection refused",
		"network",
		"timed out",
		"connection reset",
		"socket",
		"no route to host",
	],
	"Disk Failure": [
		"disk",
		"ioexception",
		"input/output error",
		"no space left",
		"corrupt",
		"checksum",
		"bad block",
	],
	"Memory Failure": [
		"outofmemory",
		"out of memory",
		"java heap space",
		"gc overhead",
		"memory",
	],
	"Permission Failure": [
		"permission denied",
		"not authorized",
		"access denied",
		"authentication failed",
		"authorization",
	],
	"Node Failure": [
		"datanode",
		"namenode",
		"node failed",
		"node lost",
		"heartbeat",
		"terminated",
	],
}

DEFAULT_LABEL = "Other"


def clean_log_text(text: str) -> str:
	"""Normalize a log line into stable model features."""
	if not isinstance(text, str):
		return ""

	text = text.lower().strip()
	text = IP_PATTERN.sub("<ip>", text)
	text = BLOCK_ID_PATTERN.sub("<block_id>", text)
	text = NUMBER_PATTERN.sub("<num>", text)
	text = MULTISPACE_PATTERN.sub(" ", text)
	return text


def infer_label(clean_text: str) -> str:
	"""Infer a weak label from keyword rules.

	This is a bootstrap strategy for Step 1. For production quality, replace or
	augment with human-validated labels.
	"""
	for label, keywords in LABEL_KEYWORDS.items():
		if any(keyword in clean_text for keyword in keywords):
			return label
	return DEFAULT_LABEL


def build_training_dataframe(log_lines: Iterable[str]) -> pd.DataFrame:
	"""Convert raw log lines into a training-ready dataframe.

	Output schema:
	- raw_log: original log line
	- clean_log: normalized text used for modeling
	- label: weakly inferred class label
	"""
	records = []
	for line in log_lines:
		raw_line = line.strip()
		if not raw_line:
			continue
		clean_line = clean_log_text(raw_line)
		label = infer_label(clean_line)
		records.append({"raw_log": raw_line, "clean_log": clean_line, "label": label})

	return pd.DataFrame(records)


def load_log_file(log_file_path: str | Path) -> List[str]:
	"""Read a raw log file as a list of lines."""
	path = Path(log_file_path)
	if not path.exists():
		raise FileNotFoundError(f"Log file not found: {path}")
	return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def build_dataset_from_file(log_file_path: str | Path) -> pd.DataFrame:
	"""Build training dataframe directly from a raw log file."""
	lines = load_log_file(log_file_path)
	return build_training_dataframe(lines)


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
	"""Persist processed data for downstream model training."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output, index=False)
	return output
