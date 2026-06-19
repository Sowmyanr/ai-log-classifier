
import sys
import requests
import json
import time
from typing import List

"""
Batch log classification using the FastAPI endpoint, with error handling.

Usage:
    python batch_predict.py input_logs.txt output_results.json

- input_logs.txt: one log line per row
- output_results.json: list of prediction results
"""

def classify_log(log_text, api_url="http://127.0.0.1:8000/predict", max_retries=3, delay=1, error_log=None):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, json={"log": log_text}, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = str(e)
            if error_log:
                error_log.write(f"Attempt {attempt} failed for log: {log_text}\nError: {last_err}\n")
            if attempt < max_retries:
                time.sleep(delay)
    return {"error": last_err, "raw_log": log_text}


def batch_classify(input_path: str, output_path: str, api_url: str = "http://127.0.0.1:8000/predict", error_log_path: str = "errors.log"):
    with open(input_path, encoding="utf-8") as f:
        logs = [line.strip() for line in f if line.strip()]
    results: List[dict] = []
    with open(error_log_path, "w", encoding="utf-8") as elog:
        for log in logs:
            result = classify_log(log, api_url, error_log=elog)
            results.append(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Processed {len(logs)} logs. Results saved to {output_path}")
    print(f"Errors (if any) logged to {error_log_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python batch_predict.py input_logs.txt output_results.json [errors.log]")
        sys.exit(1)
    error_log_path = sys.argv[3] if len(sys.argv) == 4 else "errors.log"
    batch_classify(sys.argv[1], sys.argv[2], error_log_path=error_log_path)
