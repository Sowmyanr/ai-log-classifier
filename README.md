# AI Infrastructure Log Diagnostic Classifier

A production-ready, Dockerized ML pipeline for classifying infrastructure/system logs into failure categories via a FastAPI endpoint.

## Features
- FastAPI backend
- Scikit-learn ML pipeline (TF-IDF + Logistic Regression)
- Dockerized for easy deployment
- Ready for cloud and MLOps upgrades

## Project Structure

```
ai-log-classifier/
├── app/
│   ├── main.py
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── preprocessing.py
│   │   ├── vectorizer.pkl
│   │   └── classifier.pkl
│   └── routes/
│       └── predict.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Quickstart

1. Clone repo & create virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Download HDFS log dataset to `data/raw/`
4. Run preprocessing, training, and start API
5. Use Docker for deployment

## API Example

Request:
```json
{
  "log": "ERROR: DataNode failed to connect"
}
```
Response:
```json
{
  "prediction": "Network Failure",
  "confidence": 0.91
}
```

## Running the API Locally

Start the FastAPI server (from project root):

```
uvicorn app.main:app --reload --port 8000
```

Health check:

```
curl http://127.0.0.1:8000/healthz
```

Predict endpoint (curl):

```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"log": "ERROR: DataNode failed to connect to NameNode due to socket timeout"}'
```

Predict endpoint (Python):

```python
import requests
resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"log": "ERROR: DataNode failed to connect to NameNode due to socket timeout"}
)
print(resp.json())
```

Interactive docs:
- Open http://127.0.0.1:8000/docs for Swagger UI
- Open http://127.0.0.1:8000/redoc for ReDoc
```
