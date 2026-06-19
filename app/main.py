"""FastAPI entrypoint for infrastructure log classification API.

Step 5: Expose model inference as a REST endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from app.model.predict import predict_log

app = FastAPI(title="AI Log Classifier", version="0.1.0")


class LogRequest(BaseModel):
    log: str


@app.post("/predict", summary="Predict log failure category")
def predict_endpoint(request: LogRequest):
    """Return predicted failure category and confidence for a log line."""
    try:
        result = predict_log(request.log)
        return JSONResponse(content={
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_log": result["raw_log"],
            "clean_log": result["clean_log"],
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/healthz", summary="Health check")
def healthz():
    return {"status": "ok"}
