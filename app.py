from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bankflow.tools import analyze_transactions_csv, find_unusual_transactions, train_transaction_categorizer

app = FastAPI(
    title="BankFlow Automator API",
    version="1.0.0",
    description="Portfolio project for automating banking transaction analysis with ML.",
)


class FileRequest(BaseModel):
    file_path: str = Field(default="data/sample_transactions.csv", description="CSV path to analyze")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: FileRequest) -> dict[str, Any]:
    try:
        return analyze_transactions_csv(request.file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train-categorizer")
def train_categorizer(request: FileRequest) -> dict[str, Any]:
    try:
        return train_transaction_categorizer(request.file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/anomalies")
def anomalies(request: FileRequest) -> dict[str, Any]:
    try:
        return find_unusual_transactions(request.file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
