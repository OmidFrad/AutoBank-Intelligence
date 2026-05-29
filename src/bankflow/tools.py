from __future__ import annotations

from typing import Any

from bankflow.anomaly import detect_anomalies
from bankflow.categorizer import predict_categories, train_category_model
from bankflow.data_loader import load_transactions
from bankflow.recurring import detect_recurring_payments
from bankflow.reporting import build_monthly_summary


def analyze_transactions_csv(file_path: str = "data/sample_transactions.csv") -> dict[str, Any]:
    """
    End-to-end banking automation tool.

    Loads bank transactions, trains a category model, predicts categories,
    detects anomalies, detects recurring payments, and creates a cashflow report.
    """
    df = load_transactions(file_path)
    model, model_result = train_category_model(df)
    scored = predict_categories(df, model)
    anomalies = detect_anomalies(scored)
    recurring = detect_recurring_payments(scored)
    report = build_monthly_summary(scored)

    return {
        "rows_analyzed": int(len(df)),
        "model_accuracy": round(model_result.accuracy, 4),
        "detected_categories": model_result.labels,
        "cashflow_report": report,
        "top_anomalies": anomalies.head(10).assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records"),
        "recurring_payments": recurring.to_dict(orient="records"),
    }


def train_transaction_categorizer(file_path: str = "data/sample_transactions.csv") -> dict[str, Any]:
    """Train only the transaction categorization model and return evaluation metrics."""
    df = load_transactions(file_path)
    _, model_result = train_category_model(df)
    return {
        "accuracy": round(model_result.accuracy, 4),
        "labels": model_result.labels,
        "classification_report": model_result.report,
    }


def find_unusual_transactions(file_path: str = "data/sample_transactions.csv") -> dict[str, Any]:
    """Return unusual or risky-looking transactions from a bank CSV file."""
    df = load_transactions(file_path)
    anomalies = detect_anomalies(df)
    return {
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies.assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records"),
    }
