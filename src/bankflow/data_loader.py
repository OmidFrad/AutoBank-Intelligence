from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "transaction_id",
    "date",
    "description",
    "merchant",
    "amount",
    "currency",
    "balance",
    "account_id",
}


def load_transactions(file_path: str | Path) -> pd.DataFrame:
    """Load and validate a bank transaction CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df[df["date"].isna()].index.tolist()[:5]
        raise ValueError(f"Invalid dates found in rows: {bad_rows}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    if df[["amount", "balance"]].isna().any().any():
        raise ValueError("Amount or balance contains non-numeric values.")

    for col in ["description", "merchant", "currency", "account_id"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["amount_abs"] = df["amount"].abs()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["is_income"] = df["amount"] > 0
    return df.sort_values("date").reset_index(drop=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
