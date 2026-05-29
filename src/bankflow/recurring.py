from __future__ import annotations

import pandas as pd


def detect_recurring_payments(df: pd.DataFrame, min_occurrences: int = 3) -> pd.DataFrame:
    """Find likely recurring payments based on merchant, amount, and monthly frequency."""
    if df.empty:
        return pd.DataFrame()

    expenses = df[df["amount"] < 0].copy()
    expenses["rounded_amount"] = expenses["amount"].round(0)
    grouped = (
        expenses.groupby(["merchant", "rounded_amount"])
        .agg(
            occurrences=("transaction_id", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            avg_amount=("amount", "mean"),
            months_seen=("month", "nunique"),
        )
        .reset_index()
    )

    candidates = grouped[
        (grouped["occurrences"] >= min_occurrences)
        & (grouped["months_seen"] >= min_occurrences)
    ].copy()

    if candidates.empty:
        return candidates

    candidates["avg_amount"] = candidates["avg_amount"].round(2)
    candidates["first_date"] = candidates["first_date"].dt.strftime("%Y-%m-%d")
    candidates["last_date"] = candidates["last_date"].dt.strftime("%Y-%m-%d")
    return candidates.sort_values(["occurrences", "avg_amount"], ascending=[False, True]).reset_index(drop=True)
