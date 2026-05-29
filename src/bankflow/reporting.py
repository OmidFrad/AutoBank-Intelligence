from __future__ import annotations

from typing import Any

import pandas as pd


def build_monthly_summary(df: pd.DataFrame, category_column: str = "predicted_category") -> dict[str, Any]:
    """Build a compact finance report from transactions."""
    category = category_column if category_column in df.columns else "label_category"

    total_income = float(df.loc[df["amount"] > 0, "amount"].sum())
    total_expenses = float(df.loc[df["amount"] < 0, "amount"].sum())
    net_cashflow = total_income + total_expenses

    monthly = (
        df.groupby("month")
        .agg(
            income=("amount", lambda s: float(s[s > 0].sum())),
            expenses=("amount", lambda s: float(s[s < 0].sum())),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
    )
    monthly["net_cashflow"] = monthly["income"] + monthly["expenses"]

    by_category = (
        df[df["amount"] < 0]
        .groupby(category)["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_cashflow": round(net_cashflow, 2),
        "spending_by_category": by_category,
        "monthly_summary": monthly.round(2).to_dict(orient="records"),
    }
