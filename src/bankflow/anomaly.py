from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Detect unusual transactions using Isolation Forest."""
    if df.empty:
        return df.copy()

    work = df.copy()
    work["day_of_month"] = work["date"].dt.day
    work["amount_direction"] = (work["amount"] > 0).astype(int)

    features = work[["amount", "amount_abs", "balance", "day_of_month", "amount_direction"]]
    scaled = StandardScaler().fit_transform(features)

    model = IsolationForest(contamination=contamination, random_state=42)
    work["anomaly_flag"] = model.fit_predict(scaled)
    work["anomaly_score"] = model.decision_function(scaled)
    anomalies = work[work["anomaly_flag"] == -1].sort_values("anomaly_score")

    columns = [
        "transaction_id",
        "date",
        "merchant",
        "description",
        "amount",
        "currency",
        "balance",
        "anomaly_score",
    ]
    return anomalies[columns].reset_index(drop=True)
