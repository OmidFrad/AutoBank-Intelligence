from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class CategoryModelResult:
    accuracy: float
    labels: list[str]
    report: dict[str, Any]


def _build_text_features(df: pd.DataFrame) -> pd.Series:
    return (
        df["description"].fillna("").astype(str)
        + " | merchant: "
        + df["merchant"].fillna("").astype(str)
        + " | amount_band: "
        + pd.cut(
            df["amount"].abs(),
            bins=[0, 10, 25, 50, 100, 250, 1000, float("inf")],
            labels=["0-10", "10-25", "25-50", "50-100", "100-250", "250-1000", "1000+"],
            include_lowest=True,
        ).astype(str)
    )


def train_category_model(df: pd.DataFrame, target_column: str = "label_category") -> tuple[Pipeline, CategoryModelResult]:
    """Train a transaction categorization model using transaction text and amount bands."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = _build_text_features(df)
    y = df[target_column].astype(str)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    result = CategoryModelResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        labels=sorted(y.unique().tolist()),
        report=classification_report(y_test, predictions, output_dict=True, zero_division=0),
    )
    return model, result


def predict_categories(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Add predicted_category to a transaction DataFrame."""
    scored = df.copy()
    scored["predicted_category"] = model.predict(_build_text_features(scored))
    return scored
