from bankflow.tools import analyze_transactions_csv, find_unusual_transactions, train_transaction_categorizer


SAMPLE_FILE = "data/sample_transactions.csv"


def test_train_transaction_categorizer_returns_accuracy():
    result = train_transaction_categorizer(SAMPLE_FILE)
    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1
    assert len(result["labels"]) > 3


def test_analyze_transactions_returns_report_sections():
    result = analyze_transactions_csv(SAMPLE_FILE)
    assert result["rows_analyzed"] > 100
    assert "cashflow_report" in result
    assert "top_anomalies" in result
    assert "recurring_payments" in result


def test_find_unusual_transactions_returns_anomalies():
    result = find_unusual_transactions(SAMPLE_FILE)
    assert "anomaly_count" in result
    assert result["anomaly_count"] > 0
