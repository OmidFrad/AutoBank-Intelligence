# AutoBank Intelligence
AI-powered banking transaction automation for intelligent financial data workflows.
AutoBank Intelligence ingests bank transaction CSV files, trains a machine learning model to categorize transactions, detects unusual transactions, identifies recurring payments, and exposes the workflow through a FastAPI API.

> This project uses synthetic sample data. Do not commit real banking data, account numbers, credentials, API keys, or customer information to GitHub.

## Why this project is useful

This project demonstrates skills that are relevant for fintech, banking analytics, and data automation roles:

- CSV ingestion and validation
- Transaction categorization with machine learning
- Anomaly detection for unusual transactions
- Recurring payment detection
- Cashflow reporting
- FastAPI API design
- Testable Python project structure

## Project structure

```text
bankflow-automator/
├── app.py
├── data/
│   └── sample_transactions.csv
├── src/
│   └── bankflow/
│       ├── anomaly.py
│       ├── categorizer.py
│       ├── cli.py
│       ├── data_loader.py
│       ├── recurring.py
│       ├── reporting.py
│       └── tools.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the command-line workflow

```bash
PYTHONPATH=src python -m bankflow.cli --file data/sample_transactions.csv
```

Save the output report:

```bash
PYTHONPATH=src python -m bankflow.cli --file data/sample_transactions.csv --output reports/report.json
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"
python -m bankflow.cli --file data/sample_transactions.csv
```

## Run the API

```bash
PYTHONPATH=src uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Example request body for `/analyze`:

```json
{
  "file_path": "data/sample_transactions.csv"
}
```

## Main API endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/health` | GET | Check API status |
| `/analyze` | POST | Run the full banking automation workflow |
| `/train-categorizer` | POST | Train and evaluate the categorization model |
| `/anomalies` | POST | Detect unusual transactions |

## Data schema

The CSV file should include these columns:

| Column | Meaning |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `date` | Transaction date |
| `description` | Transaction text from bank statement |
| `merchant` | Merchant or counterparty |
| `amount` | Positive income or negative expense |
| `currency` | Currency code, for example EUR |
| `balance` | Account balance after transaction |
| `account_id` | Synthetic account identifier |
| `label_category` | Training label for the ML model |

## Example output

```json
{
  "rows_analyzed": 153,
  "model_accuracy": 0.92,
  "detected_categories": ["Bank Fees", "Dining", "Entertainment", "Groceries"],
  "cashflow_report": {
    "total_income": 20180.0,
    "total_expenses": -13750.0,
    "net_cashflow": 6430.0
  },
  "top_anomalies": [],
  "recurring_payments": []
}
```

## Suggested GitHub repository description

Machine learning banking automation project that categorizes transactions, detects anomalies, identifies recurring payments, and exposes the workflow through FastAPI.

## Suggested next improvements

- Add Streamlit dashboard
- Add database storage with SQLite or PostgreSQL
- Add Plaid Sandbox or GoCardless Bank Account Data sandbox connector
- Add Dockerfile
- Add GitHub Actions CI
- Add time-series cashflow forecasting
- Add SHAP explanations for model interpretability
