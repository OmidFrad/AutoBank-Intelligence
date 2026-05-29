from __future__ import annotations

import argparse
import json
from pathlib import Path

from bankflow.tools import analyze_transactions_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bank transactions from a CSV file.")
    parser.add_argument("--file", default="data/sample_transactions.csv", help="Path to transaction CSV file")
    parser.add_argument("--output", default="", help="Optional path to save the JSON report")
    args = parser.parse_args()

    result = analyze_transactions_csv(args.file)
    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    print(formatted)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted, encoding="utf-8")
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
