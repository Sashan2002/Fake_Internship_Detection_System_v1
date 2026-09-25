"""
Evaluation entry point. Runs a fitted model over a given split and writes
a structured JSON report to ml/experiments/runs/<run_id>/.

Per Section 17/18, this script is meant to be called on the VALIDATION
split during model development and only ONCE on the TEST split after the
model and evaluation protocol are frozen ('Do not report final test
performance before the model and evaluation protocol are frozen.').
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.evaluation.metrics import compute_metrics

RUNS_DIR = Path(__file__).resolve().parents[2] / "ml" / "experiments" / "runs"


def evaluate_predictions(y_true, y_prob, run_name: str, split_name: str, extra_meta: dict = None) -> dict:
    report = compute_metrics(y_true, y_prob).to_dict()
    report["run_name"] = run_name
    report["split"] = split_name
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if extra_meta:
        report["meta"] = extra_meta

    if split_name == "test":
        report["_warning"] = (
            "Test-set result. Confirm the model and evaluation protocol were "
            "frozen BEFORE this run, per Section 17/18 of the implementation spec."
        )

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_metrics.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved predictions against ground truth.")
    parser.add_argument("--predictions_csv", required=True,
                         help="CSV with columns: y_true, y_prob")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    report = evaluate_predictions(df["y_true"], df["y_prob"], args.run_name, args.split)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
