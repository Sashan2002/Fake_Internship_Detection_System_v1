"""
Generates a small SYNTHETIC dataset matching the approved 44-column
research-ready schema (Section 21) purely so the rest of the pipeline
(schema validation, splitting, baseline training, API wiring, tests) can be
exercised end-to-end before the real EMSCAD-derived research dataset is
placed at ml/data/raw/ (dev sequence steps 2-3).

THIS IS NOT REAL DATA AND MUST NOT BE USED FOR ANY REPORTED RESULT.
Replace with the genuine, approved research-ready dataset before running
any experiment intended for evaluation (Section 19 checklist item:
'Dataset placed in ml/data/raw/').
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from ml.preprocessing.feature_engineering import engineer_features
from ml.preprocessing.dataset_split import compute_duplicate_group_id, duplicate_aware_split

RANDOM_SEED = 42

LEGIT_TITLES = [
    "Marketing Intern", "Software Engineering Intern", "Finance Summer Intern",
    "HR Internship Programme", "Data Analyst Intern", "Graphic Design Intern",
]
FRAUD_TITLES = [
    "Work From Home - Earn $500/Day!!", "URGENT Internship - No Experience Needed",
    "Easy Money Internship - Immediate Start", "Internship - Send Bank Details to Apply",
]

LEGIT_DESC = (
    "We are looking for a motivated intern to join our team for a structured "
    "{months}-month programme. You will work alongside senior staff on real "
    "projects, receive mentorship, and gain hands-on experience in {field}."
)
FRAUD_DESC = (
    "No experience needed!!! Earn money fast working from home. Contact us "
    "immediately by email or phone to secure your spot, limited slots available, "
    "send your details today to start earning."
)


def _make_row(job_id: int, fraudulent: int, rng: random.Random) -> dict:
    if fraudulent:
        title = rng.choice(FRAUD_TITLES)
        description = FRAUD_DESC
        company_profile = "" if rng.random() < 0.6 else "Fast growing company."
        requirements = "" if rng.random() < 0.7 else "None."
        benefits = "High pay, flexible hours!!!"
        has_company_logo = 0
        has_questions = 0
        salary_range = "$500-$1000" if rng.random() < 0.5 else ""
        location = "" if rng.random() < 0.4 else "Remote"
    else:
        title = rng.choice(LEGIT_TITLES)
        field = title.split()[0]
        description = LEGIT_DESC.format(months=rng.choice([3, 6, 12]), field=field)
        company_profile = "Established organisation operating since " + str(rng.randint(1980, 2015)) + "."
        requirements = "Currently enrolled in a relevant degree programme."
        benefits = "Mentorship, training, and potential full-time offer."
        has_company_logo = 1
        has_questions = 1
        salary_range = f"${rng.randint(15,25)},000-${rng.randint(26,35)},000"
        location = rng.choice(["London, UK", "New York, US", "Remote", "Berlin, DE"])

    return {
        "job_id": job_id,
        "title": title,
        "location": location,
        "department": rng.choice(["Marketing", "Engineering", "Finance", "", "HR"]),
        "salary_range": salary_range,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "telecommuting": rng.choice([0, 1]),
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
        "employment_type": rng.choice(["Internship", "Full-time", ""]),
        "required_experience": rng.choice(["Internship", "Entry level", ""]),
        "required_education": rng.choice(["Bachelor's Degree", "", "High School"]),
        "industry": rng.choice(["Technology", "Finance", "Marketing", ""]),
        "function": rng.choice(["Engineering", "Marketing", "Administrative", ""]),
        "fraudulent": fraudulent,
    }


def generate(n_rows: int = 400, fraud_rate: float = 0.12, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    labels = [1] * n_fraud + [0] * n_legit
    rng.shuffle(labels)

    rows = [_make_row(job_id=i + 1, fraudulent=label, rng=rng) for i, label in enumerate(labels)]
    df = pd.DataFrame(rows)

    df = engineer_features(df)
    df["duplicate_group_id"] = compute_duplicate_group_id(df)
    df = duplicate_aware_split(df, random_seed=seed)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic research-ready dataset for development.")
    parser.add_argument("--n_rows", type=int, default=400)
    parser.add_argument("--fraud_rate", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output",
        default="ml/data/raw/emscad_research_ready.csv",
        help="Output path (default matches RAW_DATASET_PATH in .env.example)",
    )
    args = parser.parse_args()

    df = generate(args.n_rows, args.fraud_rate, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {out_path} (44 columns, dev/test use only).")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
