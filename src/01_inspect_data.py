

from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the accepted LendingClub loans CSV file
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)

# Number of rows to load for initial inspection
SAMPLE_SIZE = 10_000


# ============================================================
# 2. CHECK THAT THE FILE EXISTS
# ============================================================

print("=" * 70)
print("CREDIT RISK CLASSIFIER: DATA INSPECTION")
print("=" * 70)

print(f"\nLooking for dataset at:\n{DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "\nDataset not found!\n"
        "Check that accepted_2007_to_2018Q4.csv is inside "
        "the data/raw/ directory."
    )

print("\nDataset file found successfully.")


# ============================================================
# 3. LOAD A SAMPLE OF THE DATA
# ============================================================

print("\n" + "=" * 70)
print(f"LOADING FIRST {SAMPLE_SIZE:,} ROWS")
print("=" * 70)

# Read only a sample to avoid using too much memory.
# Low-memory mode is disabled to improve consistency in type inference.
df = pd.read_csv(
    DATA_PATH,
    nrows=SAMPLE_SIZE,
    low_memory=False
)

print("\nSample loaded successfully.")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]:,}")


# ============================================================
# 4. INSPECT THE COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(column)


# ============================================================
# 5. DISPLAY THE FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head().to_string())


# ============================================================
# 6. INSPECT DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())


# ============================================================
# 7. INSPECT MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_counts = df.isna().sum()
missing_counts = missing_counts.sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percentage": (
        missing_counts / len(df) * 100
    ).round(2)
})

missing_report = missing_report[
    missing_report["missing_count"] > 0
]

if missing_report.empty:
    print("No missing values found in this sample.")
else:
    print(missing_report.head(25).to_string())


# ============================================================
# 8. INSPECT LOAN STATUS
# ============================================================

print("\n" + "=" * 70)
print("LOAN STATUS DISTRIBUTION")
print("=" * 70)

if "loan_status" in df.columns:

    print("\nLoan status counts:")

    print(
        df["loan_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nLoan status percentages:")

    print(
        (
            df["loan_status"]
            .value_counts(normalize=True, dropna=False)
            * 100
        ).round(2).to_string()
    )

else:
    print(
        "The column 'loan_status' was not found. "
        "Check the column names above."
    )


# ============================================================
# 9. CHECK OUR REQUIRED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("REQUESTED FEATURE AVAILABILITY")
print("=" * 70)

# These are the expected LendingClub column names.
# Some project features will need to be derived later.

requested_features = {
    "income": ["annual_inc"],
    "loan_amount": ["loan_amnt"],
    "loan_term": ["term"],
    "interest_rate": ["int_rate"],
    "installment": ["installment"],
    "debt_to_income": ["dti"],
    "employment_length": ["emp_length"],
    "credit_history_length": [
        "earliest_cr_line",
        "issue_d"
    ],
    "open_accounts": ["open_acc"],
    "past_delinquencies": ["delinq_2yrs"],
    "home_ownership": ["home_ownership"],
    "loan_purpose": ["purpose"]
}

for project_feature, required_columns in requested_features.items():

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if not missing_columns:
        status = "AVAILABLE"
    else:
        status = (
            "MISSING: " + ", ".join(missing_columns)
        )

    print(f"{project_feature:28} {status}")


# ============================================================
# 10. INSPECT POSSIBLE TARGET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE INVESTIGATION")
print("=" * 70)

print(
    "\nWe will investigate loan_status before creating "
    "the binary default target."
)

if "loan_status" in df.columns:

    print("\nObserved loan_status values:")

    statuses = (
        df["loan_status"]
        .value_counts(dropna=False)
        .index
        .tolist()
    )

    for status in statuses:
        print(f"- {status}")

    print(
        "\nIMPORTANT: These values have NOT been converted "
        "into default = 1 or default = 0."
    )

    print(
        "We must first decide which loan outcomes count as "
        "defaults, non-defaults, or ambiguous outcomes."
    )

else:
    print(
        "No loan_status column found. We must identify "
        "the correct outcome field before modelling."
    )


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"""
File inspected:
{DATA_PATH.name}

Rows inspected: {len(df):,}
Columns found: {df.shape[1]:,}

This was a sample inspection only.
The full dataset has not been loaded or modified.

Next step:
Review the loan_status distribution and feature availability
before defining the binary default target.
""")
