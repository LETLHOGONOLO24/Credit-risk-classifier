
from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = (
    PROCESSED_DIR / "loan_default_model_data.csv"
)

CHUNK_SIZE = 100_000

# Our initial target definition
TARGET_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1,
}

# Original LendingClub columns required for this project
SOURCE_COLUMNS = [
    "loan_status",
    "annual_inc",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "dti",
    "emp_length",
    "earliest_cr_line",
    "issue_d",  # Preserve loan issue date
    "open_acc",
    "delinq_2yrs",
    "home_ownership",
    "purpose",
]

# Consistent feature names for our project
FEATURE_MAPPING = {
    "annual_inc": "income",
    "loan_amnt": "loan_amount",
    "term": "loan_term",
    "int_rate": "interest_rate",
    "installment": "installment",
    "dti": "debt_to_income",
    "emp_length": "employment_length",
    "open_acc": "open_accounts",
    "delinq_2yrs": "past_delinquencies",
    "home_ownership": "home_ownership",
    "purpose": "loan_purpose",
}


# ============================================================
# 2. CHECK INPUT AND PREPARE OUTPUT
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_PATH}"
    )

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Remove an old output so rerunning the script doesn't
# accidentally append duplicate records.
if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

print("=" * 70)
print("PREPARING THE LOAN DEFAULT MODELLING DATASET")
print("=" * 70)

print(f"\nInput:  {RAW_PATH}")
print(f"Output: {OUTPUT_PATH}")


# ============================================================
# 3. INITIALIZE COUNTERS
# ============================================================

total_rows = 0
eligible_rows = 0
excluded_status_counts = {}

target_counts = {
    0: 0,
    1: 0,
}

credit_date_missing = 0
first_chunk = True


# ============================================================
# 4. READ AND PROCESS THE CSV IN CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        RAW_PATH,
        usecols=SOURCE_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):

    total_rows += len(chunk)

    # --------------------------------------------------------
    # 4.1 Count statuses excluded from the initial classifier
    # --------------------------------------------------------

    status_counts = chunk["loan_status"].value_counts(
        dropna=False
    )

    for status, count in status_counts.items():

        if status not in TARGET_MAPPING:
            excluded_status_counts[status] = (
                excluded_status_counts.get(status, 0)
                + int(count)
            )

    # --------------------------------------------------------
    # 4.2 Keep only loans with selected final outcomes
    # --------------------------------------------------------

    eligible = chunk[
        chunk["loan_status"].isin(TARGET_MAPPING)
    ].copy()

    if eligible.empty:
        print(
            f"Processed chunk {chunk_number:,}: "
            f"{total_rows:,} source rows examined"
        )
        continue

    # --------------------------------------------------------
    # 4.3 Create the binary target
    # --------------------------------------------------------

    eligible["default"] = (
        eligible["loan_status"]
        .map(TARGET_MAPPING)
        .astype("int8")
    )

    # --------------------------------------------------------
    # 4.4 Derive credit history length in years
    # --------------------------------------------------------

    # LendingClub dates are commonly represented as
    # strings such as "Dec-2015" and "Jan-1998".
    issue_date = pd.to_datetime(
        eligible["issue_d"],
        format="%b-%Y",
        errors="coerce",
    )

    earliest_credit_date = pd.to_datetime(
        eligible["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    months_of_history = (
        (issue_date.dt.year - earliest_credit_date.dt.year)
        * 12
        + issue_date.dt.month
        - earliest_credit_date.dt.month
    )

    # Negative values indicate inconsistent dates.
    # Treat these as missing rather than inventing a value.
    invalid_history = months_of_history < 0

    months_of_history = months_of_history.mask(
        invalid_history
    )

    eligible["credit_history_length"] = (
        months_of_history / 12
    )

    credit_date_missing += int(
        eligible["credit_history_length"].isna().sum()
    )

    # --------------------------------------------------------
    # 4.5 Keep the requested features and target
    # --------------------------------------------------------

    eligible = eligible.rename(
        columns=FEATURE_MAPPING
    )

    # Preserve the original loan issue date as metadata.
    # It will be used for chronological splitting, NOT as a predictor.
    eligible["issue_date"] = pd.to_datetime(
        eligible["issue_d"],
        format="%b-%Y",
        errors="coerce",
    )

    feature_columns = list(FEATURE_MAPPING.values())

    model_df = eligible[
        feature_columns
        + [
            "credit_history_length",
            "default",
            "issue_date",
        ]
    ].copy()

    # --------------------------------------------------------
    # 4.6 Standardize basic numeric representations
    # --------------------------------------------------------

    # Convert loan term, e.g. "36 months", to 36.
    model_df["loan_term"] = pd.to_numeric(
        model_df["loan_term"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    # Convert interest rate, e.g. "13.56%", to 13.56.
    model_df["interest_rate"] = pd.to_numeric(
        model_df["interest_rate"]
        .astype("string")
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    # Ensure numeric features have numeric types.
    numeric_columns = [
        "income",
        "loan_amount",
        "installment",
        "debt_to_income",
        "open_accounts",
        "past_delinquencies",
        "credit_history_length",
    ]

    for column in numeric_columns:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce",
        )

    # Keep employment_length and the categorical variables
    # as strings. We will preprocess them later.

    # --------------------------------------------------------
    # 4.7 Update counters and save this chunk
    # --------------------------------------------------------

    eligible_rows += len(model_df)

    counts = model_df["default"].value_counts()

    for label, count in counts.items():
        target_counts[int(label)] += int(count)

    model_df.to_csv(
        OUTPUT_PATH,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )

    first_chunk = False

    print(
        f"Processed chunk {chunk_number:,} | "
        f"Source rows: {total_rows:,} | "
        f"Eligible loans saved: {eligible_rows:,}"
    )


# ============================================================
# 5. REPORT THE FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION SUMMARY")
print("=" * 70)

print(f"Source rows examined: {total_rows:,}")
print(f"Eligible loans saved: {eligible_rows:,}")
print(f"Excluded rows: {total_rows - eligible_rows:,}")

print("\nTarget counts:")

for label, description in [
    (0, "Fully Paid"),
    (1, "Charged Off"),
]:
    print(
        f"default = {label} ({description}): "
        f"{target_counts[label]:,}"
    )

if eligible_rows > 0:

    print("\nTarget proportions:")

    for label in [0, 1]:
        percentage = (
            target_counts[label] / eligible_rows * 100
        )

        print(
            f"default = {label}: {percentage:.2f}%"
        )

    print(
        "\nMissing credit-history lengths: "
        f"{credit_date_missing:,}"
    )

    # Inspect missing values in the saved modelling dataset.
    print("\nReading saved dataset to summarize missing values...")

    final_df = pd.read_csv(
        OUTPUT_PATH,
        low_memory=False,
    )

    missing = final_df.isna().sum()
    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print("\nMissing values by feature:")

    if missing.empty:
        print("No missing values found.")
    else:
        missing_report = pd.DataFrame({
            "missing_count": missing,
            "missing_percentage": (
                missing / len(final_df) * 100
            ).round(2),
        })

        print(missing_report.to_string())

    print("\nFinal dataset shape:", final_df.shape)

    print("\nFirst five rows:")
    print(final_df.head().to_string(index=False))

    print(f"\nSaved dataset: {OUTPUT_PATH}")

else:
    print(
        "\nNo eligible loans were found. "
        "Check the source column names and loan statuses."
    )

print("\nPreparation complete.")
