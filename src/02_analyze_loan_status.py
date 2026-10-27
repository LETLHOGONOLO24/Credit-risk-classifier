
from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accepted_2007_to_2018Q4.csv"
)

CHUNK_SIZE = 100_000

# For the first version of our classifier:
# 0 = Fully Paid
# 1 = Charged Off
ELIGIBLE_STATUSES = ["Fully Paid", "Charged Off"]


# ============================================================
# 2. CHECK THE FILE
# ============================================================

print("=" * 70)
print("FULL DATASET LOAN STATUS ANALYSIS")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find dataset: {DATA_PATH}"
    )


# ============================================================
# 3. COUNT STATUSES IN CHUNKS
# ============================================================

status_counts = {}
total_rows = 0

print("\nReading the full CSV in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=["loan_status"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):
    total_rows += len(chunk)

    counts = chunk["loan_status"].value_counts(
        dropna=False
    )

    for status, count in counts.items():
        status_counts[status] = (
            status_counts.get(status, 0) + int(count)
        )

    print(
        f"Processed chunk {chunk_number:,} | "
        f"Rows processed: {total_rows:,}"
    )


# ============================================================
# 4. DISPLAY FULL DATASET STATUS DISTRIBUTION
# ============================================================

status_series = pd.Series(
    status_counts,
    name="number_of_loans"
).sort_values(ascending=False)

status_report = pd.DataFrame({
    "number_of_loans": status_series,
    "percentage": (
        status_series / total_rows * 100
    ).round(2)
})

print("\n" + "=" * 70)
print("FULL DATASET LOAN STATUS DISTRIBUTION")
print("=" * 70)

print(f"Total rows: {total_rows:,}\n")
print(status_report.to_string())


# ============================================================
# 5. DEFINE THE INITIAL ELIGIBLE POPULATION
# ============================================================

fully_paid = status_counts.get("Fully Paid", 0)
charged_off = status_counts.get("Charged Off", 0)

eligible_total = fully_paid + charged_off

excluded_total = total_rows - eligible_total

print("\n" + "=" * 70)
print("INITIAL BINARY CLASSIFICATION POPULATION")
print("=" * 70)

print(f"Fully Paid (target = 0): {fully_paid:,}")
print(f"Charged Off (target = 1): {charged_off:,}")
print(f"Eligible loans: {eligible_total:,}")
print(f"Other statuses excluded: {excluded_total:,}")

if eligible_total > 0:
    default_rate = charged_off / eligible_total
    non_default_rate = fully_paid / eligible_total

    print(
        f"\nObserved default rate among eligible loans: "
        f"{default_rate:.2%}"
    )

    print(
        f"Observed non-default rate among eligible loans: "
        f"{non_default_rate:.2%}"
    )

    print(
        f"\nDefault-to-non-default ratio: "
        f"1:{fully_paid / charged_off:.2f}"
        if charged_off > 0
        else "\nNo charged-off loans were found."
    )

print("\nAnalysis complete.")
print(
    "No rows have been changed, relabelled, or saved."
)
