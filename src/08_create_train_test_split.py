
from pathlib import Path

import pandas as pd

# --------------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "loan_default_model_data_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

TEST_START_DATE = pd.Timestamp("2017-01-01")

# --------------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["issue_date"] = pd.to_datetime(
    df["issue_date"],
    errors="coerce",
)

if df["issue_date"].isna().any():
    raise ValueError("Missing issue dates detected.")

if not df["default"].isin([0, 1]).all():
    raise ValueError("Unexpected values found in default target.")

# --------------------------------------------------------
# 3. Split chronologically
# --------------------------------------------------------

train_df = df[df["issue_date"] < TEST_START_DATE].copy()

test_df = df[df["issue_date"] >= TEST_START_DATE].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or test set is empty.")

# --------------------------------------------------------
# 4. Save the datasets
# --------------------------------------------------------

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# --------------------------------------------------------
# 5. Report the split
# --------------------------------------------------------

print("=" * 65)
print("CHRONOLOGICAL TRAIN/TEST SPLIT")
print("=" * 65)

for name, data in [("TRAIN", train_df), ("TEST", test_df)]:
    print(f"\n{name} SET")
    print("-" * 40)
    print(f"Rows: {len(data):,}")
    print(f"First issue date: {data['issue_date'].min().date()}")
    print(f"Last issue date:  {data['issue_date'].max().date()}")
    print(f"Default rate: {data['default'].mean() * 100:.2f}%")

    print("\nTarget counts:")
    print(data["default"].value_counts().sort_index().to_string())

    print("\nTarget proportions (%):")
    print(
        (data["default"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .to_string()
    )

print("\nSaved files:")
print(TRAIN_PATH)
print(TEST_PATH)

print("\nChronological split complete.")
