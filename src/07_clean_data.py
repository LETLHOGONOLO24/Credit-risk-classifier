
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(r"C:\Users\HLOGIZNBUCKS\Downloads\Credit-risk-classifier\data\processed\loan_default_model_data.csv")
OUTPUT_PATH = Path(r"C:\Users\HLOGIZNBUCKS\Downloads\Credit-risk-classifier\data\processed\loan_default_model_data_clean.csv")

df = pd.read_csv(INPUT_PATH)

print("=" * 65)
print("DATA CLEANING")
print("=" * 65)

# Record the original values before changing anything.
original_dti = df["debt_to_income"].copy()

negative_dti_count = (original_dti < 0).sum()
sentinel_999_count = (original_dti == 999).sum()
extreme_dti_count = (
    (original_dti > 100) & (original_dti < 999)
).sum()

# Replace negative DTI values and the provisional 999 sentinel
# with missing values. Do not delete the affected loans.
df.loc[df["debt_to_income"] < 0, "debt_to_income"] = float("nan")
df.loc[df["debt_to_income"] == 999, "debt_to_income"] = float("nan")

print(f"Negative DTI values replaced: {negative_dti_count}")
print(f"DTI values equal to 999 replaced: {sentinel_999_count}")
print(f"DTI values above 100 but below 999 retained: {extreme_dti_count}")

print("\nMissing values after cleaning:")
print(df.isna().sum()[df.isna().sum() > 0])

# Keep all rows. Imputation will be learned from training data
# later, rather than calculated here on the complete dataset.
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")
print("No rows were deleted.")
