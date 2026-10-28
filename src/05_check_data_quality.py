from pathlib import Path

import pandas as pd

DATA_PATH = Path(r"C:\Users\HLOGIZNBUCKS\Downloads\Credit-risk-classifier\data\processed\loan_default_model_data.csv")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "income",
        "debt_to_income",
        "loan_amount",
        "default",
    ],
)

print("\n" + "=" * 65)
print("DATA QUALITY CHECKS")
print("=" * 65)

# 1. Investigate debt-to-income values
print("\nDebt-to-income values:")
print("Number equal to 999:", (df["debt_to_income"] == 999).sum())
print("Number above 100:", (df["debt_to_income"] > 100).sum())
print("Number missing:", df["debt_to_income"].isna().sum())

print("\nLoans with debt-to-income equal to 999:")
print(
    df.loc[
        df["debt_to_income"] == 999,
        ["income", "loan_amount", "debt_to_income", "default"],
    ].head(10).to_string(index=False)
)

# 2. Investigate zero income
zero_income = df["income"] == 0

print("\nIncome equal to zero:")
print("Number of loans:", zero_income.sum())

if zero_income.any():
    print(
        "Default rate:",
        round(df.loc[zero_income, "default"].mean() * 100, 2),
        "%",
    )

# 3. Inspect the highest income records
print("\nTop 10 reported incomes:")
print(
    df.nlargest(10, "income")[
        ["income", "loan_amount", "debt_to_income", "default"]
    ].to_string(index=False)
)

# 4. Compare debt-to-income distributions by outcome
print("\nDebt-to-income summary by target:")
print(
    df.groupby("default")["debt_to_income"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .round(2)
)

print("\nChecks complete.")