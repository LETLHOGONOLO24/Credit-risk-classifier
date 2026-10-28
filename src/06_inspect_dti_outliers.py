from pathlib import Path
import pandas as pd

DATA_PATH = Path(r"C:\Users\HLOGIZNBUCKS\Downloads\Credit-risk-classifier\data\processed\loan_default_model_data.csv")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "income",
        "loan_amount",
        "debt_to_income",
        "interest_rate",
        "default",
    ],
)

dti = df["debt_to_income"]

print("Negative DTI values:", (dti < 0).sum())
print("DTI equal to -1:", (dti == -1).sum())
print("DTI between 100 and 999:", ((dti > 100) & (dti < 999)).sum())
print("DTI equal to 999:", (dti == 999).sum())

print("\nNegative DTI records:")
print(
    df.loc[dti < 0]
    .head(15)
    .to_string(index=False)
)

print("\nDTI above 100 but below 999:")
print(
    df.loc[(dti > 100) & (dti < 999)]
    .sort_values("debt_to_income", ascending=False)
    .head(15)
    .to_string(index=False)
)