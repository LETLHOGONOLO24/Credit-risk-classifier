
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------
# 1. Paths
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# --------------------------------------------------------
# 2. Load train and test data
# --------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "default"
METADATA = "issue_date"

# --------------------------------------------------------
# 3. Define predictors
# --------------------------------------------------------

numeric_features = [
    "income",
    "loan_amount",
    "loan_term",
    "interest_rate",
    "installment",
    "debt_to_income",
    "open_accounts",
    "past_delinquencies",
    "credit_history_length",
]

categorical_features = [
    "employment_length",
    "home_ownership",
    "loan_purpose",
]

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns]
y_train = train_df[TARGET]

X_test = test_df[feature_columns]
y_test = test_df[TARGET]

# --------------------------------------------------------
# 4. Numerical preprocessing
# --------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# --------------------------------------------------------
# 5. Categorical preprocessing
# --------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

# --------------------------------------------------------
# 6. Combine both preprocessing pipelines
# --------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

# --------------------------------------------------------
# 7. Fit on training data only
# --------------------------------------------------------

X_train_transformed = preprocessor.fit_transform(X_train)

# Apply the already-fitted transformations to test data.
X_test_transformed = preprocessor.transform(X_test)

# --------------------------------------------------------
# 8. Verify the result
# --------------------------------------------------------

print("=" * 65)
print("PREPROCESSING PIPELINE CHECK")
print("=" * 65)

print(f"Training rows: {X_train.shape[0]:,}")
print(f"Test rows: {X_test.shape[0]:,}")

print(f"Original predictors: {X_train.shape[1]}")
print(
    "Transformed training shape:",
    X_train_transformed.shape,
)
print(
    "Transformed test shape:",
    X_test_transformed.shape,
)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

print("\nMissing values remaining after preprocessing:")
print(
    "Training:",
    pd.isna(X_train_transformed.data).sum()
    if hasattr(X_train_transformed, "data")
    else pd.isna(X_train_transformed).sum(),
)

print("\nPipeline check complete.")
