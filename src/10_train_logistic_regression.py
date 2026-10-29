
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------
# 1. Paths and data
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "default"

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

features = numeric_features + categorical_features

X_train = train_df[features]
y_train = train_df[TARGET]

X_test = test_df[features]
y_test = test_df[TARGET]

# --------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

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
# 3. Build the complete model pipeline
# --------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight=None,
                random_state=42,
            ),
        ),
    ]
)

# --------------------------------------------------------
# 4. Train
# --------------------------------------------------------

print("Training Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")

# --------------------------------------------------------
# 5. Predict on the untouched test set
# --------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------------
# 6. Evaluate
# --------------------------------------------------------

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION — TEST RESULTS")
print("=" * 65)

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(
    "Average precision (PR-AUC summary): "
    f"{average_precision_score(y_test, y_prob):.4f}"
)

print("\nConfusion matrix:")
print("Rows = actual class; columns = predicted class")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fully Paid", "Charged Off"],
        digits=4,
        zero_division=0,
    )
)
