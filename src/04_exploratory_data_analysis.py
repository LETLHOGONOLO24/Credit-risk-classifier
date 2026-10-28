
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "loan_default_model_data.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

CHUNK_SIZE = 100_000
SAMPLE_PER_CHUNK = 2_000
RANDOM_STATE = 42

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "employment_length",
    "home_ownership",
    "loan_purpose",
]

TARGET = "default"


# ============================================================
# 2. CHECK FILES AND CREATE OUTPUT DIRECTORIES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}"
    )

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 3. READ THE FULL DATASET IN CHUNKS
# ============================================================

total_rows = 0
target_counts = {0: 0, 1: 0}
missing_counts = {}
sample_parts = []

print("\nReading dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):
    total_rows += len(chunk)

    # Count the target using the entire dataset.
    counts = chunk[TARGET].value_counts()

    for label, count in counts.items():
        target_counts[int(label)] += int(count)

    # Count missing values using the entire dataset.
    for column, count in chunk.isna().sum().items():
        missing_counts[column] = (
            missing_counts.get(column, 0) + int(count)
        )

    # Take a manageable sample for plots.
    sample_parts.append(
        chunk.sample(
            n=min(SAMPLE_PER_CHUNK, len(chunk)),
            random_state=RANDOM_STATE + chunk_number,
        )
    )

    print(
        f"Processed {total_rows:,} rows"
    )

sample_df = pd.concat(
    sample_parts,
    ignore_index=True,
)

print(f"\nFull dataset rows: {total_rows:,}")
print(f"Sample rows for visualizations: {len(sample_df):,}")


# ============================================================
# 4. SAVE CLASS DISTRIBUTION
# ============================================================

class_report = pd.DataFrame({
    "default": [0, 1],
    "description": ["Fully Paid", "Charged Off"],
    "count": [
        target_counts[0],
        target_counts[1],
    ],
})

class_report["percentage"] = (
    class_report["count"] / total_rows * 100
).round(2)

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(class_report.to_string(index=False))

class_report.to_csv(
    TABLES_DIR / "class_distribution.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    class_report["description"],
    class_report["count"],
)

ax.set_title("Loan Outcome Distribution")
ax.set_xlabel("Loan outcome")
ax.set_ylabel("Number of loans")

ax.bar_label(
    bars,
    labels=[
        f"{count:,}\n({percentage:.2f}%)"
        for count, percentage in zip(
            class_report["count"],
            class_report["percentage"],
        )
    ],
    padding=3,
)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "class_distribution.png",
    dpi=150,
)
plt.close(fig)


# ============================================================
# 5. MISSING-VALUE ANALYSIS
# ============================================================

missing_report = pd.DataFrame({
    "missing_count": pd.Series(missing_counts),
})

missing_report["missing_percentage"] = (
    missing_report["missing_count"] / total_rows * 100
).round(2)

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False,
)

print("\n" + "=" * 70)
print("MISSING-VALUE REPORT")
print("=" * 70)

print(missing_report.to_string())

missing_report.to_csv(
    TABLES_DIR / "missing_values.csv"
)


# ============================================================
# 6. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 70)

numeric_summary = sample_df[
    NUMERIC_FEATURES
].describe().T

print(numeric_summary.round(2).to_string())

numeric_summary.to_csv(
    TABLES_DIR / "numeric_summary_sample.csv"
)


# ============================================================
# 7. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

print("\nCreating numerical feature histograms...")

for feature in NUMERIC_FEATURES:

    values = pd.to_numeric(
        sample_df[feature],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values,
        bins=40,
        edgecolor="black",
    )

    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"distribution_{feature}.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 8. DEFAULT RATE BY CATEGORICAL FEATURE
# ============================================================

print("\nCalculating default rates by category...")

categorical_reports = {}

for feature in CATEGORICAL_FEATURES:

    grouped = (
        sample_df
        .groupby(feature, dropna=False)[TARGET]
        .agg(
            loan_count="count",
            default_rate="mean",
        )
        .reset_index()
    )

    grouped["default_rate_pct"] = (
        grouped["default_rate"] * 100
    ).round(2)

    # Save the number of loans alongside the default rate.
    grouped = grouped.sort_values(
        "default_rate",
        ascending=False,
    )

    categorical_reports[feature] = grouped

    print(f"\nDefault rate by {feature}:")
    print(grouped.to_string(index=False))

    grouped.to_csv(
        TABLES_DIR / f"default_rate_by_{feature}.csv",
        index=False,
    )

    # Plot categories in descending default-rate order.
    plot_data = grouped.copy()

    plot_data[feature] = (
        plot_data[feature]
        .fillna("Missing")
        .astype(str)
    )

    fig_width = max(
        9,
        min(16, len(plot_data) * 0.65),
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 5)
    )

    bars = ax.bar(
        plot_data[feature],
        plot_data["default_rate_pct"],
    )

    ax.set_title(f"Observed Default Rate by {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Default rate (%)")

    ax.tick_params(
        axis="x",
        labelrotation=45,
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"default_rate_by_{feature}.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 9. NUMERICAL FEATURES BY TARGET
# ============================================================

print("\nCreating numerical feature comparisons...")

for feature in NUMERIC_FEATURES:

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, description in [
        (0, "Fully Paid"),
        (1, "Charged Off"),
    ]:
        values = pd.to_numeric(
            sample_df.loc[
                sample_df[TARGET] == label,
                feature,
            ],
            errors="coerce",
        ).dropna()

        ax.hist(
            values,
            bins=35,
            alpha=0.5,
            label=description,
        )

    ax.set_title(f"{feature} by Loan Outcome")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"{feature}_by_default.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"Full dataset: {total_rows:,} rows")
print(f"Visualization sample: {len(sample_df):,} rows")

print(f"\nFigures saved to:\n{FIGURES_DIR}")
print(f"\nTables saved to:\n{TABLES_DIR}")

print(
    "\nReview the figures and tables before deciding "
    "how to preprocess and model the data."
)
