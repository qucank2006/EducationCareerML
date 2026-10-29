from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TARGET = "Starting_Salary"
MAJOR_COLUMN = "Field_of_Study"


def generate_eda(df):
    """Generate exploratory charts from the available dataset columns."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Overall starting salary distribution
    if TARGET in df.columns:
        salary = pd.to_numeric(df[TARGET], errors="coerce").dropna()
        if not salary.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(salary, bins=30, edgecolor="black")
            plt.title("Distribution of Starting Salary")
            plt.xlabel("Starting Salary")
            plt.ylabel("Number of records")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "starting_salary_distribution.png", dpi=150)
            plt.close()

    # 2. Starting salary by field of study
    if TARGET in df.columns and MAJOR_COLUMN in df.columns:
        salary_data = df[[MAJOR_COLUMN, TARGET]].copy()
        salary_data[TARGET] = pd.to_numeric(salary_data[TARGET], errors="coerce")
        salary_data = salary_data.dropna(subset=[MAJOR_COLUMN, TARGET])

        if not salary_data.empty:
            summary = (
                salary_data.groupby(MAJOR_COLUMN, observed=True)[TARGET]
                .agg(["count", "mean", "median"])
                .sort_values("mean", ascending=False)
            )
            summary.to_csv(RESULTS_DIR / "salary_by_field_of_study.csv")

            # Keep the chart readable when a dataset contains many categories.
            chart_data = summary.head(15).sort_values("mean")
            plt.figure(figsize=(10, max(5, len(chart_data) * 0.4)))
            plt.barh(chart_data.index.astype(str), chart_data["mean"])
            plt.title("Average Starting Salary by Field of Study (Top 15)")
            plt.xlabel("Average Starting Salary")
            plt.ylabel("Field of Study")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "salary_by_field_of_study.png", dpi=150)
            plt.close()

    # 3. Correlation matrix for numeric columns, including the target when available.
    numeric_data = df.select_dtypes(include="number")
    if numeric_data.shape[1] >= 2:
        correlation = numeric_data.corr(numeric_only=True)
        correlation.to_csv(RESULTS_DIR / "correlation_matrix.csv")

        figure_size = max(7, min(14, 0.8 * len(correlation.columns)))
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=90)
        ax.set_yticklabels(correlation.columns)
        ax.set_title("Correlation Matrix of Numeric Features")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=150)
        plt.close(fig)

    print("EDA outputs saved in:", RESULTS_DIR)
