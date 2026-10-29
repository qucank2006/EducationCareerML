from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def plot_feature_importance(model, X_test, y_test):
    """
    Estimate feature importance by shuffling each original input column
    on the held-out test set. Larger importance means shuffling that column
    worsened the model score more. Negative values can occur.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    importance_df.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )

    chart_data = importance_df.sort_values("Importance_Mean")
    plt.figure(figsize=(10, max(5, len(chart_data) * 0.45)))
    plt.barh(
        chart_data["Feature"],
        chart_data["Importance_Mean"],
        xerr=chart_data["Importance_Std"],
        capsize=3,
    )
    plt.axvline(0, linewidth=1)
    plt.title("Feature Importance (Permutation Importance - Random Forest)")
    plt.xlabel("Increase in MAE after shuffling (higher = more important)")
    plt.ylabel("Input feature")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print("\nFEATURE IMPORTANCE (Permutation Importance)")
    print(importance_df.to_string(index=False))
    print("Saved: results/feature_importance.csv")
    print("Saved: results/feature_importance.png")

    return importance_df
