
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
)

def evaluate_models(models, X_test, y_test):
    RESULTS_DIR.mkdir(exist_ok=True)

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error_custom(
                y_test, y_pred
            ),
            "RMSE": np.sqrt(
                np.mean((np.asarray(y_test) - y_pred) ** 2)
            ),
            "R2": r2_score_custom(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE")

    print("\nMODEL EVALUATION")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    for metric in ["MAE", "RMSE", "R2"]:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Model"], results_df[metric])
        plt.title(f"Model Comparison - {metric}")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"{metric.lower()}_comparison.png",
            dpi=150
        )
        plt.close()

    return results_df

from sklearn.metrics import mean_absolute_error, r2_score

def mean_absolute_error_custom(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2_score_custom(y_true, y_pred):
    return r2_score(y_true, y_pred)
