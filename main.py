
from pathlib import Path
import joblib

from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.train import train_models
from src.evaluate import evaluate_models

def main():
    print("=== EDUCATION CAREER ML ===")

    # Load dataset
    df = load_data()

    # Preprocessing
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor
    ) = prepare_data(df)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train
    models = train_models(
        X_train, y_train, preprocessor
    )

    # Evaluate
    results = evaluate_models(
        models, X_test, y_test
    )

    # Save best model
    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(
        best_model,
        model_dir / "best_model.pkl"
    )

    print(f"\nBest model: {best_name}")
    print("Saved: models/best_model.pkl")
    print("Finished!")

if __name__ == "__main__":
    main()
