
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_models(X_train, y_train, preprocessor):
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training: {name}")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    return trained_models
