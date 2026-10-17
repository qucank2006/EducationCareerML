
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET = "Starting_Salary"

FEATURES = [
    "Age",
    "Gender",
    "High_School_GPA",
    "SAT_Score",
    "University_Ranking",
    "University_GPA",
    "Field_of_Study",
    "Internships_Completed",
    "Projects_Completed",
    "Certifications",
    "Soft_Skills_Score",
    "Networking_Score"
]

def prepare_data(df):
    if TARGET not in df.columns:
        raise ValueError(
            f"Dataset thieu cot {TARGET}"
        )

    features = [
        col for col in FEATURES
        if col in df.columns
    ]

    if not features:
        raise ValueError("Khong tim thay dac trung")

    data = df[features + [TARGET]].copy()

    data[TARGET] = pd.to_numeric(
        data[TARGET], errors="coerce"
    )

    data = data.dropna(subset=[TARGET])

    X = data[features]
    y = data[TARGET]

    numeric_cols = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train, X_test,
        y_train, y_test,
        preprocessor
    )
