"""Generate synthetic churn data, train models, evaluate, and save artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    build_preprocessor, load_dataset,
)
from src.evaluation import calculate_metrics

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"


def generate_synthetic_data(n_rows=1500, random_state=42):
    """Create clearly labelled synthetic subscription customer records."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_rows)
    monthly = np.clip(rng.normal(72, 22, n_rows), 18, 150).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows,
                          p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber", "DSL", "No"], n_rows, p=[0.52, 0.32, 0.16])
    tickets = np.clip(rng.poisson(1.5, n_rows), 0, 10)
    late = np.clip(rng.poisson(0.8, n_rows), 0, 8)
    usage = np.clip(rng.normal(180, 75, n_rows), 5, 500).round(1)
    total = (monthly * tenure * rng.uniform(0.88, 1.12, n_rows)).round(2)

    # Probabilistic target generation creates learnable but non-deterministic patterns.
    logit = (
        -2.0
        + 1.05 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.12 * tickets
        + 0.18 * late
        - 0.025 * tenure
        + 0.008 * (monthly - 70)
        + 0.45 * (internet == "Fiber")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    data = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_rows + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": contract,
        "internet_service": internet,
        "support_tickets": tickets,
        "late_payments": late,
        "monthly_data_gb": usage,
        "churn": churn,
    })
    # A small amount of realistic missingness exercises the imputation pipeline.
    for column in ["monthly_charges", "total_charges", "internet_service"]:
        mask = rng.random(n_rows) < 0.015
        data.loc[mask, column] = np.nan
    return data


def train():
    for folder in [ROOT / "data", ROOT / "models", ROOT / "reports"]:
        folder.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        generate_synthetic_data().to_csv(DATA_PATH, index=False)
        print(f"Generated synthetic dataset: {DATA_PATH}")
    data = load_dataset(DATA_PATH)
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    logistic = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    forest = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=1
        )),
    ])

    logistic_cv = cross_val_score(logistic, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    param_grid = {
        "classifier__n_estimators": [40, 70],
        "classifier__max_depth": [None, 8],
    }
    search = GridSearchCV(
        forest, param_grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    metrics = calculate_metrics(best_model, x_test, y_test)
    metrics.update({
        "dataset_type": "Synthetic demonstration data",
        "dataset_rows": int(len(data)),
        "positive_class": "Churned",
        "random_forest_best_params": search.best_params_,
        "random_forest_cv_roc_auc_mean": float(search.best_score_),
        "logistic_regression_cv_roc_auc_mean": float(logistic_cv.mean()),
        "logistic_regression_cv_roc_auc_std": float(logistic_cv.std()),
        "train_samples": int(len(x_train)),
        "note": "All held-out metrics are calculated from actual test-set predictions.",
    })
    joblib.dump({
        "model": best_model,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "model_name": "Tuned Random Forest",
        "threshold": 0.50,
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\nHeld-out test metrics:")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    train()
