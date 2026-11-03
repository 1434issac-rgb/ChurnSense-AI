"""Command-line prediction helper."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"


def predict_customer(customer):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    row = pd.DataFrame([customer])
    probability = float(model.predict_proba(row)[0, 1])
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    return {"churn_probability": probability, "risk": risk,
            "prediction": int(probability >= bundle.get("threshold", 0.5))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict churn for one customer.")
    parser.add_argument("--tenure", type=int, default=12)
    parser.add_argument("--monthly", type=float, default=79.9)
    parser.add_argument("--total", type=float, default=958.8)
    parser.add_argument("--contract", choices=["Month-to-month", "One year", "Two year"],
                        default="Month-to-month")
    parser.add_argument("--internet", choices=["Fiber", "DSL", "No"], default="Fiber")
    parser.add_argument("--tickets", type=int, default=2)
    parser.add_argument("--late-payments", type=int, default=1)
    parser.add_argument("--usage", type=float, default=180.0)
    args = parser.parse_args()
    customer = {
        "tenure_months": args.tenure, "monthly_charges": args.monthly,
        "total_charges": args.total, "contract_type": args.contract,
        "internet_service": args.internet, "support_tickets": args.tickets,
        "late_payments": args.late_payments, "monthly_data_gb": args.usage,
    }
    print(predict_customer(customer))
