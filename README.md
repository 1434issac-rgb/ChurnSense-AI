# ChurnSense AI — Intelligent Customer Churn Prediction

ChurnSense AI is a GitHub-ready, end-to-end binary classification project that estimates the likelihood that a subscription customer will churn. It includes synthetic data generation, preprocessing, model comparison, cross-validation, hyperparameter tuning, held-out evaluation, a saved model, a CLI prediction utility, and a Streamlit dashboard.

> **Dataset transparency:** The included customer records are synthetically generated for demonstration and learning. They are not real customer records and do not represent any actual company.

## Problem statement

Customer churn reduces recurring revenue and can indicate gaps in customer experience. This project demonstrates how historical customer attributes can be used to estimate churn risk and support retention prioritization.

## Objectives

- Build a reproducible binary classification workflow.
- Compare Logistic Regression with a tuned Random Forest.
- Handle missing numeric and categorical values in a preprocessing pipeline.
- Evaluate on a stratified, held-out test set.
- Present customer-level predictions and model diagnostics in a dashboard.

## Technology

Python 3.10+, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib, Git, and GitHub.

## Dataset

`data/customer_churn.csv` contains 1,500 synthetic rows by default, with:
- `customer_id`: synthetic customer identifier
- `tenure_months`: subscription tenure
- `monthly_charges`, `total_charges`: billing values
- `contract_type`: month-to-month, one year, or two year
- `internet_service`: Fiber, DSL, or No
- `support_tickets`, `late_payments`: service and payment indicators
- `monthly_data_gb`: monthly data consumption
- `churn`: target label (`1` = churned, `0` = stayed)

A small proportion of feature values are intentionally missing to exercise imputation. The target is generated probabilistically from selected customer attributes. The data generator and random seed are in `train.py`.

## Machine learning workflow

1. Generate the synthetic dataset if it is not present.
2. Split features and target; create a stratified 80/20 train-test split.
3. Impute numeric values with medians and categorical values with most-frequent values.
4. Standardize numeric features and one-hot encode categorical features.
5. Compare Logistic Regression using 3-fold stratified cross-validation.
6. Tune Random Forest hyperparameters using 3-fold stratified cross-validation and ROC-AUC scoring.
7. Evaluate the selected model on the untouched test set.
8. Save the fitted pipeline and calculated metrics.

Reported metrics include accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix. The values in `reports/model_metrics.json` are generated from actual held-out predictions when you run training; they are not manually invented. Because the dataset is synthetic, performance should not be interpreted as evidence of real-world business performance.

## Installation (macOS / VS Code)

Open the project folder in VS Code, then run these commands in its integrated terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This creates or refreshes:
- `data/customer_churn.csv`
- `models/churn_model.joblib`
- `reports/model_metrics.json`

## Launch the dashboard

```bash
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) to open in your browser.

## Optional: predict from the terminal

```bash
python predict.py --tenure 8 --monthly 89.5 --total 716 --contract Month-to-month --internet Fiber --tickets 3 --late-payments 2 --usage 220
```

## Dashboard

The Streamlit application includes:
- Overview cards and churn/contract distributions
- Customer input form and churn probability
- Low / Medium / High risk categorization
- Held-out model metrics and confusion matrix
- Random Forest feature importance and CV model comparison

A preview image is included at `assets/dashboard_preview.png`. The dashboard itself is interactive and is rendered by Streamlit.

## Project structure

```text
ChurnSense-AI/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/customer_churn.csv
├── models/churn_model.joblib
├── notebooks/exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── reports/model_metrics.json
└── assets/dashboard_preview.png
```

## Notes and limitations

- Predictions are estimates and should not be used as the sole basis for customer decisions.
- Synthetic data can validate code execution but cannot establish real-world predictive value.
- For production use, replace the synthetic dataset with appropriately governed, consented, representative data; review fairness, privacy, calibration, drift, and business costs.
- The risk bands (below 40%, 40–70%, and 70%+) are illustrative thresholds.

## Future improvements

- Train and validate on a real, documented churn dataset.
- Add probability calibration and threshold optimization based on retention costs.
- Add explainability with SHAP and per-customer explanations.
- Add model monitoring, drift detection, and experiment tracking.
- Add authentication and role-based access for production deployment.

## License

Choose and add a license before distributing this repository.
