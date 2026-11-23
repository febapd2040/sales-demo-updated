"""Shared logic for app.py (Streamlit dashboard) and program.py (console version)."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Ad_Spend", "Price", "Economy", "Competition"]
TARGET = "Sales"
EXAMPLE = dict(zip(FEATURES, [3500, 8, 70, 5]))
LIMITATIONS = [
    "The dataset is simulated and does not represent a real business.",
    "Only 90 records are available, so results may change with other data.",
    "Correlation does not prove that a factor causes sales to change.",
    "Related input features can make individual effects hard to separate.",
    "Test metrics are estimates based on this particular train/test split.",
    "Predictions can be uncertain even within the observed data range.",
]


def load_data(path=None):
    path = path or Path(__file__).resolve().parent / "data.csv"
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 10:
        raise ValueError("At least 10 valid records are required.")
    return df.reset_index(drop=True)


def relationship_label(r):
    strength = "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.3 else None
    if not strength:
        return "Weak or no linear relationship"
    return f"{strength} {'positive' if r > 0 else 'negative'} relationship"


def train(data):
    """Fit the scaled model (+ an unscaled one for original-unit coefficients)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return dict(
        model=model, X_train=X_tr, X_test=X_te, y_test=y_te, pred=pred,
        mae=mean_absolute_error(y_te, pred), r2=r2_score(y_te, pred),
        std_coef=model[-1].coef_,
        raw_coef=LinearRegression().fit(X_tr, y_tr).coef_)


def sections(data, corr, r):
    """Ordered {title: text} used for both the console output and the .txt report."""
    coefs = lambda c: "\n".join(f"{f}: {v:.3f}" for f, v in zip(FEATURES, c))
    example_pred = r["model"].predict(pd.DataFrame([EXAMPLE]))[0]
    return {
        "BUSINESS DATASET": (
            f"{data.to_string(index=False)}\n\nDataset rows: {len(data)}\n"
            f"Dataset columns: {data.shape[1]}\n"
            "Data source: Simulated sample data for demonstration"),
        "CORRELATION MATRIX": corr.to_string(),
        "MODEL EVALUATION": (
            f"Training records: {len(r['X_train'])}\nTesting records: {len(r['X_test'])}\n"
            f"MAE = {r['mae']:.3f} sales units\nR² = {r['r2']:.3f}\n"
            "MAE is the average absolute prediction error; lower is better.\n"
            "R² measures how well the model explains sales variation on the test set."),
        "CORRELATION-BASED BUSINESS INSIGHTS": "\n".join(
            f"{f} -> {relationship_label(corr.loc[f, TARGET])} "
            f"(corr = {corr.loc[f, TARGET]:.2f})" for f in FEATURES),
        "MODEL COEFFICIENTS": (
            f"Standardised:\n{coefs(r['std_coef'])}\n\nOriginal units:\n{coefs(r['raw_coef'])}\n\n"
            "A coefficient estimates the change in predicted Sales for a one-unit\n"
            "increase in that feature, while the other features are held constant.\n"
            "These are associations, not proof of cause and effect."),
        "EXAMPLE SALES PREDICTION": (
            "\n".join(f"{k} = {v}" for k, v in EXAMPLE.items())
            + f"\nPredicted Sales = {example_pred:.2f}"),
        "LIMITATIONS": "\n".join(f"- {x}" for x in LIMITATIONS),
    }


def report_text(data, corr, r):
    body = "\n\n".join(f"{t}\n{body}" for t, body in sections(data, corr, r).items())
    return ("BUSINESS PREDICTOR | COMPLETE BUSINESS ANALYSIS REPORT\n" + "=" * 55
            + "\n\n" + body + "\n\nNote: Simulated educational data. Predictions are "
            "exploratory estimates and do not prove causation or guarantee real-world results.")