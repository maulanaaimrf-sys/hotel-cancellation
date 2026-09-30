"""Shared helpers used by both train_model.py and app.py.

Keeping the cleaning logic in one place guarantees the app sees exactly the
same data the model was trained on.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_PATH = "data/hotel_bookings_dataset.csv"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Columns dropped because they are only known AFTER the booking outcome (data leakage)
# or would let the model memorise specific dates.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type",
                "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"]

CAT_COLS = ["hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "customer_type", "country_grp"]


def load_and_clean(path=DATA_PATH):
    """Reproduce the notebook's Section 3. Returns (clean_df, steps)."""
    raw = pd.read_csv(path)
    steps = [{"Step": "Raw data", "Rows": len(raw), "Cancellation rate": raw["is_canceled"].mean()}]

    df = raw.drop_duplicates().reset_index(drop=True)
    steps.append({"Step": "Remove duplicates", "Rows": len(df), "Cancellation rate": df["is_canceled"].mean()})

    df["children"] = df["children"].fillna(0)
    df = df.dropna(subset=["country"]).drop(columns=["agent", "company"]).reset_index(drop=True)
    steps.append({"Step": "Handle missing values", "Rows": len(df), "Cancellation rate": df["is_canceled"].mean()})

    no_guest = (df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)
    df = df[~no_guest]
    df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]
    df = df[df["market_segment"] != "Undefined"]
    df = df[df["distribution_channel"] != "Undefined"].reset_index(drop=True)
    steps.append({"Step": "Remove anomalies", "Rows": len(df), "Cancellation rate": df["is_canceled"].mean()})

    df = df.drop(columns=LEAKAGE_COLS)
    return df, steps


def predict_risk(new_bookings, artifact):
    """Score new bookings. Needs the raw booking columns (including 'country')."""
    data = new_bookings.copy()
    if "country" in data.columns:
        top = artifact["top_countries"]
        data["country_grp"] = data["country"].apply(lambda c: c if c in top else "Other")

    missing = [c for c in artifact["feature_columns"] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[artifact["feature_columns"]].copy()
    data["children"] = data["children"].fillna(0)

    prob = artifact["model"].predict_proba(data)[:, 1]
    result = new_bookings.copy()
    result["risk_score"] = prob.round(4)
    result["risk_label"] = np.where(prob >= artifact["threshold"],
                                    "At risk of cancelling", "Likely to show up")
    return result


def cost_of_predictions(y_true, y_pred, retention_cost=1, empty_room_cost=10):
    """Same cost logic as the notebook (Section 11.3).

    FP = guest flagged as risky but would have come  -> wasted retention effort
    FN = guest cancels without being flagged         -> unexpected empty room
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    loss = fp * retention_cost + fn * empty_room_cost
    gain = tn * retention_cost + tp * empty_room_cost
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "Net benefit": gain - loss}


def threshold_table(y_true, prob, retention_cost=1, empty_room_cost=10):
    """Net benefit for every threshold from 0.10 to 0.625 (same grid as the notebook)."""
    rows = []
    for t in np.arange(0.10, 0.65, 0.025):
        row = cost_of_predictions(y_true, (prob >= t).astype(int), retention_cost, empty_room_cost)
        row["Threshold"] = round(float(t), 3)
        rows.append(row)
    return pd.DataFrame(rows)[["Threshold", "TN", "FP", "FN", "TP", "Net benefit"]]
