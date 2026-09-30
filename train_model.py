"""Rebuild the final model from the notebook (Sections 3, 8, 10, 11, 12).

Run once:  python train_model.py
Output:    model/final_model_hotel_cancellation.pkl

The hyperparameters below are the best ones found by RandomizedSearchCV in the
notebook (Section 10.2), so no tuning is repeated here.
"""
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import CAT_COLS, load_and_clean, threshold_table

MODEL_PATH = "model/final_model_hotel_cancellation.pkl"

# 1. Clean data (identical to notebook Section 3)
df, _ = load_and_clean()

# 2. Preprocessing (Section 8): top-15 countries + "Other", one-hot + scaling
top_countries = df["country"].value_counts().head(15).index.tolist()
df["country_grp"] = df["country"].apply(lambda c: c if c in top_countries else "Other")
df = df.drop(columns=["country"])

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]
num_cols = [c for c in X.columns if c not in CAT_COLS]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ("num", StandardScaler(), num_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 3. Final model: tuned LightGBM (Section 10.2 best params)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1,
        subsample=0.85, num_leaves=63, n_estimators=350,
        learning_rate=0.05, colsample_bytree=0.7,
    )),
])
model.fit(X_train, y_train)

# 4. Evaluate on the test set + pick the cost-optimal threshold (Section 11.3)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f} | PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

table = threshold_table(y_test, y_prob)
best = table.loc[table["Net benefit"].idxmax()]
print(f"Best threshold: {best['Threshold']} | Net benefit: {best['Net benefit']:.0f}")
print(best[["TN", "FP", "FN", "TP"]].astype(int).to_dict())

# 5. Feature importance (top 15, native LightGBM importance)
names = model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(model.named_steps["classifier"].feature_importances_, index=names)
importance = importance.sort_values(ascending=False).head(15)

# 6. Save everything the app needs
artifact = {
    "model": model,
    "model_name": "LightGBM",
    "threshold": float(best["Threshold"]),
    "feature_columns": X.columns.tolist(),
    "cat_cols": CAT_COLS,
    "num_cols": num_cols,
    "top_countries": top_countries,
    # used by the app's Model page (live cost calculator + feature importance)
    "y_test": y_test.to_numpy(),
    "y_prob_test": y_prob,
    "feature_importance": importance.to_dict(),
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)
print("Saved ->", MODEL_PATH)
