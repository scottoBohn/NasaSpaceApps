#!/usr/bin/env python3
"""
Train classifier on train.pkl, validate on val.pkl, test on test.pkl.
Outputs per-row predictions & probabilities for EACH test sample,
plus global metrics: accuracy, std dev, Lin's CCC.
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Path setup (adjust if needed) ---
BASE_DIR = r"C:\Users\btau4\OneDrive\Desktop"
TRAIN_PATH = os.path.join(BASE_DIR, "train.pkl")
VAL_PATH   = os.path.join(BASE_DIR, "val.pkl")
TEST_PATH  = os.path.join(BASE_DIR, "test.pkl")

TARGET = "koi_pdisposition"

def load_df(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "dataframe" in data:
        return data["dataframe"]
    return data

print("🔹 Loading datasets...")
train_df = load_df(TRAIN_PATH)
val_df   = load_df(VAL_PATH)
test_df  = load_df(TEST_PATH)

# --- Basic hygiene ---
for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if TARGET not in df.columns:
        raise ValueError(f"[{df_name}] Missing target column: {TARGET}")

train_df = train_df.dropna(subset=[TARGET])
val_df   = val_df.dropna(subset=[TARGET])
test_df  = test_df.dropna(subset=[TARGET])

# --- Encode target ---
le = LabelEncoder()
train_df[TARGET] = le.fit_transform(train_df[TARGET])
val_df[TARGET]   = le.transform(val_df[TARGET])
test_df[TARGET]  = le.transform(test_df[TARGET])

# --- Split X/y ---
X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_val,   y_val   = val_df.drop(columns=[TARGET]),   val_df[TARGET]
X_test,  y_test  = test_df.drop(columns=[TARGET]),  test_df[TARGET]

# --- Encode categoricals -> codes; fillna; scale ---
def prep(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df.fillna(0)

X_train = prep(X_train)
X_val   = prep(X_val)
X_test  = prep(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# --- Model ---
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("🔹 Training...")
model.fit(X_train_s, y_train)

# --- Validation quick check ---
val_preds = model.predict(X_val_s)
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.3f}")

# --- Test predictions (per-row) ---
test_preds = model.predict(X_test_s)
test_probs = model.predict_proba(X_test_s)  # shape: [n_rows, n_classes]
test_acc   = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc:.3f}")

# --- Global stats requested ---
std_dev = np.std(test_preds)  # std of numeric label predictions
def lins_ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = np.cov(x, y, bias=False)[0,1]
    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)

ccc = lins_ccc(y_test, test_preds)
print(f"Std Dev (pred labels): {std_dev:.4f}")
print(f"Lin's CCC: {ccc:.4f}")

# --- Build per-row detailed output ---
# Start from original X_test (unscaled) so columns are readable
detailed = X_test.copy()

# Add actual & predicted (string labels)
detailed["Actual"]    = le.inverse_transform(y_test)
detailed["Predicted"] = le.inverse_transform(test_preds)

# Correct / Error flags
detailed["Correct"] = (detailed["Actual"] == detailed["Predicted"]).astype(int)
detailed["Error"]   = 1 - detailed["Correct"]

# Confidence (max class probability)
max_conf = test_probs.max(axis=1)
detailed["Confidence"] = max_conf

# Per-class probability columns (aligned with le.classes_)
for i, cls in enumerate(le.classes_):
    detailed[f"proba_{cls}"] = test_probs[:, i]

# Save per-row CSV
perrow_csv = os.path.join(BASE_DIR, "exoplanet_test_predictions_detailed.csv")
detailed.to_csv(perrow_csv, index=False)
print(f"✅ Per-row predictions saved → {perrow_csv}")

# --- Also print a quick summary report ---
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# --- Save a tiny metrics CSV too (optional) ---
metrics_csv = os.path.join(BASE_DIR, "exoplanet_test_metrics.csv")
pd.DataFrame({
    "metric": ["test_accuracy", "pred_std_dev", "lins_ccc"],
    "value":  [test_acc,        float(std_dev), float(ccc)],
}).to_csv(metrics_csv, index=False)
print(f"✅ Metrics saved → {metrics_csv}")
