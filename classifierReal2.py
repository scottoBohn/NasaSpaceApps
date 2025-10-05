#!/usr/bin/env python3
"""
Exoplanet classifier (strict CLI: no hard-coded filenames/paths)

REQUIRES:
  --train <path>  --val <path>  --test <path>

Optional:
  --target <colname>            (default: koi_pdisposition)
  --drop-col <col> (repeatable) (e.g., --drop-col host_id --drop-col planet_id)
  --outdir <dir>                (default: .)
  --prefix <str>                (default: exoplanet)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- CLI (all data files required) ----------
parser = argparse.ArgumentParser(description="Exoplanet classifier (strict CLI).")
parser.add_argument("--train", required=True, help="Path to TRAIN PKL")
parser.add_argument("--val",   required=True, help="Path to VALIDATION PKL")
parser.add_argument("--test",  required=True, help="Path to TEST PKL")

parser.add_argument("--target", default="koi_pdisposition",
                    help="Target column name (default: koi_pdisposition)")
parser.add_argument("--drop-col", dest="drop_cols", action="append", default=[],
                    help="Column to drop from features (repeatable)")
parser.add_argument("--outdir", default=".", help="Where to write outputs (default: .)")
parser.add_argument("--prefix", default="exoplanet", help="Output filename prefix (default: exoplanet)")
args = parser.parse_args()

TRAIN_PATH = Path(args.train).resolve()
VAL_PATH   = Path(args.val).resolve()
TEST_PATH  = Path(args.test).resolve()
OUTDIR     = Path(args.outdir).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)
TARGET     = args.target
DROP_COLS  = ["host_id", "host_name", "planet_id"]
if args.drop_cols:
    # merge & de-dup
    DROP_COLS = list(dict.fromkeys(DROP_COLS + args.drop_cols))

# ---------- Existence check ----------
for name, p in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
    if not p.exists():
        raise SystemExit(f"[ERROR] {name} file not found: {p}")

print("[INFO] Using files:")
print("  TRAIN:", TRAIN_PATH)
print("  VAL:  ", VAL_PATH)
print("  TEST: ", TEST_PATH)
print(f"[INFO] Target: {TARGET}")
print(f"[INFO] Outdir: {OUTDIR}")
print(f"[INFO] Drop columns: {DROP_COLS or '[]'}")

# ---------- Loaders ----------
def load_df(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "dataframe" in data:
        return data["dataframe"]
    if isinstance(data, pd.DataFrame):
        return data
    raise SystemExit(f"[ERROR] {path} must be a DataFrame PKL or dict with key 'dataframe'.")

print("🔹 Loading datasets...")
train_df = load_df(TRAIN_PATH)
val_df   = load_df(VAL_PATH)
test_df  = load_df(TEST_PATH)

# ---------- Basic hygiene ----------
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if TARGET not in df.columns:
        raise SystemExit(f"[ERROR] [{name}] missing target column: {TARGET}")
    before = len(df)
    df.dropna(subset=[TARGET], inplace=True)
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows from {name} with NaN target.")

# ---------- Encode target (handle unseen labels safely) ----------
le = LabelEncoder()
train_df[TARGET] = le.fit_transform(train_df[TARGET])

def safe_transform(labels: pd.Series, le_: LabelEncoder) -> np.ndarray:
    known = set(le_.classes_)
    fallback = le_.classes_[0]
    labels = labels.astype(object).where(labels.isin(known), other=fallback)
    return le_.transform(labels)

val_df[TARGET]  = safe_transform(val_df[TARGET], le)
test_df[TARGET] = safe_transform(test_df[TARGET], le)

# ---------- Features / labels ----------
def Xy(df: pd.DataFrame):
    cols_to_drop = [c for c in [TARGET] + DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[TARGET]
    return X, y

X_train, y_train = Xy(train_df)
X_val,   y_val   = Xy(val_df)
X_test,  y_test  = Xy(test_df)

print(f"[INFO] Feature count: {X_train.shape[1]}")

# ---------- Preprocess ----------
def prep(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- Model ----------
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("🔹 Training...")
model.fit(X_train_s, y_train)

# ---------- Metrics ----------
val_preds = model.predict(X_val_s)
test_preds = model.predict(X_test_s)
test_probs = model.predict_proba(X_test_s)

val_acc  = accuracy_score(y_val,  val_preds)
test_acc = accuracy_score(y_test, test_preds)
print(f"Validation Accuracy: {val_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

def lins_ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = np.cov(x, y, bias=False)[0, 1]
    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)

std_dev = float(np.std(test_preds))
ccc     = float(lins_ccc(y_test, test_preds))
print(f"Std Dev (pred labels): {std_dev:.4f}")
print(f"Lin's CCC: {ccc:.4f}")

# ---------- Per-row detailed output ----------
detailed = X_test.copy()
detailed["Actual"]    = le.inverse_transform(y_test)
detailed["Predicted"] = le.inverse_transform(test_preds)
detailed["Correct"]   = (detailed["Actual"] == detailed["Predicted"]).astype(int)
detailed["Error"]     = 1 - detailed["Correct"]
detailed["Confidence"] = test_probs.max(axis=1)
for i, cls in enumerate(le.classes_):
    detailed[f"proba_{cls}"] = test_probs[:, i]

perrow_csv = OUTDIR / f"{args.prefix}_test_predictions_detailed.csv"
detailed.to_csv(perrow_csv, index=False)
print(f"✅ Per-row predictions saved → {perrow_csv}")

# ---------- Summary report ----------
print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# ---------- Tiny metrics CSV ----------
metrics_csv = OUTDIR / f"{args.prefix}_test_metrics.csv"
pd.DataFrame({
    "metric": ["val_accuracy", "test_accuracy", "pred_std_dev", "lins_ccc"],
    "value":  [float(val_acc), float(test_acc), std_dev, ccc],
}).to_csv(metrics_csv, index=False)
print(f"✅ Metrics saved → {metrics_csv}")
