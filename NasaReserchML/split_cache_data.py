#!/usr/bin/env python3
"""
Splits a cache.pkl dataset into 60% train, 20% validation, and 20% test sets.
Saves them as separate pickle files.
"""

import pickle
import os
from sklearn.model_selection import train_test_split

# --- Path setup ---
BASE_DIR = r"C:\Users\btau4\OneDrive\Desktop"
CACHE_FILE = os.path.join(BASE_DIR, "cache.pkl")

# --- Load cached dataframe ---
print(f"Loading data from {CACHE_FILE} ...")
with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

# Some cache files from previous scripts contain the data under 'dataframe'
df = cache["dataframe"] if isinstance(cache, dict) and "dataframe" in cache else cache

print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

# --- Split data ---
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")

# --- Save results ---
train_path = os.path.join(BASE_DIR, "train.pkl")
val_path = os.path.join(BASE_DIR, "val.pkl")
test_path = os.path.join(BASE_DIR, "test.pkl")

for subset, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
    with open(path, "wb") as f:
        pickle.dump({"dataframe": eval(f"{subset}_df")}, f)
    print(f"Saved {subset} data → {path}")

print("✅ Split complete.")
