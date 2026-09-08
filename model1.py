import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_NAME = "synthetic_mastitis_model1_dataset_v2.csv"

df = pd.read_csv(FILE_NAME)

print("=" * 70)
print("MODEL 1 - BOVINE MASTITIS DAILY RISK PREDICTION")
print("=" * 70)

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 70)
print(f"Total rows: {len(df)}")
print(f"Total cows: {df['cow_id'].nunique()}")

# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "cow_id",
    "date",
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent",
    "previous_mastitis",
    "days_since_last_mastitis",
    "mastitis_today"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print(col)
    raise SystemExit

print("\nAll required columns are present.")

# ============================================================
# 3. BASIC VALIDATION
# ============================================================

print("\nTARGET DISTRIBUTION")
print("-" * 70)

target_counts = df["mastitis_today"].value_counts()

print(target_counts)

print("\nTarget percentages:")
print(
    df["mastitis_today"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

features = [
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent",
    "previous_mastitis",
    "days_since_last_mastitis"
]

target = "mastitis_today"

X = df[features]
y = df[target]

# ============================================================
# 5. COW-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\nCOW-LEVEL TRAIN / TEST SPLIT")
print("-" * 70)

unique_cows = df["cow_id"].unique()

train_cows, test_cows = train_test_split(
    unique_cows,
    test_size=0.20,
    random_state=42
)

train_df = df[df["cow_id"].isin(train_cows)].copy()
test_df = df[df["cow_id"].isin(test_cows)].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print(f"Total cows       : {len(unique_cows)}")
print(f"Training cows    : {len(train_cows)}")
print(f"Testing cows     : {len(test_cows)}")

print(f"\nTraining rows    : {len(train_df)}")
print(f"Testing rows     : {len(test_df)}")

# ============================================================
# 6. VERIFY NO COW OVERLAP
# ============================================================

overlap = set(train_cows).intersection(set(test_cows))

print(f"\nCow overlap      : {len(overlap)}")

if len(overlap) != 0:
    print("ERROR: Cow leakage detected!")
    raise SystemExit
else:
    print("No cow leakage detected.")

# ============================================================
# 7. TARGET DISTRIBUTION AFTER SPLIT
# ============================================================

print("\nTRAINING TARGET DISTRIBUTION")
print("-" * 70)
print(y_train.value_counts())

print("\nTESTING TARGET DISTRIBUTION")
print("-" * 70)
print(y_test.value_counts())

# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\nTRAINING MODEL")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed.")

# ============================================================
# 9. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of mastitis = class 1 probability
y_prob = model.predict_proba(X_test)[:, 1]

# Convert probability to percentage
risk_percentage = y_prob * 100

# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\nMODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy              : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision             : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall / Sensitivity  : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score              : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC               : {roc_auc:.4f}")

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("=" * 70)

print("                 Predicted")
print("                 Healthy   Mastitis")
print(f"Actual Healthy   {cm[0][0]:8d}   {cm[0][1]:8d}")
print(f"Actual Mastitis  {cm[1][0]:8d}   {cm[1][1]:8d}")

# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Healthy", "Mastitis"],
        zero_division=0
    )
)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\nFEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

for _, row in importance_df.iterrows():
    print(
        f"{row['feature']:35s} : "
        f"{row['importance']:.4f}"
    )

# ============================================================
# 15. EXAMPLE RISK OUTPUT
# ============================================================

print("\nEXAMPLE MODEL 1 OUTPUT")
print("=" * 70)

example_count = min(10, len(test_df))

example_results = test_df[
    ["cow_id", "date", "mastitis_today"]
].iloc[:example_count].copy()

example_results["predicted_risk_percent"] = (
    risk_percentage[:example_count]
)

example_results["predicted_class"] = y_pred[:example_count]

print(example_results.to_string(index=False))

# ============================================================
# 16. SAVE MODEL 1
# ============================================================

import joblib

MODEL_FILE = "mastitis_model1.pkl"

joblib.dump(model, MODEL_FILE)

print("\nMODEL SAVED")
print("-" * 70)
print(f"Saved as: {MODEL_FILE}")

print("\nMODEL 1 TRAINING COMPLETE")
print("=" * 70)