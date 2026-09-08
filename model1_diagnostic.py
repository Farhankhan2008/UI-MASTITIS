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
    confusion_matrix
)


# ============================================================
# MODEL 1 DIAGNOSTIC TEST
# ============================================================
# Purpose:
# Determine how much Model 1 depends on:
#
# 1. ALL FEATURES
# 2. SENSOR/ENVIRONMENT FEATURES ONLY
# 3. HISTORICAL FEATURES ONLY
#
# IMPORTANT:
# The train/test split is performed at COW level.
# The same cow will NEVER appear in both training and testing.
# ============================================================


FILE_NAME = "synthetic_mastitis_model1_dataset_v2.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 80)
print("MODEL 1 DIAGNOSTIC TEST")
print("BOVINE MASTITIS DAILY RISK PREDICTION")
print("=" * 80)

df = pd.read_csv(FILE_NAME)

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 80)

print(f"Total rows : {len(df)}")
print(f"Total cows : {df['cow_id'].nunique()}")

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
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: The following columns are missing:")

    for column in missing_columns:
        print(f" - {column}")

    raise SystemExit


print("\nAll required columns are present.")


# ============================================================
# 3. DEFINE FEATURE GROUPS
# ============================================================

# ------------------------------------------------------------
# FULL MODEL
# ------------------------------------------------------------

full_features = [
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent",
    "previous_mastitis",
    "days_since_last_mastitis"
]


# ------------------------------------------------------------
# SENSOR / ENVIRONMENT ONLY
# ------------------------------------------------------------
# Historical mastitis variables removed.
#
# This tells us whether the physiological/environmental
# measurements themselves contain useful predictive information.
# ------------------------------------------------------------

sensor_features = [
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent"
]


# ------------------------------------------------------------
# HISTORICAL FEATURES ONLY
# ------------------------------------------------------------
# This tells us how much predictive information is contained
# in previous mastitis history alone.
# ------------------------------------------------------------

history_features = [
    "previous_mastitis",
    "days_since_last_mastitis"
]


target = "mastitis_today"


# ============================================================
# 4. COW-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("=" * 80)
print("COW-LEVEL TRAIN / TEST SPLIT")
print("=" * 80)

unique_cows = df["cow_id"].unique()

train_cows, test_cows = train_test_split(
    unique_cows,
    test_size=0.20,
    random_state=42
)

train_df = df[
    df["cow_id"].isin(train_cows)
].copy()

test_df = df[
    df["cow_id"].isin(test_cows)
].copy()


print(f"\nTotal cows       : {len(unique_cows)}")
print(f"Training cows    : {len(train_cows)}")
print(f"Testing cows     : {len(test_cows)}")

print(f"\nTraining rows    : {len(train_df)}")
print(f"Testing rows     : {len(test_df)}")


# ============================================================
# 5. VERIFY NO COW LEAKAGE
# ============================================================

overlap = set(train_cows).intersection(
    set(test_cows)
)

print(f"\nCow overlap      : {len(overlap)}")

if len(overlap) > 0:

    print("\nERROR: Cow leakage detected!")
    raise SystemExit

else:

    print("No cow leakage detected.")


# ============================================================
# 6. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print("\nTraining target:")
print(train_df[target].value_counts())

print("\nTesting target:")
print(test_df[target].value_counts())


# ============================================================
# 7. MODEL TRAINING FUNCTION
# ============================================================

def train_and_evaluate(
    model_name,
    feature_list
):

    print("\n")
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    print("\nFeatures used:")

    for feature in feature_list:
        print(f" - {feature}")


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X_train = train_df[feature_list]
    y_train = train_df[target]

    X_test = test_df[feature_list]
    y_test = test_df[target]


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nMODEL PERFORMANCE")
    print("-" * 80)

    print(
        f"Accuracy             : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision            : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall / Sensitivity : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-Score             : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC              : "
        f"{roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("-" * 80)

    print("                 Predicted")
    print("                 Healthy   Mastitis")

    print(
        f"Actual Healthy   "
        f"{cm[0][0]:8d}   "
        f"{cm[0][1]:8d}"
    )

    print(
        f"Actual Mastitis  "
        f"{cm[1][0]:8d}   "
        f"{cm[1][1]:8d}"
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nFEATURE IMPORTANCE")
    print("-" * 80)

    importance_df = pd.DataFrame({
        "feature": feature_list,
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


    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_negatives": int(cm[0][0]),
        "true_positives": int(cm[1][1])
    }


# ============================================================
# 8. RUN THREE DIAGNOSTIC MODELS
# ============================================================

results = []


# ------------------------------------------------------------
# TEST 1: FULL MODEL
# ------------------------------------------------------------

results.append(
    train_and_evaluate(
        "TEST 1 - FULL MODEL",
        full_features
    )
)


# ------------------------------------------------------------
# TEST 2: SENSOR / ENVIRONMENT ONLY
# ------------------------------------------------------------

results.append(
    train_and_evaluate(
        "TEST 2 - SENSOR / ENVIRONMENT ONLY",
        sensor_features
    )
)


# ------------------------------------------------------------
# TEST 3: HISTORY ONLY
# ------------------------------------------------------------

results.append(
    train_and_evaluate(
        "TEST 3 - HISTORY ONLY",
        history_features
    )
)


# ============================================================
# 9. FINAL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 80)
print("FINAL DIAGNOSTIC COMPARISON")
print("=" * 80)

print()

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. INTERPRETATION
# ============================================================

print("\n")
print("=" * 80)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 80)


full_auc = results[0]["roc_auc"]
sensor_auc = results[1]["roc_auc"]
history_auc = results[2]["roc_auc"]


print("\nROC-AUC comparison:")

print(f"Full model       : {full_auc:.4f}")
print(f"Sensor-only      : {sensor_auc:.4f}")
print(f"History-only     : {history_auc:.4f}")


# ------------------------------------------------------------
# Difference between full and sensor-only
# ------------------------------------------------------------

auc_difference = full_auc - sensor_auc

print(
    f"\nFull vs Sensor-only ROC-AUC difference: "
    f"{auc_difference:.4f}"
)


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

print("\nINTERPRETATION")
print("-" * 80)


if history_auc >= 0.90:

    print(
        "WARNING: Historical features alone have very strong "
        "predictive power."
    )

    print(
        "We should investigate whether the synthetic dataset "
        "makes mastitis episodes too predictable from history."
    )

else:

    print(
        "Historical features alone do not completely explain "
        "mastitis prediction."
    )


if sensor_auc >= 0.90:

    print(
        "GOOD: Sensor/environment features alone provide "
        "strong predictive information."
    )

elif sensor_auc >= 0.80:

    print(
        "REASONABLE: Sensor/environment features provide "
        "useful predictive information."
    )

else:

    print(
        "WARNING: Sensor/environment-only performance is "
        "relatively weak."
    )


if full_auc >= 0.90:

    print(
        "GOOD: The complete Model 1 has strong discrimination."
    )

else:

    print(
        "The complete Model 1 may require further investigation."
    )


# ============================================================
# 11. CHECK WHETHER HISTORY DOMINATES
# ============================================================

print("\n")
print("=" * 80)
print("HISTORY DOMINANCE CHECK")
print("=" * 80)

if history_auc > sensor_auc + 0.10:

    print(
        "\nWARNING:"
    )

    print(
        "Historical features are substantially stronger than "
        "sensor/environment features."
    )

    print(
        "Do NOT modify the dataset yet."
    )

    print(
        "We should inspect the dataset generation and "
        "feature relationships before proceeding."
    )

else:

    print(
        "\nNo severe history dominance detected."
    )

    print(
        "Sensor/environment features are contributing useful "
        "predictive information."
    )


# ============================================================
# 12. CHECK FULL MODEL IMPROVEMENT
# ============================================================

print("\n")
print("=" * 80)
print("FULL MODEL IMPROVEMENT CHECK")
print("=" * 80)

full_vs_sensor = full_auc - sensor_auc
full_vs_history = full_auc - history_auc

print(
    f"\nFull model improvement over sensor-only : "
    f"{full_vs_sensor:.4f} ROC-AUC"
)

print(
    f"Full model improvement over history-only : "
    f"{full_vs_history:.4f} ROC-AUC"
)


if full_vs_sensor >= 0.02:

    print(
        "\nHistory features provide meaningful additional "
        "information beyond the sensor variables."
    )

else:

    print(
        "\nAdding history features provides only a small "
        "ROC-AUC improvement over sensor/environment data."
    )


# ============================================================
# 13. FINAL DECISION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL DECISION")
print("=" * 80)

print(
    """
This diagnostic does NOT replace the original Model 1.

It is only used to understand what the model is learning.

After reviewing these results, we will decide whether:

1. Keep the current Model 1
2. Adjust the feature configuration
3. Investigate the synthetic data
4. Proceed to Model 2
"""
)


# ============================================================
# 14. SAVE DIAGNOSTIC RESULTS
# ============================================================

results_df.to_csv(
    "model1_diagnostic_results.csv",
    index=False
)

print("\nDiagnostic results saved as:")
print("model1_diagnostic_results.csv")


print("\n")
print("=" * 80)
print("MODEL 1 DIAGNOSTIC TEST COMPLETE")
print("=" * 80)

