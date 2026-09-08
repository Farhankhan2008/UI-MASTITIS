import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)

MODEL_7D_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model2_v4_7d.pkl"
)

MODEL_14D_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model2_v4_14d.pkl"
)

DATASET_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_classification_dataset.csv"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_feature_importance.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HIGH_RISK_THRESHOLD = 50.0

RANDOM_STATE = 42

N_ESTIMATORS = 500


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 78)
print("                 MODEL 2 V4 TRAINING")
print("          LONG-TERM HIGH-RISK CLASSIFICATION")
print("=" * 78)

print()
print("Purpose:")
print("Predict whether a cow will enter the HIGH-RISK state")
print("within the next 7 days or 14 days.")
print()

print("High-risk definition:")
print(f"Model 1 risk >= {HIGH_RISK_THRESHOLD:.0f}%")
print()

print("Input dataset:")
print(INPUT_FILE)
print()


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nERROR: Dataset not found:\n{INPUT_FILE}\n"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 78)
print("STEP 1 — DATASET LOADED")
print("=" * 78)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Unique cows       : {df['cow_id'].nunique():,}")

print()


# ============================================================
# STEP 2 — VALIDATE REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "cow_id",
    "date",
    "risk_today"
]

for column in REQUIRED_COLUMNS:
    if column not in df.columns:
        raise ValueError(
            f"ERROR: Required column missing: {column}"
        )

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# STEP 3 — CHECK DATA INTEGRITY
# ============================================================

print("=" * 78)
print("STEP 2 — DATA INTEGRITY CHECK")
print("=" * 78)

duplicate_count = df.duplicated(
    subset=["cow_id", "date"]
).sum()

missing_count = df.isnull().sum().sum()

print(f"Duplicate cow/date records : {duplicate_count}")
print(f"Missing cells              : {missing_count}")

if duplicate_count != 0:
    raise ValueError(
        "ERROR: Duplicate cow/date records detected."
    )

if missing_count != 0:
    raise ValueError(
        "ERROR: Missing values detected."
    )

print("Data integrity check PASSED.")

print()


# ============================================================
# STEP 4 — DEFINE SAFE FEATURES
# ============================================================

# These are FUTURE TARGETS from V3.
#
# They MUST NOT be used as Model 2 V4 input features.
#
# Otherwise the model would see future information during
# training, causing target leakage.

FUTURE_TARGET_COLUMNS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d",
    "risk_14d"
]

EXCLUDED_COLUMNS = [
    "cow_id",
    "date"
] + FUTURE_TARGET_COLUMNS

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


print("=" * 78)
print("STEP 3 — FEATURE SELECTION")
print("=" * 78)

print(f"Total V3 columns       : {len(df.columns)}")
print(f"Excluded columns       : {len(EXCLUDED_COLUMNS)}")
print(f"V4 input features      : {len(FEATURE_COLUMNS)}")

print()

print("Excluded:")
for column in EXCLUDED_COLUMNS:
    print(f"  - {column}")

print()


# ============================================================
# STEP 5 — CREATE STRICT FUTURE HIGH-RISK TARGETS
# ============================================================

print("=" * 78)
print("STEP 4 — CREATING V4 TARGETS")
print("=" * 78)

print()
print("7-day target:")
print(
    "1 = cow reaches >= 50% risk on at least one of the "
    "NEXT 7 complete days"
)

print()
print("14-day target:")
print(
    "1 = cow reaches >= 50% risk on at least one of the "
    "NEXT 14 complete days"
)

print()
print("Important:")
print("Only COMPLETE future windows are used.")
print("Partial windows at the end of a cow's history are discarded.")
print()


# ------------------------------------------------------------
# Target creation function
# ------------------------------------------------------------

def create_future_high_risk_target(
    cow_df,
    horizon_days,
    threshold
):
    """
    Creates a strict future high-risk target.

    For each current day:

        target = 1

    if ANY of the next `horizon_days` daily risk values
    is >= threshold.

    A row is valid only when ALL future days exist.
    """

    cow_df = cow_df.sort_values("date").reset_index(drop=True)

    target_values = []
    valid_values = []

    dates = cow_df["date"].tolist()
    risks = cow_df["risk_today"].to_numpy()

    n = len(cow_df)

    for i in range(n):

        future_start = i + 1
        future_end = i + 1 + horizon_days

        # Not enough future days
        if future_end > n:
            target_values.append(np.nan)
            valid_values.append(False)
            continue

        # ----------------------------------------------------
        # Check that future records are actually consecutive
        # calendar days.
        # ----------------------------------------------------

        expected_dates = [
            dates[i] + pd.Timedelta(days=j)
            for j in range(1, horizon_days + 1)
        ]

        actual_dates = dates[
            future_start:future_end
        ]

        if actual_dates != expected_dates:
            target_values.append(np.nan)
            valid_values.append(False)
            continue

        # ----------------------------------------------------
        # Get future risk values
        # ----------------------------------------------------

        future_risks = risks[
            future_start:future_end
        ]

        # ----------------------------------------------------
        # High-risk event
        # ----------------------------------------------------

        high_risk_event = int(
            np.any(
                future_risks >= threshold
            )
        )

        target_values.append(high_risk_event)
        valid_values.append(True)

    result = cow_df.copy()

    result[f"high_risk_within_{horizon_days}d"] = target_values

    result[f"valid_{horizon_days}d"] = valid_values

    return result


# ============================================================
# CREATE 7-DAY TARGET
# ============================================================

groups_7d = []

for cow_id, cow_df in df.groupby("cow_id"):

    result = create_future_high_risk_target(
        cow_df,
        horizon_days=7,
        threshold=HIGH_RISK_THRESHOLD
    )

    groups_7d.append(result)

df_7d = pd.concat(
    groups_7d,
    ignore_index=True
)


# ============================================================
# CREATE 14-DAY TARGET
# ============================================================

groups_14d = []

for cow_id, cow_df in df.groupby("cow_id"):

    result = create_future_high_risk_target(
        cow_df,
        horizon_days=14,
        threshold=HIGH_RISK_THRESHOLD
    )

    groups_14d.append(result)


df_14d = pd.concat(
    groups_14d,
    ignore_index=True
)


# ============================================================
# MERGE TARGETS
# ============================================================

TARGET_7D = "high_risk_within_7d"
TARGET_14D = "high_risk_within_14d"

VALID_7D = "valid_7d"
VALID_14D = "valid_14d"


target_7d = df_7d[
    ["cow_id", "date", TARGET_7D, VALID_7D]
]

target_14d = df_14d[
    ["cow_id", "date", TARGET_14D, VALID_14D]
]


df_v4 = df.merge(
    target_7d,
    on=["cow_id", "date"],
    how="left"
)

df_v4 = df_v4.merge(
    target_14d,
    on=["cow_id", "date"],
    how="left"
)


# ============================================================
# DATASET SUMMARY
# ============================================================

valid_7d_count = df_v4[VALID_7D].sum()
valid_14d_count = df_v4[VALID_14D].sum()

positive_7d = (
    df_v4.loc[
        df_v4[VALID_7D],
        TARGET_7D
    ].sum()
)

positive_14d = (
    df_v4.loc[
        df_v4[VALID_14D],
        TARGET_14D
    ].sum()
)


print("=" * 78)
print("V4 TARGET SUMMARY")
print("=" * 78)

print()
print("7-DAY")
print(f"Valid rows           : {valid_7d_count:,}")
print(f"High-risk events     : {int(positive_7d):,}")
print(
    f"Positive rate        : "
    f"{positive_7d / valid_7d_count * 100:.2f}%"
)

print()

print("14-DAY")
print(f"Valid rows           : {valid_14d_count:,}")
print(f"High-risk events     : {int(positive_14d):,}")
print(
    f"Positive rate        : "
    f"{positive_14d / valid_14d_count * 100:.2f}%"
)

print()


# ============================================================
# SAVE V4 CLASSIFICATION DATASET
# ============================================================

df_v4.to_csv(
    DATASET_OUTPUT_FILE,
    index=False
)

print(
    f"V4 classification dataset saved:\n"
    f"{DATASET_OUTPUT_FILE}"
)

print()


# ============================================================
# STEP 6 — COW-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("=" * 78)
print("STEP 5 — COW-LEVEL TRAIN / TEST SPLIT")
print("=" * 78)

unique_cows = sorted(
    df_v4["cow_id"].unique()
)

train_cows, test_cows = train_test_split(
    unique_cows,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_cows = set(train_cows)
test_cows = set(test_cows)

print(f"Total cows       : {len(unique_cows)}")
print(f"Training cows    : {len(train_cows)}")
print(f"Testing cows     : {len(test_cows)}")

overlap = train_cows.intersection(test_cows)

print(f"Cow overlap      : {len(overlap)}")

if len(overlap) != 0:
    raise ValueError(
        "ERROR: Cow leakage detected between train/test!"
    )

print("Cow-level split PASSED.")

print()


# ============================================================
# HELPER FUNCTION — BUILD DATA
# ============================================================

def prepare_training_data(
    dataframe,
    target_column,
    valid_column
):

    valid_df = dataframe[
        dataframe[valid_column] == True
    ].copy()

    X = valid_df[FEATURE_COLUMNS].copy()

    y = valid_df[target_column].astype(int)

    train_mask = valid_df["cow_id"].isin(
        train_cows
    )

    test_mask = valid_df["cow_id"].isin(
        test_cows
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return (
        valid_df,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_v4_classifier(
    dataframe,
    target_column,
    valid_column,
    horizon_label
):

    print("=" * 78)
    print(f"MODEL 2 V4 — {horizon_label} HIGH-RISK CLASSIFIER")
    print("=" * 78)

    (
        valid_df,
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_training_data(
        dataframe,
        target_column,
        valid_column
    )

    print()
    print("DATA")
    print(f"Total valid rows     : {len(valid_df):,}")
    print(f"Training rows        : {len(X_train):,}")
    print(f"Testing rows         : {len(X_test):,}")

    print()

    print("TRAINING CLASS BALANCE")
    print(
        f"Class 0             : "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        f"Class 1             : "
        f"{int((y_train == 1).sum()):,}"
    )

    print(
        f"Positive rate       : "
        f"{y_train.mean() * 100:.2f}%"
    )

    print()

    print("TEST CLASS BALANCE")
    print(
        f"Class 0             : "
        f"{int((y_test == 0).sum()):,}"
    )

    print(
        f"Class 1             : "
        f"{int((y_test == 1).sum()):,}"
    )

    print(
        f"Positive rate       : "
        f"{y_test.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("TRAINING RANDOM FOREST...")
    print()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_probability >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
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
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(f"{horizon_label} V4 PERFORMANCE")
    print("=" * 78)

    print()
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print(f"ROC-AUC             : {roc_auc:.4f}")
    print(f"PR-AUC              : {pr_auc:.4f}")

    print()
    print("CONFUSION MATRIX")
    print()
    print(f"True Negatives      : {tn:,}")
    print(f"False Positives     : {fp:,}")
    print(f"False Negatives     : {fn:,}")
    print(f"True Positives      : {tp:,}")

    print()
    print("CLASSIFICATION REPORT")
    print()

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not High Risk",
                "High Risk"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # HIGH-RISK DETECTION
    # --------------------------------------------------------

    actual_high_risk = int(
        (y_test == 1).sum()
    )

    detected_high_risk = int(
        tp
    )

    missed_high_risk = int(
        fn
    )

    print("=" * 78)
    print(f"{horizon_label} HIGH-RISK DETECTION")
    print("=" * 78)

    print()
    print(
        f"Actual high-risk events : "
        f"{actual_high_risk:,}"
    )

    print(
        f"Detected high-risk      : "
        f"{detected_high_risk:,}"
    )

    print(
        f"Missed high-risk        : "
        f"{missed_high_risk:,}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target": target_column,
        "horizon": horizon_label,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "classification_threshold": 0.50,
        "random_state": RANDOM_STATE
    }

    if horizon_label == "7-day":
        model_file = MODEL_7D_FILE
    else:
        model_file = MODEL_14D_FILE

    joblib.dump(
        model_package,
        model_file
    )

    print()
    print(
        f"Model saved:\n{model_file}"
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    metrics = {
        "horizon": horizon_label,
        "total_valid_rows": len(valid_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "actual_high_risk": actual_high_risk,
        "predicted_high_risk": int(y_pred.sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    predictions = valid_df.loc[
        X_test.index,
        ["cow_id", "date", "risk_today"]
    ].copy()

    predictions["horizon"] = horizon_label

    predictions["actual_high_risk"] = y_test.values

    predictions["predicted_probability"] = y_probability

    predictions["predicted_high_risk"] = y_pred

    return (
        metrics,
        predictions,
        model
    )


# ============================================================
# STEP 7 — TRAIN 7-DAY MODEL
# ============================================================

results_7d, predictions_7d, model_7d = train_v4_classifier(
    df_v4,
    TARGET_7D,
    VALID_7D,
    "7-day"
)


# ============================================================
# STEP 8 — TRAIN 14-DAY MODEL
# ============================================================

results_14d, predictions_14d, model_14d = train_v4_classifier(
    df_v4,
    TARGET_14D,
    VALID_14D,
    "14-day"
)


# ============================================================
# STEP 9 — SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    [
        results_7d,
        results_14d
    ]
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    [
        predictions_7d,
        predictions_14d
    ],
    ignore_index=True
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# STEP 10 — FEATURE IMPORTANCE
# ============================================================

importance_7d = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance_7d": model_7d.feature_importances_
})

importance_14d = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance_14d": model_14d.feature_importances_
})

feature_importance_df = importance_7d.merge(
    importance_14d,
    on="feature"
)

feature_importance_df[
    "average_importance"
] = (
    feature_importance_df["importance_7d"]
    +
    feature_importance_df["importance_14d"]
) / 2

feature_importance_df = feature_importance_df.sort_values(
    "average_importance",
    ascending=False
)

feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("                    MODEL 2 V4 COMPLETE")
print("=" * 78)

print()

print("7-DAY CLASSIFIER")
print(
    f"Precision : {results_7d['precision']:.4f}"
)
print(
    f"Recall    : {results_7d['recall']:.4f}"
)
print(
    f"F1        : {results_7d['f1']:.4f}"
)
print(
    f"ROC-AUC   : {results_7d['roc_auc']:.4f}"
)
print(
    f"PR-AUC    : {results_7d['pr_auc']:.4f}"
)
print(
    f"FN        : {results_7d['false_negatives']}"
)

print()

print("14-DAY CLASSIFIER")
print(
    f"Precision : {results_14d['precision']:.4f}"
)
print(
    f"Recall    : {results_14d['recall']:.4f}"
)
print(
    f"F1        : {results_14d['f1']:.4f}"
)
print(
    f"ROC-AUC   : {results_14d['roc_auc']:.4f}"
)
print(
    f"PR-AUC    : {results_14d['pr_auc']:.4f}"
)
print(
    f"FN        : {results_14d['false_negatives']}"
)

print()
print("=" * 78)
print("FILES CREATED")
print("=" * 78)

print()
print(MODEL_7D_FILE)
print(MODEL_14D_FILE)
print(DATASET_OUTPUT_FILE)
print(RESULTS_FILE)
print(PREDICTIONS_FILE)
print(FEATURE_IMPORTANCE_FILE)

print()
print("=" * 78)
print("DO NOT CHANGE MODEL 1 OR MODEL 2 V3 FILES.")
print("V4 is a separate long-term classification layer.")
print("=" * 78)
print()