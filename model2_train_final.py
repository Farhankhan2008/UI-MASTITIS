import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesClassifier
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)

# Final model files
MODEL_24H = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_24h.pkl"
)

MODEL_48H = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_48h.pkl"
)

MODEL_3D = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_3d.pkl"
)

MODEL_7D = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_7d.pkl"
)

MODEL_14D = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_14d.pkl"
)

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_config.pkl"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_final_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

HIGH_RISK_THRESHOLD = 50.0

# Warning thresholds selected from V4 analysis
WARNING_THRESHOLD_7D = 0.20
WARNING_THRESHOLD_14D = 0.30


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("FINAL MODEL 2 TRAINING")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)

print(f"Rows              : {len(df)}")
print(f"Columns            : {len(df.columns)}")
print(f"Unique cows        : {df['cow_id'].nunique()}")
print(f"Missing values     : {df.isna().sum().sum()}")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

TARGET_COLUMNS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d",
    "risk_14d"
]

EXCLUDED_COLUMNS = [
    "cow_id",
    "date"
] + TARGET_COLUMNS


FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(f"Number of features : {len(FEATURE_COLUMNS)}")

print("\nFeatures:")

for i, feature in enumerate(FEATURE_COLUMNS, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# PREPARE MODEL DATA
# ============================================================

df_model = df[
    ["cow_id", "date"] +
    FEATURE_COLUMNS +
    TARGET_COLUMNS
].copy()

df_model = df_model.dropna().reset_index(drop=True)

print("\nRows after preparation:", len(df_model))
print("Unique cows:", df_model["cow_id"].nunique())


# ============================================================
# COW-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("COW-LEVEL TRAIN / TEST SPLIT")
print("=" * 80)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        df_model,
        groups=df_model["cow_id"]
    )
)

train_df = df_model.iloc[train_idx].copy()
test_df = df_model.iloc[test_idx].copy()

train_cows = set(train_df["cow_id"])
test_cows = set(test_df["cow_id"])

print(f"Train cows : {len(train_cows)}")
print(f"Test cows  : {len(test_cows)}")
print(f"Cow overlap: {len(train_cows.intersection(test_cows))}")

print(f"Train rows : {len(train_df)}")
print(f"Test rows  : {len(test_df)}")


# ============================================================
# FEATURE MATRICES
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# FUNCTION — REGRESSION TRAINING
# ============================================================

def train_regression_model(
    horizon,
    model,
    model_path
):

    target_column = f"risk_{horizon}"

    print("\n" + "=" * 80)
    print(f"TRAINING FINAL REGRESSION MODEL — {horizon}")
    print("=" * 80)

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    # Keep risk within valid range
    predictions = np.clip(
        predictions,
        0,
        100
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # High-risk evaluation
    actual_high = (
        y_test >= HIGH_RISK_THRESHOLD
    ).astype(int)

    predicted_high = (
        predictions >= HIGH_RISK_THRESHOLD
    ).astype(int)

    precision = precision_score(
        actual_high,
        predicted_high,
        zero_division=0
    )

    recall = recall_score(
        actual_high,
        predicted_high,
        zero_division=0
    )

    f1 = f1_score(
        actual_high,
        predicted_high,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        actual_high,
        predicted_high,
        labels=[0, 1]
    ).ravel()

    # Save model
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "horizon": horizon,
            "model_type": "regression",
            "risk_range": [0, 100],
            "high_risk_threshold": HIGH_RISK_THRESHOLD
        },
        model_path
    )

    print("\nRESULTS")
    print("-" * 50)
    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"R2               : {r2:.4f}")
    print(f"High-risk Precision: {precision:.4f}")
    print(f"High-risk Recall   : {recall:.4f}")
    print(f"High-risk F1       : {f1:.4f}")
    print(f"TN               : {tn}")
    print(f"FP               : {fp}")
    print(f"FN               : {fn}")
    print(f"TP               : {tp}")

    print(f"\nModel saved:")
    print(model_path)

    results.append({
        "horizon": horizon,
        "problem_type": "regression",
        "model": type(model).__name__,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "precision_high_risk": precision,
        "recall_high_risk": recall,
        "F1_high_risk": f1,
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan
    })


# ============================================================
# FUNCTION — CREATE FUTURE HIGH-RISK TARGET
# ============================================================

def create_future_high_risk_target(
    source_df,
    days
):

    target = pd.Series(
        np.nan,
        index=source_df.index,
        dtype=float
    )

    valid = pd.Series(
        False,
        index=source_df.index,
        dtype=bool
    )

    for cow_id, group in source_df.groupby(
        "cow_id",
        sort=False
    ):

        group = group.sort_values("date")

        risks = group["risk_today"].to_numpy()
        indexes = group.index.to_numpy()

        n = len(group)

        for i in range(n):

            end = i + days

            if end >= n:
                continue

            future_risks = risks[
                i + 1:end + 1
            ]

            if len(future_risks) != days:
                continue

            target.loc[indexes[i]] = int(
                np.max(future_risks)
                >= HIGH_RISK_THRESHOLD
            )

            valid.loc[indexes[i]] = True

    return target, valid


# ============================================================
# TRAIN 24H
# ============================================================

train_regression_model(
    "24h",
    RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    MODEL_24H
)


# ============================================================
# TRAIN 48H
# ============================================================

train_regression_model(
    "48h",
    RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    MODEL_48H
)


# ============================================================
# TRAIN 3D
# ============================================================

train_regression_model(
    "3d",
    RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    MODEL_3D
)


# ============================================================
# BUILD FUTURE HIGH-RISK TARGETS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING FUTURE HIGH-RISK TARGETS")
print("=" * 80)

target_7d, valid_7d = create_future_high_risk_target(
    df_model,
    7
)

target_14d, valid_14d = create_future_high_risk_target(
    df_model,
    14
)


# ============================================================
# FUNCTION — CLASSIFICATION TRAINING
# ============================================================

def train_classification_model(
    horizon,
    target_series,
    valid_series,
    model,
    model_path,
    warning_threshold
):

    print("\n" + "=" * 80)
    print(f"TRAINING FINAL CLASSIFIER — {horizon}")
    print("=" * 80)

    valid_indices = df_model.index[
        valid_series
    ]

    train_valid = [
        index
        for index in valid_indices
        if index in set(train_df.index)
    ]

    test_valid = [
        index
        for index in valid_indices
        if index in set(test_df.index)
    ]

    X_train_cls = df_model.loc[
        train_valid,
        FEATURE_COLUMNS
    ]

    X_test_cls = df_model.loc[
        test_valid,
        FEATURE_COLUMNS
    ]

    y_train_cls = target_series.loc[
        train_valid
    ].astype(int)

    y_test_cls = target_series.loc[
        test_valid
    ].astype(int)

    print(f"Valid train rows : {len(X_train_cls)}")
    print(f"Valid test rows  : {len(X_test_cls)}")

    print(
        f"Train positives  : {y_train_cls.sum()}"
    )

    print(
        f"Test positives   : {y_test_cls.sum()}"
    )

    print("\nTraining model...")

    model.fit(
        X_train_cls,
        y_train_cls
    )

    probabilities = model.predict_proba(
        X_test_cls
    )[:, 1]

    # Classification threshold used for evaluation
    predictions = (
        probabilities >= warning_threshold
    ).astype(int)

    precision = precision_score(
        y_test_cls,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_cls,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_cls,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test_cls,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test_cls,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test_cls,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Save model
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "horizon": horizon,
            "model_type": "classification",
            "target_definition": (
                f"1 if any future Model 1 risk "
                f">= {HIGH_RISK_THRESHOLD}% "
                f"within {horizon}"
            ),
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "warning_probability_threshold": warning_threshold
        },
        model_path
    )

    print("\nRESULTS")
    print("-" * 50)
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1               : {f1:.4f}")
    print(f"ROC-AUC          : {roc_auc:.4f}")
    print(f"PR-AUC           : {pr_auc:.4f}")
    print(f"TN               : {tn}")
    print(f"FP               : {fp}")
    print(f"FN               : {fn}")
    print(f"TP               : {tp}")

    print(
        f"\nWarning threshold: {warning_threshold:.2f}"
    )

    print("\nModel saved:")
    print(model_path)

    results.append({
        "horizon": horizon,
        "problem_type": "classification",
        "model": type(model).__name__,
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "precision_high_risk": precision,
        "recall_high_risk": recall,
        "F1_high_risk": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    })


# ============================================================
# TRAIN 7D CLASSIFIER
# ============================================================

train_classification_model(
    "7d",
    target_7d,
    valid_7d,

    ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    MODEL_7D,

    WARNING_THRESHOLD_7D
)


# ============================================================
# TRAIN 14D CLASSIFIER
# ============================================================

train_classification_model(
    "14d",
    target_14d,
    valid_14d,

    ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    MODEL_14D,

    WARNING_THRESHOLD_14D
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model1": {
        "dataset": "synthetic_mastitis_model1_dataset_v2.csv",
        "model_file": "mastitis_model1.pkl"
    },

    "model2": {
        "dataset": "model2_forecasting_dataset_v3.csv",

        "24h": {
            "file": "mastitis_model2_final_24h.pkl",
            "type": "regression",
            "output": "risk_percentage"
        },

        "48h": {
            "file": "mastitis_model2_final_48h.pkl",
            "type": "regression",
            "output": "risk_percentage"
        },

        "3d": {
            "file": "mastitis_model2_final_3d.pkl",
            "type": "regression",
            "output": "risk_percentage"
        },

        "7d": {
            "file": "mastitis_model2_final_7d.pkl",
            "type": "classification",
            "output": "high_risk_probability",
            "warning_threshold": WARNING_THRESHOLD_7D
        },

        "14d": {
            "file": "mastitis_model2_final_14d.pkl",
            "type": "classification",
            "output": "high_risk_probability",
            "warning_threshold": WARNING_THRESHOLD_14D
        },

        "high_risk_definition": "risk >= 50%"
    },

    "features": FEATURE_COLUMNS,

    "architecture": {
        "24h": "RandomForestRegressor",
        "48h": "RandomForestRegressor",
        "3d": "RandomForestRegressor",
        "7d": "ExtraTreesClassifier",
        "14d": "ExtraTreesClassifier"
    }
}


joblib.dump(
    config,
    CONFIG_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL 2 TRAINING COMPLETE")
print("=" * 80)

print("\nFINAL MODEL FILES")
print("-" * 50)

print(f"24h  : {MODEL_24H}")
print(f"48h  : {MODEL_48H}")
print(f"3d   : {MODEL_3D}")
print(f"7d   : {MODEL_7D}")
print(f"14d  : {MODEL_14D}")

print(f"\nConfig : {CONFIG_FILE}")
print(f"Results: {RESULTS_FILE}")

print("\n" + "=" * 80)
print("FINAL ARCHITECTURE")
print("=" * 80)

print("""
24h  -> Random Forest Regression
       Output: predicted mastitis risk %

48h  -> Random Forest Regression
       Output: predicted mastitis risk %

3d   -> Random Forest Regression
       Output: predicted mastitis risk %

7d   -> Extra Trees Classification
       Output: probability of entering high-risk
               state within next 7 days

14d  -> Extra Trees Classification
       Output: probability of entering high-risk
               state within next 14 days
""")

print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print("""
Model 1 was NOT modified.

Existing V3/V4 Model 2 files were NOT modified.

The new files are separate final Model 2 files.
""")

print("=" * 80)