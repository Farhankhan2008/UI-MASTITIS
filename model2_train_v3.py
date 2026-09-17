import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

DATASET_FILE = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)

# Output files
RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_v3_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "model2_v3_test_predictions.csv"
)

HIGH_RISK_FILE = os.path.join(
    BASE_DIR,
    "model2_v3_high_risk_analysis.csv"
)


# ============================================================
# MODEL 2 V3 FEATURES
# ============================================================

MODEL2_FEATURES = [
    "risk_today",

    "risk_1d_ago",
    "risk_2d_ago",
    "risk_3d_ago",
    "risk_4d_ago",
    "risk_5d_ago",
    "risk_6d_ago",
    "risk_7d_ago",
    "risk_8d_ago",
    "risk_9d_ago",
    "risk_10d_ago",
    "risk_11d_ago",
    "risk_12d_ago",
    "risk_13d_ago",
    "risk_14d_ago",

    "risk_change_1d",
    "risk_change_2d",
    "risk_change_3d",
    "risk_change_7d",
    "risk_change_14d",

    "risk_3day_mean",
    "risk_3day_max",
    "risk_3day_min",
    "risk_std_3d",

    "risk_7day_mean",
    "risk_7day_max",
    "risk_7day_min",
    "risk_std_7d",

    "risk_14day_mean",
    "risk_14day_max",
    "risk_14day_min",
    "risk_std_14d",

    "risk_trend_3d",
    "risk_slope_3d",
    "risk_slope_7d",
    "risk_slope_14d",

    "risk_acceleration_2d",
    "risk_acceleration_3d",

    "consecutive_rising_days",
    "consecutive_falling_days",

    "high_risk_count_3d",
    "high_risk_count_7d",
    "high_risk_count_14d",

    "recent_high_risk_3d",
    "recent_high_risk_7d",
    "recent_high_risk_14d",

    "distance_from_3day_max",
    "distance_from_7day_max",
    "distance_from_14day_max",

    "risk_vs_3day_mean",
    "risk_vs_7day_mean",
    "risk_vs_14day_mean",

    "risk_7day_mean_vs_14day_mean",
    "risk_3day_mean_vs_7day_mean"
]


# ============================================================
# FORECAST TARGETS
# ============================================================

TARGETS = {
    "24h": "risk_24h",
    "48h": "risk_48h",
    "3d": "risk_3d",
    "7d": "risk_7d",
    "14d": "risk_14d"
}


# ============================================================
# RANDOM FOREST CONFIGURATION
# ============================================================

MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted, naive_prediction):
    """
    Calculate regression and high-risk classification metrics.
    """

    # --------------------------------------------------------
    # Regression metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    naive_mae = mean_absolute_error(
        actual,
        naive_prediction
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            actual,
            naive_prediction
        )
    )

    naive_r2 = r2_score(
        actual,
        naive_prediction
    )

    if naive_mae != 0:
        mae_improvement = (
            (naive_mae - mae)
            / naive_mae
        ) * 100
    else:
        mae_improvement = 0.0

    # --------------------------------------------------------
    # High-risk classification
    # --------------------------------------------------------

    HIGH_RISK_THRESHOLD = 50.0

    actual_high_risk = (
        np.asarray(actual) >= HIGH_RISK_THRESHOLD
    ).astype(int)

    predicted_high_risk = (
        np.asarray(predicted) >= HIGH_RISK_THRESHOLD
    ).astype(int)

    precision = precision_score(
        actual_high_risk,
        predicted_high_risk,
        zero_division=0
    )

    recall = recall_score(
        actual_high_risk,
        predicted_high_risk,
        zero_division=0
    )

    f1 = f1_score(
        actual_high_risk,
        predicted_high_risk,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        actual_high_risk,
        predicted_high_risk,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = np.mean(
        np.asarray(predicted)
        - np.asarray(actual)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,

        "Naive_MAE": naive_mae,
        "Naive_RMSE": naive_rmse,
        "Naive_R2": naive_r2,

        "MAE_Improvement_Percent": mae_improvement,

        "High_Risk_Precision": precision,
        "High_Risk_Recall": recall,
        "High_Risk_F1": f1,

        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,

        "Actual_High_Risk_Count": int(
            actual_high_risk.sum()
        ),

        "Predicted_High_Risk_Count": int(
            predicted_high_risk.sum()
        ),

        "Mean_Bias": bias
    }


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("              MODEL 2 V3 TRAINING")
print("=" * 70)
print()

print("Dataset:")
print(DATASET_FILE)
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):
    print("ERROR: V3 dataset was not found.")
    print()
    print("Expected file:")
    print(DATASET_FILE)
    print()
    print("Make sure model2_prepare_v3.py was run successfully.")
    raise SystemExit(1)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading V3 dataset...")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print(
    f"Total rows        : {len(df)}"
)
print(
    f"Total columns     : {len(df.columns)}"
)
print(
    f"Unique cows       : {df['cow_id'].nunique()}"
)
print()


# ============================================================
# VALIDATE FEATURES
# ============================================================

print("Validating dataset columns...")

missing_features = [
    feature
    for feature in MODEL2_FEATURES
    if feature not in df.columns
]

missing_targets = [
    target
    for target in TARGETS.values()
    if target not in df.columns
]

if missing_features:
    print()
    print("ERROR: Missing Model 2 V3 features:")
    for feature in missing_features:
        print(" -", feature)

    raise SystemExit(1)


if missing_targets:
    print()
    print("ERROR: Missing target columns:")
    for target in missing_targets:
        print(" -", target)

    raise SystemExit(1)


print(
    f"Features available : {len(MODEL2_FEATURES)}"
)

print(
    f"Targets available  : {len(TARGETS)}"
)

print("Column validation passed.")
print()


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("Checking missing values...")

feature_missing = df[MODEL2_FEATURES].isna().sum().sum()

target_missing = df[
    list(TARGETS.values())
].isna().sum().sum()

if feature_missing > 0:
    print(
        f"ERROR: {feature_missing} missing feature values found."
    )
    raise SystemExit(1)

if target_missing > 0:
    print(
        f"ERROR: {target_missing} missing target values found."
    )
    raise SystemExit(1)

print("No missing feature/target values found.")
print()


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("Checking duplicate cow/date records...")

duplicates = df.duplicated(
    subset=["cow_id", "date"]
).sum()

if duplicates > 0:
    print(
        f"ERROR: {duplicates} duplicate cow/date records found."
    )
    raise SystemExit(1)

print("No duplicate cow/date records found.")
print()


# ============================================================
# COW-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("=" * 70)
print("                    DATA SPLIT")
print("=" * 70)
print()

unique_cows = sorted(
    df["cow_id"].unique()
)

total_cows = len(unique_cows)

train_cow_count = int(
    total_cows * 0.80
)

test_cow_count = (
    total_cows - train_cow_count
)

# Same deterministic split approach as previous versions
rng = np.random.RandomState(42)

shuffled_cows = rng.permutation(
    unique_cows
)

train_cows = shuffled_cows[
    :train_cow_count
]

test_cows = shuffled_cows[
    train_cow_count:
]

train_cows = set(train_cows)
test_cows = set(test_cows)

train_df = df[
    df["cow_id"].isin(train_cows)
].copy()

test_df = df[
    df["cow_id"].isin(test_cows)
].copy()


print(
    f"Total cows       : {total_cows}"
)

print(
    f"Training cows    : {len(train_cows)}"
)

print(
    f"Testing cows     : {len(test_cows)}"
)

print()

print(
    f"Training rows    : {len(train_df)}"
)

print(
    f"Testing rows     : {len(test_df)}"
)

print()


# ============================================================
# VERIFY NO COW LEAKAGE
# ============================================================

overlap = train_cows.intersection(
    test_cows
)

if len(overlap) > 0:
    print("ERROR: Cow leakage detected!")
    raise SystemExit(1)

print("Cow-level split verified.")
print("Train/test cow overlap : 0")
print()


# ============================================================
# PREPARE X
# ============================================================

X_train = train_df[
    MODEL2_FEATURES
].copy()

X_test = test_df[
    MODEL2_FEATURES
].copy()


# ============================================================
# PREPARE TARGETS
# ============================================================

print("Preparing forecast targets...")

Y_train = {}
Y_test = {}

for horizon, target_column in TARGETS.items():

    Y_train[horizon] = train_df[
        target_column
    ].values

    Y_test[horizon] = test_df[
        target_column
    ].values


print("Targets prepared.")
print()


# ============================================================
# TRAIN MODELS
# ============================================================

print("=" * 70)
print("                    MODEL TRAINING")
print("=" * 70)
print()

results = []
all_predictions = []

trained_models = {}


for horizon, target_column in TARGETS.items():

    print()
    print("-" * 70)
    print(f"TRAINING {horizon} FORECAST MODEL")
    print("-" * 70)

    print(
        f"Target column : {target_column}"
    )

    print(
        f"Features      : {len(MODEL2_FEATURES)}"
    )

    print(
        f"Training rows : {len(X_train)}"
    )

    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=MODEL_PARAMETERS[
            "n_estimators"
        ],

        max_depth=MODEL_PARAMETERS[
            "max_depth"
        ],

        min_samples_split=MODEL_PARAMETERS[
            "min_samples_split"
        ],

        min_samples_leaf=MODEL_PARAMETERS[
            "min_samples_leaf"
        ],

        random_state=MODEL_PARAMETERS[
            "random_state"
        ],

        n_jobs=MODEL_PARAMETERS[
            "n_jobs"
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training Random Forest...")

    model.fit(
        X_train,
        Y_train[horizon]
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # Keep forecast between 0 and 100
    predictions = np.clip(
        predictions,
        0,
        100
    )

    # --------------------------------------------------------
    # Naive baseline
    # --------------------------------------------------------

    naive_prediction = test_df[
        "risk_today"
    ].values

    naive_prediction = np.clip(
        naive_prediction,
        0,
        100
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        Y_test[horizon],
        predictions,
        naive_prediction
    )

    metrics["Horizon"] = horizon
    metrics["Target"] = target_column

    results.append(metrics)

    # --------------------------------------------------------
    # Store model
    # --------------------------------------------------------

    model_filename = os.path.join(
        BASE_DIR,
        f"mastitis_model2_v3_{horizon}.pkl"
    )

    joblib.dump(
        model,
        model_filename
    )

    trained_models[horizon] = model

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    prediction_df = test_df[
        ["cow_id", "date"]
    ].copy()

    prediction_df["horizon"] = horizon

    prediction_df[
        "actual_risk"
    ] = Y_test[horizon]

    prediction_df[
        "predicted_risk"
    ] = predictions

    prediction_df[
        "naive_risk"
    ] = naive_prediction

    all_predictions.append(
        prediction_df
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()
    print("RESULTS")
    print()

    print(
        f"MAE                  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE                 : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²                   : {metrics['R2']:.4f}"
    )

    print()

    print(
        f"Naive MAE            : {metrics['Naive_MAE']:.4f}"
    )

    print(
        f"Naive RMSE           : {metrics['Naive_RMSE']:.4f}"
    )

    print(
        f"Naive R²             : {metrics['Naive_R2']:.4f}"
    )

    print(
        f"MAE improvement      : "
        f"{metrics['MAE_Improvement_Percent']:.2f}%"
    )

    print()

    print("HIGH-RISK DETECTION (>= 50%)")

    print(
        f"Precision            : "
        f"{metrics['High_Risk_Precision']:.4f}"
    )

    print(
        f"Recall               : "
        f"{metrics['High_Risk_Recall']:.4f}"
    )

    print(
        f"F1                   : "
        f"{metrics['High_Risk_F1']:.4f}"
    )

    print()

    print(
        f"Actual high-risk    : "
        f"{metrics['Actual_High_Risk_Count']}"
    )

    print(
        f"Predicted high-risk : "
        f"{metrics['Predicted_High_Risk_Count']}"
    )

    print()

    print("CONFUSION MATRIX")

    print(
        f"True Negatives  : "
        f"{metrics['True_Negatives']}"
    )

    print(
        f"False Positives : "
        f"{metrics['False_Positives']}"
    )

    print(
        f"False Negatives : "
        f"{metrics['False_Negatives']}"
    )

    print(
        f"True Positives   : "
        f"{metrics['True_Positives']}"
    )

    print()

    print(
        f"Mean bias            : "
        f"{metrics['Mean_Bias']:.4f}"
    )

    print()

    print(
        f"Model saved as: "
        f"{model_filename}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

# Put horizon first
results_df = results_df[
    [
        "Horizon",
        "Target",
        "MAE",
        "RMSE",
        "R2",
        "Naive_MAE",
        "Naive_RMSE",
        "Naive_R2",
        "MAE_Improvement_Percent",
        "High_Risk_Precision",
        "High_Risk_Recall",
        "High_Risk_F1",
        "True_Negatives",
        "False_Positives",
        "False_Negatives",
        "True_Positives",
        "Actual_High_Risk_Count",
        "Predicted_High_Risk_Count",
        "Mean_Bias"
    ]
]


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print()
print("=" * 70)
print("              RESULTS SAVED")
print("=" * 70)
print()

print(
    f"Results file: {RESULTS_FILE}"
)


# ============================================================
# SAVE ALL TEST PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"Predictions file: {PREDICTIONS_FILE}"
)


# ============================================================
# HIGH-RISK ANALYSIS
# ============================================================

print()
print("=" * 70)
print("                 HIGH-RISK ANALYSIS")
print("=" * 70)
print()

high_risk_rows = []

HIGH_RISK_THRESHOLD = 50.0


for horizon in TARGETS.keys():

    horizon_data = predictions_df[
        predictions_df["horizon"] == horizon
    ].copy()

    actual = horizon_data[
        "actual_risk"
    ].values

    predicted = horizon_data[
        "predicted_risk"
    ].values

    actual_high = (
        actual >= HIGH_RISK_THRESHOLD
    )

    predicted_high = (
        predicted >= HIGH_RISK_THRESHOLD
    )

    # --------------------------------------------------------
    # Actual high-risk cases
    # --------------------------------------------------------

    actual_high_count = int(
        actual_high.sum()
    )

    predicted_high_count = int(
        predicted_high.sum()
    )

    true_positive_count = int(
        np.sum(
            actual_high & predicted_high
        )
    )

    false_positive_count = int(
        np.sum(
            ~actual_high & predicted_high
        )
    )

    false_negative_count = int(
        np.sum(
            actual_high & ~predicted_high
        )
    )

    # --------------------------------------------------------
    # High-risk actual cases
    # --------------------------------------------------------

    if actual_high_count > 0:

        actual_high_values = actual[
            actual_high
        ]

        predicted_on_actual_high = predicted[
            actual_high
        ]

        mean_actual_high = np.mean(
            actual_high_values
        )

        mean_predicted_high = np.mean(
            predicted_on_actual_high
        )

        high_risk_gap = (
            mean_actual_high
            - mean_predicted_high
        )

        high_risk_mae = mean_absolute_error(
            actual_high_values,
            predicted_on_actual_high
        )

    else:

        mean_actual_high = 0.0
        mean_predicted_high = 0.0
        high_risk_gap = 0.0
        high_risk_mae = 0.0

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    high_risk_rows.append({

        "Horizon": horizon,

        "Actual_High_Risk_Count":
            actual_high_count,

        "Predicted_High_Risk_Count":
            predicted_high_count,

        "True_Positive_Count":
            true_positive_count,

        "False_Positive_Count":
            false_positive_count,

        "False_Negative_Count":
            false_negative_count,

        "Mean_Actual_Risk_High_Risk_Cases":
            mean_actual_high,

        "Mean_Predicted_Risk_On_High_Risk_Cases":
            mean_predicted_high,

        "High_Risk_Underprediction_Gap":
            high_risk_gap,

        "High_Risk_MAE":
            high_risk_mae
    })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"{horizon} Forecast"
    )

    print(
        f"Actual high-risk cases    : "
        f"{actual_high_count}"
    )

    print(
        f"Predicted high-risk cases : "
        f"{predicted_high_count}"
    )

    print(
        f"True positives             : "
        f"{true_positive_count}"
    )

    print(
        f"False positives            : "
        f"{false_positive_count}"
    )

    print(
        f"False negatives            : "
        f"{false_negative_count}"
    )

    print(
        f"Mean actual high-risk     : "
        f"{mean_actual_high:.2f}%"
    )

    print(
        f"Mean predicted on them    : "
        f"{mean_predicted_high:.2f}%"
    )

    print(
        f"Underprediction gap       : "
        f"{high_risk_gap:.2f} percentage points"
    )

    print(
        f"High-risk MAE             : "
        f"{high_risk_mae:.2f}"
    )

    print()


high_risk_df = pd.DataFrame(
    high_risk_rows
)

high_risk_df.to_csv(
    HIGH_RISK_FILE,
    index=False
)

print(
    f"High-risk analysis saved: "
    f"{HIGH_RISK_FILE}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("                  FEATURE IMPORTANCE")
print("=" * 70)
print()


feature_importance_tables = []


for horizon, model in trained_models.items():

    importance_df = pd.DataFrame({
        "Feature": MODEL2_FEATURES,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print()
    print("-" * 70)
    print(f"TOP FEATURES — {horizon}")
    print("-" * 70)

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    importance_df[
        "Horizon"
    ] = horizon

    feature_importance_tables.append(
        importance_df
    )


feature_importance_df = pd.concat(
    feature_importance_tables,
    ignore_index=True
)


FEATURE_IMPORTANCE_FILE = os.path.join(
    BASE_DIR,
    "model2_v3_feature_importance.csv"
)

feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print()
print(
    f"Feature importance saved: "
    f"{FEATURE_IMPORTANCE_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("                  MODEL 2 V3 SUMMARY")
print("=" * 70)
print()

print(
    results_df[
        [
            "Horizon",
            "MAE",
            "RMSE",
            "R2",
            "Naive_MAE",
            "Naive_R2",
            "MAE_Improvement_Percent",
            "High_Risk_Precision",
            "High_Risk_Recall",
            "High_Risk_F1"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("=" * 70)
print("                    FILES CREATED")
print("=" * 70)
print()

for horizon in TARGETS.keys():

    print(
        f"mastitis_model2_v3_{horizon}.pkl"
    )

print()
print(
    "model2_v3_test_predictions.csv"
)

print(
    "model2_v3_results.csv"
)

print(
    "model2_v3_high_risk_analysis.csv"
)

print(
    "model2_v3_feature_importance.csv"
)

print()
print("=" * 70)
print("             MODEL 2 V3 TRAINING COMPLETE")
print("=" * 70)
print()