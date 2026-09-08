
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
    "model2_forecasting_dataset_v2.csv"
)

# Output model files
MODEL_FILES = {
    "risk_24h": os.path.join(
        BASE_DIR,
        "mastitis_model2_v2_24h.pkl"
    ),
    "risk_48h": os.path.join(
        BASE_DIR,
        "mastitis_model2_v2_48h.pkl"
    ),
    "risk_3d": os.path.join(
        BASE_DIR,
        "mastitis_model2_v2_3d.pkl"
    ),
    "risk_7d": os.path.join(
        BASE_DIR,
        "mastitis_model2_v2_7d.pkl"
    )
}

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_test_predictions.csv"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_results.csv"
)

HIGH_RISK_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_high_risk_analysis.csv"
)


# ============================================================
# MODEL 2 FEATURES
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

    "risk_change_1d",
    "risk_change_2d",
    "risk_change_3d",

    "risk_3day_mean",
    "risk_3day_max",
    "risk_3day_min",
    "risk_std_3d",

    "risk_7day_mean",
    "risk_7day_max",
    "risk_7day_min",
    "risk_std_7d",

    "risk_trend_3d",

    "risk_slope_3d",
    "risk_slope_7d",

    "consecutive_rising_days",
    "consecutive_falling_days",

    "high_risk_count_3d",
    "high_risk_count_7d",

    "recent_high_risk_3d",
    "recent_high_risk_7d",

    "distance_from_3day_max",
    "distance_from_7day_max",

    "risk_vs_3day_mean",
    "risk_vs_7day_mean"
]


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "risk_24h": "24h",
    "risk_48h": "48h",
    "risk_3d": "3d",
    "risk_7d": "7d"
}


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

HIGH_RISK_THRESHOLD = 50.0


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 78)
print("                 MODEL 2 V2 TRAINING")
print("=" * 78)

print()
print("Purpose:")
print("Train an enhanced temporal forecasting system")
print("using historical Model 1 mastitis risk percentages.")

print()
print("Forecast horizons:")
print("24 hours")
print("48 hours")
print("3 days")
print("7 days")

print()
print("High-risk threshold:", HIGH_RISK_THRESHOLD, "%")


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 78)
print("                    FILE CHECK")
print("=" * 78)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"\nERROR: Dataset not found:\n{DATASET_FILE}"
    )

print()
print("PASS: Enhanced Model 2 dataset found.")
print(DATASET_FILE)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 78)
print("                  LOADING DATASET")
print("=" * 78)

df = pd.read_csv(DATASET_FILE)

print()
print("Dataset loaded successfully.")
print("Rows :", f"{len(df):,}")
print("Cows :", df["cow_id"].nunique())


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print()
print("=" * 78)
print("                COLUMN VALIDATION")
print("=" * 78)

required_columns = (
    ["cow_id", "date"]
    + MODEL2_FEATURES
    + list(TARGETS.keys())
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print()
    print("ERROR: Missing columns:")
    for column in missing_columns:
        print(" -", column)

    raise ValueError(
        "Dataset does not contain all required columns."
    )

print()
print("PASS: All required columns found.")

print()
print("Model 2 features:", len(MODEL2_FEATURES))
print("Future targets   :", len(TARGETS))


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print()
print("=" * 78)
print("                 BASIC VALIDATION")
print("=" * 78)

missing_cells = int(df[required_columns].isna().sum().sum())

duplicate_rows = int(
    df.duplicated(
        subset=["cow_id", "date"]
    ).sum()
)

print()
print("Missing cells       :", missing_cells)
print("Duplicate cow/date  :", duplicate_rows)
print("Unique cows         :", df["cow_id"].nunique())

if missing_cells != 0:
    raise ValueError(
        "ERROR: Missing cells detected."
    )

if duplicate_rows != 0:
    raise ValueError(
        "ERROR: Duplicate cow/date records detected."
    )

print()
print("PASS: Dataset structure is valid.")


# ============================================================
# SORT DATA
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

X = df[MODEL2_FEATURES].copy()


# ============================================================
# COW-LEVEL TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 78)
print("                COW-LEVEL DATA SPLIT")
print("=" * 78)

unique_cows = sorted(
    df["cow_id"].unique()
)

rng = np.random.RandomState(
    RANDOM_STATE
)

rng.shuffle(unique_cows)

split_index = int(
    len(unique_cows) * (1 - TEST_SIZE)
)

train_cows = unique_cows[:split_index]
test_cows = unique_cows[split_index:]

train_cows = set(train_cows)
test_cows = set(test_cows)

train_mask = df["cow_id"].isin(train_cows)
test_mask = df["cow_id"].isin(test_cows)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

print()
print("Total cows :", len(unique_cows))
print("Train cows :", len(train_cows))
print("Test cows  :", len(test_cows))

print()
print("Training rows :", f"{len(X_train):,}")
print("Testing rows  :", f"{len(X_test):,}")

overlap = train_cows.intersection(test_cows)

print()
print("Cow overlap:", len(overlap))

if len(overlap) != 0:
    raise ValueError(
        "ERROR: Cow leakage detected between train and test."
    )

print()
print("PASS: No cow overlap between train and test.")


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_forecasting_model(
    target_column,
    horizon_name
):

    print()
    print("-" * 78)
    print(f"                 TRAINING {horizon_name}")
    print("-" * 78)

    y = df[target_column]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print()
    print("Target:", target_column)

    print()
    print("Training target statistics:")
    print("Mean :", f"{y_train.mean():.2f}%")
    print("Std  :", f"{y_train.std():.2f}%")
    print("Min  :", f"{y_train.min():.2f}%")
    print("Max  :", f"{y_train.max():.2f}%")

    # --------------------------------------------------------
    # RANDOM FOREST REGRESSOR
    # --------------------------------------------------------

    print()
    print("Creating Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # Keep risk within valid range
    predictions = np.clip(
        predictions,
        0,
        100
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NAIVE BASELINE
    # --------------------------------------------------------
    # Baseline:
    # Future risk = today's risk

    naive_predictions = df.loc[
        test_mask,
        "risk_today"
    ].values

    naive_predictions = np.clip(
        naive_predictions,
        0,
        100
    )

    naive_mae = mean_absolute_error(
        y_test,
        naive_predictions
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            naive_predictions
        )
    )

    naive_r2 = r2_score(
        y_test,
        naive_predictions
    )

    # --------------------------------------------------------
    # MAE IMPROVEMENT
    # --------------------------------------------------------

    if naive_mae != 0:
        mae_improvement = (
            (naive_mae - mae)
            / naive_mae
        ) * 100
    else:
        mae_improvement = 0

    # --------------------------------------------------------
    # HIGH-RISK CLASSIFICATION
    # --------------------------------------------------------

    actual_high = (
        y_test.values >= HIGH_RISK_THRESHOLD
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

    cm = confusion_matrix(
        actual_high,
        predicted_high,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("MODEL PERFORMANCE")
    print("-" * 50)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print()
    print("NAIVE BASELINE")
    print("-" * 50)

    print(
        f"MAE  : {naive_mae:.4f}"
    )

    print(
        f"RMSE : {naive_rmse:.4f}"
    )

    print(
        f"R²   : {naive_r2:.4f}"
    )

    print()
    print(
        f"MAE improvement over naive: "
        f"{mae_improvement:.2f}%"
    )

    print()
    print("HIGH-RISK DETECTION")
    print("-" * 50)

    print(
        f"Threshold              : "
        f">= {HIGH_RISK_THRESHOLD:.1f}%"
    )

    print(
        f"Actual high-risk cases : "
        f"{actual_high.sum():,}"
    )

    print(
        f"Predicted high-risk    : "
        f"{predicted_high.sum():,}"
    )

    print(
        f"Precision              : "
        f"{precision:.4f}"
    )

    print(
        f"Recall                 : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score              : "
        f"{f1:.4f}"
    )

    print()
    print("Confusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "                 Low    High"
    )
    print(
        f"Actual Low       {tn:5d}  {fp:5d}"
    )
    print(
        f"Actual High      {fn:5d}  {tp:5d}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": MODEL2_FEATURES,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("TOP FEATURE IMPORTANCE")
    print("-" * 50)

    for _, row in importance.head(10).iterrows():

        print(
            f"{row['feature']:<30}"
            f"{row['importance']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = MODEL_FILES[target_column]

    joblib.dump(
        model,
        model_file
    )

    print()
    print("Model saved:")
    print(model_file)

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    result = {
        "horizon": horizon_name,
        "target": target_column,

        "mae": mae,
        "rmse": rmse,
        "r2": r2,

        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_r2": naive_r2,

        "mae_improvement_percent":
            mae_improvement,

        "high_risk_precision":
            precision,

        "high_risk_recall":
            recall,

        "high_risk_f1":
            f1,

        "actual_high_risk_count":
            int(actual_high.sum()),

        "predicted_high_risk_count":
            int(predicted_high.sum()),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)
    }

    return (
        model,
        predictions,
        result
    )


# ============================================================
# TRAIN ALL FOUR MODELS
# ============================================================

all_results = []

prediction_data = df.loc[
    test_mask,
    ["cow_id", "date"]
].copy()

prediction_data = prediction_data.reset_index(
    drop=True
)


for target_column, horizon_name in TARGETS.items():

    model, predictions, result = (
        train_forecasting_model(
            target_column,
            horizon_name
        )
    )

    all_results.append(result)

    prediction_data[
        f"actual_{horizon_name}"
    ] = df.loc[
        test_mask,
        target_column
    ].values

    prediction_data[
        f"predicted_{horizon_name}"
    ] = predictions

    # Baseline
    prediction_data[
        f"naive_{horizon_name}"
    ] = df.loc[
        test_mask,
        "risk_today"
    ].values


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print()
print("=" * 78)
print("                 SAVING PREDICTIONS")
print("=" * 78)

prediction_data.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print()
print("Test predictions saved:")
print(PREDICTIONS_FILE)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print()
print("Results saved:")
print(RESULTS_FILE)


# ============================================================
# HIGH-RISK ANALYSIS FILE
# ============================================================

high_risk_rows = []

for _, row in prediction_data.iterrows():

    for horizon in ["24h", "48h", "3d", "7d"]:

        actual = row[
            f"actual_{horizon}"
        ]

        predicted = row[
            f"predicted_{horizon}"
        ]

        actual_high = (
            actual >= HIGH_RISK_THRESHOLD
        )

        predicted_high = (
            predicted >= HIGH_RISK_THRESHOLD
        )

        high_risk_rows.append({

            "cow_id":
                row["cow_id"],

            "date":
                row["date"],

            "horizon":
                horizon,

            "actual_risk":
                actual,

            "predicted_risk":
                predicted,

            "absolute_error":
                abs(actual - predicted),

            "actual_high_risk":
                int(actual_high),

            "predicted_high_risk":
                int(predicted_high),

            "correct_high_risk_detection":
                int(
                    actual_high
                    and predicted_high
                ),

            "missed_high_risk":
                int(
                    actual_high
                    and not predicted_high
                ),

            "false_high_risk":
                int(
                    not actual_high
                    and predicted_high
                )
        })


high_risk_df = pd.DataFrame(
    high_risk_rows
)

high_risk_df.to_csv(
    HIGH_RISK_FILE,
    index=False
)

print()
print("High-risk analysis saved:")
print(HIGH_RISK_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("                  FINAL MODEL 2 V2 SUMMARY")
print("=" * 78)

summary_columns = [
    "horizon",
    "mae",
    "rmse",
    "r2",
    "naive_mae",
    "naive_rmse",
    "naive_r2",
    "mae_improvement_percent",
    "high_risk_precision",
    "high_risk_recall",
    "high_risk_f1"
]

print()

print(
    results_df[
        summary_columns
    ].round(4).to_string(
        index=False
    )
)


# ============================================================
# SIMPLE INTERPRETATION
# ============================================================

print()
print("=" * 78)
print("                    INTERPRETATION")
print("=" * 78)

for _, row in results_df.iterrows():

    horizon = row["horizon"]
    mae = row["mae"]
    r2 = row["r2"]
    improvement = row[
        "mae_improvement_percent"
    ]

    recall = row[
        "high_risk_recall"
    ]

    print()

    print(
        f"{horizon}:"
    )

    print(
        f"  MAE improvement over naive: "
        f"{improvement:.2f}%"
    )

    print(
        f"  R²: {r2:.4f}"
    )

    print(
        f"  High-risk recall: "
        f"{recall:.4f}"
    )

    if horizon == "24h":

        if r2 >= 0.70:
            print(
                "  Assessment: STRONG"
            )
        else:
            print(
                "  Assessment: MODERATE"
            )

    elif horizon == "48h":

        if r2 >= 0.50:
            print(
                "  Assessment: GOOD"
            )
        else:
            print(
                "  Assessment: MODERATE"
            )

    elif horizon == "3d":

        if r2 >= 0.30:
            print(
                "  Assessment: USEFUL"
            )
        else:
            print(
                "  Assessment: LIMITED"
            )

    elif horizon == "7d":

        if r2 >= 0.30:
            print(
                "  Assessment: USEFUL"
            )
        elif r2 >= 0:
            print(
                "  Assessment: WEAK"
            )
        else:
            print(
                "  Assessment: CAUTION"
            )


# ============================================================
# FILE SUMMARY
# ============================================================

print()
print("=" * 78)
print("                    OUTPUT FILES")
print("=" * 78)

print()

for target_column, model_file in MODEL_FILES.items():

    print(
        f"{target_column:<12}: "
        f"{model_file}"
    )

print()
print(
    "Predictions    : "
    f"{PREDICTIONS_FILE}"
)

print(
    "Results        : "
    f"{RESULTS_FILE}"
)

print(
    "High-risk      : "
    f"{HIGH_RISK_FILE}"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 78)
print("             MODEL 2 V2 TRAINING COMPLETED")
print("=" * 78)

print()
print("All four forecasting models have been trained.")

print()
print("IMPORTANT:")
print("Model 2 predicts future Model 1 risk scores.")
print("It is not a clinically validated diagnostic system.")

print()
print("Next step:")
print("Review the Model 2 V2 metrics and compare them")
print("against the original Model 2 results.")

print("=" * 78)
print()
