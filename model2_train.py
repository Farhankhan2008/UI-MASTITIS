
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MODEL 2 - FUTURE MASTITIS RISK FORECASTING
# ============================================================
#
# Model 2 receives historical Model 1 risk percentages and
# forecasts future Model 1 risk:
#
#       Historical Risk
#             |
#             v
#          MODEL 2
#             |
#      +------+------+------+
#      |      |      |      |
#     24h    48h     3d     7d
#
#
# IMPORTANT:
#   Model 2 does NOT directly predict mastitis_today.
#
#   Model 1:
#       Sensor + history data
#              ->
#       Today's mastitis risk %
#
#   Model 2:
#       Historical Model 1 risk %
#              ->
#       Future risk %
#
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

DATASET_FILE = "model2_forecasting_dataset.csv"

MODEL_24H_FILE = "mastitis_model2_24h.pkl"
MODEL_48H_FILE = "mastitis_model2_48h.pkl"
MODEL_3D_FILE = "mastitis_model2_3d.pkl"
MODEL_7D_FILE = "mastitis_model2_7d.pkl"

PREDICTIONS_FILE = "model2_test_predictions.csv"

RESULTS_FILE = "model2_results.csv"


# ============================================================
# HISTORICAL MODEL 1 RISK FEATURES
# ============================================================
#
# These are allowed inputs.
#
# They contain only current/past risk information.
#
# ============================================================

FEATURE_COLUMNS = [
    "risk_today",
    "risk_1d_ago",
    "risk_2d_ago",
    "risk_3d_ago",
    "risk_7d_ago",

    "risk_3day_mean",
    "risk_3day_max",
    "risk_3day_min",

    "risk_7day_mean",
    "risk_7day_max",
    "risk_7day_min",

    "risk_trend_3d"
]


# ============================================================
# FORECAST TARGETS
# ============================================================

TARGETS = {
    "24h": "risk_24h",
    "48h": "risk_48h",
    "3d": "risk_3d",
    "7d": "risk_7d"
}


# ============================================================
# MODEL PARAMETERS
# ============================================================

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 5

MIN_SAMPLES_LEAF = 2

RANDOM_STATE = 42


# ============================================================
# START
# ============================================================

print("=" * 75)
print("             MODEL 2 - RISK FORECASTING")
print("=" * 75)


# ============================================================
# 1. CHECK DATASET
# ============================================================

print("\n1. CHECKING DATASET")
print("-" * 75)

if not os.path.exists(DATASET_FILE):
    print(f"ERROR: Dataset not found:")
    print(f"       {DATASET_FILE}")
    print("\nMake sure model2_prepare.py was run successfully.")
    raise SystemExit(1)

print(f"Dataset found: {DATASET_FILE}")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n2. LOADING MODEL 2 DATASET")
print("-" * 75)

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 3. REQUIRED COLUMN VALIDATION
# ============================================================

print("\n3. VALIDATING COLUMNS")
print("-" * 75)

required_columns = [
    "cow_id",
    "date"
] + FEATURE_COLUMNS + list(TARGETS.values())

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("ERROR: Missing required columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit(1)

print("All required columns are present.")


# ============================================================
# 4. DATE PROCESSING
# ============================================================

print("\n4. PROCESSING DATES")
print("-" * 75)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():
    print("ERROR: Invalid dates found.")
    raise SystemExit(1)

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)

print("Dates processed successfully.")


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print("\n5. BASIC DATA VALIDATION")
print("-" * 75)

print(
    f"Unique cows: {df['cow_id'].nunique():,}"
)

missing_values = df[
    FEATURE_COLUMNS + list(TARGETS.values())
].isna().sum().sum()

print(f"Missing feature/target values: {missing_values}")

if missing_values > 0:
    print("ERROR: Missing values detected.")
    raise SystemExit(1)

duplicate_count = df.duplicated(
    subset=["cow_id", "date"]
).sum()

print(f"Duplicate cow/date records: {duplicate_count}")

if duplicate_count > 0:
    print("ERROR: Duplicate cow/date records detected.")
    raise SystemExit(1)


# ============================================================
# 6. COW-LEVEL TRAIN/TEST SPLIT
# ============================================================
#
# VERY IMPORTANT:
#
# We do NOT randomly split rows.
#
# If rows from the same cow appeared in both training and
# testing, Model 2 could partially memorize that cow's
# historical risk behavior.
#
# Instead:
#
#       80% cows -> training
#       20% cows -> testing
#
# ============================================================

print("\n6. COW-LEVEL TRAIN/TEST SPLIT")
print("-" * 75)

unique_cows = sorted(
    df["cow_id"].unique()
)

rng = np.random.RandomState(
    RANDOM_STATE
)

shuffled_cows = unique_cows.copy()

rng.shuffle(shuffled_cows)

total_cows = len(shuffled_cows)

train_cow_count = int(
    total_cows * 0.80
)

train_cows = set(
    shuffled_cows[:train_cow_count]
)

test_cows = set(
    shuffled_cows[train_cow_count:]
)


train_df = df[
    df["cow_id"].isin(train_cows)
].copy()

test_df = df[
    df["cow_id"].isin(test_cows)
].copy()


print(f"Total cows      : {total_cows:,}")
print(f"Training cows   : {len(train_cows):,}")
print(f"Testing cows    : {len(test_cows):,}")

print(f"\nTraining rows   : {len(train_df):,}")
print(f"Testing rows    : {len(test_df):,}")


# ============================================================
# 7. VERIFY NO COW OVERLAP
# ============================================================

overlap = (
    set(train_df["cow_id"].unique())
    &
    set(test_df["cow_id"].unique())
)

print(
    f"\nCow overlap     : {len(overlap)}"
)

if len(overlap) != 0:
    print("ERROR: Cow leakage detected!")
    raise SystemExit(1)

print("Cow-level split validation: PASSED")


# ============================================================
# 8. PREPARE INPUT FEATURES
# ============================================================

print("\n7. PREPARING MODEL 2 INPUT FEATURES")
print("-" * 75)

X_train = train_df[
    FEATURE_COLUMNS
].copy()

X_test = test_df[
    FEATURE_COLUMNS
].copy()

print("Model 2 input features:")

for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")


# ============================================================
# 9. DISPLAY WHAT IS NOT BEING USED
# ============================================================

print("\nIMPORTANT TARGET-LEAKAGE CHECK")
print("-" * 75)

print("The following future columns are NOT used as inputs:")

for target in TARGETS.values():
    print(f"  - {target}")

print("\nCow ID and date are also NOT used as numerical inputs.")


# ============================================================
# 10. FUNCTION FOR TRAINING ONE FORECAST MODEL
# ============================================================

def train_forecast_model(
    horizon_name,
    target_column
):

    print("\n")
    print("=" * 75)
    print(
        f"TRAINING MODEL 2 - {horizon_name.upper()} FORECAST"
    )
    print("=" * 75)

    y_train = train_df[
        target_column
    ].values

    y_test = test_df[
        target_column
    ].values

    # --------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------

    print("\nTarget statistics:")

    print(
        f"Training mean : {np.mean(y_train):.2f}%"
    )

    print(
        f"Testing mean  : {np.mean(y_test):.2f}%"
    )

    print(
        f"Training min  : {np.min(y_train):.2f}%"
    )

    print(
        f"Training max  : {np.max(y_train):.2f}%"
    )

    # --------------------------------------------------------
    # Create Random Forest Regressor
    # --------------------------------------------------------

    print("\nCreating Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("Generating predictions...")

    predictions = model.predict(
        X_test
    )

    # Keep forecast within valid risk range
    predictions = np.clip(
        predictions,
        0,
        100
    )

    # --------------------------------------------------------
    # Model metrics
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
    # Naive baseline
    # --------------------------------------------------------
    #
    # Baseline:
    #
    #       Future risk = risk_today
    #
    # Model 2 should ideally outperform this.
    #
    # --------------------------------------------------------

    naive_predictions = test_df[
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
    # Improvement
    # --------------------------------------------------------

    if naive_mae != 0:
        mae_improvement = (
            (naive_mae - mae)
            / naive_mae
        ) * 100
    else:
        mae_improvement = 0

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nMODEL PERFORMANCE")
    print("-" * 50)

    print(f"Forecast horizon : {horizon_name}")

    print("\nModel 2:")
    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"R²               : {r2:.4f}")

    print("\nNaive baseline:")
    print(f"MAE              : {naive_mae:.4f}")
    print(f"RMSE             : {naive_rmse:.4f}")
    print(f"R²               : {naive_r2:.4f}")

    print(
        f"\nMAE improvement over baseline: "
        f"{mae_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nFEATURE IMPORTANCE")
    print("-" * 50)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<25} "
            f"{row['importance']:.4f}"
        )

    # --------------------------------------------------------
    # Create prediction dataframe
    # --------------------------------------------------------

    prediction_df = test_df[
        ["cow_id", "date"]
    ].copy()

    prediction_df[
        f"actual_{horizon_name}"
    ] = y_test

    prediction_df[
        f"predicted_{horizon_name}"
    ] = predictions

    prediction_df[
        f"naive_{horizon_name}"
    ] = naive_predictions

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    metrics = {
        "horizon": horizon_name,
        "target": target_column,
        "model_mae": mae,
        "model_rmse": rmse,
        "model_r2": r2,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_r2": naive_r2,
        "mae_improvement_percent": mae_improvement
    }

    return (
        model,
        prediction_df,
        metrics
    )


# ============================================================
# 11. TRAIN ALL FOUR MODELS
# ============================================================

print("\n")
print("=" * 75)
print("STARTING FOUR FORECAST MODELS")
print("=" * 75)


all_predictions = test_df[
    ["cow_id", "date"]
].copy()

all_results = []


# ============================================================
# 24 HOURS
# ============================================================

model_24h, predictions_24h, results_24h = train_forecast_model(
    "24h",
    TARGETS["24h"]
)

all_predictions = all_predictions.merge(
    predictions_24h,
    on=["cow_id", "date"],
    how="left"
)

all_results.append(
    results_24h
)


# ============================================================
# 48 HOURS
# ============================================================

model_48h, predictions_48h, results_48h = train_forecast_model(
    "48h",
    TARGETS["48h"]
)

all_predictions = all_predictions.merge(
    predictions_48h,
    on=["cow_id", "date"],
    how="left"
)

all_results.append(
    results_48h
)


# ============================================================
# 3 DAYS
# ============================================================

model_3d, predictions_3d, results_3d = train_forecast_model(
    "3d",
    TARGETS["3d"]
)

all_predictions = all_predictions.merge(
    predictions_3d,
    on=["cow_id", "date"],
    how="left"
)

all_results.append(
    results_3d
)


# ============================================================
# 7 DAYS
# ============================================================

model_7d, predictions_7d, results_7d = train_forecast_model(
    "7d",
    TARGETS["7d"]
)

all_predictions = all_predictions.merge(
    predictions_7d,
    on=["cow_id", "date"],
    how="left"
)

all_results.append(
    results_7d
)


# ============================================================
# 12. SAVE MODELS
# ============================================================

print("\n")
print("=" * 75)
print("SAVING TRAINED MODEL 2 MODELS")
print("=" * 75)

joblib.dump(
    model_24h,
    MODEL_24H_FILE
)

print(
    f"24h model saved: {MODEL_24H_FILE}"
)


joblib.dump(
    model_48h,
    MODEL_48H_FILE
)

print(
    f"48h model saved: {MODEL_48H_FILE}"
)


joblib.dump(
    model_3d,
    MODEL_3D_FILE
)

print(
    f"3-day model saved: {MODEL_3D_FILE}"
)


joblib.dump(
    model_7d,
    MODEL_7D_FILE
)

print(
    f"7-day model saved: {MODEL_7D_FILE}"
)


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

print("\nSAVING TEST PREDICTIONS")
print("-" * 75)

all_predictions = all_predictions.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)

all_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"Test predictions saved: {PREDICTIONS_FILE}"
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

print("\nSAVING MODEL RESULTS")
print("-" * 75)

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    f"Results saved: {RESULTS_FILE}"
)


# ============================================================
# 15. FINAL PERFORMANCE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("                 FINAL MODEL 2 RESULTS")
print("=" * 75)

summary_columns = [
    "horizon",
    "model_mae",
    "model_rmse",
    "model_r2",
    "naive_mae",
    "naive_rmse",
    "naive_r2",
    "mae_improvement_percent"
]

print(
    results_df[
        summary_columns
    ].round(4).to_string(index=False)
)


# ============================================================
# 16. CHECK WHETHER MODEL BEATS BASELINE
# ============================================================

print("\n")
print("=" * 75)
print("              BASELINE COMPARISON")
print("=" * 75)

for _, row in results_df.iterrows():

    horizon = row["horizon"]

    model_mae = row["model_mae"]

    baseline_mae = row["naive_mae"]

    if model_mae < baseline_mae:

        print(
            f"{horizon:>4}: MODEL 2 BEATS BASELINE"
        )

    elif model_mae > baseline_mae:

        print(
            f"{horizon:>4}: MODEL 2 DOES NOT BEAT BASELINE"
        )

    else:

        print(
            f"{horizon:>4}: MODEL 2 = BASELINE"
        )


# ============================================================
# 17. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 75)
print("              SAMPLE TEST PREDICTIONS")
print("=" * 75)

display_columns = [
    "cow_id",
    "date",
    "actual_24h",
    "predicted_24h",
    "actual_48h",
    "predicted_48h",
    "actual_3d",
    "predicted_3d",
    "actual_7d",
    "predicted_7d"
]

print(
    all_predictions[
        display_columns
    ].head(20).round(2).to_string(index=False)
)


# ============================================================
# 18. CHECK PREDICTION RANGE
# ============================================================

print("\n")
print("=" * 75)
print("              PREDICTION RANGE CHECK")
print("=" * 75)

prediction_columns = [
    "predicted_24h",
    "predicted_48h",
    "predicted_3d",
    "predicted_7d"
]

for column in prediction_columns:

    print(
        f"{column:<20} "
        f"min={all_predictions[column].min():.2f}% "
        f"max={all_predictions[column].max():.2f}%"
    )


# ============================================================
# 19. FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 75)
print("                 FINAL VALIDATION")
print("=" * 75)

prediction_missing = all_predictions[
    prediction_columns
].isna().sum().sum()

print(
    f"Missing predictions: {prediction_missing}"
)

if prediction_missing == 0:
    print("Prediction validation: PASSED")
else:
    print("WARNING: Missing predictions detected.")


# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("             MODEL 2 TRAINING COMPLETE")
print("=" * 75)

print(f"""
Dataset:
    {DATASET_FILE}

Total rows:
    {len(df):,}

Training cows:
    {len(train_cows):,}

Testing cows:
    {len(test_cows):,}

Training rows:
    {len(train_df):,}

Testing rows:
    {len(test_df):,}

Cow overlap:
    {len(overlap)}

Forecast models:
    24 hours
    48 hours
    3 days
    7 days

Saved models:
    {MODEL_24H_FILE}
    {MODEL_48H_FILE}
    {MODEL_3D_FILE}
    {MODEL_7D_FILE}

Saved predictions:
    {PREDICTIONS_FILE}

Saved results:
    {RESULTS_FILE}
""")

print("=" * 75)
print("DO NOT MODIFY THE RESULTS YET.")
print("Send the COMPLETE TERMINAL OUTPUT to Nat.")
print("=" * 75)
