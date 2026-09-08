import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "model2_test_predictions.csv"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_results.csv"
)

OUTPUT_DIAGNOSTIC_FILE = os.path.join(
    BASE_DIR,
    "model2_diagnostic_results.csv"
)

OUTPUT_RANGE_FILE = os.path.join(
    BASE_DIR,
    "model2_risk_range_analysis.csv"
)

OUTPUT_HIGH_RISK_FILE = os.path.join(
    BASE_DIR,
    "model2_high_risk_predictions.csv"
)


# ============================================================
# HORIZONS
# ============================================================

HORIZONS = {
    "24h": "risk_24h",
    "48h": "risk_48h",
    "3d": "risk_3d",
    "7d": "risk_7d",
}


# ============================================================
# THRESHOLDS
# ============================================================

HIGH_RISK_THRESHOLD = 50.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_separator(char="=", length=78):
    print(char * length)


def safe_r2(y_true, y_pred):
    """
    Calculate R2 safely.
    """
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE and R2.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = safe_r2(y_true, y_pred)

    return mae, rmse, r2


def calculate_high_risk_metrics(y_true, y_pred):
    """
    Convert continuous risk predictions into binary
    high-risk / not-high-risk predictions.

    High risk = risk >= 50%
    """

    actual_binary = (
        np.asarray(y_true) >= HIGH_RISK_THRESHOLD
    ).astype(int)

    predicted_binary = (
        np.asarray(y_pred) >= HIGH_RISK_THRESHOLD
    ).astype(int)

    precision = precision_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    recall = recall_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    f1 = f1_score(
        actual_binary,
        predicted_binary,
        zero_division=0
    )

    cm = confusion_matrix(
        actual_binary,
        predicted_binary,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "actual_high_risk_count": int(actual_binary.sum()),
        "predicted_high_risk_count": int(predicted_binary.sum()),
    }


def get_risk_range(value):
    """
    Convert risk percentage into a risk range.
    """

    if value < 20:
        return "0-20%"

    elif value < 50:
        return "20-50%"

    elif value < 80:
        return "50-80%"

    else:
        return "80-100%"


# ============================================================
# HEADER
# ============================================================

print()
print_separator()
print("              MODEL 2 DIAGNOSTIC ANALYSIS")
print_separator()

print()
print("Purpose:")
print("Evaluate whether Model 2 is useful as a future mastitis")
print("risk-warning system, not just whether it has good")
print("numerical regression metrics.")
print()

print(f"High-risk threshold: >= {HIGH_RISK_THRESHOLD}%")
print()


# ============================================================
# CHECK FILE
# ============================================================

print_separator("-")

if not os.path.exists(PREDICTIONS_FILE):
    print()
    print("ERROR:")
    print("model2_test_predictions.csv was not found.")
    print()
    print("Expected location:")
    print(PREDICTIONS_FILE)
    print()
    print("Run model2_train.py first.")
    print()
    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Model 2 prediction data...")

predictions = pd.read_csv(PREDICTIONS_FILE)

print("Prediction file loaded successfully.")
print(f"Rows: {len(predictions):,}")
print()

print("Columns found:")
print(list(predictions.columns))
print()


# ============================================================
# DETERMINE COLUMN STRUCTURE
# ============================================================

# The training script may use names such as:
#
# actual_24h / predicted_24h
# actual_48h / predicted_48h
# actual_3d / predicted_3d
# actual_7d / predicted_7d
#
# We detect these automatically.

column_map = {}

for horizon in HORIZONS.keys():

    possible_actual = [
        f"actual_{horizon}",
        f"risk_{horizon}_actual",
        f"actual_risk_{horizon}",
    ]

    possible_predicted = [
        f"predicted_{horizon}",
        f"risk_{horizon}_predicted",
        f"predicted_risk_{horizon}",
    ]

    actual_col = None
    predicted_col = None

    for col in possible_actual:
        if col in predictions.columns:
            actual_col = col
            break

    for col in possible_predicted:
        if col in predictions.columns:
            predicted_col = col
            break

    if actual_col is not None and predicted_col is not None:
        column_map[horizon] = {
            "actual": actual_col,
            "predicted": predicted_col
        }


# ============================================================
# VALIDATE HORIZONS
# ============================================================

print_separator("-")
print("HORIZON COLUMN DETECTION")
print_separator("-")

for horizon in HORIZONS.keys():

    if horizon in column_map:

        print(
            f"{horizon:>4} : "
            f"actual={column_map[horizon]['actual']} | "
            f"predicted={column_map[horizon]['predicted']}"
        )

    else:

        print(
            f"{horizon:>4} : NOT FOUND"
        )

print()


if len(column_map) == 0:

    print("ERROR:")
    print("Could not identify actual/predicted columns.")
    print()
    print("Please check model2_test_predictions.csv.")
    print()

    raise SystemExit(1)


# ============================================================
# BASIC DATA QUALITY CHECK
# ============================================================

print_separator("-")
print("DATA QUALITY CHECK")
print_separator("-")

print(
    f"Total prediction rows: "
    f"{len(predictions):,}"
)

if "cow_id" in predictions.columns:

    print(
        f"Unique cows: "
        f"{predictions['cow_id'].nunique():,}"
    )

missing_total = predictions.isnull().sum().sum()

print(
    f"Missing cells: {missing_total:,}"
)

print()


# ============================================================
# STORAGE
# ============================================================

diagnostic_results = []
range_results = []
high_risk_rows = []


# ============================================================
# ANALYZE EACH HORIZON
# ============================================================

for horizon, cols in column_map.items():

    actual_col = cols["actual"]
    predicted_col = cols["predicted"]

    y_true = pd.to_numeric(
        predictions[actual_col],
        errors="coerce"
    )

    y_pred = pd.to_numeric(
        predictions[predicted_col],
        errors="coerce"
    )

    valid_mask = (
        y_true.notna()
        & y_pred.notna()
    )

    y_true = y_true[valid_mask].to_numpy()
    y_pred = y_pred[valid_mask].to_numpy()

    # Clip predictions to valid risk range.
    y_pred = np.clip(
        y_pred,
        0,
        100
    )

    # --------------------------------------------------------
    # REGRESSION METRICS
    # --------------------------------------------------------

    mae, rmse, r2 = calculate_regression_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # HIGH RISK METRICS
    # --------------------------------------------------------

    high_risk = calculate_high_risk_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    errors = y_pred - y_true

    mean_actual = np.mean(y_true)
    mean_predicted = np.mean(y_pred)

    median_actual = np.median(y_true)
    median_predicted = np.median(y_pred)

    mean_error = np.mean(errors)

    mean_absolute_error_value = np.mean(
        np.abs(errors)
    )

    # --------------------------------------------------------
    # HIGH-RISK UNDERPREDICTION
    # --------------------------------------------------------

    actual_high_mask = (
        y_true >= HIGH_RISK_THRESHOLD
    )

    if np.any(actual_high_mask):

        high_risk_actual_mean = np.mean(
            y_true[actual_high_mask]
        )

        high_risk_predicted_mean = np.mean(
            y_pred[actual_high_mask]
        )

        high_risk_underprediction = (
            high_risk_actual_mean
            - high_risk_predicted_mean
        )

    else:

        high_risk_actual_mean = 0
        high_risk_predicted_mean = 0
        high_risk_underprediction = 0

    # --------------------------------------------------------
    # PRINT HORIZON REPORT
    # --------------------------------------------------------

    print()
    print_separator()
    print(f"                 {horizon} FORECAST")
    print_separator()

    print()

    print("REGRESSION PERFORMANCE")
    print("-" * 50)

    print(
        f"MAE              : {mae:.4f}%"
    )

    print(
        f"RMSE             : {rmse:.4f}%"
    )

    print(
        f"R²               : {r2:.4f}"
    )

    print()

    print("RISK DISTRIBUTION")
    print("-" * 50)

    print(
        f"Actual mean      : {mean_actual:.2f}%"
    )

    print(
        f"Predicted mean   : {mean_predicted:.2f}%"
    )

    print(
        f"Actual median    : {median_actual:.2f}%"
    )

    print(
        f"Predicted median : {median_predicted:.2f}%"
    )

    print(
        f"Mean error       : {mean_error:.2f}%"
    )

    print()

    print("HIGH-RISK DETECTION")
    print("-" * 50)

    print(
        f"Threshold        : >= {HIGH_RISK_THRESHOLD:.0f}%"
    )

    print(
        f"Actual high-risk : "
        f"{high_risk['actual_high_risk_count']:,}"
    )

    print(
        f"Predicted high-risk : "
        f"{high_risk['predicted_high_risk_count']:,}"
    )

    print()

    print(
        f"Precision        : "
        f"{high_risk['precision']:.4f}"
    )

    print(
        f"Recall           : "
        f"{high_risk['recall']:.4f}"
    )

    print(
        f"F1-score         : "
        f"{high_risk['f1']:.4f}"
    )

    print()

    print("CONFUSION MATRIX")
    print("-" * 50)

    print("                 Predicted")
    print("                 Low   High")
    print(
        f"Actual Low       "
        f"{high_risk['tn']:5d} "
        f"{high_risk['fp']:5d}"
    )

    print(
        f"Actual High      "
        f"{high_risk['fn']:5d} "
        f"{high_risk['tp']:5d}"
    )

    print()

    print("HIGH-RISK UNDERPREDICTION CHECK")
    print("-" * 50)

    print(
        f"Actual high-risk mean    : "
        f"{high_risk_actual_mean:.2f}%"
    )

    print(
        f"Predicted high-risk mean : "
        f"{high_risk_predicted_mean:.2f}%"
    )

    print(
        f"Underprediction gap      : "
        f"{high_risk_underprediction:.2f}%"
    )

    # --------------------------------------------------------
    # STORE DIAGNOSTIC RESULT
    # --------------------------------------------------------

    diagnostic_results.append({

        "horizon": horizon,

        "rows_evaluated": len(y_true),

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "actual_mean": mean_actual,

        "predicted_mean": mean_predicted,

        "actual_median": median_actual,

        "predicted_median": median_predicted,

        "mean_error": mean_error,

        "mean_absolute_error": mean_absolute_error_value,

        "high_risk_threshold": HIGH_RISK_THRESHOLD,

        "actual_high_risk_count":
            high_risk["actual_high_risk_count"],

        "predicted_high_risk_count":
            high_risk["predicted_high_risk_count"],

        "high_risk_precision":
            high_risk["precision"],

        "high_risk_recall":
            high_risk["recall"],

        "high_risk_f1":
            high_risk["f1"],

        "true_negative":
            high_risk["tn"],

        "false_positive":
            high_risk["fp"],

        "false_negative":
            high_risk["fn"],

        "true_positive":
            high_risk["tp"],

        "actual_high_risk_mean":
            high_risk_actual_mean,

        "predicted_high_risk_mean":
            high_risk_predicted_mean,

        "high_risk_underprediction_gap":
            high_risk_underprediction,
    })

    # --------------------------------------------------------
    # RISK RANGE ANALYSIS
    # --------------------------------------------------------

    range_labels = [
        "0-20%",
        "20-50%",
        "50-80%",
        "80-100%"
    ]

    print()
    print("PERFORMANCE BY ACTUAL RISK RANGE")
    print("-" * 70)

    for risk_range in range_labels:

        range_mask = np.array([
            get_risk_range(value) == risk_range
            for value in y_true
        ])

        count = int(range_mask.sum())

        if count == 0:

            print(
                f"{risk_range:>8} : "
                f"No samples"
            )

            continue

        range_actual = y_true[range_mask]
        range_predicted = y_pred[range_mask]

        range_mae = mean_absolute_error(
            range_actual,
            range_predicted
        )

        range_actual_mean = np.mean(
            range_actual
        )

        range_predicted_mean = np.mean(
            range_predicted
        )

        range_bias = (
            range_predicted_mean
            - range_actual_mean
        )

        print(
            f"{risk_range:>8} : "
            f"Count={count:5d} | "
            f"MAE={range_mae:7.2f}% | "
            f"Actual Mean={range_actual_mean:7.2f}% | "
            f"Pred Mean={range_predicted_mean:7.2f}% | "
            f"Bias={range_bias:+7.2f}%"
        )

        range_results.append({

            "horizon": horizon,

            "risk_range": risk_range,

            "count": count,

            "mae": range_mae,

            "actual_mean": range_actual_mean,

            "predicted_mean": range_predicted_mean,

            "bias": range_bias,
        })

    # --------------------------------------------------------
    # SAVE HIGH-RISK INDIVIDUAL PREDICTIONS
    # --------------------------------------------------------

    high_mask = (
        y_true >= HIGH_RISK_THRESHOLD
    )

    if np.any(high_mask):

        temp = predictions.loc[
            valid_mask
        ].copy()

        temp["actual_risk"] = y_true
        temp["predicted_risk"] = y_pred

        temp["horizon"] = horizon

        temp["prediction_error"] = (
            temp["predicted_risk"]
            - temp["actual_risk"]
        )

        temp["absolute_error"] = np.abs(
            temp["prediction_error"]
        )

        temp["actual_high_risk"] = (
            temp["actual_risk"]
            >= HIGH_RISK_THRESHOLD
        )

        temp["predicted_high_risk"] = (
            temp["predicted_risk"]
            >= HIGH_RISK_THRESHOLD
        )

        temp["missed_high_risk"] = (
            temp["actual_high_risk"]
            & ~temp["predicted_high_risk"]
        )

        high_risk_rows.append(
            temp[
                temp["actual_high_risk"]
            ].copy()
        )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

diagnostic_df = pd.DataFrame(
    diagnostic_results
)

range_df = pd.DataFrame(
    range_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

diagnostic_df.to_csv(
    OUTPUT_DIAGNOSTIC_FILE,
    index=False
)

range_df.to_csv(
    OUTPUT_RANGE_FILE,
    index=False
)


if high_risk_rows:

    high_risk_df = pd.concat(
        high_risk_rows,
        ignore_index=True
    )

    high_risk_df = high_risk_df.sort_values(
        by=[
            "horizon",
            "absolute_error"
        ],
        ascending=[
            True,
            False
        ]
    )

    high_risk_df.to_csv(
        OUTPUT_HIGH_RISK_FILE,
        index=False
    )

else:

    high_risk_df = pd.DataFrame()

    high_risk_df.to_csv(
        OUTPUT_HIGH_RISK_FILE,
        index=False
    )


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print()
print_separator()
print("                 FINAL DIAGNOSTIC SUMMARY")
print_separator()

summary_columns = [
    "horizon",
    "mae",
    "rmse",
    "r2",
    "high_risk_precision",
    "high_risk_recall",
    "high_risk_f1",
    "actual_high_risk_count",
    "predicted_high_risk_count",
]

print()

print(
    diagnostic_df[
        summary_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

print_separator()
print("                 AUTOMATIC INTERPRETATION")
print_separator()

print()

for _, row in diagnostic_df.iterrows():

    horizon = row["horizon"]

    mae = row["mae"]

    r2 = row["r2"]

    precision = row["high_risk_precision"]

    recall = row["high_risk_recall"]

    f1 = row["high_risk_f1"]

    underprediction = (
        row["high_risk_underprediction_gap"]
    )

    print(f"{horizon}:")

    # --------------------------------------------------------
    # Regression quality
    # --------------------------------------------------------

    if horizon == "24h":

        if r2 >= 0.70 and mae <= 10:

            print(
                "  - Strong short-term forecasting performance."
            )

        elif r2 >= 0.50:

            print(
                "  - Reasonable short-term forecasting performance."
            )

        else:

            print(
                "  - Short-term forecasting performance needs improvement."
            )

    elif horizon == "48h":

        if r2 >= 0.50:

            print(
                "  - Good 48-hour forecasting performance."
            )

        elif r2 >= 0.25:

            print(
                "  - Moderate 48-hour forecasting performance."
            )

        else:

            print(
                "  - Weak 48-hour forecasting performance."
            )

    elif horizon == "3d":

        if r2 >= 0.50:

            print(
                "  - Strong 3-day forecasting performance."
            )

        elif r2 >= 0.25:

            print(
                "  - Moderate 3-day forecasting performance."
            )

        else:

            print(
                "  - Weak 3-day forecasting performance."
            )

    elif horizon == "7d":

        if r2 >= 0.25:

            print(
                "  - Useful 7-day regression performance."
            )

        elif r2 >= 0:

            print(
                "  - Weak 7-day regression performance."
            )

        else:

            print(
                "  - 7-day R² is negative; long-term numerical "
                "forecasting is weak."
            )

    # --------------------------------------------------------
    # High-risk detection
    # --------------------------------------------------------

    print(
        f"  - High-risk precision: {precision:.3f}"
    )

    print(
        f"  - High-risk recall:    {recall:.3f}"
    )

    print(
        f"  - High-risk F1:        {f1:.3f}"
    )

    # --------------------------------------------------------
    # Warning about high-risk underprediction
    # --------------------------------------------------------

    if underprediction >= 20:

        print(
            f"  - WARNING: high-risk cases are being "
            f"underpredicted by about {underprediction:.1f} percentage points."
        )

    elif underprediction >= 10:

        print(
            f"  - NOTICE: some high-risk underprediction "
            f"is present ({underprediction:.1f} points)."
        )

    else:

        print(
            "  - High-risk predictions are not strongly underpredicted."
        )

    print()


# ============================================================
# OVERALL MODEL VERDICT
# ============================================================

print_separator()
print("                     OVERALL VERDICT")
print_separator()

print()

# Extract rows

row_24h = diagnostic_df[
    diagnostic_df["horizon"] == "24h"
]

row_48h = diagnostic_df[
    diagnostic_df["horizon"] == "48h"
]

row_3d = diagnostic_df[
    diagnostic_df["horizon"] == "3d"
]

row_7d = diagnostic_df[
    diagnostic_df["horizon"] == "7d"
]


def get_value(df, column):

    if len(df) == 0:
        return np.nan

    return float(
        df.iloc[0][column]
    )


r2_24 = get_value(row_24h, "r2")
r2_48 = get_value(row_48h, "r2")
r2_3d = get_value(row_3d, "r2")
r2_7d = get_value(row_7d, "r2")

recall_24 = get_value(
    row_24h,
    "high_risk_recall"
)

recall_48 = get_value(
    row_48h,
    "high_risk_recall"
)

recall_3d = get_value(
    row_3d,
    "high_risk_recall"
)

recall_7d = get_value(
    row_7d,
    "high_risk_recall"
)


print("1. 24-hour forecast:")

if (
    not np.isnan(r2_24)
    and r2_24 >= 0.70
):

    print(
        "   PASS - 24-hour forecast is strong."
    )

else:

    print(
        "   REVIEW - 24-hour forecast needs attention."
    )


print()

print("2. 48-hour forecast:")

if (
    not np.isnan(r2_48)
    and r2_48 >= 0.50
):

    print(
        "   PASS - 48-hour forecast is useful."
    )

else:

    print(
        "   REVIEW - 48-hour forecast is moderate/weak."
    )


print()

print("3. 3-day forecast:")

if (
    not np.isnan(r2_3d)
    and r2_3d >= 0.25
):

    print(
        "   PASS - 3-day forecast contains useful predictive information."
    )

else:

    print(
        "   REVIEW - 3-day forecast is weak."
    )


print()

print("4. 7-day forecast:")

if (
    not np.isnan(r2_7d)
    and r2_7d >= 0.25
):

    print(
        "   PASS - 7-day forecast has useful regression performance."
    )

elif (
    not np.isnan(r2_7d)
    and r2_7d >= 0
):

    print(
        "   CAUTION - 7-day forecast is weak."
    )

else:

    print(
        "   CAUTION - 7-day R² is negative."
    )

    print(
        "   Do NOT describe the 7-day forecast as highly accurate."
    )


print()

print("5. High-risk detection:")

valid_recalls = [
    value
    for value in [
        recall_24,
        recall_48,
        recall_3d,
        recall_7d
    ]
    if not np.isnan(value)
]

if valid_recalls:

    average_recall = np.mean(
        valid_recalls
    )

    print(
        f"   Average high-risk recall: "
        f"{average_recall:.3f}"
    )

    if average_recall >= 0.70:

        print(
            "   PASS - model detects a large portion of high-risk cases."
        )

    elif average_recall >= 0.50:

        print(
            "   MODERATE - model detects some high-risk cases, "
            "but misses a significant number."
        )

    else:

        print(
            "   WARNING - high-risk detection is weak."
        )


# ============================================================
# IMPORTANT SCIENTIFIC WARNING
# ============================================================

print()
print_separator()
print("                 IMPORTANT INTERPRETATION")
print_separator()

print()

print(
    "Model 2 predicts future MODEL 1 RISK SCORES."
)

print(
    "It does NOT directly predict clinically confirmed mastitis."
)

print()

print(
    "Therefore, the forecast should be described as:"
)

print(
    '  \"Predicted future mastitis risk based on the Model 1 risk trend.\"'
)

print()

print(
    "Do NOT describe Model 2 as a clinically validated "
    "mastitis probability model."
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print_separator()
print("                 FILES CREATED")
print_separator()

print()

print(
    f"1. {OUTPUT_DIAGNOSTIC_FILE}"
)

print(
    f"2. {OUTPUT_RANGE_FILE}"
)

print(
    f"3. {OUTPUT_HIGH_RISK_FILE}"
)

print()

print("Diagnostic analysis completed successfully.")

print()
print_separator()
print("                         DONE")
print_separator()

