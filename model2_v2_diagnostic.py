
import os
import warnings
import numpy as np
import pandas as pd

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

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_test_predictions.csv"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_diagnostic_results.csv"
)

RISK_RANGE_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_risk_range_analysis.csv"
)

HIGH_RISK_FILE = os.path.join(
    BASE_DIR,
    "model2_v2_high_risk_cases.csv"
)

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "model2_v1_vs_v2_comparison.csv"
)

HIGH_RISK_THRESHOLD = 50.0


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 78)
print("                 MODEL 2 V2 DIAGNOSTIC")
print("=" * 78)

print()
print("Purpose:")
print("Diagnose Model 2 V2 forecasting performance.")
print()
print("This script will NOT modify:")
print("- Model 1")
print("- Model 1 dataset")
print("- Model 2 V2 models")
print("- Model 2 V2 prediction file")


# ============================================================
# CHECK FILE
# ============================================================

print()
print("=" * 78)
print("                    FILE CHECK")
print("=" * 78)

if not os.path.exists(PREDICTIONS_FILE):

    raise FileNotFoundError(
        f"\nERROR: Prediction file not found:\n"
        f"{PREDICTIONS_FILE}"
    )

print()
print("PASS: V2 prediction file found.")
print(PREDICTIONS_FILE)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("                  LOADING DATA")
print("=" * 78)

df = pd.read_csv(
    PREDICTIONS_FILE
)

print()
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))

print()
print("Columns:")
for column in df.columns:
    print(" -", column)


# ============================================================
# HORIZONS
# ============================================================

HORIZONS = {
    "24h": {
        "actual": "actual_24h",
        "predicted": "predicted_24h",
        "naive": "naive_24h"
    },
    "48h": {
        "actual": "actual_48h",
        "predicted": "predicted_48h",
        "naive": "naive_48h"
    },
    "3d": {
        "actual": "actual_3d",
        "predicted": "predicted_3d",
        "naive": "naive_3d"
    },
    "7d": {
        "actual": "actual_7d",
        "predicted": "predicted_7d",
        "naive": "naive_7d"
    }
}


# ============================================================
# COLUMN CHECK
# ============================================================

print()
print("=" * 78)
print("                COLUMN VALIDATION")
print("=" * 78)

required_columns = [
    "cow_id",
    "date"
]

for horizon in HORIZONS.values():

    required_columns.extend([
        horizon["actual"],
        horizon["predicted"],
        horizon["naive"]
    ])


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
        "Prediction file is incomplete."
    )

print()
print("PASS: All required prediction columns found.")


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 78)
print("                 BASIC VALIDATION")
print("=" * 78)

missing_cells = int(
    df[required_columns]
    .isna()
    .sum()
    .sum()
)

duplicate_rows = int(
    df.duplicated(
        subset=["cow_id", "date"]
    ).sum()
)

unique_cows = df["cow_id"].nunique()

print()
print("Unique cows:", unique_cows)
print("Missing cells:", missing_cells)
print("Duplicate cow/date:", duplicate_rows)

if missing_cells > 0:
    raise ValueError(
        "ERROR: Missing prediction values detected."
    )

if duplicate_rows > 0:
    raise ValueError(
        "ERROR: Duplicate cow/date records detected."
    )

print()
print("PASS: Prediction file is structurally valid.")


# ============================================================
# STORAGE
# ============================================================

diagnostic_results = []
risk_range_results = []
high_risk_cases = []


# ============================================================
# ANALYZE EACH HORIZON
# ============================================================

for horizon, columns in HORIZONS.items():

    actual_col = columns["actual"]
    predicted_col = columns["predicted"]
    naive_col = columns["naive"]

    actual = df[actual_col].values
    predicted = df[predicted_col].values
    naive = df[naive_col].values

    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    naive_mae = mean_absolute_error(
        actual,
        naive
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            actual,
            naive
        )
    )

    naive_r2 = r2_score(
        actual,
        naive
    )

    mae_improvement = (
        (naive_mae - mae)
        / naive_mae
    ) * 100 if naive_mae != 0 else 0

    # --------------------------------------------------------
    # ERROR / BIAS
    # --------------------------------------------------------

    errors = predicted - actual

    mean_bias = np.mean(errors)

    mean_absolute_error_value = np.mean(
        np.abs(errors)
    )

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    actual_high = (
        actual >= HIGH_RISK_THRESHOLD
    ).astype(int)

    predicted_high = (
        predicted >= HIGH_RISK_THRESHOLD
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
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    predicted_mean = np.mean(
        predicted
    )

    predicted_median = np.median(
        predicted
    )

    predicted_min = np.min(
        predicted
    )

    predicted_max = np.max(
        predicted
    )

    actual_mean = np.mean(
        actual
    )

    actual_median = np.median(
        actual
    )

    actual_min = np.min(
        actual
    )

    actual_max = np.max(
        actual
    )

    # --------------------------------------------------------
    # HIGH-RISK ACTUAL MEAN VS PREDICTED
    # --------------------------------------------------------

    high_mask = (
        actual >= HIGH_RISK_THRESHOLD
    )

    actual_high_count = int(
        high_mask.sum()
    )

    predicted_high_count = int(
        predicted_high.sum()
    )

    if actual_high_count > 0:

        actual_high_mean = np.mean(
            actual[high_mask]
        )

        predicted_high_mean = np.mean(
            predicted[high_mask]
        )

        high_risk_underprediction = (
            actual_high_mean
            - predicted_high_mean
        )

    else:

        actual_high_mean = 0
        predicted_high_mean = 0
        high_risk_underprediction = 0

    # --------------------------------------------------------
    # PRINT HORIZON
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(f"                    {horizon} ANALYSIS")
    print("=" * 78)

    print()
    print("OVERALL PERFORMANCE")
    print("-" * 50)

    print(
        f"MAE                 : {mae:.4f}"
    )

    print(
        f"RMSE                : {rmse:.4f}"
    )

    print(
        f"R²                  : {r2:.4f}"
    )

    print(
        f"Naive MAE           : {naive_mae:.4f}"
    )

    print(
        f"Naive RMSE          : {naive_rmse:.4f}"
    )

    print(
        f"Naive R²            : {naive_r2:.4f}"
    )

    print(
        f"MAE improvement     : {mae_improvement:.2f}%"
    )

    print()
    print("BIAS")
    print("-" * 50)

    print(
        f"Mean bias           : {mean_bias:.4f}"
    )

    print(
        f"Absolute error mean : "
        f"{mean_absolute_error_value:.4f}"
    )

    if mean_bias < -2:
        print(
            "Interpretation      : UNDERPREDICTION"
        )

    elif mean_bias > 2:
        print(
            "Interpretation      : OVERPREDICTION"
        )

    else:
        print(
            "Interpretation      : RELATIVELY BALANCED"
        )

    print()
    print("ACTUAL RISK DISTRIBUTION")
    print("-" * 50)

    print(
        f"Mean                : {actual_mean:.2f}%"
    )

    print(
        f"Median              : {actual_median:.2f}%"
    )

    print(
        f"Minimum             : {actual_min:.2f}%"
    )

    print(
        f"Maximum             : {actual_max:.2f}%"
    )

    print()
    print("PREDICTED RISK DISTRIBUTION")
    print("-" * 50)

    print(
        f"Mean                : {predicted_mean:.2f}%"
    )

    print(
        f"Median              : {predicted_median:.2f}%"
    )

    print(
        f"Minimum             : {predicted_min:.2f}%"
    )

    print(
        f"Maximum             : {predicted_max:.2f}%"
    )

    print()
    print("HIGH-RISK DETECTION")
    print("-" * 50)

    print(
        f"Threshold           : "
        f">= {HIGH_RISK_THRESHOLD:.1f}%"
    )

    print(
        f"Actual high-risk    : "
        f"{actual_high_count:,}"
    )

    print(
        f"Predicted high-risk : "
        f"{predicted_high_count:,}"
    )

    print(
        f"Precision           : {precision:.4f}"
    )

    print(
        f"Recall              : {recall:.4f}"
    )

    print(
        f"F1                  : {f1:.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 50)

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

    print()
    print("ACTUAL HIGH-RISK CASE ANALYSIS")
    print("-" * 50)

    print(
        f"Actual high-risk mean    : "
        f"{actual_high_mean:.2f}%"
    )

    print(
        f"Predicted on those cases : "
        f"{predicted_high_mean:.2f}%"
    )

    print(
        f"Underprediction gap      : "
        f"{high_risk_underprediction:.2f} points"
    )

    # --------------------------------------------------------
    # RISK RANGES
    # --------------------------------------------------------

    ranges = [
        ("0-20", 0, 20),
        ("20-50", 20, 50),
        ("50-80", 50, 80),
        ("80-100", 80, 100)
    ]

    print()
    print("RISK RANGE ANALYSIS")
    print("-" * 78)

    print(
        f"{'Range':<12}"
        f"{'Count':>8}"
        f"{'MAE':>12}"
        f"{'Actual Mean':>15}"
        f"{'Pred Mean':>15}"
        f"{'Bias':>12}"
    )

    for range_name, lower, upper in ranges:

        if upper == 100:

            mask = (
                (actual >= lower)
                & (actual <= upper)
            )

        else:

            mask = (
                (actual >= lower)
                & (actual < upper)
            )

        count = int(mask.sum())

        if count > 0:

            range_actual = actual[mask]
            range_predicted = predicted[mask]

            range_mae = mean_absolute_error(
                range_actual,
                range_predicted
            )

            range_actual_mean = np.mean(
                range_actual
            )

            range_pred_mean = np.mean(
                range_predicted
            )

            range_bias = (
                range_pred_mean
                - range_actual_mean
            )

        else:

            range_mae = 0
            range_actual_mean = 0
            range_pred_mean = 0
            range_bias = 0

        print(
            f"{range_name:<12}"
            f"{count:>8}"
            f"{range_mae:>12.2f}"
            f"{range_actual_mean:>15.2f}"
            f"{range_pred_mean:>15.2f}"
            f"{range_bias:>12.2f}"
        )

        risk_range_results.append({

            "horizon": horizon,
            "risk_range": range_name,
            "count": count,
            "mae": range_mae,
            "actual_mean": range_actual_mean,
            "predicted_mean": range_pred_mean,
            "bias": range_bias
        })

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    diagnostic_results.append({

        "horizon": horizon,

        "mae": mae,
        "rmse": rmse,
        "r2": r2,

        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_r2": naive_r2,

        "mae_improvement_percent":
            mae_improvement,

        "mean_bias":
            mean_bias,

        "actual_mean":
            actual_mean,

        "actual_median":
            actual_median,

        "actual_min":
            actual_min,

        "actual_max":
            actual_max,

        "predicted_mean":
            predicted_mean,

        "predicted_median":
            predicted_median,

        "predicted_min":
            predicted_min,

        "predicted_max":
            predicted_max,

        "actual_high_risk_count":
            actual_high_count,

        "predicted_high_risk_count":
            predicted_high_count,

        "high_risk_precision":
            precision,

        "high_risk_recall":
            recall,

        "high_risk_f1":
            f1,

        "actual_high_risk_mean":
            actual_high_mean,

        "predicted_on_actual_high_risk_mean":
            predicted_high_mean,

        "high_risk_underprediction_gap":
            high_risk_underprediction,

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)
    })

    # --------------------------------------------------------
    # COLLECT MISSED HIGH-RISK CASES
    # --------------------------------------------------------

    for index in range(len(df)):

        if (
            actual[index] >= HIGH_RISK_THRESHOLD
            and predicted[index] < HIGH_RISK_THRESHOLD
        ):

            high_risk_cases.append({

                "cow_id":
                    df.iloc[index]["cow_id"],

                "date":
                    df.iloc[index]["date"],

                "horizon":
                    horizon,

                "actual_risk":
                    actual[index],

                "predicted_risk":
                    predicted[index],

                "error":
                    predicted[index]
                    - actual[index],

                "absolute_error":
                    abs(
                        predicted[index]
                        - actual[index]
                    )
            })


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

print()
print("=" * 78)
print("                 SAVING DIAGNOSTICS")
print("=" * 78)

diagnostic_df = pd.DataFrame(
    diagnostic_results
)

diagnostic_df.to_csv(
    RESULTS_FILE,
    index=False
)

print()
print("Diagnostic results saved:")
print(RESULTS_FILE)


risk_range_df = pd.DataFrame(
    risk_range_results
)

risk_range_df.to_csv(
    RISK_RANGE_FILE,
    index=False
)

print()
print("Risk range analysis saved:")
print(RISK_RANGE_FILE)


high_risk_df = pd.DataFrame(
    high_risk_cases
)

high_risk_df.to_csv(
    HIGH_RISK_FILE,
    index=False
)

print()
print("Missed high-risk cases saved:")
print(HIGH_RISK_FILE)


# ============================================================
# V1 VS V2 COMPARISON
# ============================================================

print()
print("=" * 78)
print("                 V1 VS V2 COMPARISON")
print("=" * 78)

# Original Model 2 results from previous training
v1_data = {
    "24h": {
        "mae": 5.9323,
        "rmse": 12.7474,
        "r2": 0.8585,
        "precision": 0.9390,
        "recall": 0.8174,
        "f1": 0.8740
    },

    "48h": {
        "mae": 10.6214,
        "rmse": 20.6125,
        "r2": 0.6282,
        "precision": 0.8987,
        "recall": 0.6488,
        "f1": 0.7536
    },

    "3d": {
        "mae": 15.0254,
        "rmse": 26.0817,
        "r2": 0.3992,
        "precision": 0.8126,
        "recall": 0.4749,
        "f1": 0.5995
    },

    "7d": {
        "mae": 22.0166,
        "rmse": 32.8981,
        "r2": -0.0141,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
}


comparison_rows = []

for _, row in diagnostic_df.iterrows():

    horizon = row["horizon"]

    v1 = v1_data[horizon]

    mae_change = (
        row["mae"]
        - v1["mae"]
    )

    rmse_change = (
        row["rmse"]
        - v1["rmse"]
    )

    r2_change = (
        row["r2"]
        - v1["r2"]
    )

    recall_change = (
        row["high_risk_recall"]
        - v1["recall"]
    )

    f1_change = (
        row["high_risk_f1"]
        - v1["f1"]
    )

    comparison_rows.append({

        "horizon": horizon,

        "v1_mae": v1["mae"],
        "v2_mae": row["mae"],
        "mae_change": mae_change,

        "v1_rmse": v1["rmse"],
        "v2_rmse": row["rmse"],
        "rmse_change": rmse_change,

        "v1_r2": v1["r2"],
        "v2_r2": row["r2"],
        "r2_change": r2_change,

        "v1_precision": v1["precision"],
        "v2_precision": row["high_risk_precision"],

        "v1_recall": v1["recall"],
        "v2_recall": row["high_risk_recall"],
        "recall_change": recall_change,

        "v1_f1": v1["f1"],
        "v2_f1": row["high_risk_f1"],
        "f1_change": f1_change
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)

print()
print(
    comparison_df.round(4).to_string(
        index=False
    )
)

print()
print("V1 vs V2 comparison saved:")
print(COMPARISON_FILE)


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 78)
print("                     FINAL VERDICT")
print("=" * 78)

for _, row in diagnostic_df.iterrows():

    horizon = row["horizon"]

    r2 = row["r2"]
    mae = row["mae"]
    improvement = row[
        "mae_improvement_percent"
    ]

    recall = row[
        "high_risk_recall"
    ]

    predicted_high = row[
        "predicted_high_risk_count"
    ]

    actual_high = row[
        "actual_high_risk_count"
    ]

    print()

    print(f"{horizon}")
    print("-" * 40)

    if horizon == "24h":

        if r2 >= 0.70 and recall >= 0.70:
            print("VERDICT: STRONG")
        else:
            print("VERDICT: MODERATE")

    elif horizon == "48h":

        if r2 >= 0.50 and recall >= 0.60:
            print("VERDICT: GOOD")
        else:
            print("VERDICT: MODERATE")

    elif horizon == "3d":

        if r2 >= 0.30:
            print("VERDICT: USEFUL")
        else:
            print("VERDICT: LIMITED")

    elif horizon == "7d":

        if (
            r2 >= 0.30
            and recall >= 0.50
        ):
            print("VERDICT: USEFUL")

        elif (
            r2 >= 0
            and recall >= 0.20
        ):
            print("VERDICT: WEAK")

        else:
            print("VERDICT: CAUTION")

    print(
        f"MAE                  : {mae:.4f}"
    )

    print(
        f"R²                   : {r2:.4f}"
    )

    print(
        f"MAE improvement      : "
        f"{improvement:.2f}%"
    )

    print(
        f"High-risk recall     : "
        f"{recall:.4f}"
    )

    print(
        f"Actual high-risk    : "
        f"{actual_high}"
    )

    print(
        f"Predicted high-risk : "
        f"{predicted_high}"
    )


# ============================================================
# 7-DAY SPECIFIC DIAGNOSIS
# ============================================================

seven_day = diagnostic_df[
    diagnostic_df["horizon"] == "7d"
].iloc[0]

print()
print("=" * 78)
print("                 7-DAY SPECIFIC DIAGNOSIS")
print("=" * 78)

print()

if seven_day["predicted_high_risk_count"] <= 10:

    print(
        "WARNING: The 7-day model is producing almost"
        " no high-risk predictions."
    )

    print()
    print(
        "This means the model is strongly conservative"
        " for long-range high-risk events."
    )

if seven_day["r2"] < 0:

    print()
    print(
        "R² is negative."
    )

    print(
        "The model does not explain the variance of"
        " the 7-day target well."
    )

if seven_day[
    "mae_improvement_percent"
] > 0:

    print()
    print(
        "However, the model still improves MAE"
        " over the naive baseline."
    )

    print(
        f"Improvement: "
        f"{seven_day['mae_improvement_percent']:.2f}%"
    )

print()
print(
    "Therefore, 7-day forecasting should currently"
    " be treated as a long-range estimate/trend,"
    " not as a reliable high-risk detector."
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print()
print("=" * 78)
print("                  RECOMMENDATION")
print("=" * 78)

print()

v2_24h = diagnostic_df[
    diagnostic_df["horizon"] == "24h"
].iloc[0]

v2_48h = diagnostic_df[
    diagnostic_df["horizon"] == "48h"
].iloc[0]

v2_3d = diagnostic_df[
    diagnostic_df["horizon"] == "3d"
].iloc[0]

v2_7d = diagnostic_df[
    diagnostic_df["horizon"] == "7d"
].iloc[0]


print(
    "24h : "
    "Suitable for the prototype."
)

print(
    "48h : "
    "Suitable for the prototype."
)

print(
    "3d  : "
    "Useful forecasting information."
)

print(
    "7d  : "
    "Use cautiously as a long-range estimate."
)

print()

if (
    v2_24h["r2"] > v1_data["24h"]["r2"]
    and
    v2_48h["r2"] > v1_data["48h"]["r2"]
    and
    v2_3d["r2"] > v1_data["3d"]["r2"]
):

    print(
        "RESULT:"
    )

    print(
        "Model 2 V2 improves the main short/"
        "medium-term forecasting horizons."
    )

else:

    print(
        "RESULT:"
    )

    print(
        "Model 2 V2 does not consistently improve"
        " the previous model."
    )

print()
print(
    "No model or dataset was modified by this diagnostic."
)

print()
print("=" * 78)
print("                 DIAGNOSTIC COMPLETED")
print("=" * 78)
print()
