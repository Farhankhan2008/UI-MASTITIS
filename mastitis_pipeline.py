
import os
import warnings
import numpy as np
import pandas as pd

# ============================================================
# SUPPRESS NON-CRITICAL WARNINGS
# ============================================================

warnings.filterwarnings("ignore")

# ============================================================
# IMPORT EXISTING MODEL 1 -> MODEL 2 INFERENCE MODULE
# ============================================================

import model1_to_model2_inference as inference

from model1_to_model2_inference import (
    predict_model1_risk,
    predict_model2
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

DATASET_FILE = os.path.join(
    BASE_DIR,
    "synthetic_mastitis_model1_dataset_v2.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "herd_predictions.csv"
)

HIGH_RISK_THRESHOLD = 50.0

WARNING_THRESHOLD_7D = 20.0
WARNING_THRESHOLD_14D = 30.0


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("MASTITIS AI COMPLETE INFERENCE PIPELINE")
print("=" * 80)

print()
print("Pipeline:")
print("  Dataset")
print("      |")
print("      v")
print("  Model 1")
print("      |")
print("      v")
print("  15-Day Risk History")
print("      |")
print("      v")
print("  Model 2")
print("      |")
print("      v")
print("  Decision Engine")
print("      |")
print("      v")
print("  Herd Risk Analysis")
print("      |")
print("      v")
print("  herd_predictions.csv")
print()


# ============================================================
# PERFORMANCE FIX
#
# Your Random Forest models were using parallel inference.
#
# With the current Python / scikit-learn / joblib environment,
# this caused repeated warnings and very slow inference.
#
# Setting n_jobs = 1:
#
# - DOES NOT retrain the model
# - DOES NOT change model parameters
# - DOES NOT change predictions
# - only changes how prediction is executed
# ============================================================

print("Configuring inference performance...")

# ------------------------------------------------------------
# Model 1
# ------------------------------------------------------------

if hasattr(inference.model1, "n_jobs"):
    inference.model1.n_jobs = 1


# ------------------------------------------------------------
# Model 2
# ------------------------------------------------------------

for horizon in [
    "24h",
    "48h",
    "3d",
    "7d",
    "14d"
]:

    model_object = inference.model2[horizon]["model"]

    if hasattr(model_object, "n_jobs"):
        model_object.n_jobs = 1


print("Inference configured for single-core prediction.")
print()


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(
    risk_percent
):

    risk_percent = float(
        risk_percent
    )

    if risk_percent <= 10.0:

        return "No Risk"

    elif risk_percent <= 25.0:

        return "Low Risk"

    elif risk_percent <= 50.0:

        return "Moderate Risk"

    else:

        return "High Risk"


# ============================================================
# ALERT LEVEL
# ============================================================

def get_alert_level(
    current_risk,
    probability_7d,
    probability_14d
):

    current_risk = float(
        current_risk
    )

    probability_7d = float(
        probability_7d
    )

    probability_14d = float(
        probability_14d
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if (
        current_risk >= HIGH_RISK_THRESHOLD
        and
        probability_7d >= WARNING_THRESHOLD_7D
        and
        probability_14d >= WARNING_THRESHOLD_14D
    ):

        return "CRITICAL"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if (
        current_risk >= HIGH_RISK_THRESHOLD
        or
        probability_7d >= WARNING_THRESHOLD_7D
        or
        probability_14d >= WARNING_THRESHOLD_14D
    ):

        return "HIGH"

    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if current_risk > 25.0:

        return "MODERATE"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if current_risk > 10.0:

        return "LOW"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "NORMAL"


# ============================================================
# RECOMMENDATION ENGINE
#
# RULE-BASED
# NOT AN ADDITIONAL ML MODEL
# ============================================================

def generate_recommendations(
    current_risk,
    probability_7d,
    probability_14d,
    milk_conductivity,
    cow_activity,
    risk_history
):

    recommendations = []

    # --------------------------------------------------------
    # CURRENT RISK
    # --------------------------------------------------------

    if current_risk > 50.0:

        recommendations.append(
            "Prioritize udder-health inspection for this cow."
        )

    elif current_risk > 25.0:

        recommendations.append(
            "Increase monitoring of this cow."
        )

    # --------------------------------------------------------
    # 7-DAY WARNING
    # --------------------------------------------------------

    if probability_7d >= WARNING_THRESHOLD_7D:

        recommendations.append(
            "Elevated 7-day high-risk probability. "
            "Increase monitoring frequency."
        )

    # --------------------------------------------------------
    # 14-DAY WARNING
    # --------------------------------------------------------

    if probability_14d >= WARNING_THRESHOLD_14D:

        recommendations.append(
            "Elevated 14-day high-risk probability. "
            "Consider an early udder-health assessment."
        )

    # --------------------------------------------------------
    # MILK CONDUCTIVITY
    # --------------------------------------------------------

    if milk_conductivity >= 5.0:

        recommendations.append(
            "Milk conductivity is elevated. "
            "Verify milk-quality indicators and consider SCC testing."
        )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    if cow_activity < 50:

        recommendations.append(
            "Cow activity is reduced. "
            "Monitor for behavioral or health changes."
        )

    # --------------------------------------------------------
    # RISK TREND
    # --------------------------------------------------------

    if len(risk_history) >= 4:

        today = risk_history[0]

        risk_3_days_ago = risk_history[3]

        if (
            today - risk_3_days_ago
        ) > 10:

            recommendations.append(
                "Mastitis risk is increasing compared with recent history."
            )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "Continue routine monitoring."
        )

    return recommendations


# ============================================================
# FIND DATASET COLUMNS
# ============================================================

def find_column(
    dataframe,
    possible_names
):

    for name in possible_names:

        if name in dataframe.columns:

            return name

    return None


# ============================================================
# PREDICT ONE COW
# ============================================================

def predict_one_cow(
    cow_id,
    cow_data,
    cow_id_column,
    date_column
):

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    cow_data = cow_data.copy()

    cow_data[date_column] = pd.to_datetime(
        cow_data[date_column],
        errors="coerce"
    )

    cow_data = cow_data.sort_values(
        date_column
    )

    # --------------------------------------------------------
    # REQUIRE 15 DAYS
    # --------------------------------------------------------

    if len(cow_data) < 15:

        raise ValueError(
            f"{cow_id} has only {len(cow_data)} records. "
            f"Model 2 requires at least 15 records."
        )

    # --------------------------------------------------------
    # TAKE MOST RECENT 15 DAYS
    #
    # Chronological order:
    #
    # oldest
    # ...
    # newest / today
    # --------------------------------------------------------

    recent_data = cow_data.tail(
        15
    ).copy()

    # --------------------------------------------------------
    # GENERATE MODEL 1 RISK FOR EACH OF THE 15 DAYS
    # --------------------------------------------------------

    chronological_risks = []

    for _, row in recent_data.iterrows():

        risk = predict_model1_risk(

            row["milk_yield_liters"],

            row["milk_conductivity_ms_cm"],

            row["cow_activity"],

            row["environment_temperature_c"],

            row["milk_temperature_c"],

            row["humidity_percent"],

            row["previous_mastitis"],

            row["days_since_last_mastitis"]
        )

        chronological_risks.append(
            risk
        )

    # --------------------------------------------------------
    # MODEL 2 REQUIRES:
    #
    # [today,
    #  1d ago,
    #  2d ago,
    #  ...
    #  14d ago]
    #
    # Therefore reverse chronological risks.
    # --------------------------------------------------------

    risk_history = list(
        reversed(
            chronological_risks
        )
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(risk_history) != 15:

        raise ValueError(
            f"{cow_id}: Model 2 risk history length "
            f"is {len(risk_history)}, expected 15."
        )

    # --------------------------------------------------------
    # CURRENT / TODAY'S RECORD
    # --------------------------------------------------------

    today_row = recent_data.iloc[-1]

    # --------------------------------------------------------
    # CURRENT MODEL 1 RISK
    # --------------------------------------------------------

    current_risk = float(
        risk_history[0]
    )

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    model2_predictions = predict_model2(
        risk_history
    )

    # --------------------------------------------------------
    # MODEL 2 VALUES
    # --------------------------------------------------------

    predicted_24h = float(
        model2_predictions["24h"]
    )

    predicted_48h = float(
        model2_predictions["48h"]
    )

    predicted_3d = float(
        model2_predictions["3d"]
    )

    probability_7d = float(
        model2_predictions["7d_probability"]
    )

    probability_14d = float(
        model2_predictions["14d_probability"]
    )

    warning_7d = bool(
        model2_predictions["7d_warning"]
    )

    warning_14d = bool(
        model2_predictions["14d_warning"]
    )

    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    risk_category = get_risk_category(
        current_risk
    )

    # --------------------------------------------------------
    # ALERT LEVEL
    # --------------------------------------------------------

    alert_level = get_alert_level(

        current_risk,

        probability_7d,

        probability_14d
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = generate_recommendations(

        current_risk,

        probability_7d,

        probability_14d,

        float(
            today_row["milk_conductivity_ms_cm"]
        ),

        float(
            today_row["cow_activity"]
        ),

        risk_history
    )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = {

        "cow_id":
            cow_id,

        "date":
            today_row[date_column],

        # ----------------------------------------------------
        # CURRENT SENSOR VALUES
        # ----------------------------------------------------

        "milk_yield_liters":
            float(
                today_row["milk_yield_liters"]
            ),

        "milk_conductivity_ms_cm":
            float(
                today_row["milk_conductivity_ms_cm"]
            ),

        "cow_activity":
            float(
                today_row["cow_activity"]
            ),

        "environment_temperature_c":
            float(
                today_row["environment_temperature_c"]
            ),

        "milk_temperature_c":
            float(
                today_row["milk_temperature_c"]
            ),

        "humidity_percent":
            float(
                today_row["humidity_percent"]
            ),

        "previous_mastitis":
            int(
                today_row["previous_mastitis"]
            ),

        "days_since_last_mastitis":
            float(
                today_row["days_since_last_mastitis"]
            ),

        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        "current_risk":
            round(
                current_risk,
                2
            ),

        "risk_category":
            risk_category,

        # ----------------------------------------------------
        # MODEL 2 SHORT-TERM
        # ----------------------------------------------------

        "24h_predicted_risk":
            round(
                predicted_24h,
                2
            ),

        "48h_predicted_risk":
            round(
                predicted_48h,
                2
            ),

        "3d_predicted_risk":
            round(
                predicted_3d,
                2
            ),

        # ----------------------------------------------------
        # MODEL 2 LONG-TERM
        # ----------------------------------------------------

        "7d_probability":
            round(
                probability_7d,
                2
            ),

        "7d_warning":
            warning_7d,

        "14d_probability":
            round(
                probability_14d,
                2
            ),

        "14d_warning":
            warning_14d,

        # ----------------------------------------------------
        # DECISION ENGINE
        # ----------------------------------------------------

        "alert_level":
            alert_level,

        "recommendations":
            " | ".join(
                recommendations
            )
    }

    return result


# ============================================================
# HERD STATUS
# ============================================================

def calculate_herd_status(
    dataframe
):

    total_cows = len(
        dataframe
    )

    if total_cows == 0:

        return "NORMAL"

    # --------------------------------------------------------
    # HIGH-RISK RATE
    # --------------------------------------------------------

    high_risk_count = int(
        (
            dataframe["current_risk"]
            >= HIGH_RISK_THRESHOLD
        ).sum()
    )

    high_risk_rate = (
        high_risk_count
        /
        total_cows
        *
        100
    )

    # --------------------------------------------------------
    # AVERAGE RISK
    # --------------------------------------------------------

    average_risk = float(
        dataframe["current_risk"].mean()
    )

    # --------------------------------------------------------
    # WARNING RATES
    # --------------------------------------------------------

    warning_7d_rate = (
        dataframe["7d_warning"]
        .astype(bool)
        .mean()
        *
        100
    )

    warning_14d_rate = (
        dataframe["14d_warning"]
        .astype(bool)
        .mean()
        *
        100
    )

    # --------------------------------------------------------
    # HERD STATUS
    # --------------------------------------------------------

    if (
        high_risk_rate >= 15
        or
        average_risk >= 35
        or
        warning_7d_rate >= 25
        or
        warning_14d_rate >= 35
    ):

        return "HIGH"

    elif (
        high_risk_rate >= 5
        or
        average_risk >= 20
        or
        warning_7d_rate >= 15
        or
        warning_14d_rate >= 25
    ):

        return "MODERATE"

    elif (
        high_risk_rate > 0
        or
        warning_7d_rate > 0
        or
        warning_14d_rate > 0
    ):

        return "LOW"

    else:

        return "NORMAL"


# ============================================================
# HERD RECOMMENDATIONS
# ============================================================

def generate_herd_recommendations(
    dataframe
):

    recommendations = []

    total_cows = len(
        dataframe
    )

    if total_cows == 0:

        return [
            "No cow predictions available."
        ]

    # --------------------------------------------------------
    # HIGH-RISK COWS
    # --------------------------------------------------------

    high_risk_count = int(
        (
            dataframe["current_risk"]
            >= HIGH_RISK_THRESHOLD
        ).sum()
    )

    high_risk_rate = (
        high_risk_count
        /
        total_cows
        *
        100
    )

    # --------------------------------------------------------
    # WARNING COWS
    # --------------------------------------------------------

    warning_7d_count = int(
        dataframe["7d_warning"]
        .astype(bool)
        .sum()
    )

    warning_14d_count = int(
        dataframe["14d_warning"]
        .astype(bool)
        .sum()
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    if high_risk_count > 0:

        recommendations.append(
            f"{high_risk_count} cow(s) are currently "
            "in the high-risk category. Prioritize "
            "individual inspection."
        )

    if high_risk_rate >= 15:

        recommendations.append(
            "A significant proportion of the herd is "
            "high risk. Review milking hygiene, equipment, "
            "housing and environmental conditions."
        )

    if warning_7d_count > 0:

        recommendations.append(
            f"{warning_7d_count} cow(s) have elevated "
            "7-day high-risk probability. Increase "
            "monitoring frequency."
        )

    if warning_14d_count > 0:

        recommendations.append(
            f"{warning_14d_count} cow(s) have elevated "
            "14-day high-risk probability. Consider "
            "early herd-level health assessment."
        )

    # --------------------------------------------------------
    # MILK CONDUCTIVITY
    # --------------------------------------------------------

    elevated_conductivity = int(
        (
            dataframe[
                "milk_conductivity_ms_cm"
            ]
            >= 5.0
        ).sum()
    )

    if elevated_conductivity > 0:

        recommendations.append(
            f"{elevated_conductivity} cow(s) show "
            "elevated milk conductivity. Verify milk "
            "quality and consider SCC testing."
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "Continue routine herd monitoring."
        )

    return recommendations


# ============================================================
# MAIN HERD PIPELINE
# ============================================================

def run_herd_pipeline():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    if not os.path.exists(
        DATASET_FILE
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    print()
    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Total rows: {len(dataframe):,}"
    )

    # ========================================================
    # FIND COW ID COLUMN
    # ========================================================

    cow_id_column = find_column(

        dataframe,

        [
            "cow_id",
            "Cow_ID",
            "cowID",
            "animal_id",
            "Animal_ID",
            "rfid_id",
            "RFID_ID"
        ]
    )

    if cow_id_column is None:

        raise ValueError(
            "Could not find cow ID column."
        )

    # ========================================================
    # FIND DATE COLUMN
    # ========================================================

    date_column = find_column(

        dataframe,

        [
            "date",
            "Date",
            "record_date",
            "measurement_date"
        ]
    )

    if date_column is None:

        raise ValueError(
            "Could not find date column."
        )

    print(
        f"Cow ID column : {cow_id_column}"
    )

    print(
        f"Date column   : {date_column}"
    )

    # ========================================================
    # CHECK REQUIRED MODEL 1 FEATURES
    # ========================================================

    required_features = [

        "milk_yield_liters",

        "milk_conductivity_ms_cm",

        "cow_activity",

        "environment_temperature_c",

        "milk_temperature_c",

        "humidity_percent",

        "previous_mastitis",

        "days_since_last_mastitis"
    ]

    missing_features = [

        feature

        for feature in required_features

        if feature not in dataframe.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required Model 1 features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    dataframe[date_column] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=[
            cow_id_column,
            date_column
        ]
    )

    unique_cows = (
        dataframe[cow_id_column]
        .unique()
    )

    total_cows = len(
        unique_cows
    )

    print()
    print(
        f"Unique cows found: {total_cows}"
    )

    print()

    # ========================================================
    # PROCESS EVERY COW
    # ========================================================

    print("=" * 80)
    print("RUNNING COW-LEVEL AI PIPELINE")
    print("=" * 80)

    results = []

    failed_cows = []

    for index, cow_id in enumerate(
        unique_cows,
        start=1
    ):

        cow_data = dataframe[
            dataframe[cow_id_column]
            ==
            cow_id
        ].copy()

        try:

            result = predict_one_cow(

                cow_id,

                cow_data,

                cow_id_column,

                date_column
            )

            results.append(
                result
            )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                index <= 10
                or
                index % 25 == 0
                or
                index == total_cows
            ):

                print(
                    f"[{index:>3}/{total_cows}] "
                    f"{cow_id} -> "
                    f"{result['current_risk']:.2f}% "
                    f"-> "
                    f"{result['risk_category']}"
                )

        except Exception as error:

            failed_cows.append(
                (
                    cow_id,
                    str(error)
                )
            )

            print(
                f"[{index:>3}/{total_cows}] "
                f"{cow_id} -> FAILED: {error}"
            )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(results) == 0:

        raise RuntimeError(
            "No cow predictions were generated."
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE COW-LEVEL PREDICTIONS
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # HERD STATISTICS
    # ========================================================

    total_predicted = len(
        results_df
    )

    no_risk_count = int(
        (
            results_df["risk_category"]
            ==
            "No Risk"
        ).sum()
    )

    low_risk_count = int(
        (
            results_df["risk_category"]
            ==
            "Low Risk"
        ).sum()
    )

    moderate_risk_count = int(
        (
            results_df["risk_category"]
            ==
            "Moderate Risk"
        ).sum()
    )

    high_risk_count = int(
        (
            results_df["risk_category"]
            ==
            "High Risk"
        ).sum()
    )

    # ========================================================
    # HERD RISK METRICS
    # ========================================================

    average_risk = float(
        results_df[
            "current_risk"
        ].mean()
    )

    median_risk = float(
        results_df[
            "current_risk"
        ].median()
    )

    minimum_risk = float(
        results_df[
            "current_risk"
        ].min()
    )

    maximum_risk = float(
        results_df[
            "current_risk"
        ].max()
    )

    high_risk_rate = (
        high_risk_count
        /
        total_predicted
        *
        100
    )

    warning_7d_count = int(
        results_df[
            "7d_warning"
        ]
        .astype(bool)
        .sum()
    )

    warning_14d_count = int(
        results_df[
            "14d_warning"
        ]
        .astype(bool)
        .sum()
    )

    warning_7d_rate = (
        warning_7d_count
        /
        total_predicted
        *
        100
    )

    warning_14d_rate = (
        warning_14d_count
        /
        total_predicted
        *
        100
    )

    average_7d_probability = float(
        results_df[
            "7d_probability"
        ].mean()
    )

    average_14d_probability = float(
        results_df[
            "14d_probability"
        ].mean()
    )

    # ========================================================
    # HERD STATUS
    # ========================================================

    herd_status = calculate_herd_status(
        results_df
    )

    # ========================================================
    # HERD RECOMMENDATIONS
    # ========================================================

    herd_recommendations = (
        generate_herd_recommendations(
            results_df
        )
    )

    # ========================================================
    # PRIORITY COWS
    # ========================================================

    priority_cows = results_df.sort_values(

        by=[
            "alert_level",
            "current_risk",
            "14d_probability"
        ],

        ascending=[
            False,
            False,
            False
        ]
    )

    # Better numerical priority sorting
    alert_priority = {

        "CRITICAL": 4,

        "HIGH": 3,

        "MODERATE": 2,

        "LOW": 1,

        "NORMAL": 0
    }

    priority_cows = (
        results_df
        .copy()
    )

    priority_cows[
        "_alert_priority"
    ] = priority_cows[
        "alert_level"
    ].map(
        alert_priority
    )

    priority_cows = priority_cows.sort_values(

        by=[
            "_alert_priority",
            "current_risk",
            "14d_probability"
        ],

        ascending=[
            False,
            False,
            False
        ]
    )

    # ========================================================
    # PRINT FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("HERD-LEVEL MASTITIS RISK REPORT")
    print("=" * 80)

    print()

    print(
        f"Total cows processed     : "
        f"{total_predicted}"
    )

    print(
        f"No Risk                  : "
        f"{no_risk_count}"
    )

    print(
        f"Low Risk                 : "
        f"{low_risk_count}"
    )

    print(
        f"Moderate Risk            : "
        f"{moderate_risk_count}"
    )

    print(
        f"High Risk                : "
        f"{high_risk_count}"
    )

    print()

    print(
        f"Average herd risk        : "
        f"{average_risk:.2f}%"
    )

    print(
        f"Median herd risk         : "
        f"{median_risk:.2f}%"
    )

    print(
        f"Minimum risk             : "
        f"{minimum_risk:.2f}%"
    )

    print(
        f"Maximum risk             : "
        f"{maximum_risk:.2f}%"
    )

    print()

    print(
        f"High-risk cows           : "
        f"{high_risk_rate:.2f}%"
    )

    print(
        f"7-day warning cows       : "
        f"{warning_7d_count}"
    )

    print(
        f"7-day warning rate       : "
        f"{warning_7d_rate:.2f}%"
    )

    print(
        f"14-day warning cows      : "
        f"{warning_14d_count}"
    )

    print(
        f"14-day warning rate      : "
        f"{warning_14d_rate:.2f}%"
    )

    print()

    print(
        f"Average 7-day probability  : "
        f"{average_7d_probability:.2f}%"
    )

    print(
        f"Average 14-day probability : "
        f"{average_14d_probability:.2f}%"
    )

    print()

    print(
        f"Herd Status              : "
        f"{herd_status}"
    )

    # ========================================================
    # PRIORITY COWS
    # ========================================================

    print()
    print("=" * 80)
    print("TOP 10 PRIORITY COWS")
    print("=" * 80)

    print()

    top_priority = priority_cows.head(
        10
    )

    for _, row in top_priority.iterrows():

        print(
            f"{str(row['cow_id']):<12}"
            f" Risk: {row['current_risk']:>6.2f}%"
            f" | Category: {row['risk_category']:<15}"
            f" | 7d: {row['7d_probability']:>6.2f}%"
            f" | 14d: {row['14d_probability']:>6.2f}%"
            f" | Alert: {row['alert_level']}"
        )

    # ========================================================
    # HERD RECOMMENDATIONS
    # ========================================================

    print()
    print("=" * 80)
    print("HERD RECOMMENDATIONS")
    print("=" * 80)

    print()

    for recommendation in herd_recommendations:

        print(
            f"- {recommendation}"
        )

    # ========================================================
    # FAILED COWS
    # ========================================================

    if failed_cows:

        print()
        print("=" * 80)
        print("FAILED COW PREDICTIONS")
        print("=" * 80)

        print()

        for cow_id, error in failed_cows:

            print(
                f"- {cow_id}: {error}"
            )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Prediction file saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        f"Successfully processed: "
        f"{total_predicted}/{total_cows} cows"
    )

    if failed_cows:

        print(
            f"Failed: {len(failed_cows)} cow(s)"
        )

    else:

        print(
            "Failed: 0 cows"
        )

    print()

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_herd_pipeline()

