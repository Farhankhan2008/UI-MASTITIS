import numpy as np
import pandas as pd


# ============================================================
# HERD RISK ENGINE
#
# Purpose:
# Aggregate individual cow predictions into a herd-level
# mastitis risk assessment.
#
# This does NOT modify or retrain Model 1 or Model 2.
#
# Expected individual-cow data:
#
# cow_id
# current_risk
# risk_category
# 7d_probability
# 14d_probability
# 7d_warning
# 14d_warning
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_RISK_THRESHOLD = 50.0

NO_RISK_MAX = 10.0
LOW_RISK_MAX = 25.0
MODERATE_RISK_MAX = 50.0


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(risk_percent):

    risk_percent = float(risk_percent)

    if risk_percent <= NO_RISK_MAX:
        return "No Risk"

    elif risk_percent <= LOW_RISK_MAX:
        return "Low Risk"

    elif risk_percent <= MODERATE_RISK_MAX:
        return "Moderate Risk"

    else:
        return "High Risk"


# ============================================================
# HERD STATUS
#
# Prototype aggregation logic.
#
# The status considers:
#   1. Percentage of high-risk cows
#   2. Average herd risk
#   3. Percentage of cows with future warnings
#
# These are prototype decision thresholds and should not be
# presented as clinically validated thresholds.
# ============================================================

def determine_herd_status(
    average_risk,
    high_risk_percentage,
    warning_7d_percentage,
    warning_14d_percentage
):

    average_risk = float(average_risk)
    high_risk_percentage = float(high_risk_percentage)
    warning_7d_percentage = float(warning_7d_percentage)
    warning_14d_percentage = float(warning_14d_percentage)

    # --------------------------------------------------------
    # HIGH HERD RISK
    # --------------------------------------------------------

    if (
        high_risk_percentage >= 15
        or average_risk >= 35
        or warning_7d_percentage >= 25
        or warning_14d_percentage >= 35
    ):

        return "HIGH"

    # --------------------------------------------------------
    # MODERATE HERD RISK
    # --------------------------------------------------------

    if (
        high_risk_percentage >= 5
        or average_risk >= 20
        or warning_7d_percentage >= 15
        or warning_14d_percentage >= 25
    ):

        return "MODERATE"

    # --------------------------------------------------------
    # LOW HERD RISK
    # --------------------------------------------------------

    if (
        high_risk_percentage > 0
        or average_risk > 10
        or warning_7d_percentage > 0
        or warning_14d_percentage > 0
    ):

        return "LOW"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "NORMAL"


# ============================================================
# HERD RECOMMENDATIONS
# ============================================================

def generate_herd_recommendations(
    herd_status,
    high_risk_count,
    warning_7d_count,
    warning_14d_count,
    average_risk
):

    recommendations = []

    # --------------------------------------------------------
    # HIGH-RISK COWS
    # --------------------------------------------------------

    if high_risk_count > 0:

        recommendations.append(
            f"Prioritize {high_risk_count} high-risk "
            f"cow(s) for individual health inspection."
        )

    # --------------------------------------------------------
    # 7-DAY WARNINGS
    # --------------------------------------------------------

    if warning_7d_count > 0:

        recommendations.append(
            f"Increase monitoring for {warning_7d_count} "
            f"cow(s) with elevated 7-day risk."
        )

    # --------------------------------------------------------
    # 14-DAY WARNINGS
    # --------------------------------------------------------

    if warning_14d_count > 0:

        recommendations.append(
            f"Review {warning_14d_count} cow(s) with "
            f"elevated 14-day high-risk probability."
        )

    # --------------------------------------------------------
    # HERD STATUS
    # --------------------------------------------------------

    if herd_status == "HIGH":

        recommendations.append(
            "Multiple herd-level risk indicators are elevated. "
            "Review herd health, milking hygiene, equipment, "
            "and environmental conditions."
        )

    elif herd_status == "MODERATE":

        recommendations.append(
            "Herd risk is moderately elevated. "
            "Continue monitoring and investigate recurring "
            "risk patterns."
        )

    elif herd_status == "LOW":

        recommendations.append(
            "Herd risk is relatively low. "
            "Continue routine monitoring."
        )

    else:

        recommendations.append(
            "Herd risk is currently within the normal prototype range."
        )

    # --------------------------------------------------------
    # AVERAGE RISK
    # --------------------------------------------------------

    if average_risk >= 30:

        recommendations.append(
            "Herd average risk is elevated. "
            "Review the highest-risk animals first."
        )

    return recommendations


# ============================================================
# BUILD HERD DATAFRAME
#
# Converts individual cow prediction dictionaries into a
# structured DataFrame.
# ============================================================

def build_herd_dataframe(cow_predictions):

    if not cow_predictions:

        raise ValueError(
            "No cow predictions were provided."
        )

    rows = []

    for cow in cow_predictions:

        if "cow_id" not in cow:

            raise ValueError(
                "Each cow prediction must contain 'cow_id'."
            )

        if "current_risk" not in cow:

            raise ValueError(
                "Each cow prediction must contain 'current_risk'."
            )

        current_risk = float(
            cow["current_risk"]
        )

        # ----------------------------------------------------
        # Automatically calculate category if not provided.
        # ----------------------------------------------------

        category = cow.get(
            "risk_category",
            get_risk_category(current_risk)
        )

        # ----------------------------------------------------
        # Future probabilities
        # ----------------------------------------------------

        probability_7d = float(
            cow.get(
                "7d_probability",
                0.0
            )
        )

        probability_14d = float(
            cow.get(
                "14d_probability",
                0.0
            )
        )

        # ----------------------------------------------------
        # Warning flags
        #
        # If warning values already exist, use them.
        # Otherwise calculate from probability.
        # ----------------------------------------------------

        warning_7d = cow.get(
            "7d_warning",
            probability_7d >= 20.0
        )

        warning_14d = cow.get(
            "14d_warning",
            probability_14d >= 30.0
        )

        rows.append({

            "cow_id": cow["cow_id"],

            "current_risk": current_risk,

            "risk_category": category,

            "7d_probability": probability_7d,

            "14d_probability": probability_14d,

            "7d_warning": bool(
                warning_7d
            ),

            "14d_warning": bool(
                warning_14d
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN HERD ANALYSIS
# ============================================================

def analyze_herd(cow_predictions):

    # --------------------------------------------------------
    # BUILD DATAFRAME
    # --------------------------------------------------------

    df = build_herd_dataframe(
        cow_predictions
    )

    total_cows = len(df)

    if total_cows == 0:

        raise ValueError(
            "Herd must contain at least one cow."
        )

    # --------------------------------------------------------
    # RISK CATEGORY COUNTS
    # --------------------------------------------------------

    no_risk_count = int(
        (df["risk_category"] == "No Risk").sum()
    )

    low_risk_count = int(
        (df["risk_category"] == "Low Risk").sum()
    )

    moderate_risk_count = int(
        (df["risk_category"] == "Moderate Risk").sum()
    )

    high_risk_count = int(
        (df["risk_category"] == "High Risk").sum()
    )

    # --------------------------------------------------------
    # BASIC HERD METRICS
    # --------------------------------------------------------

    average_risk = float(
        df["current_risk"].mean()
    )

    median_risk = float(
        df["current_risk"].median()
    )

    maximum_risk = float(
        df["current_risk"].max()
    )

    minimum_risk = float(
        df["current_risk"].min()
    )

    # --------------------------------------------------------
    # HIGH-RISK PERCENTAGE
    # --------------------------------------------------------

    high_risk_percentage = (
        high_risk_count
        /
        total_cows
        *
        100.0
    )

    # --------------------------------------------------------
    # FUTURE WARNING COUNTS
    # --------------------------------------------------------

    warning_7d_count = int(
        df["7d_warning"].sum()
    )

    warning_14d_count = int(
        df["14d_warning"].sum()
    )

    # --------------------------------------------------------
    # FUTURE WARNING PERCENTAGES
    # --------------------------------------------------------

    warning_7d_percentage = (
        warning_7d_count
        /
        total_cows
        *
        100.0
    )

    warning_14d_percentage = (
        warning_14d_count
        /
        total_cows
        *
        100.0
    )

    # --------------------------------------------------------
    # AVERAGE FUTURE PROBABILITIES
    # --------------------------------------------------------

    average_7d_probability = float(
        df["7d_probability"].mean()
    )

    average_14d_probability = float(
        df["14d_probability"].mean()
    )

    # --------------------------------------------------------
    # HERD STATUS
    # --------------------------------------------------------

    herd_status = determine_herd_status(
        average_risk=average_risk,
        high_risk_percentage=high_risk_percentage,
        warning_7d_percentage=warning_7d_percentage,
        warning_14d_percentage=warning_14d_percentage
    )

    # --------------------------------------------------------
    # PRIORITY COWS
    #
    # Highest current risk first.
    # --------------------------------------------------------

    priority_df = df.sort_values(
        by=[
            "current_risk",
            "7d_probability",
            "14d_probability"
        ],
        ascending=False
    )

    priority_cows = (
        priority_df
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # HIGH-RISK COW LIST
    # --------------------------------------------------------

    high_risk_cows = (
        df[
            df["current_risk"]
            >= HIGH_RISK_THRESHOLD
        ]
        .sort_values(
            by="current_risk",
            ascending=False
        )
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # HERD RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = generate_herd_recommendations(
        herd_status=herd_status,
        high_risk_count=high_risk_count,
        warning_7d_count=warning_7d_count,
        warning_14d_count=warning_14d_count,
        average_risk=average_risk
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "total_cows": total_cows,

        "no_risk_count": no_risk_count,

        "low_risk_count": low_risk_count,

        "moderate_risk_count": moderate_risk_count,

        "high_risk_count": high_risk_count,

        "average_risk": round(
            average_risk,
            2
        ),

        "median_risk": round(
            median_risk,
            2
        ),

        "minimum_risk": round(
            minimum_risk,
            2
        ),

        "maximum_risk": round(
            maximum_risk,
            2
        ),

        "high_risk_percentage": round(
            high_risk_percentage,
            2
        ),

        "warning_7d_count": warning_7d_count,

        "warning_7d_percentage": round(
            warning_7d_percentage,
            2
        ),

        "warning_14d_count": warning_14d_count,

        "warning_14d_percentage": round(
            warning_14d_percentage,
            2
        ),

        "average_7d_probability": round(
            average_7d_probability,
            2
        ),

        "average_14d_probability": round(
            average_14d_probability,
            2
        ),

        "herd_status": herd_status,

        "priority_cows": priority_cows,

        "high_risk_cows": high_risk_cows,

        "recommendations": recommendations,

        "dataframe": df
    }

    return result


# ============================================================
# PRINT HERD REPORT
# ============================================================

def print_herd_report(result):

    print()
    print("=" * 80)
    print("HERD RISK ASSESSMENT")
    print("=" * 80)

    print()

    print(
        f"Total cows              : "
        f"{result['total_cows']}"
    )

    print()

    print("RISK DISTRIBUTION")
    print("-" * 50)

    print(
        f"No Risk                 : "
        f"{result['no_risk_count']}"
    )

    print(
        f"Low Risk                : "
        f"{result['low_risk_count']}"
    )

    print(
        f"Moderate Risk           : "
        f"{result['moderate_risk_count']}"
    )

    print(
        f"High Risk               : "
        f"{result['high_risk_count']}"
    )

    print()

    print("HERD METRICS")
    print("-" * 50)

    print(
        f"Average herd risk      : "
        f"{result['average_risk']:.2f}%"
    )

    print(
        f"Median herd risk       : "
        f"{result['median_risk']:.2f}%"
    )

    print(
        f"Minimum risk           : "
        f"{result['minimum_risk']:.2f}%"
    )

    print(
        f"Maximum risk           : "
        f"{result['maximum_risk']:.2f}%"
    )

    print(
        f"High-risk cows         : "
        f"{result['high_risk_percentage']:.2f}%"
    )

    print()

    print("FORECAST WARNINGS")
    print("-" * 50)

    print(
        f"7-day warning cows     : "
        f"{result['warning_7d_count']}"
    )

    print(
        f"7-day warning rate     : "
        f"{result['warning_7d_percentage']:.2f}%"
    )

    print(
        f"14-day warning cows    : "
        f"{result['warning_14d_count']}"
    )

    print(
        f"14-day warning rate    : "
        f"{result['warning_14d_percentage']:.2f}%"
    )

    print()

    print(
        f"Average 7-day risk     : "
        f"{result['average_7d_probability']:.2f}%"
    )

    print(
        f"Average 14-day risk    : "
        f"{result['average_14d_probability']:.2f}%"
    )

    print()

    print("HERD STATUS")
    print("-" * 50)

    print(
        f"Status                  : "
        f"{result['herd_status']}"
    )

    print()

    print("TOP PRIORITY COWS")
    print("-" * 80)

    for index, cow in enumerate(
        result["priority_cows"],
        start=1
    ):

        print(
            f"{index:2d}. "
            f"{cow['cow_id']:10s} "
            f"{cow['current_risk']:6.2f}% "
            f"{cow['risk_category']:15s} "
            f"7d={cow['7d_probability']:6.2f}% "
            f"14d={cow['14d_probability']:6.2f}%"
        )

    print()

    print("HIGH-RISK COWS")
    print("-" * 80)

    if len(result["high_risk_cows"]) == 0:

        print(
            "No cows currently exceed the "
            f"{HIGH_RISK_THRESHOLD:.0f}% high-risk threshold."
        )

    else:

        for cow in result["high_risk_cows"]:

            print(
                f"{cow['cow_id']:10s} "
                f"{cow['current_risk']:6.2f}% "
                f"{cow['risk_category']}"
            )

    print()

    print("HERD RECOMMENDATIONS")
    print("-" * 80)

    for recommendation in result["recommendations"]:

        print(
            f"- {recommendation}"
        )

    print()

    print("=" * 80)
    print("HERD RISK ASSESSMENT COMPLETE")
    print("=" * 80)


# ============================================================
# DEMO TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # DEMO COW PREDICTIONS
    #
    # These simulate the output of your existing
    # Model 1 + Model 2 + Decision Engine.
    #
    # Later, these values will come automatically from
    # real cow/RFID/sensor data.
    # --------------------------------------------------------

    demo_cow_predictions = [

        {
            "cow_id": "COW_001",
            "current_risk": 41.27,
            "risk_category": "Moderate Risk",
            "7d_probability": 29.86,
            "14d_probability": 46.68,
            "7d_warning": True,
            "14d_warning": True
        },

        {
            "cow_id": "COW_002",
            "current_risk": 12.43,
            "risk_category": "Low Risk",
            "7d_probability": 8.20,
            "14d_probability": 15.40,
            "7d_warning": False,
            "14d_warning": False
        },

        {
            "cow_id": "COW_003",
            "current_risk": 76.81,
            "risk_category": "High Risk",
            "7d_probability": 72.40,
            "14d_probability": 84.10,
            "7d_warning": True,
            "14d_warning": True
        },

        {
            "cow_id": "COW_004",
            "current_risk": 8.21,
            "risk_category": "No Risk",
            "7d_probability": 4.20,
            "14d_probability": 8.10,
            "7d_warning": False,
            "14d_warning": False
        },

        {
            "cow_id": "COW_005",
            "current_risk": 33.50,
            "risk_category": "Moderate Risk",
            "7d_probability": 25.30,
            "14d_probability": 39.70,
            "7d_warning": True,
            "14d_warning": True
        },

        {
            "cow_id": "COW_006",
            "current_risk": 5.80,
            "risk_category": "No Risk",
            "7d_probability": 3.10,
            "14d_probability": 6.70,
            "7d_warning": False,
            "14d_warning": False
        },

        {
            "cow_id": "COW_007",
            "current_risk": 18.70,
            "risk_category": "Low Risk",
            "7d_probability": 12.50,
            "14d_probability": 22.80,
            "7d_warning": False,
            "14d_warning": False
        },

        {
            "cow_id": "COW_008",
            "current_risk": 61.40,
            "risk_category": "High Risk",
            "7d_probability": 55.20,
            "14d_probability": 70.30,
            "7d_warning": True,
            "14d_warning": True
        },

        {
            "cow_id": "COW_009",
            "current_risk": 23.10,
            "risk_category": "Low Risk",
            "7d_probability": 17.20,
            "14d_probability": 28.10,
            "7d_warning": False,
            "14d_warning": False
        },

        {
            "cow_id": "COW_010",
            "current_risk": 54.60,
            "risk_category": "High Risk",
            "7d_probability": 48.70,
            "14d_probability": 65.20,
            "7d_warning": True,
            "14d_warning": True
        }
    ]

    # --------------------------------------------------------
    # ANALYZE HERD
    # --------------------------------------------------------

    herd_result = analyze_herd(
        demo_cow_predictions
    )

    # --------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------

    print_herd_report(
        herd_result
    )