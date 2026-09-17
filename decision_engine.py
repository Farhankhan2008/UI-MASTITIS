import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_RISK_THRESHOLD = 50.0

WARNING_THRESHOLD_7D = 20.0
WARNING_THRESHOLD_14D = 30.0


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(risk_percent):

    risk_percent = float(risk_percent)

    if risk_percent <= 10:
        return "No Risk"

    elif risk_percent <= 25:
        return "Low Risk"

    elif risk_percent <= 50:
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

    current_risk = float(current_risk)
    probability_7d = float(probability_7d)
    probability_14d = float(probability_14d)

    # Critical
    if (
        current_risk >= HIGH_RISK_THRESHOLD
        and probability_7d >= WARNING_THRESHOLD_7D
        and probability_14d >= WARNING_THRESHOLD_14D
    ):
        return "CRITICAL"

    # High
    if (
        current_risk >= HIGH_RISK_THRESHOLD
        or probability_7d >= WARNING_THRESHOLD_7D
        or probability_14d >= WARNING_THRESHOLD_14D
    ):
        return "HIGH"

    # Moderate
    if current_risk > 25:
        return "MODERATE"

    # Low
    if current_risk > 10:
        return "LOW"

    return "NORMAL"


# ============================================================
# INDIVIDUAL COW RECOMMENDATIONS
# ============================================================

def generate_cow_recommendations(
    current_risk,
    probability_7d,
    probability_14d,
    milk_conductivity=None,
    cow_activity=None,
    risk_trend=None
):

    current_risk = float(current_risk)
    probability_7d = float(probability_7d)
    probability_14d = float(probability_14d)

    recommendations = []

    # --------------------------------------------------------
    # CURRENT RISK
    # --------------------------------------------------------

    if current_risk > HIGH_RISK_THRESHOLD:

        recommendations.append(
            "Prioritize this cow for udder-health inspection."
        )

    elif current_risk > 25:

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
    #
    # We do not diagnose based on this alone.
    # It is only a decision-support signal.
    # --------------------------------------------------------

    if milk_conductivity is not None:

        if milk_conductivity >= 5.0:

            recommendations.append(
                "Milk conductivity is elevated. "
                "Verify milk-quality indicators and consider SCC testing."
            )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    if cow_activity is not None:

        if cow_activity < 50:

            recommendations.append(
                "Activity level is relatively low. "
                "Monitor the cow for behavioral or health changes."
            )

    # --------------------------------------------------------
    # RISK TREND
    # --------------------------------------------------------

    if risk_trend is not None:

        if risk_trend > 10:

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
# COMPLETE INDIVIDUAL DECISION
# ============================================================

def analyze_cow(
    current_risk,
    probability_7d,
    probability_14d,
    milk_conductivity=None,
    cow_activity=None,
    risk_trend=None
):

    category = get_risk_category(
        current_risk
    )

    alert_level = get_alert_level(
        current_risk,
        probability_7d,
        probability_14d
    )

    recommendations = generate_cow_recommendations(
        current_risk=current_risk,
        probability_7d=probability_7d,
        probability_14d=probability_14d,
        milk_conductivity=milk_conductivity,
        cow_activity=cow_activity,
        risk_trend=risk_trend
    )

    return {
        "risk_percent": round(
            float(current_risk),
            2
        ),

        "risk_category": category,

        "7d_probability": round(
            float(probability_7d),
            2
        ),

        "14d_probability": round(
            float(probability_14d),
            2
        ),

        "7d_warning": (
            float(probability_7d)
            >= WARNING_THRESHOLD_7D
        ),

        "14d_warning": (
            float(probability_14d)
            >= WARNING_THRESHOLD_14D
        ),

        "alert_level": alert_level,

        "recommendations": recommendations
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = analyze_cow(
        current_risk=41.27,
        probability_7d=29.86,
        probability_14d=46.68,
        milk_conductivity=5.2,
        cow_activity=72,
        risk_trend=20.0
    )

    print("=" * 70)
    print("DECISION ENGINE TEST")
    print("=" * 70)

    print()
    print(
        f"Current Risk       : "
        f"{result['risk_percent']:.2f}%"
    )

    print(
        f"Risk Category      : "
        f"{result['risk_category']}"
    )

    print(
        f"7-Day Probability  : "
        f"{result['7d_probability']:.2f}%"
    )

    print(
        f"7-Day Warning      : "
        f"{'YES' if result['7d_warning'] else 'NO'}"
    )

    print(
        f"14-Day Probability : "
        f"{result['14d_probability']:.2f}%"
    )

    print(
        f"14-Day Warning     : "
        f"{'YES' if result['14d_warning'] else 'NO'}"
    )

    print(
        f"Alert Level        : "
        f"{result['alert_level']}"
    )

    print()
    print("Recommendations:")
    print("-" * 70)

    for recommendation in result["recommendations"]:

        print(
            f"- {recommendation}"
        )

    print()
    print("=" * 70)
    print("DECISION ENGINE TEST COMPLETE")
    print("=" * 70)