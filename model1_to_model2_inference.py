import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL1_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model1.pkl"
)

MODEL2_FILES = {
    "24h": os.path.join(BASE_DIR, "mastitis_model2_final_24h.pkl"),
    "48h": os.path.join(BASE_DIR, "mastitis_model2_final_48h.pkl"),
    "3d": os.path.join(BASE_DIR, "mastitis_model2_final_3d.pkl"),
    "7d": os.path.join(BASE_DIR, "mastitis_model2_v4_7d.pkl"),
    "14d": os.path.join(BASE_DIR, "mastitis_model2_v4_14d.pkl")
}


# ============================================================
# MODEL 1 FEATURES
# ============================================================

MODEL1_FEATURES = [
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent",
    "previous_mastitis",
    "days_since_last_mastitis"
]


# ============================================================
# MODEL 2 FEATURES
# EXACT ORDER USED DURING TRAINING
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


HIGH_RISK_THRESHOLD = 50.0


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 80)
print("MODEL 1 -> MODEL 2 INFERENCE PIPELINE")
print("=" * 80)

print()
print("Loading models...")

model1 = joblib.load(MODEL1_FILE)

model2 = {}

for horizon, path in MODEL2_FILES.items():

    model2[horizon] = joblib.load(path)

print("Model 1 loaded.")
print("Model 2 models loaded.")

print(
    f"Model 2 feature count: "
    f"{len(model2['24h']['features'])}"
)


# ============================================================
# MODEL 1 PREDICTION
# ============================================================

def predict_model1_risk(
    milk_yield_liters,
    milk_conductivity_ms_cm,
    cow_activity,
    environment_temperature_c,
    milk_temperature_c,
    humidity_percent,
    previous_mastitis,
    days_since_last_mastitis
):

    input_data = pd.DataFrame(
        [[
            milk_yield_liters,
            milk_conductivity_ms_cm,
            cow_activity,
            environment_temperature_c,
            milk_temperature_c,
            humidity_percent,
            previous_mastitis,
            days_since_last_mastitis
        ]],
        columns=MODEL1_FEATURES
    )

    probability = model1.predict_proba(
        input_data
    )[0, 1]

    return float(
        np.clip(
            probability * 100.0,
            0,
            100
        )
    )


# ============================================================
# SLOPE
# EXACT LOGIC USED IN MODEL2_PREPARE_V3.PY
# ============================================================

def calculate_slope(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 2:
        return 0.0

    x = np.arange(
        len(values),
        dtype=float
    )

    try:

        slope = np.polyfit(
            x,
            values,
            1
        )[0]

        if (
            np.isnan(slope)
            or np.isinf(slope)
        ):
            return 0.0

        return float(slope)

    except Exception:

        return 0.0


# ============================================================
# RISING STREAK
# EXACT LOGIC USED IN V3
# ============================================================

def calculate_rising_streak(values):

    values = np.asarray(
        values,
        dtype=float
    )

    result = np.zeros(
        len(values),
        dtype=float
    )

    current_streak = 0

    for i in range(1, len(values)):

        if (
            not np.isnan(values[i])
            and not np.isnan(values[i - 1])
            and values[i] > values[i - 1]
        ):

            current_streak += 1

        else:

            current_streak = 0

        result[i] = current_streak

    return int(result[-1])


# ============================================================
# FALLING STREAK
# EXACT LOGIC USED IN V3
# ============================================================

def calculate_falling_streak(values):

    values = np.asarray(
        values,
        dtype=float
    )

    result = np.zeros(
        len(values),
        dtype=float
    )

    current_streak = 0

    for i in range(1, len(values)):

        if (
            not np.isnan(values[i])
            and not np.isnan(values[i - 1])
            and values[i] < values[i - 1]
        ):

            current_streak += 1

        else:

            current_streak = 0

        result[i] = current_streak

    return int(result[-1])


# ============================================================
# MODEL 2 FEATURE GENERATION
#
# IMPORTANT:
# risks must be:
#
# [today,
#  1d ago,
#  2d ago,
#  ...
#  14d ago]
#
# ============================================================

def calculate_model2_features(risks):

    if len(risks) != 15:

        raise ValueError(
            "Model 2 requires exactly 15 risk values:"
            " today + previous 14 days."
        )

    risks = np.asarray(
        risks,
        dtype=float
    )

    risks = np.clip(
        risks,
        0,
        100
    )

    today = risks[0]

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    risk_1d = risks[1]
    risk_2d = risks[2]
    risk_3d = risks[3]
    risk_7d = risks[7]
    risk_14d = risks[14]

    # --------------------------------------------------------
    # RISK CHANGES
    # --------------------------------------------------------

    risk_change_1d = (
        today - risk_1d
    )

    risk_change_2d = (
        today - risk_2d
    )

    risk_change_3d = (
        today - risk_3d
    )

    risk_change_7d = (
        today - risk_7d
    )

    risk_change_14d = (
        today - risk_14d
    )

    # --------------------------------------------------------
    # HISTORICAL WINDOWS
    #
    # EXACTLY MATCHES:
    # shifted_risk = group.shift(1)
    #
    # Therefore TODAY IS NOT INCLUDED.
    # --------------------------------------------------------

    history_3d = risks[1:4]
    history_7d = risks[1:8]
    history_14d = risks[1:15]

    # --------------------------------------------------------
    # 3-DAY STATISTICS
    # --------------------------------------------------------

    risk_3day_mean = np.mean(
        history_3d
    )

    risk_3day_max = np.max(
        history_3d
    )

    risk_3day_min = np.min(
        history_3d
    )

    risk_std_3d = np.std(
        history_3d,
        ddof=1
    )

    # --------------------------------------------------------
    # 7-DAY STATISTICS
    # --------------------------------------------------------

    risk_7day_mean = np.mean(
        history_7d
    )

    risk_7day_max = np.max(
        history_7d
    )

    risk_7day_min = np.min(
        history_7d
    )

    risk_std_7d = np.std(
        history_7d,
        ddof=1
    )

    # --------------------------------------------------------
    # 14-DAY STATISTICS
    # --------------------------------------------------------

    risk_14day_mean = np.mean(
        history_14d
    )

    risk_14day_max = np.max(
        history_14d
    )

    risk_14day_min = np.min(
        history_14d
    )

    risk_std_14d = np.std(
        history_14d,
        ddof=1
    )

    # --------------------------------------------------------
    # TREND
    #
    # EXACT V3:
    #
    # risk_trend_3d =
    #     risk_1d_ago - risk_3d_ago
    # --------------------------------------------------------

    risk_trend_3d = (
        risk_1d - risk_3d
    )

    # --------------------------------------------------------
    # SLOPES
    #
    # Training uses shifted history.
    #
    # Historical values need to be in chronological order:
    #
    # 3d slope:
    # 3d ago -> 2d ago -> 1d ago
    # --------------------------------------------------------

    slope_3d_values = risks[
        3:0:-1
    ]

    slope_7d_values = risks[
        7:0:-1
    ]

    slope_14d_values = risks[
        14:0:-1
    ]

    risk_slope_3d = calculate_slope(
        slope_3d_values
    )

    risk_slope_7d = calculate_slope(
        slope_7d_values
    )

    risk_slope_14d = calculate_slope(
        slope_14d_values
    )

    # --------------------------------------------------------
    # ACCELERATION
    # EXACT V3 FORMULAS
    # --------------------------------------------------------

    risk_acceleration_2d = (
        risk_change_1d
        -
        (
            risk_1d
            -
            risk_2d
        )
    )

    risk_acceleration_3d = (
        risk_change_1d
        -
        (
            risk_2d
            -
            risk_3d
        )
    )

    # --------------------------------------------------------
    # CONSECUTIVE MOVEMENT
    #
    # Exact V3 logic uses the complete sequence:
    # 14d ago -> ... -> today
    # --------------------------------------------------------

    chronological_risks = risks[
        ::-1
    ]

    consecutive_rising_days = (
        calculate_rising_streak(
            chronological_risks
        )
    )

    consecutive_falling_days = (
        calculate_falling_streak(
            chronological_risks
        )
    )

    # --------------------------------------------------------
    # HIGH-RISK HISTORY
    #
    # TODAY IS EXCLUDED.
    # --------------------------------------------------------

    historical_high_risk = (
        history_14d >= HIGH_RISK_THRESHOLD
    ).astype(int)

    high_risk_count_3d = int(
        np.sum(
            history_3d >= HIGH_RISK_THRESHOLD
        )
    )

    high_risk_count_7d = int(
        np.sum(
            history_7d >= HIGH_RISK_THRESHOLD
        )
    )

    high_risk_count_14d = int(
        np.sum(
            historical_high_risk
        )
    )

    recent_high_risk_3d = int(
        high_risk_count_3d > 0
    )

    recent_high_risk_7d = int(
        high_risk_count_7d > 0
    )

    recent_high_risk_14d = int(
        high_risk_count_14d > 0
    )

    # --------------------------------------------------------
    # DISTANCE FROM HISTORICAL MAXIMUM
    # --------------------------------------------------------

    distance_from_3day_max = (
        today -
        risk_3day_max
    )

    distance_from_7day_max = (
        today -
        risk_7day_max
    )

    distance_from_14day_max = (
        today -
        risk_14day_max
    )

    # --------------------------------------------------------
    # CURRENT RISK VS HISTORICAL MEAN
    # --------------------------------------------------------

    risk_vs_3day_mean = (
        today -
        risk_3day_mean
    )

    risk_vs_7day_mean = (
        today -
        risk_7day_mean
    )

    risk_vs_14day_mean = (
        today -
        risk_14day_mean
    )

    # --------------------------------------------------------
    # LONG-TERM TREND COMPARISON
    # --------------------------------------------------------

    risk_7day_mean_vs_14day_mean = (
        risk_7day_mean -
        risk_14day_mean
    )

    risk_3day_mean_vs_7day_mean = (
        risk_3day_mean -
        risk_7day_mean
    )

    # --------------------------------------------------------
    # BUILD EXACT 54-FEATURE DICTIONARY
    # --------------------------------------------------------

    features = {

        "risk_today": today,

        "risk_1d_ago": risks[1],
        "risk_2d_ago": risks[2],
        "risk_3d_ago": risks[3],
        "risk_4d_ago": risks[4],
        "risk_5d_ago": risks[5],
        "risk_6d_ago": risks[6],
        "risk_7d_ago": risks[7],

        "risk_8d_ago": risks[8],
        "risk_9d_ago": risks[9],
        "risk_10d_ago": risks[10],
        "risk_11d_ago": risks[11],
        "risk_12d_ago": risks[12],
        "risk_13d_ago": risks[13],
        "risk_14d_ago": risks[14],

        "risk_change_1d": risk_change_1d,
        "risk_change_2d": risk_change_2d,
        "risk_change_3d": risk_change_3d,
        "risk_change_7d": risk_change_7d,
        "risk_change_14d": risk_change_14d,

        "risk_3day_mean": risk_3day_mean,
        "risk_3day_max": risk_3day_max,
        "risk_3day_min": risk_3day_min,
        "risk_std_3d": risk_std_3d,

        "risk_7day_mean": risk_7day_mean,
        "risk_7day_max": risk_7day_max,
        "risk_7day_min": risk_7day_min,
        "risk_std_7d": risk_std_7d,

        "risk_14day_mean": risk_14day_mean,
        "risk_14day_max": risk_14day_max,
        "risk_14day_min": risk_14day_min,
        "risk_std_14d": risk_std_14d,

        "risk_trend_3d": risk_trend_3d,
        "risk_slope_3d": risk_slope_3d,
        "risk_slope_7d": risk_slope_7d,
        "risk_slope_14d": risk_slope_14d,

        "risk_acceleration_2d":
            risk_acceleration_2d,

        "risk_acceleration_3d":
            risk_acceleration_3d,

        "consecutive_rising_days":
            consecutive_rising_days,

        "consecutive_falling_days":
            consecutive_falling_days,

        "high_risk_count_3d":
            high_risk_count_3d,

        "high_risk_count_7d":
            high_risk_count_7d,

        "high_risk_count_14d":
            high_risk_count_14d,

        "recent_high_risk_3d":
            recent_high_risk_3d,

        "recent_high_risk_7d":
            recent_high_risk_7d,

        "recent_high_risk_14d":
            recent_high_risk_14d,

        "distance_from_3day_max":
            distance_from_3day_max,

        "distance_from_7day_max":
            distance_from_7day_max,

        "distance_from_14day_max":
            distance_from_14day_max,

        "risk_vs_3day_mean":
            risk_vs_3day_mean,

        "risk_vs_7day_mean":
            risk_vs_7day_mean,

        "risk_vs_14day_mean":
            risk_vs_14day_mean,

        "risk_7day_mean_vs_14day_mean":
            risk_7day_mean_vs_14day_mean,

        "risk_3day_mean_vs_7day_mean":
            risk_3day_mean_vs_7day_mean
    }

    # --------------------------------------------------------
    # FINAL ORDER CHECK
    # --------------------------------------------------------

    feature_vector = pd.DataFrame(
        [[
            features[name]
            for name in MODEL2_FEATURES
        ]],
        columns=MODEL2_FEATURES
    )

    return feature_vector


# ============================================================
# MODEL 2 PREDICTION
# ============================================================

def predict_model2(
    risk_history
):

    features = calculate_model2_features(
        risk_history
    )

    predictions = {}

    # --------------------------------------------------------
    # 24 HOURS
    # --------------------------------------------------------

    model_24h = model2["24h"]["model"]

    predictions["24h"] = float(
        np.clip(
            model_24h.predict(
                features
            )[0],
            0,
            100
        )
    )

    # --------------------------------------------------------
    # 48 HOURS
    # --------------------------------------------------------

    model_48h = model2["48h"]["model"]

    predictions["48h"] = float(
        np.clip(
            model_48h.predict(
                features
            )[0],
            0,
            100
        )
    )

    # --------------------------------------------------------
    # 3 DAYS
    # --------------------------------------------------------

    model_3d = model2["3d"]["model"]

    predictions["3d"] = float(
        np.clip(
            model_3d.predict(
                features
            )[0],
            0,
            100
        )
    )

    # --------------------------------------------------------
    # 7 DAYS
    # --------------------------------------------------------

    model_7d = model2["7d"]["model"]

    probability_7d = (
        model_7d.predict_proba(
            features
        )[0, 1]
    )

    predictions["7d_probability"] = (
        float(
            probability_7d * 100
        )
    )

    WARNING_THRESHOLD_7D = 0.20

    predictions["7d_warning"] = probability_7d >= WARNING_THRESHOLD_7D

    # --------------------------------------------------------
    # 14 DAYS
    # --------------------------------------------------------

    model_14d = model2["14d"]["model"]

    probability_14d = (
        model_14d.predict_proba(
            features
        )[0, 1]
    )

    predictions["14d_probability"] = (
        float(
            probability_14d * 100
        )
    )

    WARNING_THRESHOLD_14D = 0.30

    predictions["14d_warning"] = probability_14d >= WARNING_THRESHOLD_14D

    return predictions


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("DEMO PREDICTION")
    print("=" * 80)

    print()
    print(
        "The values below are DEMO sensor values."
    )

    print(
        "Replace them with actual sensor/RFID data later."
    )

    # --------------------------------------------------------
    # DEMO COW
    # --------------------------------------------------------

    cow_id = "COW_001"

    # --------------------------------------------------------
    # DEMO SENSOR VALUES
    # --------------------------------------------------------

    milk_yield_liters = 18.5

    milk_conductivity_ms_cm = 5.2

    cow_activity = 72

    environment_temperature_c = 30.5

    milk_temperature_c = 38.7

    humidity_percent = 68

    previous_mastitis = 0

    days_since_last_mastitis = 45

    # --------------------------------------------------------
    # PREVIOUS MODEL 1 RISKS
    #
    # [today placeholder,
    #  1d ago,
    #  ...
    #  14d ago]
    #
    # Today's value will be replaced by Model 1.
    # --------------------------------------------------------

    previous_14_risks = [
        12.0,
        11.0,
        10.0,
        13.0,
        15.0,
        14.0,
        12.0,
        11.0,
        10.0,
        9.0,
        8.0,
        9.0,
        10.0,
        11.0
    ]

    # --------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------

    today_risk = predict_model1_risk(
        milk_yield_liters,
        milk_conductivity_ms_cm,
        cow_activity,
        environment_temperature_c,
        milk_temperature_c,
        humidity_percent,
        previous_mastitis,
        days_since_last_mastitis
    )

    # Today + previous 14 days
    risk_history = [
        today_risk
    ] + previous_14_risks

    print()
    print("=" * 80)
    print("MODEL 1")
    print("=" * 80)

    print()
    print(
        f"Cow ID: {cow_id}"
    )

    print(
        f"Today's Model 1 mastitis risk: "
        f"{today_risk:.2f}%"
    )

    print()
    print("Risk history used by Model 2:")

    print(
        f"Today          : "
        f"{risk_history[0]:.2f}%"
    )

    for day in range(1, 15):

        print(
            f"{day} day(s) ago   : "
            f"{risk_history[day]:.2f}%"
        )

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    predictions = predict_model2(
        risk_history
    )

    print()
    print("=" * 80)
    print("MODEL 2 FORECAST")
    print("=" * 80)

    print()
    print("SHORT-TERM FORECAST")
    print("-" * 50)

    print(
        f"24-hour predicted risk : "
        f"{predictions['24h']:.2f}%"
    )

    print(
        f"48-hour predicted risk : "
        f"{predictions['48h']:.2f}%"
    )

    print(
        f"3-day predicted risk   : "
        f"{predictions['3d']:.2f}%"
    )

    print()
    print("LONG-TERM WARNING")
    print("-" * 50)

    print(
        f"7-day high-risk probability  : "
        f"{predictions['7d_probability']:.2f}%"
    )

    print(
        f"7-day warning                : "
        f"{'YES' if predictions['7d_warning'] else 'NO'}"
    )

    print(
        f"14-day high-risk probability : "
        f"{predictions['14d_probability']:.2f}%"
    )

    print(
        f"14-day warning               : "
        f"{'YES' if predictions['14d_warning'] else 'NO'}"
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print()
    print("24h / 48h / 3d:")

    print(
        "    These are predicted mastitis risk percentages."
    )

    print()
    print("7d:")

    print(
        "    Probability that the cow enters a high-risk state"
    )

    print(
        "    (Model 1 risk >= 50%) within the next 7 days."
    )

    print()
    print("14d:")

    print(
        "    Probability that the cow enters a high-risk state"
    )

    print(
        "    (Model 1 risk >= 50%) within the next 14 days."
    )

    print()
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)