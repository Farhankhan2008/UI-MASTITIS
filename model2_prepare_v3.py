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

MODEL1_DATASET = os.path.join(
    BASE_DIR,
    "synthetic_mastitis_model1_dataset_v2.csv"
)

MODEL1_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model1.pkl"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)


# ============================================================
# MODEL 1 INPUT FEATURES
# ============================================================

MODEL1_FEATURES = [
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent"
]


# ============================================================
# MODEL 2 V3 FEATURES
# ============================================================

MODEL2_FEATURES = [

    # --------------------------------------------------------
    # CURRENT + LAGGED MODEL 1 RISK
    # --------------------------------------------------------
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


    # --------------------------------------------------------
    # SHORT-TERM RISK CHANGES
    # --------------------------------------------------------
    "risk_change_1d",
    "risk_change_2d",
    "risk_change_3d",
    "risk_change_7d",
    "risk_change_14d",


    # --------------------------------------------------------
    # ROLLING 3-DAY STATISTICS
    # --------------------------------------------------------
    "risk_3day_mean",
    "risk_3day_max",
    "risk_3day_min",
    "risk_std_3d",


    # --------------------------------------------------------
    # ROLLING 7-DAY STATISTICS
    # --------------------------------------------------------
    "risk_7day_mean",
    "risk_7day_max",
    "risk_7day_min",
    "risk_std_7d",


    # --------------------------------------------------------
    # ROLLING 14-DAY STATISTICS
    # --------------------------------------------------------
    "risk_14day_mean",
    "risk_14day_max",
    "risk_14day_min",
    "risk_std_14d",


    # --------------------------------------------------------
    # TREND / SLOPE
    # --------------------------------------------------------
    "risk_trend_3d",
    "risk_slope_3d",
    "risk_slope_7d",
    "risk_slope_14d",


    # --------------------------------------------------------
    # RISK ACCELERATION
    # --------------------------------------------------------
    "risk_acceleration_2d",
    "risk_acceleration_3d",


    # --------------------------------------------------------
    # CONSECUTIVE MOVEMENT
    # --------------------------------------------------------
    "consecutive_rising_days",
    "consecutive_falling_days",


    # --------------------------------------------------------
    # HIGH-RISK HISTORY
    # --------------------------------------------------------
    "high_risk_count_3d",
    "high_risk_count_7d",
    "high_risk_count_14d",

    "recent_high_risk_3d",
    "recent_high_risk_7d",
    "recent_high_risk_14d",


    # --------------------------------------------------------
    # DISTANCE FROM RECENT MAXIMUM
    # --------------------------------------------------------
    "distance_from_3day_max",
    "distance_from_7day_max",
    "distance_from_14day_max",


    # --------------------------------------------------------
    # CURRENT RISK VS HISTORICAL AVERAGE
    # --------------------------------------------------------
    "risk_vs_3day_mean",
    "risk_vs_7day_mean",
    "risk_vs_14day_mean",


    # --------------------------------------------------------
    # LONG-TERM TREND COMPARISON
    # --------------------------------------------------------
    "risk_7day_mean_vs_14day_mean",

    "risk_3day_mean_vs_7day_mean"
]


# ============================================================
# FUTURE TARGETS
# ============================================================

TARGET_COLUMNS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d",
    "risk_14d"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_slope(values):
    """
    Calculate linear regression slope for a sequence.

    Returns 0 when there is insufficient information.
    """

    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=float)

    try:
        slope = np.polyfit(x, values, 1)[0]

        if np.isnan(slope) or np.isinf(slope):
            return 0.0

        return float(slope)

    except Exception:
        return 0.0


# ============================================================
# CONSECUTIVE RISING DAYS
# ============================================================

def calculate_rising_streak(values):
    """
    Calculate the number of consecutive rising days
    ending at the current day.

    Example:
    [5, 7, 9] -> 2
    """

    values = list(values)

    if len(values) < 2:
        return 0

    streak = 0

    for i in range(len(values) - 1, 0, -1):

        if values[i] > values[i - 1]:
            streak += 1
        else:
            break

    return streak


# ============================================================
# CONSECUTIVE FALLING DAYS
# ============================================================

def calculate_falling_streak(values):
    """
    Calculate the number of consecutive falling days
    ending at the current day.

    Example:
    [10, 8, 5] -> 2
    """

    values = list(values)

    if len(values) < 2:
        return 0

    streak = 0

    for i in range(len(values) - 1, 0, -1):

        if values[i] < values[i - 1]:
            streak += 1
        else:
            break

    return streak


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 78)
print("                 MODEL 2 V3 DATASET PREPARATION")
print("=" * 78)

print()
print("Purpose:")
print("Create an enhanced temporal forecasting dataset for:")
print("24 hours, 48 hours, 3 days, 7 days and 14 days.")

print()
print("IMPORTANT:")
print("- Model 1 will NOT be modified.")
print("- Model 1 dataset will NOT be modified.")
print("- Model 2 V1 files will NOT be modified.")
print("- Model 2 V2 files will NOT be modified.")

print()
print("=" * 78)
print("                       FILE CHECK")
print("=" * 78)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL1_DATASET):
    raise FileNotFoundError(
        f"Model 1 dataset not found:\n{MODEL1_DATASET}"
    )

if not os.path.exists(MODEL1_FILE):
    raise FileNotFoundError(
        f"Model 1 model file not found:\n{MODEL1_FILE}"
    )

print()
print("PASS: Model 1 dataset found.")
print(MODEL1_DATASET)

print()
print("PASS: Model 1 model found.")
print(MODEL1_FILE)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("                       LOADING DATA")
print("=" * 78)

df = pd.read_csv(MODEL1_DATASET)

print()
print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# BASIC MODEL 1 COLUMN CHECK
# ============================================================

required_model1_columns = [
    "cow_id",
    "date"
] + MODEL1_FEATURES + [
    "mastitis_today"
]

missing_columns = [
    col
    for col in required_model1_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required Model 1 columns:\n"
        + "\n".join(missing_columns)
    )

print()
print("PASS: All required Model 1 columns found.")


# ============================================================
# DATE CONVERSION
# ============================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError(
        "ERROR: Invalid date values detected."
    )


# ============================================================
# DUPLICATE CHECK BEFORE PROCESSING
# ============================================================

duplicate_count = df.duplicated(
    subset=["cow_id", "date"]
).sum()

print()
print(f"Duplicate cow/date records: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "ERROR: Duplicate cow/date records found."
    )

print("PASS: No duplicate cow/date records.")


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# LOAD MODEL 1
# ============================================================

print()
print("=" * 78)
print("                    LOADING MODEL 1")
print("=" * 78)

model1 = joblib.load(MODEL1_FILE)

print()
print("PASS: Model 1 loaded successfully.")


# ============================================================
# GENERATE MODEL 1 RISK
# ============================================================

print()
print("=" * 78)
print("                 GENERATING MODEL 1 RISK")
print("=" * 78)

X_model1 = df[MODEL1_FEATURES]

risk_probability = model1.predict_proba(X_model1)[:, 1]

df["model1_risk_percent"] = (
    np.clip(risk_probability * 100.0, 0, 100)
)

print()
print("PASS: Model 1 risk generated.")

print(
    f"Risk minimum : {df['model1_risk_percent'].min():.2f}%"
)

print(
    f"Risk maximum : {df['model1_risk_percent'].max():.2f}%"
)

print(
    f"Risk mean    : {df['model1_risk_percent'].mean():.2f}%"
)


# ============================================================
# CREATE TEMPORARY RISK SERIES
# ============================================================

print()
print("=" * 78)
print("                 CREATING TEMPORAL FEATURES")
print("=" * 78)

group = df.groupby("cow_id")["model1_risk_percent"]


# ============================================================
# LAG FEATURES
# ============================================================

print()
print("Creating lag features...")

for days in range(1, 15):

    df[f"risk_{days}d_ago"] = (
        group.shift(days)
    )


# Current risk
df["risk_today"] = df["model1_risk_percent"]


# ============================================================
# RISK CHANGE FEATURES
# ============================================================

print("Creating risk-change features...")

df["risk_change_1d"] = (
    df["risk_today"] -
    df["risk_1d_ago"]
)

df["risk_change_2d"] = (
    df["risk_today"] -
    df["risk_2d_ago"]
)

df["risk_change_3d"] = (
    df["risk_today"] -
    df["risk_3d_ago"]
)

df["risk_change_7d"] = (
    df["risk_today"] -
    df["risk_7d_ago"]
)

df["risk_change_14d"] = (
    df["risk_today"] -
    df["risk_14d_ago"]
)


# ============================================================
# SHIFTED HISTORY SERIES
# ============================================================

# IMPORTANT:
# All rolling statistics use shift(1).
#
# Therefore the current day's risk is NOT included
# in historical rolling calculations.
#
# This prevents information leakage.

shifted_risk = group.shift(1)


# ============================================================
# ROLLING 3-DAY FEATURES
# ============================================================

print("Creating 3-day rolling features...")

df["risk_3day_mean"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=3, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["risk_3day_max"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=3, min_periods=3)
    .max()
    .reset_index(level=0, drop=True)
)

df["risk_3day_min"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=3, min_periods=3)
    .min()
    .reset_index(level=0, drop=True)
)

df["risk_std_3d"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=3, min_periods=3)
    .std()
    .reset_index(level=0, drop=True)
)


# ============================================================
# ROLLING 7-DAY FEATURES
# ============================================================

print("Creating 7-day rolling features...")

df["risk_7day_mean"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=7, min_periods=7)
    .mean()
    .reset_index(level=0, drop=True)
)

df["risk_7day_max"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=7, min_periods=7)
    .max()
    .reset_index(level=0, drop=True)
)

df["risk_7day_min"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=7, min_periods=7)
    .min()
    .reset_index(level=0, drop=True)
)

df["risk_std_7d"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=7, min_periods=7)
    .std()
    .reset_index(level=0, drop=True)
)


# ============================================================
# ROLLING 14-DAY FEATURES
# ============================================================

print("Creating 14-day rolling features...")

df["risk_14day_mean"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=14, min_periods=14)
    .mean()
    .reset_index(level=0, drop=True)
)

df["risk_14day_max"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=14, min_periods=14)
    .max()
    .reset_index(level=0, drop=True)
)

df["risk_14day_min"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=14, min_periods=14)
    .min()
    .reset_index(level=0, drop=True)
)

df["risk_std_14d"] = (
    shifted_risk
    .groupby(df["cow_id"])
    .rolling(window=14, min_periods=14)
    .std()
    .reset_index(level=0, drop=True)
)


# ============================================================
# TREND FEATURES
# ============================================================

print("Creating trend and slope features...")


def add_group_slope(group_df, window):
    return (
        group_df
        .rolling(window=window, min_periods=window)
        .apply(calculate_slope, raw=True)
    )


df["risk_slope_3d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: add_group_slope(x.shift(1), 3)
    )
)

df["risk_slope_7d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: add_group_slope(x.shift(1), 7)
    )
)

df["risk_slope_14d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: add_group_slope(x.shift(1), 14)
    )
)

# Existing-style trend
df["risk_trend_3d"] = (
    df["risk_1d_ago"] -
    df["risk_3d_ago"]
)


# ============================================================
# RISK ACCELERATION
# ============================================================

print("Creating risk acceleration features...")

df["risk_acceleration_2d"] = (
    df["risk_change_1d"] -
    (
        df["risk_1d_ago"] -
        df["risk_2d_ago"]
    )
)

df["risk_acceleration_3d"] = (
    df["risk_change_1d"] -
    (
        df["risk_2d_ago"] -
        df["risk_3d_ago"]
    )
)


# ============================================================
# CONSECUTIVE RISING/FALLING DAYS
# ============================================================

print("Creating rising/falling streak features...")


def rising_streak_series(series):
    values = series.to_numpy(dtype=float)

    result = np.zeros(len(values), dtype=float)

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

    return pd.Series(
        result,
        index=series.index
    )


def falling_streak_series(series):
    values = series.to_numpy(dtype=float)

    result = np.zeros(len(values), dtype=float)

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

    return pd.Series(
        result,
        index=series.index
    )


df["consecutive_rising_days"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(rising_streak_series)
)

df["consecutive_falling_days"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(falling_streak_series)
)


# ============================================================
# HIGH-RISK FEATURES
# ============================================================

print("Creating high-risk history features...")

HIGH_RISK_THRESHOLD = 50.0

high_risk_indicator = (
    df["model1_risk_percent"] >= HIGH_RISK_THRESHOLD
).astype(int)

high_risk_group = (
    high_risk_indicator
    .groupby(df["cow_id"])
)


# 3-day count
df["high_risk_count_3d"] = (
    high_risk_indicator
    .groupby(df["cow_id"])
    .shift(1)
    .groupby(df["cow_id"])
    .rolling(window=3, min_periods=3)
    .sum()
    .reset_index(level=0, drop=True)
)

# 7-day count
df["high_risk_count_7d"] = (
    high_risk_indicator
    .groupby(df["cow_id"])
    .shift(1)
    .groupby(df["cow_id"])
    .rolling(window=7, min_periods=7)
    .sum()
    .reset_index(level=0, drop=True)
)

# 14-day count
df["high_risk_count_14d"] = (
    high_risk_indicator
    .groupby(df["cow_id"])
    .shift(1)
    .groupby(df["cow_id"])
    .rolling(window=14, min_periods=14)
    .sum()
    .reset_index(level=0, drop=True)
)


# ============================================================
# RECENT HIGH-RISK INDICATORS
# ============================================================

df["recent_high_risk_3d"] = (
    df["high_risk_count_3d"] > 0
).astype(int)

df["recent_high_risk_7d"] = (
    df["high_risk_count_7d"] > 0
).astype(int)

df["recent_high_risk_14d"] = (
    df["high_risk_count_14d"] > 0
).astype(int)


# ============================================================
# DISTANCE FROM RECENT MAXIMUM
# ============================================================

print("Creating distance-from-maximum features...")

df["distance_from_3day_max"] = (
    df["risk_today"] -
    df["risk_3day_max"]
)

df["distance_from_7day_max"] = (
    df["risk_today"] -
    df["risk_7day_max"]
)

df["distance_from_14day_max"] = (
    df["risk_today"] -
    df["risk_14day_max"]
)


# ============================================================
# CURRENT RISK VS HISTORICAL MEAN
# ============================================================

print("Creating risk-vs-average features...")

df["risk_vs_3day_mean"] = (
    df["risk_today"] -
    df["risk_3day_mean"]
)

df["risk_vs_7day_mean"] = (
    df["risk_today"] -
    df["risk_7day_mean"]
)

df["risk_vs_14day_mean"] = (
    df["risk_today"] -
    df["risk_14day_mean"]
)


# ============================================================
# LONG-TERM TREND COMPARISON
# ============================================================

print("Creating long-term trend comparison features...")

df["risk_7day_mean_vs_14day_mean"] = (
    df["risk_7day_mean"] -
    df["risk_14day_mean"]
)

df["risk_3day_mean_vs_7day_mean"] = (
    df["risk_3day_mean"] -
    df["risk_7day_mean"]
)


# ============================================================
# FUTURE TARGETS
# ============================================================

print()
print("=" * 78)
print("                    CREATING FUTURE TARGETS")
print("=" * 78)

# Future Model 1 risk.
#
# shift(-1)  = 24 hours ahead
# shift(-2)  = 48 hours ahead
# shift(-3)  = 3 days ahead
# shift(-7)  = 7 days ahead
# shift(-14) = 14 days ahead

df["risk_24h"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .shift(-1)
)

df["risk_48h"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .shift(-2)
)

df["risk_3d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .shift(-3)
)

df["risk_7d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .shift(-7)
)

df["risk_14d"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .shift(-14)
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

FINAL_COLUMNS = [
    "cow_id",
    "date"
] + MODEL2_FEATURES + TARGET_COLUMNS


missing_final_columns = [
    col
    for col in FINAL_COLUMNS
    if col not in df.columns
]

if missing_final_columns:
    raise ValueError(
        "ERROR: Some final columns were not created:\n"
        + "\n".join(missing_final_columns)
    )


# ============================================================
# CREATE FINAL DATASET
# ============================================================

final_df = df[FINAL_COLUMNS].copy()


# ============================================================
# REMOVE ROWS WITHOUT REQUIRED HISTORY/FUTURE
# ============================================================

rows_before = len(final_df)

final_df = final_df.dropna(
    subset=MODEL2_FEATURES + TARGET_COLUMNS
).reset_index(drop=True)

rows_after = len(final_df)

rows_removed = rows_before - rows_after


# ============================================================
# CLIP RISK VALUES
# ============================================================

risk_columns = [
    col
    for col in final_df.columns
    if col.startswith("risk_")
    or col == "model1_risk_percent"
]

for col in risk_columns:

    if col in final_df.columns:
        final_df[col] = np.clip(
            final_df[col],
            0,
            100
        )


# ============================================================
# INTEGER FEATURES
# ============================================================

integer_features = [
    "consecutive_rising_days",
    "consecutive_falling_days",
    "high_risk_count_3d",
    "high_risk_count_7d",
    "high_risk_count_14d",
    "recent_high_risk_3d",
    "recent_high_risk_7d",
    "recent_high_risk_14d"
]

for col in integer_features:

    if col in final_df.columns:
        final_df[col] = (
            final_df[col]
            .round()
            .astype(int)
        )


# ============================================================
# FINAL SORT
# ============================================================

final_df = final_df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 78)
print("                       FINAL VALIDATION")
print("=" * 78)

print()
print(f"Rows before history/future filtering : {rows_before:,}")
print(f"Rows after filtering                 : {rows_after:,}")
print(f"Rows removed                          : {rows_removed:,}")

unique_cows = final_df["cow_id"].nunique()

print()
print(f"Unique cows                           : {unique_cows}")

rows_per_cow = final_df.groupby(
    "cow_id"
).size()

print(
    f"Minimum rows per cow                 : "
    f"{rows_per_cow.min()}"
)

print(
    f"Maximum rows per cow                 : "
    f"{rows_per_cow.max()}"
)

print(
    f"Mean rows per cow                    : "
    f"{rows_per_cow.mean():.2f}"
)


# ============================================================
# MISSING VALUES
# ============================================================

missing_cells = int(
    final_df.isna().sum().sum()
)

print()
print(f"Missing cells                         : {missing_cells}")

if missing_cells != 0:
    missing_summary = (
        final_df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print()
    print("Missing values by column:")
    print(
        missing_summary[
            missing_summary > 0
        ]
    )

    raise ValueError(
        "ERROR: Missing values remain in final dataset."
    )

print("PASS: No missing cells.")


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicate_final = final_df.duplicated(
    subset=["cow_id", "date"]
).sum()

print()
print(
    f"Duplicate cow/date records            : "
    f"{duplicate_final}"
)

if duplicate_final != 0:
    raise ValueError(
        "ERROR: Duplicate cow/date records remain."
    )

print("PASS: No duplicate cow/date records.")


# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_features = [
    col
    for col in MODEL2_FEATURES
    if col not in final_df.columns
]

missing_targets = [
    col
    for col in TARGET_COLUMNS
    if col not in final_df.columns
]

print()
print(
    f"Expected Model 2 V3 features           : "
    f"{len(MODEL2_FEATURES)}"
)

print(
    f"Actual Model 2 V3 features             : "
    f"{len([c for c in MODEL2_FEATURES if c in final_df.columns])}"
)

if missing_features:
    raise ValueError(
        "ERROR: Missing Model 2 features:\n"
        + "\n".join(missing_features)
    )

if missing_targets:
    raise ValueError(
        "ERROR: Missing target columns:\n"
        + "\n".join(missing_targets)
    )

print("PASS: All Model 2 V3 features exist.")
print("PASS: All future targets exist.")


# ============================================================
# TARGET RANGE VALIDATION
# ============================================================

print()
print("TARGET RANGES")
print("-" * 50)

for target in TARGET_COLUMNS:

    print(
        f"{target:<15} "
        f"min={final_df[target].min():.2f} "
        f"max={final_df[target].max():.2f} "
        f"mean={final_df[target].mean():.2f}"
    )

    if (
        final_df[target].min() < 0
        or final_df[target].max() > 100
    ):
        raise ValueError(
            f"ERROR: Target {target} contains "
            f"values outside 0-100."
        )


print()
print("PASS: All target values are within 0-100.")


# ============================================================
# FEATURE RANGE VALIDATION
# ============================================================

print()
print("RISK FEATURE RANGE")

risk_feature_check = [
    "risk_today",
    "risk_1d_ago",
    "risk_7d_ago",
    "risk_14d_ago",
    "risk_3day_mean",
    "risk_7day_mean",
    "risk_14day_mean"
]

for feature in risk_feature_check:

    print(
        f"{feature:<25} "
        f"min={final_df[feature].min():.2f} "
        f"max={final_df[feature].max():.2f}"
    )

    if (
        final_df[feature].min() < 0
        or final_df[feature].max() > 100
    ):
        raise ValueError(
            f"ERROR: Feature {feature} "
            f"contains values outside 0-100."
        )

print()
print("PASS: Risk feature ranges are valid.")


# ============================================================
# COW CONSISTENCY CHECK
# ============================================================

print()
print("COW HISTORY VALIDATION")

cow_counts = final_df.groupby(
    "cow_id"
)["date"].nunique()

if cow_counts.nunique() == 1:

    print(
        f"Rows per cow are consistent: "
        f"{cow_counts.iloc[0]}"
    )

else:

    print(
        "WARNING: Rows per cow are not identical."
    )

print(
    f"Date range: "
    f"{final_df['date'].min().date()} "
    f"to "
    f"{final_df['date'].max().date()}"
)


# ============================================================
# EXPECTED ROW COUNT CHECK
# ============================================================

original_cows = df["cow_id"].nunique()

expected_rows_per_cow = 60 - 14 - 14

expected_total_rows = (
    original_cows *
    expected_rows_per_cow
)

print()
print("EXPECTED ROW COUNT CHECK")
print("-" * 50)

print(
    f"Original cows                : "
    f"{original_cows}"
)

print(
    f"Expected rows per cow        : "
    f"{expected_rows_per_cow}"
)

print(
    f"Expected total rows          : "
    f"{expected_total_rows:,}"
)

print(
    f"Actual final rows            : "
    f"{len(final_df):,}"
)

if len(final_df) == expected_total_rows:

    print()
    print("PASS: Row count matches expected value.")

else:

    print()
    print(
        "WARNING: Actual row count differs from "
        "the simple expected calculation."
    )


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

print()
print("=" * 78)
print("                    LEAKAGE VALIDATION")
print("=" * 78)

# Future targets must NOT be Model 2 features.

leakage_targets = [
    target
    for target in TARGET_COLUMNS
    if target in MODEL2_FEATURES
]

if leakage_targets:

    raise ValueError(
        "ERROR: Future target leakage detected:\n"
        + "\n".join(leakage_targets)
    )

print()
print("PASS: Future target columns are not Model 2 features.")


# ============================================================
# FUTURE TARGET LOGIC CHECK
# ============================================================

print()
print("Checking future target alignment...")

sample_cow = final_df["cow_id"].iloc[0]

sample = (
    final_df[
        final_df["cow_id"] == sample_cow
    ]
    .sort_values("date")
    .head(1)
)

if len(sample) == 1:

    row = sample.iloc[0]

    print()
    print(f"Sample cow: {sample_cow}")
    print(f"Sample date: {row['date'].date()}")

    print()
    print(
        f"risk_today : "
        f"{row['risk_today']:.2f}%"
    )

    print(
        f"risk_24h   : "
        f"{row['risk_24h']:.2f}%"
    )

    print(
        f"risk_48h   : "
        f"{row['risk_48h']:.2f}%"
    )

    print(
        f"risk_3d    : "
        f"{row['risk_3d']:.2f}%"
    )

    print(
        f"risk_7d    : "
        f"{row['risk_7d']:.2f}%"
    )

    print(
        f"risk_14d   : "
        f"{row['risk_14d']:.2f}%"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 78)
print("                    DATASET SUMMARY")
print("=" * 78)

print()
print(f"Final rows                 : {len(final_df):,}")
print(f"Unique cows                : {unique_cows}")
print(f"Number of Model 2 features : {len(MODEL2_FEATURES)}")
print(f"Number of targets          : {len(TARGET_COLUMNS)}")

print()
print("MODEL 2 V3 FEATURES")
print("-" * 50)

for i, feature in enumerate(
    MODEL2_FEATURES,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )

print()
print("TARGETS")
print("-" * 50)

for target in TARGET_COLUMNS:
    print(f"- {target}")


# ============================================================
# SAVE DATASET
# ============================================================

print()
print("=" * 78)
print("                      SAVING DATASET")
print("=" * 78)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("PASS: Model 2 V3 dataset saved.")

print()
print(OUTPUT_FILE)


# ============================================================
# FINAL FILE CHECK
# ============================================================

if not os.path.exists(OUTPUT_FILE):

    raise RuntimeError(
        "ERROR: Output file was not created."
    )

saved_size = os.path.getsize(
    OUTPUT_FILE
)

print()
print(
    f"Output file size           : "
    f"{saved_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# SAMPLE OUTPUT
# ============================================================

print()
print("=" * 78)
print("                     SAMPLE OUTPUT")
print("=" * 78)

display_columns = [
    "cow_id",
    "date",
    "risk_today",
    "risk_1d_ago",
    "risk_7d_ago",
    "risk_14d_ago",
    "risk_3day_mean",
    "risk_7day_mean",
    "risk_14day_mean",
    "risk_slope_7d",
    "risk_slope_14d",
    "risk_7d",
    "risk_14d"
]

print()

print(
    final_df[
        display_columns
    ].head(10).to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 78)
print("                    PREPARATION COMPLETE")
print("=" * 78)

print()
print("MODEL 2 V3 DATASET SUCCESSFULLY CREATED.")

print()
print("Next file:")
print(
    "model2_forecasting_dataset_v3.csv"
)

print()
print("Targets:")
print("- 24 hours")
print("- 48 hours")
print("- 3 days")
print("- 7 days")
print("- 14 days")

print()
print("IMPORTANT:")
print("Do NOT train Model 2 V3 until this preparation")
print("script completes successfully and its output")
print("has been checked.")

print()
print("=" * 78)