import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

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
    "model2_forecasting_dataset_v2.csv"
)


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
# FUTURE TARGETS
# ============================================================

TARGET_COLUMNS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]


# ============================================================
# PRINT HELPER
# ============================================================

def separator(character="=", length=78):
    print(character * length)


# ============================================================
# HEADER
# ============================================================

print()
separator()
print("           MODEL 2 ENHANCED DATASET PREPARATION")
separator()

print()
print("Purpose:")
print("Create an enhanced temporal dataset for Model 2.")
print()
print("Original Model 1 dataset will NOT be modified.")
print()


# ============================================================
# CHECK FILES
# ============================================================

print("Checking required files...")
print()

if not os.path.exists(MODEL1_DATASET):
    print("ERROR: Model 1 dataset not found:")
    print(MODEL1_DATASET)
    raise SystemExit(1)

if not os.path.exists(MODEL1_FILE):
    print("ERROR: Model 1 model file not found:")
    print(MODEL1_FILE)
    raise SystemExit(1)

print("PASS: Model 1 dataset found.")
print("PASS: Model 1 model found.")
print()


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading Model 1 dataset...")

df = pd.read_csv(MODEL1_DATASET)

print("Dataset loaded successfully.")
print(
    f"Rows : {len(df):,}"
)
print(
    f"Cows : {df['cow_id'].nunique():,}"
)
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "cow_id",
    "date"
] + MODEL1_FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("ERROR: Missing columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit(1)

print("PASS: All required columns found.")
print()


# ============================================================
# DATE CONVERSION
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():

    print("ERROR: Invalid dates detected.")
    raise SystemExit(1)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# LOAD MODEL 1
# ============================================================

print("Loading trained Model 1...")

model1 = joblib.load(MODEL1_FILE)

print("Model 1 loaded successfully.")
print()


# ============================================================
# GENERATE MODEL 1 RISK
# ============================================================

print("Generating Model 1 risk percentages...")

X_model1 = df[MODEL1_FEATURES].copy()

probabilities = model1.predict_proba(
    X_model1
)[:, 1]

df["model1_risk_percent"] = (
    probabilities * 100.0
)

df["model1_risk_percent"] = (
    df["model1_risk_percent"].clip(0, 100)
)

print("Model 1 risk generation completed.")

print(
    f"Minimum risk : "
    f"{df['model1_risk_percent'].min():.2f}%"
)

print(
    f"Maximum risk : "
    f"{df['model1_risk_percent'].max():.2f}%"
)

print(
    f"Average risk : "
    f"{df['model1_risk_percent'].mean():.2f}%"
)

print()


# ============================================================
# BASIC RISK FEATURES
# ============================================================

print("Creating historical risk features...")

# Current risk
df["risk_today"] = (
    df["model1_risk_percent"]
)


# Historical lags
df["risk_1d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(1)
)

df["risk_2d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(2)
)

df["risk_3d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(3)
)

df["risk_4d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(4)
)

df["risk_5d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(5)
)

df["risk_6d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(6)
)

df["risk_7d_ago"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(7)
)

print("Historical lag features created.")
print()


# ============================================================
# RISK CHANGE FEATURES
# ============================================================

print("Creating risk change features...")

df["risk_change_1d"] = (
    df["risk_today"]
    - df["risk_1d_ago"]
)

df["risk_change_2d"] = (
    df["risk_today"]
    - df["risk_2d_ago"]
)

df["risk_change_3d"] = (
    df["risk_today"]
    - df["risk_3d_ago"]
)

print("Risk change features created.")
print()


# ============================================================
# HISTORICAL SERIES
# ============================================================

# Shift by one day first.
#
# This is important:
#
# At today's prediction time, the rolling historical
# statistics should represent previous days, not future days.
#
# risk_today itself is still available as a feature.

historical_risk = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(1)
)


# ============================================================
# 3-DAY ROLLING FEATURES
# ============================================================

print("Creating 3-day rolling statistics...")

df["risk_3day_mean"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

df["risk_3day_max"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=3,
            min_periods=3
        ).max()
    )
)

df["risk_3day_min"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=3,
            min_periods=3
        ).min()
    )
)

df["risk_std_3d"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=3,
            min_periods=3
        ).std()
    )
)

print("3-day rolling statistics created.")
print()


# ============================================================
# 7-DAY ROLLING FEATURES
# ============================================================

print("Creating 7-day rolling statistics...")

df["risk_7day_mean"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)

df["risk_7day_max"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=7
        ).max()
    )
)

df["risk_7day_min"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=7
        ).min()
    )
)

df["risk_std_7d"] = (
    historical_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=7
        ).std()
    )
)

print("7-day rolling statistics created.")
print()


# ============================================================
# EXISTING 3-DAY TREND
# ============================================================

print("Creating trend features...")

df["risk_trend_3d"] = (
    df["risk_today"]
    - df["risk_3d_ago"]
)


# ============================================================
# SLOPE FUNCTION
# ============================================================

def calculate_slope(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 2:
        return np.nan

    x = np.arange(
        len(values),
        dtype=float
    )

    slope = np.polyfit(
        x,
        values,
        1
    )[0]

    return slope


# ============================================================
# 3-DAY SLOPE
# ============================================================

print("Creating 3-day risk slope...")

df["risk_slope_3d"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=3,
            min_periods=3
        )
        .apply(
            calculate_slope,
            raw=True
        )
    )
)


# ============================================================
# 7-DAY SLOPE
# ============================================================

print("Creating 7-day risk slope...")

df["risk_slope_7d"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .apply(
            calculate_slope,
            raw=True
        )
    )
)

print("Slope features created.")
print()


# ============================================================
# CONSECUTIVE RISING DAYS
# ============================================================

print("Creating consecutive trend features...")


def rising_count(series):

    values = series.to_numpy()

    result = np.zeros(
        len(values),
        dtype=float
    )

    count = 0

    for i in range(1, len(values)):

        if values[i] > values[i - 1]:

            count += 1

        else:

            count = 0

        result[i] = count

    return pd.Series(
        result,
        index=series.index
    )


def falling_count(series):

    values = series.to_numpy()

    result = np.zeros(
        len(values),
        dtype=float
    )

    count = 0

    for i in range(1, len(values)):

        if values[i] < values[i - 1]:

            count += 1

        else:

            count = 0

        result[i] = count

    return pd.Series(
        result,
        index=series.index
    )


df["consecutive_rising_days"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ]
    .transform(
        rising_count
    )
)

df["consecutive_falling_days"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ]
    .transform(
        falling_count
    )
)

print("Consecutive trend features created.")
print()


# ============================================================
# HIGH-RISK FLAG
# ============================================================

high_risk_flag = (
    df["model1_risk_percent"] >= 50
).astype(int)


# ============================================================
# HISTORICAL HIGH-RISK FLAG
# ============================================================

historical_high_risk = (
    high_risk_flag
    .groupby(df["cow_id"])
    .shift(1)
)


# ============================================================
# HIGH-RISK COUNT - 3 DAYS
# ============================================================

print("Creating historical high-risk features...")

df["high_risk_count_3d"] = (
    historical_high_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=3,
            min_periods=3
        ).sum()
    )
)


# ============================================================
# HIGH-RISK COUNT - 7 DAYS
# ============================================================

df["high_risk_count_7d"] = (
    historical_high_risk
    .groupby(df["cow_id"])
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=7
        ).sum()
    )
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


# ============================================================
# DISTANCE FROM RECENT MAXIMUM
# ============================================================

print("Creating distance-from-peak features...")

df["distance_from_3day_max"] = (
    df["risk_today"]
    - df["risk_3day_max"]
)

df["distance_from_7day_max"] = (
    df["risk_today"]
    - df["risk_7day_max"]
)


# ============================================================
# DIFFERENCE FROM RECENT MEAN
# ============================================================

df["risk_vs_3day_mean"] = (
    df["risk_today"]
    - df["risk_3day_mean"]
)

df["risk_vs_7day_mean"] = (
    df["risk_today"]
    - df["risk_7day_mean"]
)


# ============================================================
# FUTURE TARGETS
# ============================================================

print("Creating future targets...")

df["risk_24h"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(-1)
)

df["risk_48h"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(-2)
)

df["risk_3d"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(-3)
)

df["risk_7d"] = (
    df.groupby("cow_id")[
        "model1_risk_percent"
    ].shift(-7)
)

print("Future targets created.")
print()


# ============================================================
# CHECK FEATURE CREATION
# ============================================================

print("Checking enhanced features...")

missing_features = [
    feature
    for feature in MODEL2_FEATURES
    if feature not in df.columns
]

if missing_features:

    print("ERROR: These Model 2 features were not created:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise SystemExit(1)

print(
    f"PASS: All {len(MODEL2_FEATURES)} Model 2 features created."
)

print()


# ============================================================
# CHECK TARGET CREATION
# ============================================================

missing_targets = [
    target
    for target in TARGET_COLUMNS
    if target not in df.columns
]

if missing_targets:

    print("ERROR: These targets were not created:")

    for target in missing_targets:
        print(f"  - {target}")

    raise SystemExit(1)

print(
    f"PASS: All {len(TARGET_COLUMNS)} future targets created."
)

print()


# ============================================================
# REMOVE ROWS WITH INSUFFICIENT HISTORY/FUTURE
# ============================================================

print("Removing rows without enough historical/future data...")

rows_before = len(df)

df = df.dropna(
    subset=MODEL2_FEATURES + TARGET_COLUMNS
).reset_index(drop=True)

rows_after = len(df)

print(
    f"Rows before filtering : {rows_before:,}"
)

print(
    f"Rows after filtering  : {rows_after:,}"
)

print(
    f"Rows removed          : "
    f"{rows_before - rows_after:,}"
)

print()


# ============================================================
# VALIDATION 1 - COW COUNT
# ============================================================

print()
separator()
print("                    DATASET VALIDATION")
separator()

print()

unique_cows = (
    df["cow_id"].nunique()
)

print(
    f"Unique cows: {unique_cows:,}"
)


# ============================================================
# VALIDATION 2 - ROWS PER COW
# ============================================================

rows_per_cow = (
    df.groupby("cow_id")
    .size()
)

print(
    f"Minimum rows per cow: "
    f"{rows_per_cow.min()}"
)

print(
    f"Maximum rows per cow: "
    f"{rows_per_cow.max()}"
)

print(
    f"Average rows per cow: "
    f"{rows_per_cow.mean():.2f}"
)


# ============================================================
# VALIDATION 3 - DUPLICATES
# ============================================================

duplicate_rows = (
    df.duplicated(
        subset=[
            "cow_id",
            "date"
        ]
    ).sum()
)

print(
    f"Duplicate cow/date rows: "
    f"{duplicate_rows}"
)


# ============================================================
# VALIDATION 4 - MISSING VALUES
# ============================================================

missing_cells = (
    df.isna().sum().sum()
)

print(
    f"Missing cells: "
    f"{missing_cells}"
)


# ============================================================
# VALIDATION 5 - DATE RANGE
# ============================================================

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# VALIDATION 6 - RISK RANGE
# ============================================================

print()
separator("-")
print("                    RISK RANGE CHECK")
separator("-")

risk_columns = [
    "model1_risk_percent",
    "risk_today",
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]

risk_check_passed = True

for column in risk_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    if minimum < 0 or maximum > 100:

        print(
            f"FAIL: {column} "
            f"range={minimum:.2f} to {maximum:.2f}"
        )

        risk_check_passed = False

    else:

        print(
            f"PASS: {column} "
            f"range={minimum:.2f} to {maximum:.2f}"
        )


# ============================================================
# VALIDATION 7 - LEAKAGE
# ============================================================

print()
separator("-")
print("                    LEAKAGE CHECK")
separator("-")

future_targets_in_features = set(
    MODEL2_FEATURES
).intersection(
    set(TARGET_COLUMNS)
)

if future_targets_in_features:

    print("FAIL: Future targets found in Model 2 features:")

    for column in future_targets_in_features:
        print(f"  - {column}")

    raise SystemExit(1)

else:

    print(
        "PASS: No future target columns are used as features."
    )


# ============================================================
# VALIDATION 8 - TARGET CORRELATION CHECK
# ============================================================

print()
separator("-")
print("                TARGET / FEATURE CHECK")
separator("-")

print()

for target in TARGET_COLUMNS:

    correlation = (
        df["risk_today"]
        .corr(df[target])
    )

    print(
        f"risk_today vs {target}: "
        f"{correlation:.4f}"
    )


# ============================================================
# TARGET STATISTICS
# ============================================================

print()
separator("-")
print("                    TARGET STATISTICS")
separator("-")

for target in TARGET_COLUMNS:

    print()
    print(target)

    print(
        f"  Mean   : {df[target].mean():.2f}%"
    )

    print(
        f"  Std    : {df[target].std():.2f}%"
    )

    print(
        f"  Median : {df[target].median():.2f}%"
    )

    print(
        f"  Min    : {df[target].min():.2f}%"
    )

    print(
        f"  Max    : {df[target].max():.2f}%"
    )


# ============================================================
# MODEL 2 FEATURE LIST
# ============================================================

print()
separator("-")
print("                    MODEL 2 FEATURES")
separator("-")

for number, feature in enumerate(
    MODEL2_FEATURES,
    start=1
):

    print(
        f"{number:2d}. {feature}"
    )


# ============================================================
# SAMPLE DATA
# ============================================================

print()
separator()
print("                     SAMPLE DATA")
separator()

sample_columns = [
    "cow_id",
    "date",
    "risk_today",
    "risk_1d_ago",
    "risk_2d_ago",
    "risk_3d_ago",
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
    "risk_vs_7day_mean",
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]

print()

print(
    df[sample_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# ROUND NUMERIC COLUMNS
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=[np.number]
    ).columns
)

df[numeric_columns] = (
    df[numeric_columns]
    .round(4)
)


# ============================================================
# SAVE
# ============================================================

print()
separator()
print("                    SAVING DATASET")
separator()

print()

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Enhanced dataset saved successfully:"
)

print(
    OUTPUT_FILE
)

print()


# ============================================================
# FINAL VALIDATION SUMMARY
# ============================================================

print()
separator()
print("                  FINAL VALIDATION SUMMARY")
separator()

print()

checks = {
    "Rows > 0":
        len(df) > 0,

    "500 cows preserved":
        unique_cows == 500,

    "No missing cells":
        missing_cells == 0,

    "No duplicate cow/date":
        duplicate_rows == 0,

    "All Model 2 features exist":
        len(missing_features) == 0,

    "All future targets exist":
        len(missing_targets) == 0,

    "No target leakage":
        len(future_targets_in_features) == 0,

    "Risk values within 0-100":
        risk_check_passed
}


all_passed = True

for check_name, passed in checks.items():

    if passed:

        print(
            f"PASS: {check_name}"
        )

    else:

        print(
            f"FAIL: {check_name}"
        )

        all_passed = False


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

separator()

if all_passed:

    print(
        "ALL VALIDATION CHECKS PASSED."
    )

    print()
    print(
        "Enhanced Model 2 dataset is ready for training."
    )

else:

    print(
        "WARNING: SOME VALIDATION CHECKS FAILED."
    )

separator()

print()

print(
    "Output file:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Original Model 1 dataset was NOT modified."
)

print()

print(
    "NEXT STEP:"
)

print(
    "Train Model 2 using model2_forecasting_dataset_v2.csv"
)

print()

