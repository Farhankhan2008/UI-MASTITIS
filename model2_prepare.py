import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# MODEL 2 DATA PREPARATION
# ============================================================
# Purpose:
#   1. Load the existing Model 1 dataset
#   2. Load the trained Model 1
#   3. Generate daily mastitis risk percentages using Model 1
#   4. Create historical risk features
#   5. Create future risk targets:
#        - 24 hours
#        - 48 hours
#        - 3 days
#        - 7 days
#   6. Save the prepared dataset for Model 2
#
# IMPORTANT:
#   - The original CSV is NOT modified.
#   - Model 2 uses Model 1 risk history.
#   - mastitis_today is NOT used as a Model 2 input.
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

DATASET_FILE = "synthetic_mastitis_model1_dataset_v2.csv"
MODEL1_FILE = "mastitis_model1.pkl"
OUTPUT_FILE = "model2_forecasting_dataset.csv"


# ============================================================
# MODEL 1 FEATURES
# ============================================================
# These MUST match the features used when Model 1 was trained.

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
# REQUIRED ORIGINAL DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "cow_id",
    "date",
    "milk_yield_liters",
    "milk_conductivity_ms_cm",
    "cow_activity",
    "environment_temperature_c",
    "milk_temperature_c",
    "humidity_percent",
    "previous_mastitis",
    "days_since_last_mastitis",
    "mastitis_today"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("              MODEL 2 DATA PREPARATION")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n1. CHECKING REQUIRED FILES")
print("-" * 70)

if not os.path.exists(DATASET_FILE):
    print(f"ERROR: Dataset file not found:")
    print(f"       {DATASET_FILE}")
    print("\nMake sure the CSV is in the same folder as this script.")
    raise SystemExit(1)

if not os.path.exists(MODEL1_FILE):
    print(f"ERROR: Model 1 file not found:")
    print(f"       {MODEL1_FILE}")
    print("\nMake sure mastitis_model1.pkl is in the same folder.")
    raise SystemExit(1)

print(f"Dataset found : {DATASET_FILE}")
print(f"Model 1 found: {MODEL1_FILE}")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n2. LOADING DATASET")
print("-" * 70)

df = pd.read_csv(DATASET_FILE)

print(f"Dataset loaded successfully!")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

print("\n3. VALIDATING DATASET COLUMNS")
print("-" * 70)

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    print("ERROR: Required columns are missing:")
    for column in missing_columns:
        print(f"  - {column}")
    raise SystemExit(1)

print("All required columns are present.")


# ============================================================
# 4. CONVERT DATE
# ============================================================

print("\n4. PROCESSING DATES")
print("-" * 70)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    print("ERROR: Invalid date values found.")
    raise SystemExit(1)

print("Date conversion successful.")


# ============================================================
# 5. SORT DATA
# ============================================================

print("\n5. SORTING COW HISTORY")
print("-" * 70)

df = df.sort_values(
    by=["cow_id", "date"]
).reset_index(drop=True)

print("Data sorted by cow_id and date.")


# ============================================================
# 6. CHECK COW HISTORY
# ============================================================

print("\n6. CHECKING COW HISTORY")
print("-" * 70)

number_of_cows = df["cow_id"].nunique()

print(f"Unique cows: {number_of_cows:,}")

days_per_cow = df.groupby("cow_id")["date"].nunique()

print(f"Minimum days per cow: {days_per_cow.min()}")
print(f"Maximum days per cow: {days_per_cow.max()}")
print(f"Average days per cow: {days_per_cow.mean():.2f}")


# ============================================================
# 7. CHECK DUPLICATE COW/DAY RECORDS
# ============================================================

duplicate_count = df.duplicated(
    subset=["cow_id", "date"]
).sum()

if duplicate_count > 0:
    print(f"\nERROR: Found {duplicate_count} duplicate cow/date records.")
    raise SystemExit(1)

print("No duplicate cow/date combinations found.")


# ============================================================
# 8. LOAD MODEL 1
# ============================================================

print("\n7. LOADING TRAINED MODEL 1")
print("-" * 70)

model1 = joblib.load(MODEL1_FILE)

print("Model 1 loaded successfully.")
print(f"Model type: {type(model1).__name__}")


# ============================================================
# 9. VALIDATE MODEL 1 FEATURES
# ============================================================

print("\n8. VALIDATING MODEL 1 FEATURES")
print("-" * 70)

print("Features supplied to Model 1:")

for feature in MODEL1_FEATURES:
    print(f"  - {feature}")


# ============================================================
# 10. GENERATE MODEL 1 RISK
# ============================================================

print("\n9. GENERATING MODEL 1 RISK PERCENTAGES")
print("-" * 70)

X_model1 = df[MODEL1_FEATURES]


# Make sure Model 1 contains class 1
if not hasattr(model1, "classes_"):
    print("ERROR: Loaded Model 1 does not contain class information.")
    raise SystemExit(1)

model_classes = list(model1.classes_)

if 1 not in model_classes:
    print("ERROR: Model 1 does not contain mastitis class = 1.")
    print(f"Available classes: {model_classes}")
    raise SystemExit(1)

mastitis_class_index = model_classes.index(1)

print(f"Model classes: {model_classes}")
print(f"Mastitis class index: {mastitis_class_index}")

print("\nGenerating predictions...")

risk_probability = model1.predict_proba(X_model1)[
    :, mastitis_class_index
]

df["model1_risk_percent"] = risk_probability * 100

# Keep values within 0-100
df["model1_risk_percent"] = df[
    "model1_risk_percent"
].clip(0, 100)

# Round for easier viewing
df["model1_risk_percent"] = df[
    "model1_risk_percent"
].round(2)

print("Model 1 risk generation completed.")


# ============================================================
# 11. BASIC RISK VALIDATION
# ============================================================

print("\n10. MODEL 1 RISK VALIDATION")
print("-" * 70)

print(
    f"Minimum risk : {df['model1_risk_percent'].min():.2f}%"
)

print(
    f"Maximum risk : {df['model1_risk_percent'].max():.2f}%"
)

print(
    f"Average risk : {df['model1_risk_percent'].mean():.2f}%"
)

missing_risk = df["model1_risk_percent"].isna().sum()

print(f"Missing risk values: {missing_risk}")

if missing_risk > 0:
    print("ERROR: Model 1 generated missing risk values.")
    raise SystemExit(1)


# ============================================================
# 12. CREATE HISTORICAL RISK FEATURES
# ============================================================
#
# These features represent information that would already
# be available at the current forecast date.
#
# risk_today     = Model 1 risk today
# risk_1d_ago    = Model 1 risk yesterday
# risk_2d_ago    = Model 1 risk 2 days ago
# risk_3d_ago    = Model 1 risk 3 days ago
# risk_7d_ago    = Model 1 risk 7 days ago
#
# IMPORTANT:
# No future risk is used here.
# ============================================================

print("\n11. CREATING HISTORICAL RISK FEATURES")
print("-" * 70)

grouped_risk = df.groupby("cow_id")["model1_risk_percent"]

df["risk_today"] = df["model1_risk_percent"]

df["risk_1d_ago"] = grouped_risk.shift(1)

df["risk_2d_ago"] = grouped_risk.shift(2)

df["risk_3d_ago"] = grouped_risk.shift(3)

df["risk_7d_ago"] = grouped_risk.shift(7)

print("Historical risk features created.")


# ============================================================
# 13. CREATE ROLLING FEATURES
# ============================================================
#
# Rolling statistics are calculated using CURRENT and
# PREVIOUS risk values only.
#
# No future information is allowed.
# ============================================================

print("\n12. CREATING ROLLING RISK FEATURES")
print("-" * 70)


# Previous 3 days
df["risk_3day_mean"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

df["risk_3day_max"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).max()
    )
)

df["risk_3day_min"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).min()
    )
)


# Previous 7 days
df["risk_7day_mean"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)

df["risk_7day_max"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=7,
            min_periods=7
        ).max()
    )
)

df["risk_7day_min"] = (
    df.groupby("cow_id")["model1_risk_percent"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=7,
            min_periods=7
        ).min()
    )
)

print("Rolling risk features created.")


# ============================================================
# 14. CREATE TREND FEATURE
# ============================================================
#
# Simple trend:
#
#     today's risk - risk 3 days ago
#
# Positive value  -> risk increasing
# Negative value  -> risk decreasing
#
# ============================================================

print("\n13. CREATING RISK TREND FEATURE")
print("-" * 70)

df["risk_trend_3d"] = (
    df["risk_today"] - df["risk_3d_ago"]
)

df["risk_trend_3d"] = df["risk_trend_3d"].round(2)

print("3-day risk trend created.")


# ============================================================
# 15. CREATE FUTURE TARGETS
# ============================================================
#
# Model 2 targets:
#
#   risk_24h = risk after 1 day
#   risk_48h = risk after 2 days
#   risk_3d  = risk after 3 days
#   risk_7d  = risk after 7 days
#
# These are targets, NOT input features.
#
# shift(-1) = next day
# shift(-2) = two days later
# shift(-3) = three days later
# shift(-7) = seven days later
#
# ============================================================

print("\n14. CREATING FUTURE FORECAST TARGETS")
print("-" * 70)

grouped_risk = df.groupby("cow_id")["model1_risk_percent"]

df["risk_24h"] = grouped_risk.shift(-1)

df["risk_48h"] = grouped_risk.shift(-2)

df["risk_3d"] = grouped_risk.shift(-3)

df["risk_7d"] = grouped_risk.shift(-7)

print("Future targets created.")


# ============================================================
# 16. ROUND GENERATED VALUES
# ============================================================

risk_columns = [
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
    "risk_trend_3d",
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]

for column in risk_columns:
    df[column] = df[column].round(2)


# ============================================================
# 17. REMOVE ROWS WITHOUT COMPLETE HISTORY/FUTURE
# ============================================================
#
# Model 2 needs:
#
#   7 days of historical information
#   AND
#   7 days of future information
#
# Therefore the first 7 days and final 7 days of each cow's
# history cannot be used for the complete 7-day forecasting
# dataset.
#
# 60 days per cow:
#
#   60 - 7 historical - 7 future = 46 usable rows
#
# 500 cows:
#
#   500 × 46 = 23,000 rows
#
# ============================================================

print("\n15. PREPARING FINAL FORECASTING RECORDS")
print("-" * 70)

HISTORICAL_FEATURES = [
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

TARGET_COLUMNS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]

columns_needed = (
    HISTORICAL_FEATURES +
    TARGET_COLUMNS
)

before_rows = len(df)

df_model2 = df.dropna(
    subset=columns_needed
).copy()

after_rows = len(df_model2)

rows_removed = before_rows - after_rows

print(f"Rows before filtering : {before_rows:,}")
print(f"Rows after filtering  : {after_rows:,}")
print(f"Rows removed          : {rows_removed:,}")


# ============================================================
# 18. VALIDATE FINAL DATASET
# ============================================================

print("\n16. FINAL DATASET VALIDATION")
print("-" * 70)

final_cows = df_model2["cow_id"].nunique()

print(f"Rows                  : {len(df_model2):,}")
print(f"Unique cows           : {final_cows:,}")

print(
    f"Date range            : "
    f"{df_model2['date'].min().date()} "
    f"to "
    f"{df_model2['date'].max().date()}"
)

missing_cells = df_model2.isna().sum().sum()

print(f"Missing cells         : {missing_cells}")

duplicate_final = df_model2.duplicated(
    subset=["cow_id", "date"]
).sum()

print(f"Duplicate cow/date    : {duplicate_final}")


# ============================================================
# 19. CHECK EXPECTED ROW COUNT
# ============================================================

days_per_cow_final = (
    df_model2.groupby("cow_id")["date"]
    .nunique()
)

print(
    f"Rows per cow          : "
    f"min={days_per_cow_final.min()}, "
    f"max={days_per_cow_final.max()}"
)

expected_rows = None

if (
    number_of_cows == 500
    and days_per_cow.min() == 60
    and days_per_cow.max() == 60
):
    expected_rows = 500 * 46

    print(f"Expected rows         : {expected_rows:,}")

    if len(df_model2) == expected_rows:
        print("Row count validation  : PASSED")
    else:
        print(
            "WARNING: Row count differs from expected value."
        )


# ============================================================
# 20. SELECT FINAL COLUMNS
# ============================================================
#
# Keep:
#
#   cow_id
#   date
#
# for identifying the forecasting record.
#
# Model 2 training later will NOT use cow_id/date as
# numerical prediction features.
#
# ============================================================

FINAL_COLUMNS = [
    "cow_id",
    "date",

    # Current and historical Model 1 risk
    "risk_today",
    "risk_1d_ago",
    "risk_2d_ago",
    "risk_3d_ago",
    "risk_7d_ago",

    # Rolling statistics
    "risk_3day_mean",
    "risk_3day_max",
    "risk_3day_min",

    "risk_7day_mean",
    "risk_7day_max",
    "risk_7day_min",

    # Trend
    "risk_trend_3d",

    # Future targets
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d"
]

df_model2 = df_model2[FINAL_COLUMNS]


# ============================================================
# 21. SORT FINAL DATASET
# ============================================================

df_model2 = df_model2.sort_values(
    by=["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# 22. SAVE MODEL 2 DATASET
# ============================================================

print("\n17. SAVING MODEL 2 DATASET")
print("-" * 70)

df_model2.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Model 2 dataset saved successfully!")
print(f"File: {OUTPUT_FILE}")


# ============================================================
# 23. DISPLAY FINAL COLUMNS
# ============================================================

print("\n18. FINAL DATASET COLUMNS")
print("-" * 70)

for index, column in enumerate(
    df_model2.columns,
    start=1
):
    print(f"{index:2}. {column}")


# ============================================================
# 24. DISPLAY SAMPLE DATA
# ============================================================

print("\n19. SAMPLE RECORDS")
print("-" * 70)

print(
    df_model2.head(10).to_string(index=False)
)


# ============================================================
# 25. TARGET STATISTICS
# ============================================================

print("\n20. FUTURE TARGET STATISTICS")
print("-" * 70)

print(
    df_model2[
        TARGET_COLUMNS
    ].describe().round(2).to_string()
)


# ============================================================
# 26. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("              MODEL 2 PREPARATION COMPLETE")
print("=" * 70)

print(f"""
Original dataset:
    {DATASET_FILE}

Model 1:
    {MODEL1_FILE}

Model 2 dataset:
    {OUTPUT_FILE}

Original rows:
    {len(df):,}

Final Model 2 rows:
    {len(df_model2):,}

Unique cows:
    {df_model2["cow_id"].nunique():,}

Historical risk features:
    {len(HISTORICAL_FEATURES)}

Future targets:
    24 hours
    48 hours
    3 days
    7 days

Missing cells:
    {df_model2.isna().sum().sum()}

Duplicate cow/date records:
    {df_model2.duplicated(subset=["cow_id", "date"]).sum()}
""")

print("=" * 70)
print("DO NOT TRAIN MODEL 2 YET.")
print("Send the COMPLETE terminal output to Nat first.")
print("=" * 70)

