import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

DATASET_FILE = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("          MODEL 2 V3 LONG-TERM TEMPORAL ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print(f"Rows        : {len(df)}")
print(f"Columns     : {len(df.columns)}")
print(f"Unique cows : {df['cow_id'].nunique()}")


# ============================================================
# PREPARE DATA
# ============================================================

required_columns = [
    "cow_id",
    "date",
    "risk_today"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# RISK HISTORY COLUMNS
# ============================================================

risk_lag_columns = [
    col for col in df.columns
    if col.startswith("risk_")
    and col.endswith("_ago")
]

print("\nRisk history columns found:")
print(len(risk_lag_columns))

print(risk_lag_columns)


# ============================================================
# CURRENT RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CURRENT RISK DISTRIBUTION")
print("=" * 70)

risk = df["risk_today"]

print(f"Minimum risk : {risk.min():.2f}%")
print(f"Maximum risk : {risk.max():.2f}%")
print(f"Mean risk    : {risk.mean():.2f}%")
print(f"Median risk  : {risk.median():.2f}%")

current_high = risk >= 50

print("\nCurrent high-risk records (>=50%):")

print(
    f"{current_high.sum()} / {len(df)} "
    f"({current_high.mean() * 100:.2f}%)"
)


# ============================================================
# FUTURE HIGH-RISK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FUTURE HIGH-RISK ANALYSIS")
print("=" * 70)

results = []

for horizon in [1, 2, 3, 5, 7, 10, 14]:

    future_max_values = []
    current_risks = []

    for cow_id, cow_df in df.groupby("cow_id"):

        cow_df = cow_df.sort_values(
            "date"
        ).reset_index(drop=True)

        risks = cow_df["risk_today"].values

        for i in range(len(risks)):

            future_start = i + 1

            future_end = min(
                i + horizon + 1,
                len(risks)
            )

            if future_start >= len(risks):
                continue

            future_values = risks[
                future_start:future_end
            ]

            if len(future_values) == 0:
                continue

            future_max = np.max(
                future_values
            )

            future_max_values.append(
                future_max
            )

            current_risks.append(
                risks[i]
            )

    future_max_values = np.array(
        future_max_values
    )

    current_risks = np.array(
        current_risks
    )

    high_risk_future = (
        future_max_values >= 50
    )

    percentage = (
        high_risk_future.mean() * 100
    )

    print(
        f"\nWithin next {horizon:2d} day(s):"
    )

    print(
        f"  Records analyzed        : "
        f"{len(future_max_values)}"
    )

    print(
        f"  Future high-risk cases  : "
        f"{high_risk_future.sum()}"
    )

    print(
        f"  Future high-risk rate   : "
        f"{percentage:.2f}%"
    )

    print(
        f"  Mean current risk       : "
        f"{current_risks.mean():.2f}%"
    )

    print(
        f"  Mean future maximum     : "
        f"{future_max_values.mean():.2f}%"
    )

    results.append({
        "Horizon_days": horizon,
        "Records": len(future_max_values),
        "Future_high_risk_cases":
            int(high_risk_future.sum()),
        "Future_high_risk_percent":
            percentage,
        "Mean_current_risk":
            current_risks.mean(),
        "Mean_future_max":
            future_max_values.mean()
    })


# ============================================================
# CURRENT RISK VS FUTURE HIGH-RISK
# ============================================================

print("\n" + "=" * 70)
print("CURRENT RISK -> FUTURE HIGH-RISK RELATIONSHIP")
print("=" * 70)


for horizon in [7, 14]:

    print(
        f"\n--- HIGH-RISK WITHIN {horizon} DAYS ---"
    )

    records = []

    for cow_id, cow_df in df.groupby("cow_id"):

        cow_df = cow_df.sort_values(
            "date"
        ).reset_index(drop=True)

        risks = cow_df["risk_today"].values

        for i in range(len(risks)):

            future_start = i + 1

            future_end = min(
                i + horizon + 1,
                len(risks)
            )

            if future_start >= len(risks):
                continue

            future_values = risks[
                future_start:future_end
            ]

            if len(future_values) == 0:
                continue

            future_high = (
                np.max(future_values) >= 50
            )

            records.append({
                "current_risk":
                    risks[i],
                "future_high":
                    future_high
            })

    temp = pd.DataFrame(records)

    bins = [
        -np.inf,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        np.inf
    ]

    labels = [
        "<=10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        ">90"
    ]

    temp["risk_group"] = pd.cut(
        temp["current_risk"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    grouped = (
        temp
        .groupby(
            "risk_group",
            observed=False
        )["future_high"]
        .agg(
            Records="count",
            Future_High_Risk_Count="sum",
            Future_High_Risk_Rate="mean"
        )
    )

    grouped["Future_High_Risk_Rate"] *= 100

    print(grouped.to_string())


# ============================================================
# TREND RELATIONSHIP
# ============================================================

print("\n" + "=" * 70)
print("RISK TREND -> FUTURE HIGH-RISK RELATIONSHIP")
print("=" * 70)

trend_columns = [
    "risk_change_1d",
    "risk_change_2d",
    "risk_change_3d",
    "risk_change_7d",
    "risk_change_14d"
]

available_trends = [
    c for c in trend_columns
    if c in df.columns
]

print("\nAvailable trend features:")
print(available_trends)


for horizon in [7, 14]:

    print(
        f"\n--- TREND ANALYSIS: "
        f"HIGH-RISK WITHIN {horizon} DAYS ---"
    )

    records = []

    for cow_id, cow_df in df.groupby("cow_id"):

        cow_df = cow_df.sort_values(
            "date"
        ).reset_index(drop=True)

        risks = cow_df["risk_today"].values

        for i in range(len(risks)):

            future_start = i + 1

            future_end = min(
                i + horizon + 1,
                len(risks)
            )

            if future_start >= len(risks):
                continue

            future_values = risks[
                future_start:future_end
            ]

            if len(future_values) == 0:
                continue

            row = {
                "future_high":
                    np.max(future_values) >= 50
            }

            for col in available_trends:
                row[col] = cow_df.iloc[i][col]

            records.append(row)

    trend_df = pd.DataFrame(records)

    for col in available_trends:

        temp = trend_df[
            [col, "future_high"]
        ].dropna()

        if len(temp) < 2:
            continue

        try:

            correlation = temp[col].corr(
                temp["future_high"].astype(int)
            )

            print(
                f"{col:25s}: "
                f"{correlation:.4f}"
            )

        except Exception:
            pass


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    BASE_DIR,
    "model2_v3_longterm_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nAnalysis saved to:\n{output_file}"
)

print("\nIMPORTANT:")
print("Model 1 remains frozen.")
print("Model 2 V3 remains unchanged.")
print("No model files were modified.")