import pandas as pd
import os

FILE = r"D:\Mastitis\herd_predictions.csv"

df = pd.read_csv(FILE)

print("=" * 80)
print("HERD PREDICTIONS VALIDATION")
print("=" * 80)

print(f"\nRows: {len(df)}")
print("\nColumns:")
print(list(df.columns))

# ------------------------------------------------------------
# Find columns safely
# ------------------------------------------------------------

def find_col(possible):
    for name in possible:
        if name in df.columns:
            return name
    return None

risk_col = find_col([
    "current_risk",
    "risk_today",
    "model1_risk"
])

risk7_col = find_col([
    "7d_probability",
    "risk_7d",
    "7_day_probability"
])

risk14_col = find_col([
    "14d_probability",
    "risk_14d",
    "14_day_probability"
])

category_col = find_col([
    "risk_category",
    "category"
])

alert_col = find_col([
    "alert_level",
    "alert"
])

print("\nDetected columns:")
print("Current risk :", risk_col)
print("7-day risk   :", risk7_col)
print("14-day risk  :", risk14_col)
print("Category     :", category_col)
print("Alert        :", alert_col)

# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

for label, col in [
    ("CURRENT RISK", risk_col),
    ("7-DAY PROBABILITY", risk7_col),
    ("14-DAY PROBABILITY", risk14_col)
]:
    if col:
        s = pd.to_numeric(df[col], errors="coerce")

        print("\n" + "-" * 80)
        print(label)
        print("-" * 80)

        print(f"Mean   : {s.mean():.2f}%")
        print(f"Median : {s.median():.2f}%")
        print(f"Min    : {s.min():.2f}%")
        print(f"Max    : {s.max():.2f}%")
        print(f"Std    : {s.std():.2f}%")

        print("\nDistribution:")
        print(f"0-10%      : {(s <= 10).sum()}")
        print(f"10-25%     : {((s > 10) & (s <= 25)).sum()}")
        print(f"25-50%     : {((s > 25) & (s <= 50)).sum()}")
        print(f">50%       : {(s > 50).sum()}")

# ------------------------------------------------------------
# Critical cross-check
# ------------------------------------------------------------

if risk_col and risk7_col and risk14_col:

    current = pd.to_numeric(df[risk_col], errors="coerce")
    risk7 = pd.to_numeric(df[risk7_col], errors="coerce")
    risk14 = pd.to_numeric(df[risk14_col], errors="coerce")

    print("\n" + "=" * 80)
    print("CURRENT RISK vs FUTURE RISK CROSS-CHECK")
    print("=" * 80)

    cases = {
        "Current <=10%, 7d >=20%":
            (current <= 10) & (risk7 >= 20),

        "Current <=10%, 14d >=30%":
            (current <= 10) & (risk14 >= 30),

        "Current <=10%, 7d >=20%, 14d >=30%":
            (current <= 10) & (risk7 >= 20) & (risk14 >= 30),

        "Current <=25%, 7d >=20%":
            (current <= 25) & (risk7 >= 20),

        "Current <=25%, 14d >=30%":
            (current <= 25) & (risk14 >= 30),

        "Current <=25%, both warnings":
            (current <= 25) & (risk7 >= 20) & (risk14 >= 30),
    }

    for name, mask in cases.items():
        print(f"{name:<50}: {mask.sum()}")

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("CORRELATION")
    print("-" * 80)

    print(f"Current vs 7d  : {current.corr(risk7):.3f}")
    print(f"Current vs 14d : {current.corr(risk14):.3f}")
    print(f"7d vs 14d      : {risk7.corr(risk14):.3f}")

# ------------------------------------------------------------
# Category distribution
# ------------------------------------------------------------

if category_col:
    print("\n" + "=" * 80)
    print("RISK CATEGORY DISTRIBUTION")
    print("=" * 80)
    print(df[category_col].value_counts())

# ------------------------------------------------------------
# Alert distribution
# ------------------------------------------------------------

if alert_col:
    print("\n" + "=" * 80)
    print("ALERT DISTRIBUTION")
    print("=" * 80)
    print(df[alert_col].value_counts())

# ------------------------------------------------------------
# Highest future-risk cows
# ------------------------------------------------------------

if risk_col and risk7_col and risk14_col:

    print("\n" + "=" * 80)
    print("TOP 15 FUTURE-RISK COWS")
    print("=" * 80)

    temp = df.copy()

    temp["_current"] = pd.to_numeric(temp[risk_col], errors="coerce")
    temp["_7d"] = pd.to_numeric(temp[risk7_col], errors="coerce")
    temp["_14d"] = pd.to_numeric(temp[risk14_col], errors="coerce")

    display_cols = []

    for c in ["cow_id", "cow", "id"]:
        if c in temp.columns:
            display_cols.append(c)
            break

    display_cols += [risk_col, risk7_col, risk14_col]

    if category_col:
        display_cols.append(category_col)

    if alert_col:
        display_cols.append(alert_col)

    print(
        temp.sort_values("_14d", ascending=False)
        [display_cols]
        .head(15)
        .to_string(index=False)
    )

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)