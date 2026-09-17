import pandas as pd
import numpy as np
import json

# Fixed random seed for reproducible non-sequential cow ID risk distribution
np.random.seed(42)

TOTAL_COWS = 100
TARGET_HIGH = 18
TARGET_MOD = 10
TARGET_LOW = 6
TARGET_HEALTHY = TOTAL_COWS - (TARGET_HIGH + TARGET_MOD + TARGET_LOW) # 66 Healthy

cow_ids = [f"COW_{i:03d}" for i in range(1, TOTAL_COWS + 1)]

# Randomly shuffle cow IDs array once with fixed seed 42
np.random.shuffle(cow_ids)

high_risk_ids = set(cow_ids[:TARGET_HIGH])
mod_risk_ids = set(cow_ids[TARGET_HIGH : TARGET_HIGH + TARGET_MOD])
low_risk_ids = set(cow_ids[TARGET_HIGH + TARGET_MOD : TARGET_HIGH + TARGET_MOD + TARGET_LOW])
healthy_ids = set(cow_ids[TARGET_HIGH + TARGET_MOD + TARGET_LOW:])

records = []

for i in range(1, TOTAL_COWS + 1):
    c_id = f"COW_{i:03d}"

    if c_id in high_risk_ids:
        cat = "High Risk"
        current_risk = float(np.random.uniform(72.0, 88.0))
        cond = float(np.random.uniform(6.9, 8.5))
        act = int(np.random.randint(180, 290))
        yield_l = float(np.random.uniform(10.0, 16.0))
        temp_c = float(np.random.uniform(39.2, 40.5))

        # Progressive distinct forecast projections without identical cap collisions
        r_24h = min(98.5, current_risk + np.random.uniform(1.5, 3.5))
        r_48h = min(98.5, r_24h + np.random.uniform(2.0, 4.0))
        r_3d = min(98.5, r_48h + np.random.uniform(2.5, 4.5))
        r_7d = min(98.5, r_3d + np.random.uniform(2.0, 3.5))
        r_14d = min(98.5, r_7d + np.random.uniform(1.5, 3.0))

    elif c_id in mod_risk_ids:
        cat = "Moderate Risk"
        current_risk = float(np.random.uniform(42.0, 68.0))
        cond = float(np.random.uniform(5.8, 6.7))
        act = int(np.random.randint(300, 390))
        yield_l = float(np.random.uniform(17.0, 22.0))
        temp_c = float(np.random.uniform(38.6, 39.1))

        r_24h = current_risk + np.random.uniform(1.0, 3.5)
        r_48h = r_24h + np.random.uniform(1.5, 4.0)
        r_3d = r_48h + np.random.uniform(2.0, 4.5)
        r_7d = r_3d + np.random.uniform(2.0, 4.0)
        r_14d = r_7d + np.random.uniform(1.5, 3.5)

    elif c_id in low_risk_ids:
        cat = "Low Risk"
        current_risk = float(np.random.uniform(22.0, 38.0))
        cond = float(np.random.uniform(5.1, 5.7))
        act = int(np.random.randint(400, 480))
        yield_l = float(np.random.uniform(21.0, 25.0))
        temp_c = float(np.random.uniform(38.2, 38.5))

        r_24h = max(5.0, current_risk + np.random.uniform(-1.5, 2.5))
        r_48h = max(5.0, r_24h + np.random.uniform(-1.0, 3.0))
        r_3d = max(5.0, r_48h + np.random.uniform(-1.0, 3.0))
        r_7d = max(5.0, r_3d + np.random.uniform(-0.5, 3.5))
        r_14d = max(5.0, r_7d + np.random.uniform(-0.5, 3.5))

    else:
        cat = "Healthy"
        current_risk = float(np.random.uniform(3.0, 18.0))
        cond = float(np.random.uniform(4.2, 5.0))
        act = int(np.random.randint(490, 650))
        yield_l = float(np.random.uniform(24.0, 32.0))
        temp_c = float(np.random.uniform(37.8, 38.3))

        r_24h = max(2.0, current_risk + np.random.uniform(-1.0, 1.5))
        r_48h = max(2.0, r_24h + np.random.uniform(-1.0, 1.5))
        r_3d = max(2.0, r_48h + np.random.uniform(-1.0, 2.0))
        r_7d = max(2.0, r_3d + np.random.uniform(-1.0, 2.0))
        r_14d = max(2.0, r_7d + np.random.uniform(-1.0, 2.5))

    # Generate 14-day history trend
    history = []
    base_hist = max(4.0, current_risk - np.random.uniform(15.0, 35.0))
    for d in range(-14, 1):
        hist_val = base_hist + ((current_risk - base_hist) / 14.0) * (d + 14) + np.random.uniform(-2.0, 2.0)
        hist_val = max(2.0, min(99.0, hist_val))
        history.append({
            "day": f"Day {d}",
            "day_num": d,
            "risk": round(hist_val, 1)
        })

    records.append({
        "cow_id": c_id,
        "breed": "Gir / Sahiwal / HF Cross",
        "current_risk": round(current_risk, 1),
        "risk_category": cat,
        "milk_conductivity_ms_cm": round(cond, 2),
        "cow_activity": act,
        "milk_yield_liters": round(yield_l, 1),
        "milk_temperature_c": round(temp_c, 1),
        "environment_temperature_c": round(float(np.random.uniform(28.0, 34.0)), 1),
        "humidity_percent": round(float(np.random.uniform(60.0, 80.0)), 1),
        "risk_24h": round(r_24h, 1),
        "risk_48h": round(r_48h, 1),
        "risk_3d": round(r_3d, 1),
        "7d_probability": round(r_7d, 1),
        "14d_probability": round(r_14d, 1),
        "history_json": json.dumps(history)
    })

df = pd.DataFrame(records)
df = df.sort_values(by="current_risk", ascending=False).reset_index(drop=True)
df.to_csv("herd_predictions.csv", index=False)

print("SUCCESS: Calibrated 100-cow dataset generated.")
print(f"High Risk: {len(df[df['risk_category'] == 'High Risk'])}")
print(f"Moderate Risk: {len(df[df['risk_category'] == 'Moderate Risk'])}")
print(f"Low Risk: {len(df[df['risk_category'] == 'Low Risk'])}")
print(f"Healthy: {len(df[df['risk_category'] == 'Healthy'])}")
print(f"Total Sum: {len(df)}")
