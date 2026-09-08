import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_test_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "model2_v4_threshold_analysis.csv"
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("             MODEL 2 V4 THRESHOLD ANALYSIS")
print("=" * 78)

print()
print(f"Input file:")
print(INPUT_FILE)
print()

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"ERROR: File not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded : {len(df):,}")
print()


# ============================================================
# VALIDATE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "horizon",
    "actual_high_risk",
    "predicted_probability"
]

for column in REQUIRED_COLUMNS:
    if column not in df.columns:
        raise ValueError(
            f"ERROR: Required column missing: {column}"
        )


# ============================================================
# ANALYZE EACH HORIZON
# ============================================================

all_results = []

for horizon in ["7-day", "14-day"]:

    horizon_df = df[
        df["horizon"] == horizon
    ].copy()

    if len(horizon_df) == 0:
        print(
            f"WARNING: No data found for {horizon}"
        )
        continue

    y_true = horizon_df[
        "actual_high_risk"
    ].astype(int).to_numpy()

    probabilities = horizon_df[
        "predicted_probability"
    ].to_numpy()

    print()
    print("=" * 78)
    print(f"{horizon.upper()} THRESHOLD ANALYSIS")
    print("=" * 78)

    print()
    print(
        f"Test records       : {len(horizon_df):,}"
    )

    print(
        f"Actual high-risk   : {y_true.sum():,}"
    )

    print(
        f"Actual positive %  : "
        f"{y_true.mean() * 100:.2f}%"
    )

    print()

    print(
        "Threshold | Precision | Recall | F1 Score | "
        "False Positives | False Negatives | Predicted High-Risk"
    )

    print("-" * 100)

    for threshold in THRESHOLDS:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        predicted_high_risk = int(
            y_pred.sum()
        )

        print(
            f"{threshold:8.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:8.4f} | "
            f"{fp:15,} | "
            f"{fn:16,} | "
            f"{predicted_high_risk:19,}"
        )

        all_results.append({
            "horizon": horizon,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "predicted_high_risk": predicted_high_risk
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

print()
print("=" * 78)
print("BEST THRESHOLD BY F1")
print("=" * 78)

for horizon in ["7-day", "14-day"]:

    horizon_results = results_df[
        results_df["horizon"] == horizon
    ].copy()

    if len(horizon_results) == 0:
        continue

    best_row = horizon_results.loc[
        horizon_results["f1"].idxmax()
    ]

    print()
    print(horizon)

    print(
        f"Best threshold     : "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Precision           : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1                  : "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"False positives     : "
        f"{int(best_row['false_positives']):,}"
    )

    print(
        f"False negatives     : "
        f"{int(best_row['false_negatives']):,}"
    )


# ============================================================
# BEST THRESHOLD BY RECALL >= 70%
# ============================================================

print()
print("=" * 78)
print("BEST THRESHOLD WITH RECALL >= 70%")
print("=" * 78)

for horizon in ["7-day", "14-day"]:

    horizon_results = results_df[
        results_df["horizon"] == horizon
    ].copy()

    eligible = horizon_results[
        horizon_results["recall"] >= 0.70
    ]

    print()
    print(horizon)

    if len(eligible) == 0:

        print(
            "No tested threshold achieved "
            "at least 70% recall."
        )

    else:

        # Among thresholds achieving >=70% recall,
        # choose the one with the best F1.

        best_row = eligible.loc[
            eligible["f1"].idxmax()
        ]

        print(
            f"Selected threshold : "
            f"{best_row['threshold']:.2f}"
        )

        print(
            f"Precision           : "
            f"{best_row['precision']:.4f}"
        )

        print(
            f"Recall              : "
            f"{best_row['recall']:.4f}"
        )

        print(
            f"F1                  : "
            f"{best_row['f1']:.4f}"
        )

        print(
            f"False positives     : "
            f"{int(best_row['false_positives']):,}"
        )

        print(
            f"False negatives     : "
            f"{int(best_row['false_negatives']):,}"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 78)

print()
print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Do NOT change the V4 models yet."
)

print(
    "Do NOT retrain anything yet."
)

print(
    "Use the threshold results to decide the final "
    "classification threshold."
)

print()
print("=" * 78)