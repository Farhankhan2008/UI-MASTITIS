import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

DATASET = os.path.join(
    BASE_DIR,
    "model2_forecasting_dataset_v3.csv"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "model2_optimization_v5_results.csv"
)

RANDOM_STATE = 42

HIGH_RISK_THRESHOLD = 50.0

CLASSIFICATION_THRESHOLDS = {
    "7d": 0.20,
    "14d": 0.30
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("MODEL 2 V5 OPTIMIZATION")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows              : {len(df)}")
print(f"Columns            : {len(df.columns)}")
print(f"Unique cows        : {df['cow_id'].nunique()}")
print(f"Missing values     : {df.isna().sum().sum()}")


# ============================================================
# SORT FIRST
# ============================================================

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# ORIGINAL FEATURES
# ============================================================

TARGETS = [
    "risk_24h",
    "risk_48h",
    "risk_3d",
    "risk_7d",
    "risk_14d"
]

DROP_COLUMNS = [
    "cow_id",
    "date"
] + TARGETS

BASE_FEATURES = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(f"Base features: {len(BASE_FEATURES)}")


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("CREATING ADDITIONAL TEMPORAL FEATURES")
print("=" * 80)


def create_temporal_features(group):

    group = group.copy()

    risk = group["risk_today"]

    # --------------------------------------------------------
    # Median features
    # --------------------------------------------------------

    group["risk_median_3d"] = (
        risk.rolling(3, min_periods=3).median()
    )

    group["risk_median_7d"] = (
        risk.rolling(7, min_periods=7).median()
    )

    group["risk_median_14d"] = (
        risk.rolling(14, min_periods=14).median()
    )

    # --------------------------------------------------------
    # Risk ranges
    # --------------------------------------------------------

    group["risk_range_3d"] = (
        group["risk_3day_max"] -
        group["risk_3day_min"]
    )

    group["risk_range_7d"] = (
        group["risk_7day_max"] -
        group["risk_7day_min"]
    )

    group["risk_range_14d"] = (
        group["risk_14day_max"] -
        group["risk_14day_min"]
    )

    # --------------------------------------------------------
    # Additional slopes
    # --------------------------------------------------------

    group["risk_slope_5d"] = (
        risk - risk.shift(4)
    ) / 4.0

    group["risk_slope_10d"] = (
        risk - risk.shift(9)
    ) / 9.0

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    group["risk_momentum_3d"] = (
        group["risk_change_1d"]
        + group["risk_change_2d"]
        + group["risk_change_3d"]
    )

    group["risk_momentum_7d"] = (
        group["risk_change_3d"]
        + group["risk_change_7d"]
    )

    group["risk_momentum_14d"] = (
        group["risk_change_7d"]
        + group["risk_change_14d"]
    )

    # --------------------------------------------------------
    # Recent vs older history
    # --------------------------------------------------------

    group["recent_vs_old_7d"] = (
        group["risk_3day_mean"] -
        (
            group["risk_7day_mean"]
            + group["risk_7day_mean"].shift(3)
        ) / 2.0
    )

    group["recent_vs_old_14d"] = (
        group["risk_7day_mean"] -
        group["risk_14day_mean"]
    )

    # --------------------------------------------------------
    # Position inside historical range
    # --------------------------------------------------------

    group["risk_position_7d"] = (
        (
            group["risk_today"] -
            group["risk_7day_min"]
        )
        /
        (
            group["risk_7day_max"] -
            group["risk_7day_min"] +
            1e-6
        )
    )

    group["risk_position_14d"] = (
        (
            group["risk_today"] -
            group["risk_14day_min"]
        )
        /
        (
            group["risk_14day_max"] -
            group["risk_14day_min"] +
            1e-6
        )
    )

    # --------------------------------------------------------
    # Volatility change
    # --------------------------------------------------------

    group["volatility_change"] = (
        group["risk_std_7d"] -
        group["risk_std_14d"]
    )

    # --------------------------------------------------------
    # Trend consistency
    # --------------------------------------------------------

    changes = group["risk_change_1d"]

    group["positive_change_count_7d"] = (
        changes.gt(0)
        .rolling(7, min_periods=7)
        .sum()
    )

    group["positive_change_count_14d"] = (
        changes.gt(0)
        .rolling(14, min_periods=14)
        .sum()
    )

    group["negative_change_count_7d"] = (
        changes.lt(0)
        .rolling(7, min_periods=7)
        .sum()
    )

    group["negative_change_count_14d"] = (
        changes.lt(0)
        .rolling(14, min_periods=14)
        .sum()
    )

    # --------------------------------------------------------
    # High-risk proximity
    # --------------------------------------------------------

    group["distance_to_high_risk"] = (
        HIGH_RISK_THRESHOLD -
        group["risk_today"]
    )

    group["high_risk_ratio_7d"] = (
        group["high_risk_count_7d"] / 7.0
    )

    group["high_risk_ratio_14d"] = (
        group["high_risk_count_14d"] / 14.0
    )

    return group


# IMPORTANT:
# group_keys=False preserves cow_id/date correctly.
# ============================================================
# APPLY TEMPORAL FEATURES — PANDAS 3.x SAFE
# ============================================================

feature_parts = []

for cow_id, cow_group in df.groupby(
    "cow_id",
    sort=False
):

    cow_group = create_temporal_features(
        cow_group.copy()
    )

    feature_parts.append(cow_group)

df = pd.concat(
    feature_parts,
    axis=0,
    ignore_index=True
)

df = df.sort_values(
    ["cow_id", "date"]
).reset_index(drop=True)


# ============================================================
# NEW FEATURES
# ============================================================

NEW_FEATURES = [
    "risk_median_3d",
    "risk_median_7d",
    "risk_median_14d",
    "risk_range_3d",
    "risk_range_7d",
    "risk_range_14d",
    "risk_slope_5d",
    "risk_slope_10d",
    "risk_momentum_3d",
    "risk_momentum_7d",
    "risk_momentum_14d",
    "recent_vs_old_7d",
    "recent_vs_old_14d",
    "risk_position_7d",
    "risk_position_14d",
    "volatility_change",
    "positive_change_count_7d",
    "positive_change_count_14d",
    "negative_change_count_7d",
    "negative_change_count_14d",
    "distance_to_high_risk",
    "high_risk_ratio_7d",
    "high_risk_ratio_14d"
]

FEATURES = BASE_FEATURES + NEW_FEATURES

FEATURES = list(dict.fromkeys(FEATURES))

print(f"\nOriginal features : {len(BASE_FEATURES)}")
print(f"New features      : {len(NEW_FEATURES)}")
print(f"Total features    : {len(FEATURES)}")


# ============================================================
# REMOVE INSUFFICIENT-HISTORY ROWS
# ============================================================

df_model = df.dropna(
    subset=FEATURES
).copy()

print(
    f"Rows after feature preparation: "
    f"{len(df_model)}"
)

print(
    f"Unique cows after preparation: "
    f"{df_model['cow_id'].nunique()}"
)


# ============================================================
# COW-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("COW-LEVEL TRAIN / TEST SPLIT")
print("=" * 80)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        df_model,
        groups=df_model["cow_id"]
    )
)

train_df = df_model.iloc[
    train_idx
].copy()

test_df = df_model.iloc[
    test_idx
].copy()

print(
    f"Train cows : "
    f"{train_df['cow_id'].nunique()}"
)

print(
    f"Test cows  : "
    f"{test_df['cow_id'].nunique()}"
)

print(
    "Cow overlap:",
    len(
        set(train_df["cow_id"])
        &
        set(test_df["cow_id"])
    )
)

print(
    f"Train rows : {len(train_df)}"
)

print(
    f"Test rows  : {len(test_df)}"
)


# ============================================================
# FEATURE MATRICES
# ============================================================

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

REGRESSION_MODELS = {

    "RandomForest": RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
}


CLASSIFICATION_MODELS = {

    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "ExtraTreesClassifier": ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# REGRESSION OPTIMIZATION
# ============================================================

REGRESSION_TARGETS = {
    "24h": "risk_24h",
    "48h": "risk_48h",
    "3d": "risk_3d",
    "7d": "risk_7d",
    "14d": "risk_14d"
}


for horizon, target in REGRESSION_TARGETS.items():

    print("\n" + "=" * 80)
    print(
        f"REGRESSION OPTIMIZATION — {horizon}"
    )
    print("=" * 80)

    y_train = train_df[target]
    y_test = test_df[target]

    for model_name, model in REGRESSION_MODELS.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.clip(
            predictions,
            0,
            100
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        actual_high = (
            y_test >=
            HIGH_RISK_THRESHOLD
        )

        predicted_high = (
            predictions >=
            HIGH_RISK_THRESHOLD
        )

        precision = precision_score(
            actual_high,
            predicted_high,
            zero_division=0
        )

        recall = recall_score(
            actual_high,
            predicted_high,
            zero_division=0
        )

        f1 = f1_score(
            actual_high,
            predicted_high,
            zero_division=0
        )

        results.append({
            "horizon": horizon,
            "problem_type": "regression",
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "precision_high_risk": precision,
            "recall_high_risk": recall,
            "F1_high_risk": f1,
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan
        })

        print(
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f} | "
            f"HighRisk F1={f1:.4f}"
        )


# ============================================================
# BUILD STRICT FUTURE HIGH-RISK TARGETS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING STRICT FUTURE HIGH-RISK TARGETS")
print("=" * 80)


df_model["high_risk_within_7d"] = np.nan
df_model["high_risk_within_14d"] = np.nan

df_model["valid_7d"] = False
df_model["valid_14d"] = False


for cow_id, group in df_model.groupby(
    "cow_id",
    sort=False
):

    indices = group.index

    risk = group[
        "risk_today"
    ]

    # --------------------------------------------------------
    # 7 DAYS
    # --------------------------------------------------------

    future7 = pd.concat(
        [
            risk.shift(-i)
            for i in range(1, 8)
        ],
        axis=1
    )

    valid7 = (
        future7.notna().all(axis=1)
    )

    target7 = (
        future7.max(axis=1)
        >= HIGH_RISK_THRESHOLD
    )

    df_model.loc[
        indices,
        "high_risk_within_7d"
    ] = target7.astype(int).values

    df_model.loc[
        indices,
        "valid_7d"
    ] = valid7.values

    # --------------------------------------------------------
    # 14 DAYS
    # --------------------------------------------------------

    future14 = pd.concat(
        [
            risk.shift(-i)
            for i in range(1, 15)
        ],
        axis=1
    )

    valid14 = (
        future14.notna().all(axis=1)
    )

    target14 = (
        future14.max(axis=1)
        >= HIGH_RISK_THRESHOLD
    )

    df_model.loc[
        indices,
        "high_risk_within_14d"
    ] = target14.astype(int).values

    df_model.loc[
        indices,
        "valid_14d"
    ] = valid14.values


# ============================================================
# CLASSIFICATION OPTIMIZATION
# ============================================================

for horizon, target_col, valid_col in [
    (
        "7d",
        "high_risk_within_7d",
        "valid_7d"
    ),
    (
        "14d",
        "high_risk_within_14d",
        "valid_14d"
    )
]:

    print("\n" + "=" * 80)
    print(
        f"CLASSIFICATION OPTIMIZATION — {horizon}"
    )
    print("=" * 80)

    train_mask = (
        train_df.index
        .intersection(
            df_model.index[
                df_model[valid_col]
            ]
        )
    )

    test_mask = (
        test_df.index
        .intersection(
            df_model.index[
                df_model[valid_col]
            ]
        )
    )

    Xtr = df_model.loc[
        train_mask,
        FEATURES
    ]

    Xte = df_model.loc[
        test_mask,
        FEATURES
    ]

    ytr = df_model.loc[
        train_mask,
        target_col
    ].astype(int)

    yte = df_model.loc[
        test_mask,
        target_col
    ].astype(int)

    print(
        f"Valid train rows : {len(Xtr)}"
    )

    print(
        f"Valid test rows  : {len(Xte)}"
    )

    print(
        f"Train positives  : {ytr.sum()}"
    )

    print(
        f"Test positives   : {yte.sum()}"
    )

    for model_name, model in CLASSIFICATION_MODELS.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            Xtr,
            ytr
        )

        probabilities = model.predict_proba(
            Xte
        )[:, 1]

        threshold = (
            CLASSIFICATION_THRESHOLDS[
                horizon
            ]
        )

        predictions = (
            probabilities >= threshold
        )

        precision = precision_score(
            yte,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            yte,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            yte,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            yte,
            probabilities
        )

        pr_auc = average_precision_score(
            yte,
            probabilities
        )

        results.append({
            "horizon": horizon,
            "problem_type": "classification",
            "model": model_name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "precision_high_risk": precision,
            "recall_high_risk": recall,
            "F1_high_risk": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        })

        print(
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MODEL 2 V5 OPTIMIZATION COMPLETE")
print("=" * 80)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_RESULTS}"
)

print("\nFULL RESULTS:")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST REGRESSION MODELS
# ============================================================

print("\n" + "=" * 80)
print("BEST REGRESSION MODEL PER HORIZON")
print("=" * 80)

reg_results = results_df[
    results_df["problem_type"]
    == "regression"
].copy()

for horizon in [
    "24h",
    "48h",
    "3d",
    "7d",
    "14d"
]:

    subset = reg_results[
        reg_results["horizon"]
        == horizon
    ]

    best = subset.sort_values(
        "MAE"
    ).iloc[0]

    print(
        f"{horizon}: "
        f"{best['model']} | "
        f"MAE={best['MAE']:.4f} | "
        f"RMSE={best['RMSE']:.4f} | "
        f"R2={best['R2']:.4f}"
    )


# ============================================================
# BEST CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 80)
print("BEST CLASSIFICATION MODEL PER HORIZON")
print("=" * 80)

clf_results = results_df[
    results_df["problem_type"]
    == "classification"
].copy()

for horizon in [
    "7d",
    "14d"
]:

    subset = clf_results[
        clf_results["horizon"]
        == horizon
    ]

    best = subset.sort_values(
        "F1_high_risk",
        ascending=False
    ).iloc[0]

    print(
        f"{horizon}: "
        f"{best['model']} | "
        f"Precision={best['precision_high_risk']:.4f} | "
        f"Recall={best['recall_high_risk']:.4f} | "
        f"F1={best['F1_high_risk']:.4f} | "
        f"ROC-AUC={best['ROC_AUC']:.4f} | "
        f"PR-AUC={best['PR_AUC']:.4f}"
    )


print("\n" + "=" * 80)
print("DO NOT REPLACE EXISTING MODELS")
print("=" * 80)

print("""
This was an optimization experiment only.

Existing Model 1 and Model 2 files were NOT modified.

Paste the complete terminal output back here.
""")