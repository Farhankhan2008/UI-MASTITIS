import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

MODEL1_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model1.pkl"
)


# ============================================================
# LOAD MODEL 1
# ============================================================

print("=" * 80)
print("MODEL 1 INSPECTION")
print("=" * 80)

print("\nModel file:")
print(MODEL1_FILE)

if not os.path.exists(MODEL1_FILE):
    print("\nERROR: Model 1 file does not exist.")
    raise SystemExit

model_data = joblib.load(MODEL1_FILE)

print("\nLoaded object type:")
print(type(model_data))


# ============================================================
# INSPECT OBJECT
# ============================================================

if isinstance(model_data, dict):

    print("\nObject is a dictionary.")

    print("\nDictionary keys:")

    for key in model_data.keys():
        print(f" - {key}")

    print("\nDetailed information:")

    for key, value in model_data.items():

        print("\n" + "-" * 60)
        print(f"KEY: {key}")
        print(f"TYPE: {type(value)}")

        if isinstance(value, (list, tuple)):
            print(f"LENGTH: {len(value)}")

            if len(value) <= 30:
                print("VALUE:")
                print(value)

        elif isinstance(value, np.ndarray):

            print(f"SHAPE: {value.shape}")
            print(f"DTYPE: {value.dtype}")

            if value.size <= 30:
                print("VALUE:")
                print(value)

        elif isinstance(value, pd.DataFrame):

            print(f"SHAPE: {value.shape}")
            print("COLUMNS:")
            print(list(value.columns))

        elif isinstance(value, pd.Series):

            print(f"LENGTH: {len(value)}")
            print("NAME:")
            print(value.name)

        else:

            text = str(value)

            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"

            print("VALUE:")
            print(text)


# ============================================================
# IF MODEL ITSELF WAS SAVED DIRECTLY
# ============================================================

else:

    print("\nObject is not a dictionary.")

    print("\nModel attributes:")

    for attribute in [
        "n_features_in_",
        "feature_names_in_",
        "classes_",
        "n_estimators",
        "max_depth"
    ]:

        if hasattr(model_data, attribute):

            value = getattr(
                model_data,
                attribute
            )

            print(f"\n{attribute}:")
            print(value)


# ============================================================
# MODEL TYPE / FEATURES
# ============================================================

print("\n" + "=" * 80)
print("MODEL 1 STRUCTURE")
print("=" * 80)

if isinstance(model_data, dict):

    model = model_data.get("model")

    if model is not None:

        print("\nActual ML model:")
        print(type(model))

        if hasattr(model, "n_features_in_"):

            print(
                "\nNumber of input features:",
                model.n_features_in_
            )

        if hasattr(model, "feature_names_in_"):

            print("\nModel feature names:")

            for i, feature in enumerate(
                model.feature_names_in_,
                start=1
            ):
                print(f"{i:02d}. {feature}")

        if hasattr(model, "classes_"):

            print("\nClasses:")
            print(model.classes_)

        if hasattr(model, "n_estimators"):

            print(
                "\nNumber of trees:",
                model.n_estimators
            )

    features = model_data.get("features")

    if features is not None:

        print("\nSaved feature list:")

        for i, feature in enumerate(
            features,
            start=1
        ):
            print(f"{i:02d}. {feature}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("MODEL 1 INSPECTION COMPLETE")
print("=" * 80)