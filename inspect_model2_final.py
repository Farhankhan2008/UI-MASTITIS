import os
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Mastitis"

MODEL_FILES = {
    "24h": os.path.join(
        BASE_DIR,
        "mastitis_model2_final_24h.pkl"
    ),

    "48h": os.path.join(
        BASE_DIR,
        "mastitis_model2_final_48h.pkl"
    ),

    "3d": os.path.join(
        BASE_DIR,
        "mastitis_model2_final_3d.pkl"
    ),

    "7d": os.path.join(
        BASE_DIR,
        "mastitis_model2_final_7d.pkl"
    ),

    "14d": os.path.join(
        BASE_DIR,
        "mastitis_model2_final_14d.pkl"
    )
}

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "mastitis_model2_final_config.pkl"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL MODEL 2 INSPECTION")
print("=" * 80)


# ============================================================
# INSPECT EACH MODEL
# ============================================================

for horizon, file_path in MODEL_FILES.items():

    print("\n" + "=" * 80)
    print(f"MODEL 2 — {horizon}")
    print("=" * 80)

    print("\nFile:")
    print(file_path)

    if not os.path.exists(file_path):

        print("\nERROR: File does not exist.")

        continue

    data = joblib.load(file_path)

    print("\nLoaded object type:")
    print(type(data))

    if isinstance(data, dict):

        print("\nDictionary keys:")

        for key in data.keys():
            print(f" - {key}")

        model = data.get("model")

        print("\nActual ML model:")

        if model is not None:

            print(type(model))

            if hasattr(model, "n_features_in_"):

                print(
                    "Number of input features:",
                    model.n_features_in_
                )

            if hasattr(model, "feature_names_in_"):

                print("\nModel feature names:")

                for i, feature in enumerate(
                    model.feature_names_in_,
                    start=1
                ):

                    print(
                        f"{i:02d}. {feature}"
                    )

            if hasattr(model, "classes_"):

                print("\nClasses:")
                print(model.classes_)

            if hasattr(model, "n_estimators"):

                print(
                    "\nNumber of trees:",
                    model.n_estimators
                )

        print("\nSaved metadata:")

        for key, value in data.items():

            if key == "model":
                continue

            print(f"\n{key}:")

            if isinstance(
                value,
                (list, tuple, np.ndarray)
            ):

                print(
                    f"Length: {len(value)}"
                )

                if len(value) <= 100:

                    print(value)

            else:

                print(value)

    else:

        print("\nWARNING:")
        print("Model was not saved as a dictionary.")


# ============================================================
# INSPECT CONFIG
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL 2 CONFIGURATION")
print("=" * 80)

print("\nConfig file:")
print(CONFIG_FILE)

if not os.path.exists(CONFIG_FILE):

    print("\nERROR: Config file does not exist.")

else:

    config = joblib.load(CONFIG_FILE)

    print("\nConfig object type:")
    print(type(config))

    print("\nConfiguration:")

    if isinstance(config, dict):

        for key, value in config.items():

            print("\n" + "-" * 60)
            print(f"{key}:")

            if isinstance(value, dict):

                for sub_key, sub_value in value.items():

                    print(
                        f"  {sub_key}: {sub_value}"
                    )

            elif isinstance(
                value,
                (list, tuple)
            ):

                print(
                    f"  Length: {len(value)}"
                )

                if len(value) <= 100:
                    print(
                        f"  {value}"
                    )

            else:

                print(
                    f"  {value}"
                )

    else:

        print(config)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL 2 INSPECTION COMPLETE")
print("=" * 80)