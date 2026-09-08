import joblib
import pandas as pd


# ============================================================
# MODEL 1 - MANUAL TESTING PROGRAM
# ============================================================
# This program loads the trained Model 1:
#
#     mastitis_model1.pkl
#
# Then it asks the user for today's cow measurements and
# predicts the probability/risk of mastitis.
#
# IMPORTANT:
# The input order MUST match the order used during training.
# ============================================================


MODEL_FILE = "mastitis_model1.pkl"


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("BOVINE MASTITIS - MODEL 1 TEST")
print("=" * 70)

try:

    model = joblib.load(MODEL_FILE)

    print("\nModel loaded successfully.")
    print(f"Model file: {MODEL_FILE}")

except FileNotFoundError:

    print("\nERROR:")
    print(f"Could not find '{MODEL_FILE}'.")
    print("Make sure test_model1.py and mastitis_model1.pkl are")
    print("in the same folder.")

    raise SystemExit


# ============================================================
# 2. INPUT FUNCTION
# ============================================================

def get_float(prompt):

    while True:

        try:

            value = float(input(prompt))

            return value

        except ValueError:

            print("Please enter a valid number.")


def get_integer(prompt):

    while True:

        try:

            value = int(input(prompt))

            return value

        except ValueError:

            print("Please enter a whole number.")


# ============================================================
# 3. GET COW DATA
# ============================================================

print("\n")
print("=" * 70)
print("ENTER TODAY'S COW DATA")
print("=" * 70)

print("\nEnter the following values:")


milk_yield = get_float(
    "\n1. Milk yield (liters): "
)


milk_conductivity = get_float(
    "2. Milk conductivity (mS/cm): "
)


cow_activity = get_float(
    "3. Cow activity: "
)


environment_temperature = get_float(
    "4. Environment temperature (°C): "
)


milk_temperature = get_float(
    "5. Milk temperature (°C): "
)


humidity = get_float(
    "6. Humidity (%): "
)


previous_mastitis = get_integer(
    "7. Previous mastitis (0 = No, 1 = Yes): "
)


days_since_last_mastitis = get_integer(
    "8. Days since last mastitis: "
)


# ============================================================
# 4. VALIDATE INPUTS
# ============================================================

if previous_mastitis not in [0, 1]:

    print(
        "\nERROR: Previous mastitis must be either 0 or 1."
    )

    raise SystemExit


if milk_yield < 0:

    print("\nERROR: Milk yield cannot be negative.")
    raise SystemExit


if milk_conductivity < 0:

    print("\nERROR: Milk conductivity cannot be negative.")
    raise SystemExit


if cow_activity < 0:

    print("\nERROR: Cow activity cannot be negative.")
    raise SystemExit


if humidity < 0 or humidity > 100:

    print("\nERROR: Humidity must be between 0 and 100.")
    raise SystemExit


if days_since_last_mastitis < 0:

    print(
        "\nERROR: Days since last mastitis cannot be negative."
    )

    raise SystemExit


# ============================================================
# 5. CREATE INPUT DATAFRAME
# ============================================================
# MUST match the exact feature order used during training.
# ============================================================

input_data = pd.DataFrame([{

    "milk_yield_liters": milk_yield,

    "milk_conductivity_ms_cm": milk_conductivity,

    "cow_activity": cow_activity,

    "environment_temperature_c": environment_temperature,

    "milk_temperature_c": milk_temperature,

    "humidity_percent": humidity,

    "previous_mastitis": previous_mastitis,

    "days_since_last_mastitis": days_since_last_mastitis

}])


# ============================================================
# 6. PREDICT
# ============================================================

try:

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

except Exception as error:

    print("\nERROR DURING PREDICTION:")
    print(error)

    raise SystemExit


# Convert probability to percentage

risk_percentage = probability * 100


# ============================================================
# 7. DETERMINE RISK LEVEL
# ============================================================
# These are prototype display thresholds.
#
# IMPORTANT:
# They are NOT clinical diagnostic thresholds.
# ============================================================

if risk_percentage < 30:

    risk_level = "LOW RISK"

elif risk_percentage < 60:

    risk_level = "MEDIUM RISK"

else:

    risk_level = "HIGH RISK"


# ============================================================
# 8. DISPLAY INPUTS
# ============================================================

print("\n")
print("=" * 70)
print("INPUT SUMMARY")
print("=" * 70)

print(
    f"\nMilk yield              : "
    f"{milk_yield:.2f} L"
)

print(
    f"Milk conductivity       : "
    f"{milk_conductivity:.2f} mS/cm"
)

print(
    f"Cow activity            : "
    f"{cow_activity:.2f}"
)

print(
    f"Environment temperature : "
    f"{environment_temperature:.2f} °C"
)

print(
    f"Milk temperature        : "
    f"{milk_temperature:.2f} °C"
)

print(
    f"Humidity                : "
    f"{humidity:.2f}%"
)

print(
    f"Previous mastitis       : "
    f"{previous_mastitis}"
)

print(
    f"Days since last mastitis: "
    f"{days_since_last_mastitis}"
)


# ============================================================
# 9. DISPLAY MODEL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 1 PREDICTION")
print("=" * 70)

print(
    f"\nMastitis risk : {risk_percentage:.2f}%"
)

print(
    f"Risk level    : {risk_level}"
)

print(
    f"Prediction    : "
    f"{'MASTITIS' if prediction == 1 else 'HEALTHY'}"
)


# ============================================================
# 10. INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if prediction == 1:

    print(
        "\nModel 1 predicts that this cow is at "
        "higher risk of mastitis today."
    )

else:

    print(
        "\nModel 1 predicts that this cow is "
        "not classified as mastitis today."
    )


print(
    "\nThe percentage shown is the model's estimated "
    "risk score, not a veterinary diagnosis."
)


# ============================================================
# 11. END
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 1 TEST COMPLETE")
print("=" * 70)

