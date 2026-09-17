import joblib
import pandas as pd
import os
import sys

# ============================================================
# MODEL 1 - MANUAL TESTING PROGRAM
# ============================================================
# This program loads trained Model 1 (mastitis_model1.pkl),
# prompts for cow sensor data with inline validation,
# and outputs the mastitis risk score & classification.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "mastitis_model1.pkl")


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("BOVINE MASTITIS - MODEL 1 TEST PROGRAM")
print("=" * 70)

try:
    model = joblib.load(MODEL_FILE)
    print("\nModel loaded successfully.")
    print(f"Model file: {MODEL_FILE}")

except FileNotFoundError:
    print(f"\nERROR: Could not find model file at '{MODEL_FILE}'.")
    print("Please make sure you have run 'python model1.py' first to train and save Model 1.")
    sys.exit(1)
except Exception as error:
    print(f"\nERROR LOADING MODEL: {error}")
    sys.exit(1)


# ============================================================
# 2. INLINE VALIDATION PROMPT FUNCTIONS
# ============================================================

def prompt_float(prompt_text, min_val=None, max_val=None):
    """Prompt user for a float with optional min/max boundary checks."""
    while True:
        try:
            val = float(input(prompt_text))
            if min_val is not None and val < min_val:
                print(f"  --> Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  --> Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  --> Invalid input. Please enter a valid number.")


def prompt_choice(prompt_text, choices=[0, 1]):
    """Prompt user for a choice integer (e.g. 0 or 1)."""
    while True:
        try:
            val = int(input(prompt_text))
            if val in choices:
                return val
            print(f"  --> Please enter one of {choices}.")
        except ValueError:
            print("  --> Invalid input. Please enter a whole number.")


# ============================================================
# 3. GET COW DATA WITH INLINE VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("ENTER TODAY'S COW DATA")
print("=" * 70)

milk_yield = prompt_float(
    "\n1. Milk yield (liters, 0 to 80): ",
    min_val=0.0, max_val=80.0
)

milk_conductivity = prompt_float(
    "2. Milk conductivity (mS/cm, 0 to 30): ",
    min_val=0.0, max_val=30.0
)

cow_activity = prompt_float(
    "3. Cow activity index (0 to 500): ",
    min_val=0.0, max_val=500.0
)

environment_temperature = prompt_float(
    "4. Environment temperature (°C, -30 to 60): ",
    min_val=-30.0, max_val=60.0
)

milk_temperature = prompt_float(
    "5. Milk temperature (°C, 0 to 45): ",
    min_val=0.0, max_val=45.0
)

humidity = prompt_float(
    "6. Humidity (%, 0 to 100): ",
    min_val=0.0, max_val=100.0
)

previous_mastitis = prompt_choice(
    "7. Previous mastitis history (0 = No, 1 = Yes): ",
    choices=[0, 1]
)

# LOGICAL FIX FOR PREVIOUS MASTITIS == 0
if previous_mastitis == 0:
    print("  --> Cow has no previous mastitis history (setting days since last mastitis = -1).")
    days_since_last_mastitis = -1.0
else:
    days_since_last_mastitis = prompt_float(
        "8. Days since last mastitis (0 or greater): ",
        min_val=0.0, max_val=3650.0
    )


# ============================================================
# 4. CREATE INPUT DATAFRAME
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
# 5. MODEL PREDICTION
# ============================================================

try:
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= 0.50 else 0
except Exception as error:
    print("\nERROR DURING PREDICTION:")
    print(error)
    sys.exit(1)

risk_percentage = probability * 100.0


# ============================================================
# 6. DETERMINE RISK LEVEL & CATEGORY
# ============================================================

if risk_percentage < 30.0:
    risk_level = "LOW RISK"
elif risk_percentage < 60.0:
    risk_level = "MEDIUM RISK"
else:
    risk_level = "HIGH RISK"


# ============================================================
# 7. DISPLAY SUMMARY & PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("INPUT SUMMARY")
print("=" * 70)
print(f"Milk yield              : {milk_yield:.2f} L")
print(f"Milk conductivity       : {milk_conductivity:.2f} mS/cm")
print(f"Cow activity            : {cow_activity:.2f}")
print(f"Environment temperature : {environment_temperature:.2f} °C")
print(f"Milk temperature        : {milk_temperature:.2f} °C")
print(f"Humidity                : {humidity:.2f}%")
print(f"Previous mastitis       : {'Yes (1)' if previous_mastitis == 1 else 'No (0)'}")
print(f"Days since last mastitis: {days_since_last_mastitis if days_since_last_mastitis >= 0 else 'N/A (-1)'}")

print("\n")
print("=" * 70)
print("MODEL 1 PREDICTION RESULTS")
print("=" * 70)
print(f"Calculated Risk Score : {risk_percentage:.2f}%")
print(f"Assigned Risk Band    : {risk_level}")
print(f"Binary Classification  : {'MASTITIS' if prediction == 1 else 'HEALTHY'} (50% cutoff)")

print("\n")
print("=" * 70)
print("INTERPRETATION & GUIDANCE")
print("=" * 70)

if prediction == 1:
    print("WARNING: Model 1 predicts this cow is at HIGH RISK of mastitis today.")
    print("Action: Perform immediate udder inspection and conduct California Mastitis Test (CMT).")
elif risk_percentage >= 30.0:
    print("ALERT: Model 1 scores this cow in the MODERATE RISK range.")
    print("Action: Increase daily sensor monitoring and check milk conductivity trend over next 24-48 hours.")
else:
    print("NORMAL: Model 1 classifies this cow as HEALTHY with low risk today.")
    print("Action: Continue routine herd monitoring.")

print("\n" + "=" * 70)
print("MODEL 1 TEST COMPLETE")
print("=" * 70)
