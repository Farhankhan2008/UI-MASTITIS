import sys
import os
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    Form
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Mastitis AI Backend",
    description="Bovine mastitis risk monitoring and forecasting API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE & AUTHENTICATION IMPORTS
# ============================================================

from database import get_connection, create_users_table

from auth import (
    authenticate_user,
    create_user,
    create_access_token,
    verify_token
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data_service import (
    load_predictions,
    get_cow,
    get_all_cows
)

from herd_risk_engine import analyze_herd


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

create_users_table()


# ============================================================
# AUTHENTICATION MODELS
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "farmer"


# ============================================================
# JWT AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/login"
)


def get_current_user(
    token: str = Depends(oauth2_scheme)
):

    username = verify_token(token)

    if username is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )

    return username


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def make_json_safe(obj):

    # Dictionary
    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    # List
    if isinstance(obj, list):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # Tuple
    if isinstance(obj, tuple):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # NumPy scalar values
    if isinstance(obj, np.generic):
        return obj.item()

    # Normal Python value
    return obj


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")
def register(user: RegisterRequest):

    success = create_user(
        user.username,
        user.password,
        user.role
    )

    if not success:

        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    return {
        "message": "User created successfully",
        "username": user.username,
        "role": user.role
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...)
):

    authenticated_user = authenticate_user(
        username,
        password
    )

    if authenticated_user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    access_token = create_access_token({
        "sub": authenticated_user["username"]
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": authenticated_user["username"],
        "role": authenticated_user["role"]
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "system": "Mastitis AI",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    try:

        df = load_predictions()

        return {
            "status": "healthy",
            "cows_available": len(df)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# ALL COWS
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/cows")
def cows(
    current_user: str = Depends(get_current_user)
):

    records = get_all_cows()

    return make_json_safe({
        "total_cows": len(records),
        "cows": records
    })


# ============================================================
# SINGLE COW
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/cow/{cow_id}")
def cow(
    cow_id: str,
    current_user: str = Depends(get_current_user)
):

    result = get_cow(cow_id)

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=f"Cow '{cow_id}' not found"
        )

    return make_json_safe(result)


# ============================================================
# CURRENT RISK
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/cow/{cow_id}/risk")
def cow_risk(
    cow_id: str,
    current_user: str = Depends(get_current_user)
):

    result = get_cow(cow_id)

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=f"Cow '{cow_id}' not found"
        )

    return make_json_safe({
        "cow_id": result["cow_id"],
        "date": result["date"],
        "current_risk": result["current_risk"],
        "risk_category": result["risk_category"],
        "alert_level": result["alert_level"],
        "recommendations": result["recommendations"]
    })


# ============================================================
# FORECAST
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/cow/{cow_id}/forecast")
def cow_forecast(
    cow_id: str,
    current_user: str = Depends(get_current_user)
):

    result = get_cow(cow_id)

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=f"Cow '{cow_id}' not found"
        )

    return make_json_safe({
        "cow_id": result["cow_id"],
        "24h_predicted_risk": result["24h_predicted_risk"],
        "48h_predicted_risk": result["48h_predicted_risk"],
        "3d_predicted_risk": result["3d_predicted_risk"],
        "7d_probability": result["7d_probability"],
        "7d_warning": result["7d_warning"],
        "14d_probability": result["14d_probability"],
        "14d_warning": result["14d_warning"]
    })


# ============================================================
# HERD RISK
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/herd-risk")
def herd_risk(
    current_user: str = Depends(get_current_user)
):

    try:

        df = load_predictions()

        cow_predictions = df.to_dict(
            orient="records"
        )

        report = analyze_herd(
            cow_predictions
        )

        # Remove internal Pandas DataFrame
        # before sending response to frontend.
        report.pop(
            "dataframe",
            None
        )

        return make_json_safe(report)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HERD PRIORITY COWS
# 🔒 LOGIN REQUIRED
# ============================================================

@app.get("/herd-risk/priority")
def herd_priority(
    current_user: str = Depends(get_current_user)
):

    try:

        df = load_predictions()

        cow_predictions = df.to_dict(
            orient="records"
        )

        report = analyze_herd(
            cow_predictions
        )

        return make_json_safe({
            "total_cows": report["total_cows"],
            "priority_cows": report["priority_cows"]
        })

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ESP32 HARDWARE TELEMETRY INGESTION & AI RISK PREDICTION
# ============================================================

from model1_to_model2_inference import predict_model1_risk

class SensorDataPayload(BaseModel):
    cow_id: str
    temperature: float = 0.0          # Ambient / DHT11 temp (°C)
    milk_temperature: float = 38.5    # Milk temp probe (°C)
    humidity: float = 0.0             # DHT11 humidity (%)
    accel_x: float = 0.0              # MPU6500 Accel X (m/s^2)
    accel_y: float = 0.0              # MPU6500 Accel Y (m/s^2)
    accel_z: float = 0.0              # MPU6500 Accel Z (m/s^2)
    gyro_x: float = 0.0               # MPU6500 Gyro X
    gyro_y: float = 0.0               # MPU6500 Gyro Y
    gyro_z: float = 0.0               # MPU6500 Gyro Z
    tds_raw: float = 0.0              # Raw TDS ADC count
    tds_voltage: float = 0.0          # TDS Sensor voltage (V)
    weight_raw: float = 0.0           # HX711 Load Cell raw count
    previous_mastitis: int = 0        # 0 = No, 1 = Yes
    days_since_last_mastitis: float = 999.0 # Days since last event
    dht_readings: int = 0
    mpu_readings: int = 0
    tds_readings: int = 0
    hx711_readings: int = 0


@app.post("/api/sensor-data")
def ingest_sensor_data(payload: SensorDataPayload):
    """
    Ingests raw hardware telemetry from ESP32, converts raw sensor metrics to exact physical units
    required by the AI model, and returns instant Random Forest Mastitis Risk predictions.
    """
    try:
        # ==========================================================
        # 1. HARDWARE SENSOR RAW TO PHYSICAL FEATURE CONVERSION
        # ==========================================================
        
        # A. Electrical Conductivity (mS/cm) from TDS Voltage
        # Calibration Formula: Milk Conductivity (mS/cm) = TDS Voltage * 3.45 (Normal range: 4.0 - 4.8 mS/cm)
        if payload.tds_voltage > 0.1:
            milk_conductivity = round(payload.tds_voltage * 3.45, 2)
        else:
            milk_conductivity = 4.35 # Default healthy baseline
            
        # B. Milk Yield (Liters) from HX711 Raw Load Cell Weight
        # Conversion Formula: Milk Yield (L) = (raw_weight - tare_offset) / scale_factor
        # Milk density ~ 1.03 kg/L. If raw count is given without scale, default to standard yield.
        if payload.weight_raw > 1000:
            raw_kg = (payload.weight_raw - 10000.0) / 21000.0  # Example calibration factor
            milk_yield = round(max(0.5, raw_kg / 1.03), 2)
        else:
            milk_yield = 12.5 # Default yield (Liters)
            
        # C. Cow Activity Index from MPU6500 Accelerometer Vector Magnitude
        # Formula: Magnitude = sqrt(ax^2 + ay^2 + az^2)
        # Normal baseline gravity is ~9.81 m/s^2. Excess movement is scaled.
        accel_mag = np.sqrt(payload.accel_x**2 + payload.accel_y**2 + payload.accel_z**2)
        if accel_mag > 0.1:
            cow_activity = round(float(accel_mag * 5.2), 2)
        else:
            cow_activity = 52.0 # Normal activity index
            
        # D. Temperatures & Humidity
        env_temp = payload.temperature if payload.temperature > 0 else 28.0
        milk_temp = payload.milk_temperature if payload.milk_temperature > 0 else 38.5
        humidity = payload.humidity if payload.humidity > 0 else 65.0

        # ==========================================================
        # 2. RUN AI MODEL INFERENCE (Random Forest Model 1)
        # ==========================================================
        risk_percentage = predict_model1_risk(
            milk_yield_liters=milk_yield,
            milk_conductivity_ms_cm=milk_conductivity,
            cow_activity=cow_activity,
            environment_temperature_c=env_temp,
            milk_temperature_c=milk_temp,
            humidity_percent=humidity,
            previous_mastitis=payload.previous_mastitis,
            days_since_last_mastitis=payload.days_since_last_mastitis
        )

        # ==========================================================
        # 3. DECISION ENGINE & ALERT LEVEL CLASSIFICATION
        # ==========================================================
        if risk_percentage <= 25.0:
            risk_category = "Low Risk"
            alert_level = "GREEN"
            recommendation = "Normal physiological metrics. Continue routine monitoring."
        elif risk_percentage <= 50.0:
            risk_category = "Moderate Risk"
            alert_level = "AMBER"
            recommendation = "Elevated conductivity or activity drop detected. Inspect udder & perform CMT."
        else:
            risk_category = "High Risk"
            alert_level = "RED"
            recommendation = "Critical mastitis risk! Immediately isolate cow, perform California Mastitis Test, and alert veterinarian."

        return make_json_safe({
            "status": "success",
            "cow_id": payload.cow_id,
            "raw_sensor_inputs": {
                "tds_voltage": payload.tds_voltage,
                "weight_raw": payload.weight_raw,
                "mpu_accel_vector": [payload.accel_x, payload.accel_y, payload.accel_z],
                "dht_temp_c": payload.temperature,
                "dht_humidity_pct": payload.humidity
            },
            "converted_ai_features": {
                "milk_conductivity_ms_cm": milk_conductivity,
                "milk_yield_liters": milk_yield,
                "cow_activity": cow_activity,
                "environment_temperature_c": env_temp,
                "milk_temperature_c": milk_temp,
                "humidity_percent": humidity
            },
            "ai_prediction": {
                "predicted_risk_percent": round(risk_percentage, 2),
                "risk_category": risk_category,
                "alert_level": alert_level,
                "recommendation": recommendation
            }
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Telemetry conversion & AI inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

