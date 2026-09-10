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

BASE_DIR = r"D:\Mastitis"

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