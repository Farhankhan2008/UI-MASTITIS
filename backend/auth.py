import os
import sqlite3
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext

from database import get_connection

# Secret key for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "mastitis_secret_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


import hashlib

def hash_password(password):
    try:
        return pwd_context.hash(password)
    except Exception:
        salt = "mastitis_salt_2026"
        return "sha256$" + hashlib.sha256((salt + password).encode('utf-8')).hexdigest()


def verify_password(plain_password, hashed_password):
    if hashed_password.startswith("sha256$"):
        salt = "mastitis_salt_2026"
        expected = "sha256$" + hashlib.sha256((salt + plain_password).encode('utf-8')).hexdigest()
        return expected == hashed_password
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


# ==========================
# User Functions
# ==========================

def create_user(username, password, role="farmer"):
    connection = get_connection()

    hashed = hash_password(password)

    try:
        connection.execute(
            """
            INSERT INTO users
            (username, password, role)
            VALUES (?, ?, ?)
            """,
            (username, hashed, role)
        )

        connection.commit()

    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

    finally:
        connection.close()

    return True


def get_user(username):
    connection = get_connection()

    user = connection.execute(
        """
        SELECT * FROM users
        WHERE username = ?
        """,
        (username,)
    ).fetchone()

    connection.close()

    return user


def authenticate_user(username, password):
    user = get_user(username)

    if not user:
        return None

    if not verify_password(
        password,
        user["password"]
    ):
        return None

    return user


# ==========================
# JWT Token
# ==========================

def create_access_token(data):
    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode = data.copy()
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

# ==========================
# Verify JWT Token
# ==========================

def verify_token(token):
    try:
        from jose import JWTError

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if username is None:
            return None

        return username

    except JWTError:
        return None