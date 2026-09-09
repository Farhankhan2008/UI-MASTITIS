from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext

from database import get_connection

# Secret key for JWT
SECRET_KEY = "mastitis_secret_key_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# ==========================
# Password Functions
# ==========================

def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


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

    except:
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