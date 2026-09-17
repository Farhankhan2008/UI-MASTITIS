import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).parent / "users.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_users_table():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'farmer'
        )
    """)

    connection.commit()

    # Seed default farmer account if table is empty
    cursor = connection.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        from auth import hash_password
        hashed = hash_password("farmer")
        connection.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("farmer", hashed, "farmer")
        )
        connection.commit()

    connection.close()