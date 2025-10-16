import os
from typing import Optional, Dict, Any

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# =========================
# Environment / Config
# =========================
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT") or 3306)

DATA_TABLE    = "wpl3_crisis_management_tool"
RESULTS_TABLE = "wpl3_crisis_management_result"

def get_db_connection():
    """
    Returns a live MySQL connection or None on failure.
    """
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            autocommit=True,   # we also commit explicitly when needed
        )
        if conn.is_connected():
            print("✅ Connected!")
            return conn
    except Error as e:
        print("❌ Failed.")
        print(f"Error connecting to MySQL: {e}")
    return None

def save_result(request_id: int, user_id: int, result_text: str) -> None:
    """
    First-write behavior:
      - Insert a new row with both `result` and `edited_result` set to `result_text`.
    If a row already exists for (request_id, user_id), update both fields (idempotent).
    This guarantees downstream readers always see a non-NULL edited_result immediately.
    """
    conn = get_db_connection()
    if conn is None:
        print("❌ No DB connection in save_result()")
        return

    try:
        cur = conn.cursor(dictionary=True)

        # Check if we already have a row for this request_id + user_id
        cur.execute(
            f"""
              SELECT id, result, edited_result
              FROM `{RESULTS_TABLE}`
              WHERE request_id = %s AND user_id = %s
              ORDER BY id DESC
              LIMIT 1
            """,
            (request_id, user_id),
        )
        existing = cur.fetchone()

        if existing:
            # Update both fields to keep them in sync on re-generation
            cur.execute(
                f"""
                  UPDATE `{RESULTS_TABLE}`
                  SET result = %s,
                      edited_result = %s,
                      updated_at = CURDATE()
                  WHERE id = %s
                """,
                (result_text, result_text, existing["id"]),
            )
        else:
            # First insert: set edited_result = result_text (NOT NULL)
            cur.execute(
                f"""
                  INSERT INTO `{RESULTS_TABLE}` (request_id, user_id, result, edited_result, date, updated_at)
                  VALUES (%s, %s, %s, %s, CURDATE(), CURDATE())
                """,
                (request_id, user_id, result_text, result_text),
            )

        conn.commit()
        print("💾 Data saved successfully")
    except Error as e:
        print(f"❌ save_result() error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def fetch_latest_result(request_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns the latest row (dict) for a given request_id, or None if not found.
    We order by id DESC first (finer granularity than DATE), then by date.
    """
    conn = get_db_connection()
    if conn is None:
        print("❌ No DB connection in fetch_latest_result()")
        return None

    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
              SELECT id, request_id, user_id, result, edited_result, date, updated_at
              FROM `{RESULTS_TABLE}`
              WHERE request_id = %s
              ORDER BY id DESC, date DESC
              LIMIT 1
            """,
            (request_id,),
        )
        row = cur.fetchone()
        print("Fetched result:", type(row))
        return row  # may be None
    except Error as e:
        print(f"❌ fetch_latest_result() error: {e}")
        return None
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
