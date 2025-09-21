import mysql.connector
from mysql.connector import Error
from datetime import timedelta
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any

load_dotenv()  # يبحث عن .env في مجلد المشروع الحالي

# =========================
# Environment / Config
# =========================
db_name = os.getenv("DB_NAME")
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")
DATA_TABLE ="wpl3_crisis_management_tool"
RESULTS_TABLE = "wpl3_crisis_management_result"

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        if connection.is_connected():
            print("✅ Connected!")
            connection.autocommit = True
            return connection
    except Error as e:
        print("❌ Failed.")
        print(f"Error connecting to MySQL: {e}")
        return None

def save_result(request_id: int, user_id: int, result_text: str) -> None:
    sql = f"""
    INSERT INTO `{RESULTS_TABLE}` (`request_id`, `user_id`, `result`, `edited_result`)
    VALUES (%s, %s, %s, %s)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (request_id, user_id, result_text, None))
            conn.commit()
            print("Data saved successfully")  # Debugging print
    finally:
        conn.close()

def fetch_latest_result(request_id: int) -> Optional[Dict[str, Any]]:
    sql = f"""
    SELECT request_id, user_id, result, edited_result, date
    FROM `{RESULTS_TABLE}`
    WHERE request_id = %s
    ORDER BY date DESC, id DESC
    LIMIT 1
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur = conn.cursor(dictionary=True) 
            cur.execute(sql, (request_id,))
            row = cur.fetchone()
            print("Fetched result:", type(row))  # Debugging print
            return row
    finally:
        cur.close()
        conn.close()