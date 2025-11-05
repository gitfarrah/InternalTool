# src/storage/sqlite_db.py
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/pages.db")

def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = get_connection()
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        source TEXT,
        title TEXT,
        text TEXT,
        sha256 TEXT,
        http_status INTEGER,
        fetch_ts TEXT,
        indexed INTEGER DEFAULT 0
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS fts_pages
    USING fts5(title, url, source, content='');
    """)
    con.commit()
    con.close()

if __name__ == "__main__":
    init_db()
    print(f"SQLite database initialized at {DB_PATH.resolve()}")
