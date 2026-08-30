import sqlite3
import json
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "pipeline.db"

def get_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialises the database schema."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Table for Pipeline Runs (History)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                status TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                config JSON,
                failures JSON,
                recovery_used TEXT,
                mttr_sec REAL
            )
        ''')
        
        # Table for Knowledge Repository Entries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_signature TEXT NOT NULL,
                root_cause_type TEXT NOT NULL,
                recovery_strategy_used TEXT NOT NULL,
                outcome TEXT NOT NULL,
                notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

# Ensure DB is initialised when module is imported
init_db()
