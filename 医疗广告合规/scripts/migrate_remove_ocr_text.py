"""
SQLite migration script to remove the legacy `ocr_text` column from `reports` table.
Usage: python scripts/migrate_remove_ocr_text.py
It will create a backup file `reports.db.bak` before applying the migration.
"""
from pathlib import Path
import shutil
import sqlite3

DB_PATH = Path('reports.db')
BACKUP_PATH = DB_PATH.with_suffix('.db.bak')

if not DB_PATH.exists():
    print(f"Database not found at {DB_PATH}. Nothing to migrate.")
    exit(1)

print(f"Backing up {DB_PATH} -> {BACKUP_PATH} ...")
shutil.copy2(DB_PATH, BACKUP_PATH)

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Check existing columns
cursor.execute("PRAGMA table_info(reports)")
cols = [r[1] for r in cursor.fetchall()]
print("Existing columns:", cols)

if 'ocr_text' not in cols:
    print('ocr_text column not present. Migration not required.')
    conn.close()
    exit(0)

print('Starting migration: removing ocr_text column...')
try:
    cursor.execute('BEGIN')
    # Create new table without ocr_text
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_text TEXT NOT NULL,
            verdict TEXT NOT NULL,
            violation_count INTEGER,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # Copy data from old table to new table (exclude ocr_text)
    cursor.execute('''
        INSERT INTO reports_new (id, timestamp, input_text, verdict, violation_count, report_json, created_at)
        SELECT id, timestamp, input_text, verdict, violation_count, report_json, created_at FROM reports
    ''')

    # Drop old table and rename
    cursor.execute('DROP TABLE reports')
    cursor.execute('ALTER TABLE reports_new RENAME TO reports')

    conn.commit()
    print('Migration completed successfully.')
except Exception as e:
    conn.rollback()
    print('Migration failed:', e)
    print('Restoring from backup...')
    shutil.copy2(BACKUP_PATH, DB_PATH)
finally:
    conn.close()
