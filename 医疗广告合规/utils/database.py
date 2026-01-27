import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path('reports.db')


def init_db():
    """初始化数据库表结构。"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_text TEXT NOT NULL,
            ocr_text TEXT,
            verdict TEXT NOT NULL,
            violation_count INTEGER,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_report(report_data):
    """保存检测报告到数据库。
    
    report_data: dict，包含 timestamp, input_text, ocr_text, verdict, 等字段。
    """
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    violation_count = len(report_data.get('violation_rule_matches', [])) + \
                      len(report_data.get('regex_matches', [])) + \
                      len(report_data.get('semantic_matches', []))
    
    cursor.execute('''
        INSERT INTO reports (timestamp, input_text, ocr_text, verdict, violation_count, report_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_data.get('timestamp'),
        report_data.get('input_text')[:500],  # 仅存储前 500 字符用于查询
        report_data.get('ocr_text', '')[:500],
        report_data.get('verdict'),
        violation_count,
        json.dumps(report_data, ensure_ascii=False),
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()


def get_reports(limit=100):
    """获取最近的报告列表。"""
    init_db()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id, timestamp, input_text, verdict, violation_count, created_at
            FROM reports
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Database query error: {e}")
        rows = []
    finally:
        conn.close()
    return rows


def get_report_by_id(report_id):
    """按 ID 获取完整报告。"""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('SELECT report_json FROM reports WHERE id = ?', (report_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def delete_report(report_id):
    """删除指定的报告。"""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()


def get_statistics():
    """获取统计信息。"""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM reports')
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM reports WHERE verdict = '疑似违规'")
    violations = cursor.fetchone()[0]
    conn.close()
    return {'total': total, 'violations': violations, 'compliant': total - violations}
