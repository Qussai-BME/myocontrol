"""
database.py - Session persistence for MyoControl Suite

Stores each analysis run (config + full engine result) in a local SQLite
database so users can browse, reload, and compare past sessions instead of
losing everything on page refresh. This is local file-based storage (no
server/credentials needed) — suitable for a single-user research or
clinical workstation, not a multi-user deployment (see note in
list_sessions() docstring if you need that).
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    label TEXT,
    source TEXT,
    gesture TEXT,
    intensity REAL,
    duration_seconds REAL,
    n_channels INTEGER,
    sampling_rate INTEGER,
    snr_db REAL,
    snr_quality TEXT,
    model_type TEXT,
    loso_accuracy_mean REAL,
    result_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    acquisition_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
"""


@contextmanager
def _connect(db_path: Optional[Path] = None):
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_session(
    result: Dict[str, Any],
    config: Any,
    acquisition_info: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    model_type: Optional[str] = None,
    loso_accuracy_mean: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Persist one analysis run. Returns the new session id."""
    acquisition_info = acquisition_info or {}
    config_dict = config.__dict__ if hasattr(config, '__dict__') else dict(config)

    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (created_at, label, source, gesture, intensity, duration_seconds,
                n_channels, sampling_rate, snr_db, snr_quality, model_type,
                loso_accuracy_mean, result_json, config_json, acquisition_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                label,
                acquisition_info.get('source'),
                acquisition_info.get('gesture'),
                acquisition_info.get('intensity'),
                result['metadata']['duration_seconds'],
                result['metadata']['n_channels'],
                result['metadata']['sampling_rate'],
                result['signal_quality']['mean_snr_db'],
                result['signal_quality']['snr_quality'],
                model_type,
                loso_accuracy_mean,
                json.dumps(result, default=str),
                json.dumps(config_dict, default=str),
                json.dumps(acquisition_info, default=str),
            ),
        )
        return int(cur.lastrowid)


def list_sessions(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return lightweight session summaries (no full result blob), newest first.

    NOTE: this file-based SQLite store has no user/auth separation — every
    caller sees every session. Fine for a single-user workstation; if this
    ever needs to serve multiple people, add a user_id column and filter on
    it before reusing this as-is.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, created_at, label, source, gesture, intensity,
                      duration_seconds, n_channels, sampling_rate, snr_db,
                      snr_quality, model_type, loso_accuracy_mean
               FROM sessions ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def load_session(session_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the full stored result + config + acquisition info for one session."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out['result'] = json.loads(out.pop('result_json'))
        out['config'] = json.loads(out.pop('config_json'))
        out['acquisition_info'] = json.loads(out.pop('acquisition_json') or '{}')
        return out


def delete_session(session_id: int, db_path: Optional[Path] = None) -> bool:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0
