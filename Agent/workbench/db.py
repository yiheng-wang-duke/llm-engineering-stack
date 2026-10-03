import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def uid():
    return uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id)
                    ON DELETE CASCADE, question TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, finished_at TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL
                    REFERENCES sessions(id) ON DELETE CASCADE, role TEXT NOT NULL,
                    content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]',
                    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL
                    REFERENCES runs(id) ON DELETE CASCADE, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, digest TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL, embedding_key TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE, ordinal INTEGER NOT NULL, content TEXT NOT NULL,
                    page INTEGER, vector TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def rows(self, sql, params=()):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def execute(self, sql, params=()):
        with self.connect() as conn:
            return conn.execute(sql, params).rowcount

    def create_session(self, title="新对话"):
        item = dict(id=uid(), title=title, created_at=now(), updated_at=now())
        self.execute("INSERT INTO sessions VALUES (:id,:title,:created_at,:updated_at)", item)
        return item

    def sessions(self):
        return self.rows("SELECT * FROM sessions ORDER BY updated_at DESC")

    def messages(self, session_id):
        items = self.rows("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,))
        for item in items:
            item["sources"] = json.loads(item["sources"])
        return items

    def start_run(self, session_id, question):
        run_id = uid()
        self.execute(
            "INSERT INTO runs VALUES (?,?,?,'running',?,NULL,NULL)",
            (run_id, session_id, question, now()),
        )
        return run_id

    def event(self, run_id, kind, payload):
        self.execute(
            "INSERT INTO events(run_id,kind,payload,created_at) VALUES (?,?,?,?)",
            (run_id, kind, json.dumps(payload, ensure_ascii=False), now()),
        )

    def finish_turn(self, run_id, session_id, question, answer, sources):
        with self.connect() as conn:
            for role, content, refs in [("user", question, []), ("assistant", answer, sources)]:
                conn.execute(
                    """INSERT INTO messages
                    (session_id,role,content,sources,run_id,created_at) VALUES (?,?,?,?,?,?)""",
                    (
                        session_id,
                        role,
                        content,
                        json.dumps(refs, ensure_ascii=False),
                        run_id,
                        now(),
                    ),
                )
            conn.execute(
                "UPDATE runs SET status='completed',finished_at=? WHERE id=?", (now(), run_id)
            )
            conn.execute(
                """UPDATE sessions SET updated_at=?,
                title=CASE WHEN title='新对话' THEN ? ELSE title END WHERE id=?""",
                (now(), question[:36], session_id),
            )

    def fail_run(self, run_id, error, status="failed"):
        self.execute(
            "UPDATE runs SET status=?,error=?,finished_at=? WHERE id=? AND status='running'",
            (status, error, now(), run_id),
        )

    def add_memory(self, content):
        content = content.strip()
        if not content or len(content) > 1000:
            raise ValueError("记忆长度需为 1–1000 字符")
        self.execute("INSERT OR IGNORE INTO memories VALUES (?,?,?)", (uid(), content, now()))
        return self.rows("SELECT * FROM memories WHERE content=?", (content,))[0]

    def memories(self):
        return self.rows("SELECT * FROM memories ORDER BY created_at DESC")

    def documents(self):
        return self.rows("""SELECT d.id,d.name,d.digest,d.embedding_key,d.created_at,
            COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
            GROUP BY d.id ORDER BY d.created_at DESC""")
