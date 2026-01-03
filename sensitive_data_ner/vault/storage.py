"""
Vault Storage Module

Provides encrypted storage backends for sensitive data mappings.
Supports SQLite (local) and PostgreSQL (enterprise).
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

try:
    import psycopg2
    from psycopg2.extras import Json
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False


@dataclass
class StoredMapping:
    """Represents a stored original-to-anonymized mapping."""
    mapping_id: str
    session_id: str
    original_encrypted: bytes  # Original value encrypted
    anonymized_value: str
    entity_type: str
    category: str
    position_start: int
    position_end: int
    key_id: str  # Which key was used
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class StoredSession:
    """Represents a stored session with full message context."""
    session_id: str
    original_message_encrypted: bytes  # Full original message
    anonymized_message: str
    entity_count: int
    risk_score: float
    key_id: str
    created_at: datetime
    accessed_at: Optional[datetime]
    access_count: int
    metadata: Dict[str, Any]
    # Retention policy fields
    retention_policy: str = "DELETE_ON_DECRYPT"  # DELETE_ON_DECRYPT, RETAIN_DAYS, RETAIN_FOREVER
    retention_days: Optional[int] = None
    decrypted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class VaultStorage(ABC):
    """Abstract base class for vault storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage (create tables, etc.)."""
        pass

    @abstractmethod
    def store_session(self, session: StoredSession) -> str:
        """Store a session record."""
        pass

    @abstractmethod
    def store_mapping(self, mapping: StoredMapping) -> str:
        """Store a mapping record."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Retrieve a session by ID."""
        pass

    @abstractmethod
    def get_mappings_for_session(self, session_id: str) -> List[StoredMapping]:
        """Get all mappings for a session."""
        pass

    @abstractmethod
    def update_session_access(self, session_id: str) -> None:
        """Update session access timestamp and count."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its mappings."""
        pass

    @abstractmethod
    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[StoredSession]:
        """List sessions with pagination."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

    @abstractmethod
    def update_session_decrypted(
        self,
        session_id: str,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session with decryption timestamp and expiration."""
        pass

    @abstractmethod
    def get_expired_sessions(self) -> List[str]:
        """Get list of expired session IDs."""
        pass

    @abstractmethod
    def delete_expired_sessions(self) -> int:
        """Delete all expired sessions. Returns count deleted."""
        pass

    @abstractmethod
    def update_session_retention(
        self,
        session_id: str,
        retention_policy: str,
        retention_days: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session retention policy."""
        pass


class SQLiteVaultStorage(VaultStorage):
    """
    SQLite-based vault storage for local/development use.

    Uses file-based SQLite with encryption at rest via application-level
    encryption of sensitive fields.
    """

    def __init__(self, db_path: str = ".vault_data/vault.db"):
        self._db_path = Path(db_path)
        self._ensure_directory()
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_directory(self) -> None:
        """Create storage directory with secure permissions."""
        if not self._db_path.parent.exists():
            self._db_path.parent.mkdir(parents=True, mode=0o700)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self._conn:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row

            # Set secure pragmas
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

        return self._conn

    def initialize(self) -> None:
        """Create database tables."""
        conn = self._get_connection()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                original_message_encrypted BLOB NOT NULL,
                anonymized_message TEXT NOT NULL,
                entity_count INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                key_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                retention_policy TEXT DEFAULT 'DELETE_ON_DECRYPT',
                retention_days INTEGER,
                decrypted_at TEXT,
                expires_at TEXT
            );

            CREATE TABLE IF NOT EXISTS mappings (
                mapping_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                original_encrypted BLOB NOT NULL,
                anonymized_value TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                category TEXT NOT NULL,
                position_start INTEGER NOT NULL,
                position_end INTEGER NOT NULL,
                key_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                ON sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_mappings_session_id
                ON mappings(session_id);
            CREATE INDEX IF NOT EXISTS idx_mappings_entity_type
                ON mappings(entity_type);
        """)

        # Add new columns if they don't exist (for migration)
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN retention_policy TEXT DEFAULT 'DELETE_ON_DECRYPT'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN retention_days INTEGER")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN decrypted_at TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN expires_at TEXT")
        except sqlite3.OperationalError:
            pass

        conn.commit()

        # Set secure file permissions
        os.chmod(self._db_path, 0o600)

    def store_session(self, session: StoredSession) -> str:
        """Store a session record."""
        conn = self._get_connection()

        conn.execute("""
            INSERT INTO sessions (
                session_id, original_message_encrypted, anonymized_message,
                entity_count, risk_score, key_id, created_at, accessed_at,
                access_count, metadata, retention_policy, retention_days,
                decrypted_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.original_message_encrypted,
            session.anonymized_message,
            session.entity_count,
            session.risk_score,
            session.key_id,
            session.created_at.isoformat(),
            session.accessed_at.isoformat() if session.accessed_at else None,
            session.access_count,
            json.dumps(session.metadata),
            session.retention_policy,
            session.retention_days,
            session.decrypted_at.isoformat() if session.decrypted_at else None,
            session.expires_at.isoformat() if session.expires_at else None
        ))

        conn.commit()
        return session.session_id

    def store_mapping(self, mapping: StoredMapping) -> str:
        """Store a mapping record."""
        conn = self._get_connection()

        conn.execute("""
            INSERT INTO mappings (
                mapping_id, session_id, original_encrypted, anonymized_value,
                entity_type, category, position_start, position_end,
                key_id, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mapping.mapping_id,
            mapping.session_id,
            mapping.original_encrypted,
            mapping.anonymized_value,
            mapping.entity_type,
            mapping.category,
            mapping.position_start,
            mapping.position_end,
            mapping.key_id,
            mapping.created_at.isoformat(),
            json.dumps(mapping.metadata)
        ))

        conn.commit()
        return mapping.mapping_id

    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Retrieve a session by ID."""
        conn = self._get_connection()

        row = conn.execute("""
            SELECT * FROM sessions WHERE session_id = ?
        """, (session_id,)).fetchone()

        if not row:
            return None

        return StoredSession(
            session_id=row["session_id"],
            original_message_encrypted=row["original_message_encrypted"],
            anonymized_message=row["anonymized_message"],
            entity_count=row["entity_count"],
            risk_score=row["risk_score"],
            key_id=row["key_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"])
                if row["accessed_at"] else None,
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            retention_policy=row["retention_policy"] if "retention_policy" in row.keys() else "DELETE_ON_DECRYPT",
            retention_days=row["retention_days"] if "retention_days" in row.keys() else None,
            decrypted_at=datetime.fromisoformat(row["decrypted_at"])
                if row.get("decrypted_at") else None,
            expires_at=datetime.fromisoformat(row["expires_at"])
                if row.get("expires_at") else None
        )

    def get_mappings_for_session(self, session_id: str) -> List[StoredMapping]:
        """Get all mappings for a session."""
        conn = self._get_connection()

        rows = conn.execute("""
            SELECT * FROM mappings WHERE session_id = ?
            ORDER BY position_start
        """, (session_id,)).fetchall()

        return [
            StoredMapping(
                mapping_id=row["mapping_id"],
                session_id=row["session_id"],
                original_encrypted=row["original_encrypted"],
                anonymized_value=row["anonymized_value"],
                entity_type=row["entity_type"],
                category=row["category"],
                position_start=row["position_start"],
                position_end=row["position_end"],
                key_id=row["key_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]

    def update_session_access(self, session_id: str) -> None:
        """Update session access timestamp and count."""
        conn = self._get_connection()

        conn.execute("""
            UPDATE sessions
            SET accessed_at = ?, access_count = access_count + 1
            WHERE session_id = ?
        """, (datetime.utcnow().isoformat(), session_id))

        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its mappings."""
        conn = self._get_connection()

        cursor = conn.execute("""
            DELETE FROM sessions WHERE session_id = ?
        """, (session_id,))

        conn.commit()
        return cursor.rowcount > 0

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[StoredSession]:
        """List sessions with pagination."""
        conn = self._get_connection()

        query = "SELECT * FROM sessions WHERE 1=1"
        params = []

        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

        return [
            StoredSession(
                session_id=row["session_id"],
                original_message_encrypted=row["original_message_encrypted"],
                anonymized_message=row["anonymized_message"],
                entity_count=row["entity_count"],
                risk_score=row["risk_score"],
                key_id=row["key_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                accessed_at=datetime.fromisoformat(row["accessed_at"])
                    if row["accessed_at"] else None,
                access_count=row["access_count"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()

        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]

        mapping_count = conn.execute(
            "SELECT COUNT(*) FROM mappings"
        ).fetchone()[0]

        entity_stats = conn.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM mappings
            GROUP BY entity_type
            ORDER BY count DESC
        """).fetchall()

        category_stats = conn.execute("""
            SELECT category, COUNT(*) as count
            FROM mappings
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_sessions": session_count,
            "total_mappings": mapping_count,
            "entities_by_type": {row[0]: row[1] for row in entity_stats},
            "entities_by_category": {row[0]: row[1] for row in category_stats},
            "storage_path": str(self._db_path),
            "storage_size_bytes": self._db_path.stat().st_size
                if self._db_path.exists() else 0
        }

    def update_session_decrypted(
        self,
        session_id: str,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session with decryption timestamp and expiration."""
        conn = self._get_connection()

        conn.execute("""
            UPDATE sessions
            SET decrypted_at = ?, expires_at = ?
            WHERE session_id = ?
        """, (
            datetime.utcnow().isoformat(),
            expires_at.isoformat() if expires_at else None,
            session_id
        ))

        conn.commit()

    def get_expired_sessions(self) -> List[str]:
        """Get list of expired session IDs."""
        conn = self._get_connection()

        now = datetime.utcnow().isoformat()
        rows = conn.execute("""
            SELECT session_id FROM sessions
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (now,)).fetchall()

        return [row["session_id"] for row in rows]

    def delete_expired_sessions(self) -> int:
        """Delete all expired sessions. Returns count deleted."""
        conn = self._get_connection()

        now = datetime.utcnow().isoformat()
        cursor = conn.execute("""
            DELETE FROM sessions
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (now,))

        conn.commit()
        return cursor.rowcount

    def update_session_retention(
        self,
        session_id: str,
        retention_policy: str,
        retention_days: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session retention policy."""
        conn = self._get_connection()

        conn.execute("""
            UPDATE sessions
            SET retention_policy = ?, retention_days = ?, expires_at = ?
            WHERE session_id = ?
        """, (
            retention_policy,
            retention_days,
            expires_at.isoformat() if expires_at else None,
            session_id
        ))

        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class PostgreSQLVaultStorage(VaultStorage):
    """
    PostgreSQL-based vault storage for enterprise use.

    Requires PostgreSQL with pgcrypto extension for additional security.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "vault",
        user: str = "vault_user",
        password: str = "",
        schema: str = "vault"
    ):
        if not HAS_POSTGRES:
            raise ImportError(
                "psycopg2 package required. "
                "Install with: pip install psycopg2-binary"
            )

        self._conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self._schema = schema
        self._conn: Optional[Any] = None

    def _get_connection(self):
        """Get database connection."""
        if not self._conn or self._conn.closed:
            self._conn = psycopg2.connect(**self._conn_params)
        return self._conn

    def initialize(self) -> None:
        """Create database tables."""
        conn = self._get_connection()
        cur = conn.cursor()

        # Create schema
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")

        # Create tables
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.sessions (
                session_id TEXT PRIMARY KEY,
                original_message_encrypted BYTEA NOT NULL,
                anonymized_message TEXT NOT NULL,
                entity_count INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                key_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                accessed_at TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                metadata JSONB,
                retention_policy TEXT DEFAULT 'DELETE_ON_DECRYPT',
                retention_days INTEGER,
                decrypted_at TIMESTAMP,
                expires_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS {self._schema}.mappings (
                mapping_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES {self._schema}.sessions(session_id)
                    ON DELETE CASCADE,
                original_encrypted BYTEA NOT NULL,
                anonymized_value TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                category TEXT NOT NULL,
                position_start INTEGER NOT NULL,
                position_end INTEGER NOT NULL,
                key_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata JSONB
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                ON {self._schema}.sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                ON {self._schema}.sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_mappings_session_id
                ON {self._schema}.mappings(session_id);
            CREATE INDEX IF NOT EXISTS idx_mappings_entity_type
                ON {self._schema}.mappings(entity_type);
        """)

        # Add new columns if they don't exist (for migration)
        for col, col_type, default in [
            ("retention_policy", "TEXT", "'DELETE_ON_DECRYPT'"),
            ("retention_days", "INTEGER", "NULL"),
            ("decrypted_at", "TIMESTAMP", "NULL"),
            ("expires_at", "TIMESTAMP", "NULL"),
        ]:
            try:
                cur.execute(f"""
                    ALTER TABLE {self._schema}.sessions
                    ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}
                """)
            except Exception:
                pass

        conn.commit()

    def store_session(self, session: StoredSession) -> str:
        """Store a session record."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            INSERT INTO {self._schema}.sessions (
                session_id, original_message_encrypted, anonymized_message,
                entity_count, risk_score, key_id, created_at, accessed_at,
                access_count, metadata, retention_policy, retention_days,
                decrypted_at, expires_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session.session_id,
            session.original_message_encrypted,
            session.anonymized_message,
            session.entity_count,
            session.risk_score,
            session.key_id,
            session.created_at,
            session.accessed_at,
            session.access_count,
            Json(session.metadata),
            session.retention_policy,
            session.retention_days,
            session.decrypted_at,
            session.expires_at
        ))

        conn.commit()
        return session.session_id

    def store_mapping(self, mapping: StoredMapping) -> str:
        """Store a mapping record."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            INSERT INTO {self._schema}.mappings (
                mapping_id, session_id, original_encrypted, anonymized_value,
                entity_type, category, position_start, position_end,
                key_id, created_at, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            mapping.mapping_id,
            mapping.session_id,
            mapping.original_encrypted,
            mapping.anonymized_value,
            mapping.entity_type,
            mapping.category,
            mapping.position_start,
            mapping.position_end,
            mapping.key_id,
            mapping.created_at,
            Json(mapping.metadata)
        ))

        conn.commit()
        return mapping.mapping_id

    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Retrieve a session by ID."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            SELECT * FROM {self._schema}.sessions WHERE session_id = %s
        """, (session_id,))

        row = cur.fetchone()
        if not row:
            return None

        return StoredSession(
            session_id=row[0],
            original_message_encrypted=bytes(row[1]),
            anonymized_message=row[2],
            entity_count=row[3],
            risk_score=row[4],
            key_id=row[5],
            created_at=row[6],
            accessed_at=row[7],
            access_count=row[8],
            metadata=row[9] or {},
            retention_policy=row[10] if len(row) > 10 else "DELETE_ON_DECRYPT",
            retention_days=row[11] if len(row) > 11 else None,
            decrypted_at=row[12] if len(row) > 12 else None,
            expires_at=row[13] if len(row) > 13 else None
        )

    def get_mappings_for_session(self, session_id: str) -> List[StoredMapping]:
        """Get all mappings for a session."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            SELECT * FROM {self._schema}.mappings
            WHERE session_id = %s
            ORDER BY position_start
        """, (session_id,))

        return [
            StoredMapping(
                mapping_id=row[0],
                session_id=row[1],
                original_encrypted=bytes(row[2]),
                anonymized_value=row[3],
                entity_type=row[4],
                category=row[5],
                position_start=row[6],
                position_end=row[7],
                key_id=row[8],
                created_at=row[9],
                metadata=row[10] or {}
            )
            for row in cur.fetchall()
        ]

    def update_session_access(self, session_id: str) -> None:
        """Update session access timestamp and count."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            UPDATE {self._schema}.sessions
            SET accessed_at = %s, access_count = access_count + 1
            WHERE session_id = %s
        """, (datetime.utcnow(), session_id))

        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its mappings."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            DELETE FROM {self._schema}.sessions WHERE session_id = %s
        """, (session_id,))

        deleted = cur.rowcount > 0
        conn.commit()
        return deleted

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[StoredSession]:
        """List sessions with pagination."""
        conn = self._get_connection()
        cur = conn.cursor()

        query = f"SELECT * FROM {self._schema}.sessions WHERE TRUE"
        params = []

        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)

        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cur.execute(query, params)

        return [
            StoredSession(
                session_id=row[0],
                original_message_encrypted=bytes(row[1]),
                anonymized_message=row[2],
                entity_count=row[3],
                risk_score=row[4],
                key_id=row[5],
                created_at=row[6],
                accessed_at=row[7],
                access_count=row[8],
                metadata=row[9] or {}
            )
            for row in cur.fetchall()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM {self._schema}.sessions")
        session_count = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {self._schema}.mappings")
        mapping_count = cur.fetchone()[0]

        cur.execute(f"""
            SELECT entity_type, COUNT(*) as count
            FROM {self._schema}.mappings
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        entity_stats = cur.fetchall()

        cur.execute(f"""
            SELECT category, COUNT(*) as count
            FROM {self._schema}.mappings
            GROUP BY category
            ORDER BY count DESC
        """)
        category_stats = cur.fetchall()

        return {
            "total_sessions": session_count,
            "total_mappings": mapping_count,
            "entities_by_type": {row[0]: row[1] for row in entity_stats},
            "entities_by_category": {row[0]: row[1] for row in category_stats},
            "backend": "postgresql"
        }

    def update_session_decrypted(
        self,
        session_id: str,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session with decryption timestamp and expiration."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            UPDATE {self._schema}.sessions
            SET decrypted_at = %s, expires_at = %s
            WHERE session_id = %s
        """, (datetime.utcnow(), expires_at, session_id))

        conn.commit()

    def get_expired_sessions(self) -> List[str]:
        """Get list of expired session IDs."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            SELECT session_id FROM {self._schema}.sessions
            WHERE expires_at IS NOT NULL AND expires_at <= %s
        """, (datetime.utcnow(),))

        return [row[0] for row in cur.fetchall()]

    def delete_expired_sessions(self) -> int:
        """Delete all expired sessions. Returns count deleted."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            DELETE FROM {self._schema}.sessions
            WHERE expires_at IS NOT NULL AND expires_at <= %s
        """, (datetime.utcnow(),))

        deleted = cur.rowcount
        conn.commit()
        return deleted

    def update_session_retention(
        self,
        session_id: str,
        retention_policy: str,
        retention_days: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Update session retention policy."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(f"""
            UPDATE {self._schema}.sessions
            SET retention_policy = %s, retention_days = %s, expires_at = %s
            WHERE session_id = %s
        """, (retention_policy, retention_days, expires_at, session_id))

        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
