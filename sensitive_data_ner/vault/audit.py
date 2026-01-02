"""
Audit Logging Module

Comprehensive audit trail for all vault operations with:
- Tamper-evident logging
- Event categorization
- Query capabilities
"""

import json
import hashlib
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import threading


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"

    # User management
    USER_CREATED = "user.created"
    USER_DEACTIVATED = "user.deactivated"
    USER_PASSWORD_CHANGED = "user.password.changed"
    USER_ACCESS_GRANTED = "user.access.granted"

    # Key management
    KEY_GENERATED = "key.generated"
    KEY_ROTATED = "key.rotated"
    KEY_DESTROYED = "key.destroyed"

    # Data operations
    DATA_STORED = "data.stored"
    DATA_RETRIEVED = "data.retrieved"
    DATA_DECRYPTED = "data.decrypted"
    DATA_DELETED = "data.deleted"

    # Session operations
    SESSION_CREATED = "session.created"
    SESSION_ACCESSED = "session.accessed"
    SESSION_DELETED = "session.deleted"

    # Security events
    SECURITY_ACCESS_DENIED = "security.access.denied"
    SECURITY_ANOMALY_DETECTED = "security.anomaly.detected"
    SECURITY_LOCKOUT = "security.lockout"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"


@dataclass
class AuditEvent:
    """Represents an audit log event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    previous_hash: str  # For tamper detection
    event_hash: str = ""

    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash for tamper detection."""
        data = (
            f"{self.event_id}|{self.event_type.value}|"
            f"{self.timestamp.isoformat()}|{self.user_id}|"
            f"{self.session_id}|{json.dumps(self.details, sort_keys=True)}|"
            f"{self.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event has not been tampered with."""
        return self.event_hash == self._compute_hash()


class AuditLogger:
    """
    Secure audit logger with tamper-evident chain.

    Features:
    - Hash chain for tamper detection
    - Compressed storage
    - Rotation and archival
    - Query interface
    """

    def __init__(
        self,
        storage_path: str = ".vault_audit",
        max_events_per_file: int = 10000,
        retention_days: int = 365
    ):
        self._storage_path = Path(storage_path)
        self._max_events = max_events_per_file
        self._retention_days = retention_days
        self._current_file: Optional[Path] = None
        self._event_count = 0
        self._last_hash = "GENESIS"
        self._lock = threading.Lock()

        self._ensure_storage()
        self._load_chain_state()

    def _ensure_storage(self) -> None:
        """Create storage with secure permissions."""
        if not self._storage_path.exists():
            self._storage_path.mkdir(parents=True, mode=0o700)

    def _load_chain_state(self) -> None:
        """Load last hash from existing logs."""
        log_files = sorted(self._storage_path.glob("audit_*.jsonl.gz"))

        if log_files:
            # Get last event from most recent file
            latest_file = log_files[-1]
            try:
                with gzip.open(latest_file, "rt") as f:
                    last_line = None
                    for line in f:
                        last_line = line

                    if last_line:
                        event_data = json.loads(last_line)
                        self._last_hash = event_data.get("event_hash", "GENESIS")
                        self._current_file = latest_file
                        self._event_count = sum(1 for _ in gzip.open(latest_file, "rt"))
            except Exception:
                pass

    def _get_current_file(self) -> Path:
        """Get current log file, creating new if needed."""
        if self._event_count >= self._max_events or not self._current_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._current_file = self._storage_path / f"audit_{timestamp}.jsonl.gz"
            self._event_count = 0

        return self._current_file

    def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            user_id: User performing action
            username: Username for readability
            session_id: Related session
            ip_address: Client IP address
            details: Additional event details

        Returns:
            Created AuditEvent
        """
        with self._lock:
            event_id = f"evt_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                username=username,
                session_id=session_id,
                ip_address=ip_address,
                details=details or {},
                previous_hash=self._last_hash
            )

            # Write to file
            log_file = self._get_current_file()
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "username": event.username,
                "session_id": event.session_id,
                "ip_address": event.ip_address,
                "details": event.details,
                "previous_hash": event.previous_hash,
                "event_hash": event.event_hash
            }

            with gzip.open(log_file, "at") as f:
                f.write(json.dumps(event_data) + "\n")

            self._last_hash = event.event_hash
            self._event_count += 1

            # Cleanup old logs
            self._cleanup_old_logs()

            return event

    def _cleanup_old_logs(self) -> None:
        """Remove logs older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self._retention_days)

        for log_file in self._storage_path.glob("audit_*.jsonl.gz"):
            try:
                # Parse timestamp from filename
                name = log_file.stem.replace("audit_", "").replace(".jsonl", "")
                file_date = datetime.strptime(name, "%Y%m%d_%H%M%S")

                if file_date < cutoff:
                    log_file.unlink()
            except (ValueError, OSError):
                continue

    def verify_chain_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> tuple[bool, List[str]]:
        """
        Verify integrity of audit chain.

        Args:
            start_date: Start of verification range
            end_date: End of verification range

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        expected_hash = "GENESIS"

        log_files = sorted(self._storage_path.glob("audit_*.jsonl.gz"))

        for log_file in log_files:
            try:
                with gzip.open(log_file, "rt") as f:
                    for line in f:
                        event_data = json.loads(line)

                        timestamp = datetime.fromisoformat(event_data["timestamp"])

                        # Check date range
                        if start_date and timestamp < start_date:
                            expected_hash = event_data["event_hash"]
                            continue
                        if end_date and timestamp > end_date:
                            break

                        # Verify chain
                        if event_data["previous_hash"] != expected_hash:
                            errors.append(
                                f"Chain break at {event_data['event_id']}: "
                                f"expected {expected_hash}, got {event_data['previous_hash']}"
                            )

                        # Verify event hash
                        event = AuditEvent(
                            event_id=event_data["event_id"],
                            event_type=AuditEventType(event_data["event_type"]),
                            timestamp=timestamp,
                            user_id=event_data.get("user_id"),
                            username=event_data.get("username"),
                            session_id=event_data.get("session_id"),
                            ip_address=event_data.get("ip_address"),
                            details=event_data.get("details", {}),
                            previous_hash=event_data["previous_hash"],
                            event_hash=event_data["event_hash"]
                        )

                        if not event.verify_integrity():
                            errors.append(
                                f"Hash mismatch at {event_data['event_id']}"
                            )

                        expected_hash = event_data["event_hash"]

            except Exception as e:
                errors.append(f"Error processing {log_file}: {str(e)}")

        return len(errors) == 0, errors

    def query(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Query audit logs.

        Args:
            event_types: Filter by event types
            user_id: Filter by user
            session_id: Filter by session
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum events to return

        Returns:
            List of matching AuditEvents
        """
        results = []
        log_files = sorted(self._storage_path.glob("audit_*.jsonl.gz"), reverse=True)

        for log_file in log_files:
            if len(results) >= limit:
                break

            try:
                with gzip.open(log_file, "rt") as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        event_data = json.loads(line)
                        timestamp = datetime.fromisoformat(event_data["timestamp"])

                        # Apply filters
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue
                        if user_id and event_data.get("user_id") != user_id:
                            continue
                        if session_id and event_data.get("session_id") != session_id:
                            continue
                        if event_types:
                            if event_data["event_type"] not in [et.value for et in event_types]:
                                continue

                        event = AuditEvent(
                            event_id=event_data["event_id"],
                            event_type=AuditEventType(event_data["event_type"]),
                            timestamp=timestamp,
                            user_id=event_data.get("user_id"),
                            username=event_data.get("username"),
                            session_id=event_data.get("session_id"),
                            ip_address=event_data.get("ip_address"),
                            details=event_data.get("details", {}),
                            previous_hash=event_data["previous_hash"],
                            event_hash=event_data["event_hash"]
                        )

                        results.append(event)

            except Exception:
                continue

        return results

    def get_user_activity(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get activity summary for a user.

        Args:
            user_id: User to query
            days: Number of days to analyze

        Returns:
            Activity summary dictionary
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        events = self.query(user_id=user_id, start_date=start_date, limit=10000)

        summary = {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(events),
            "event_counts": {},
            "sessions_accessed": set(),
            "last_activity": None,
            "security_events": []
        }

        for event in events:
            # Count by type
            event_type = event.event_type.value
            summary["event_counts"][event_type] = \
                summary["event_counts"].get(event_type, 0) + 1

            # Track sessions
            if event.session_id:
                summary["sessions_accessed"].add(event.session_id)

            # Track security events
            if event.event_type.value.startswith("security."):
                summary["security_events"].append({
                    "event_id": event.event_id,
                    "type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "details": event.details
                })

            # Update last activity
            if not summary["last_activity"] or event.timestamp > summary["last_activity"]:
                summary["last_activity"] = event.timestamp

        summary["sessions_accessed"] = list(summary["sessions_accessed"])
        if summary["last_activity"]:
            summary["last_activity"] = summary["last_activity"].isoformat()

        return summary

    def export_logs(
        self,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Export logs to file.

        Args:
            output_path: Output file path
            start_date: Start of export range
            end_date: End of export range

        Returns:
            Number of events exported
        """
        events = self.query(
            start_date=start_date,
            end_date=end_date,
            limit=1000000
        )

        with open(output_path, "w") as f:
            for event in events:
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "username": event.username,
                    "session_id": event.session_id,
                    "ip_address": event.ip_address,
                    "details": event.details
                }
                f.write(json.dumps(event_data) + "\n")

        return len(events)
