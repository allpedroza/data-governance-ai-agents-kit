"""
Schema Drift Detector

Detects changes in data schema over time:
- Added columns
- Removed columns
- Type changes
- Nullable changes
- Name changes (fuzzy matching)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ChangeType(Enum):
    """Types of schema changes"""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"
    COLUMN_RENAMED = "column_renamed"  # Detected via fuzzy matching
    ORDER_CHANGED = "order_changed"


class ChangeSeverity(Enum):
    """Severity of schema changes"""
    INFO = "info"           # Non-breaking change
    WARNING = "warning"     # Potentially breaking
    CRITICAL = "critical"   # Breaking change


@dataclass
class SchemaChange:
    """Represents a single schema change"""
    change_type: ChangeType
    severity: ChangeSeverity
    column_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "severity": self.severity.value,
            "column_name": self.column_name,
            "old_value": str(self.old_value) if self.old_value else None,
            "new_value": str(self.new_value) if self.new_value else None,
            "message": self.message
        }


@dataclass
class SchemaSnapshot:
    """Point-in-time snapshot of a schema"""
    table_name: str
    columns: Dict[str, Dict[str, Any]]  # column_name -> {type, nullable, ...}
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": self.columns,
            "timestamp": self.timestamp,
            "version": self.version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaSnapshot":
        return cls(
            table_name=data.get("table_name", ""),
            columns=data.get("columns", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            version=data.get("version", 1),
            metadata=data.get("metadata", {})
        )

    def get_hash(self) -> str:
        """Get hash of schema for comparison"""
        schema_str = json.dumps(self.columns, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()


@dataclass
class DriftReport:
    """Report of schema drift analysis"""
    table_name: str
    has_drift: bool
    changes: List[SchemaChange]
    old_version: int
    new_version: int
    old_timestamp: str
    new_timestamp: str
    breaking_changes: int
    warning_changes: int
    info_changes: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "has_drift": self.has_drift,
            "changes": [c.to_dict() for c in self.changes],
            "old_version": self.old_version,
            "new_version": self.new_version,
            "old_timestamp": self.old_timestamp,
            "new_timestamp": self.new_timestamp,
            "breaking_changes": self.breaking_changes,
            "warning_changes": self.warning_changes,
            "info_changes": self.info_changes,
            "summary": self.summary
        }


class SchemaDriftDetector:
    """
    Detects schema changes between snapshots

    Features:
    - Snapshot persistence
    - Change detection and classification
    - Breaking change identification
    - Fuzzy matching for renamed columns

    Usage:
        detector = SchemaDriftDetector(persist_dir="./schema_history")

        # Take initial snapshot
        detector.snapshot("orders", current_schema)

        # Later, check for drift
        report = detector.detect_drift("orders", new_schema)

        if report.has_drift:
            print(f"Found {len(report.changes)} changes")
    """

    # Type compatibility matrix (old_type -> compatible new_types)
    TYPE_COMPATIBILITY = {
        "int": ["int", "bigint", "float", "double", "string"],
        "bigint": ["bigint", "float", "double", "string"],
        "float": ["float", "double", "string"],
        "double": ["double", "string"],
        "string": ["string"],
        "boolean": ["boolean", "string", "int"],
        "date": ["date", "datetime", "timestamp", "string"],
        "datetime": ["datetime", "timestamp", "string"],
        "timestamp": ["timestamp", "string"],
    }

    def __init__(self, persist_dir: str = "./schema_history"):
        """
        Initialize Schema Drift Detector

        Args:
            persist_dir: Directory to persist schema snapshots
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: Dict[str, List[SchemaSnapshot]] = {}
        self._load_snapshots()

    def _load_snapshots(self) -> None:
        """Load existing snapshots from disk"""
        for file_path in self.persist_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                table_name = data.get("table_name", file_path.stem)
                snapshots = [
                    SchemaSnapshot.from_dict(s)
                    for s in data.get("snapshots", [])
                ]
                self._snapshots[table_name] = snapshots
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    def _save_snapshots(self, table_name: str) -> None:
        """Save snapshots for a table to disk"""
        file_path = self.persist_dir / f"{table_name.replace('.', '_')}.json"
        data = {
            "table_name": table_name,
            "snapshots": [s.to_dict() for s in self._snapshots.get(table_name, [])]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def snapshot(
        self,
        table_name: str,
        schema: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> SchemaSnapshot:
        """
        Take a snapshot of the current schema

        Args:
            table_name: Name of the table
            schema: Schema definition {column_name: {type, nullable, ...}}
            metadata: Optional metadata about the snapshot

        Returns:
            SchemaSnapshot object
        """
        if table_name not in self._snapshots:
            self._snapshots[table_name] = []

        # Determine version
        existing = self._snapshots[table_name]
        version = existing[-1].version + 1 if existing else 1

        # Create snapshot
        snap = SchemaSnapshot(
            table_name=table_name,
            columns=schema,
            version=version,
            metadata=metadata or {}
        )

        # Check if actually changed
        if existing and existing[-1].get_hash() == snap.get_hash():
            return existing[-1]  # No change, return existing

        self._snapshots[table_name].append(snap)
        self._save_snapshots(table_name)

        return snap

    def snapshot_from_dataframe(
        self,
        table_name: str,
        df: Any,  # pandas DataFrame
        metadata: Optional[Dict[str, Any]] = None
    ) -> SchemaSnapshot:
        """
        Take snapshot from a pandas DataFrame

        Args:
            table_name: Name of the table
            df: pandas DataFrame
            metadata: Optional metadata

        Returns:
            SchemaSnapshot
        """
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            nullable = df[col].isna().any()
            schema[col] = {
                "type": dtype,
                "nullable": bool(nullable),
                "position": list(df.columns).index(col)
            }
        return self.snapshot(table_name, schema, metadata)

    def snapshot_from_sql(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],  # From SQLAlchemy inspector
        metadata: Optional[Dict[str, Any]] = None
    ) -> SchemaSnapshot:
        """
        Take snapshot from SQL column definitions

        Args:
            table_name: Name of the table
            columns: List of column dicts from SQLAlchemy
            metadata: Optional metadata

        Returns:
            SchemaSnapshot
        """
        schema = {}
        for i, col in enumerate(columns):
            schema[col["name"]] = {
                "type": str(col.get("type", "unknown")),
                "nullable": col.get("nullable", True),
                "primary_key": col.get("primary_key", False),
                "default": str(col.get("default")) if col.get("default") else None,
                "position": i
            }
        return self.snapshot(table_name, schema, metadata)

    def get_latest_snapshot(self, table_name: str) -> Optional[SchemaSnapshot]:
        """Get the most recent snapshot for a table"""
        snapshots = self._snapshots.get(table_name, [])
        return snapshots[-1] if snapshots else None

    def get_snapshot_history(
        self,
        table_name: str,
        limit: int = 10
    ) -> List[SchemaSnapshot]:
        """Get snapshot history for a table"""
        snapshots = self._snapshots.get(table_name, [])
        return snapshots[-limit:]

    def _normalize_type(self, dtype: str) -> str:
        """Normalize data type for comparison"""
        dtype_lower = dtype.lower()

        # Map to canonical types
        if "int" in dtype_lower and "big" in dtype_lower:
            return "bigint"
        if "int" in dtype_lower:
            return "int"
        if "float" in dtype_lower or "double" in dtype_lower:
            return "float"
        if "bool" in dtype_lower:
            return "boolean"
        if "timestamp" in dtype_lower:
            return "timestamp"
        if "datetime" in dtype_lower:
            return "datetime"
        if "date" in dtype_lower:
            return "date"
        if "str" in dtype_lower or "object" in dtype_lower or "varchar" in dtype_lower:
            return "string"

        return dtype_lower

    def _is_breaking_type_change(self, old_type: str, new_type: str) -> bool:
        """Check if type change is breaking"""
        old_normalized = self._normalize_type(old_type)
        new_normalized = self._normalize_type(new_type)

        if old_normalized == new_normalized:
            return False

        compatible = self.TYPE_COMPATIBILITY.get(old_normalized, [old_normalized])
        return new_normalized not in compatible

    def _find_renamed_column(
        self,
        old_columns: Set[str],
        new_columns: Set[str],
        old_schema: Dict[str, Dict[str, Any]],
        new_schema: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        Try to detect renamed columns using heuristics

        Returns list of (old_name, new_name) tuples
        """
        renames = []
        removed = old_columns - new_columns
        added = new_columns - old_columns

        for old_col in removed:
            old_info = old_schema[old_col]
            old_type = self._normalize_type(old_info.get("type", ""))

            for new_col in added:
                new_info = new_schema[new_col]
                new_type = self._normalize_type(new_info.get("type", ""))

                # Same type and position suggests rename
                if old_type == new_type:
                    if old_info.get("position") == new_info.get("position"):
                        renames.append((old_col, new_col))
                        break

                    # Similar names (e.g., "user_id" -> "userId")
                    old_normalized = old_col.lower().replace("_", "")
                    new_normalized = new_col.lower().replace("_", "")
                    if old_normalized == new_normalized:
                        renames.append((old_col, new_col))
                        break

        return renames

    def detect_drift(
        self,
        table_name: str,
        new_schema: Dict[str, Dict[str, Any]],
        baseline_version: Optional[int] = None
    ) -> DriftReport:
        """
        Detect schema drift between current and new schema

        Args:
            table_name: Name of the table
            new_schema: New schema to compare
            baseline_version: Compare against specific version (default: latest)

        Returns:
            DriftReport with detected changes
        """
        snapshots = self._snapshots.get(table_name, [])

        if not snapshots:
            # No baseline - treat as first snapshot
            self.snapshot(table_name, new_schema)
            return DriftReport(
                table_name=table_name,
                has_drift=False,
                changes=[],
                old_version=0,
                new_version=1,
                old_timestamp="",
                new_timestamp=datetime.now().isoformat(),
                breaking_changes=0,
                warning_changes=0,
                info_changes=0,
                summary="First schema snapshot recorded"
            )

        # Get baseline
        if baseline_version:
            baseline = next((s for s in snapshots if s.version == baseline_version), None)
            if not baseline:
                baseline = snapshots[-1]
        else:
            baseline = snapshots[-1]

        old_schema = baseline.columns
        old_columns = set(old_schema.keys())
        new_columns = set(new_schema.keys())

        changes: List[SchemaChange] = []

        # Detect renames first
        renames = self._find_renamed_column(old_columns, new_columns, old_schema, new_schema)
        renamed_old = {r[0] for r in renames}
        renamed_new = {r[1] for r in renames}

        for old_name, new_name in renames:
            changes.append(SchemaChange(
                change_type=ChangeType.COLUMN_RENAMED,
                severity=ChangeSeverity.WARNING,
                column_name=old_name,
                old_value=old_name,
                new_value=new_name,
                message=f"Column '{old_name}' appears to be renamed to '{new_name}'"
            ))

        # Detect removed columns
        for col in old_columns - new_columns - renamed_old:
            changes.append(SchemaChange(
                change_type=ChangeType.COLUMN_REMOVED,
                severity=ChangeSeverity.CRITICAL,
                column_name=col,
                old_value=old_schema[col],
                new_value=None,
                message=f"Column '{col}' was removed"
            ))

        # Detect added columns
        for col in new_columns - old_columns - renamed_new:
            nullable = new_schema[col].get("nullable", True)
            severity = ChangeSeverity.INFO if nullable else ChangeSeverity.WARNING

            changes.append(SchemaChange(
                change_type=ChangeType.COLUMN_ADDED,
                severity=severity,
                column_name=col,
                old_value=None,
                new_value=new_schema[col],
                message=f"Column '{col}' was added" + (" (NOT NULL)" if not nullable else "")
            ))

        # Detect type and nullable changes for existing columns
        for col in old_columns & new_columns:
            old_info = old_schema[col]
            new_info = new_schema[col]

            old_type = old_info.get("type", "unknown")
            new_type = new_info.get("type", "unknown")

            # Type change
            if self._normalize_type(old_type) != self._normalize_type(new_type):
                is_breaking = self._is_breaking_type_change(old_type, new_type)
                changes.append(SchemaChange(
                    change_type=ChangeType.TYPE_CHANGED,
                    severity=ChangeSeverity.CRITICAL if is_breaking else ChangeSeverity.WARNING,
                    column_name=col,
                    old_value=old_type,
                    new_value=new_type,
                    message=f"Column '{col}' type changed from {old_type} to {new_type}" +
                            (" (BREAKING)" if is_breaking else "")
                ))

            # Nullable change
            old_nullable = old_info.get("nullable", True)
            new_nullable = new_info.get("nullable", True)

            if old_nullable != new_nullable:
                # Becoming NOT NULL is breaking
                severity = ChangeSeverity.WARNING if new_nullable else ChangeSeverity.CRITICAL
                changes.append(SchemaChange(
                    change_type=ChangeType.NULLABLE_CHANGED,
                    severity=severity,
                    column_name=col,
                    old_value=old_nullable,
                    new_value=new_nullable,
                    message=f"Column '{col}' changed from " +
                            ("nullable" if old_nullable else "NOT NULL") + " to " +
                            ("nullable" if new_nullable else "NOT NULL")
                ))

        # Count by severity
        breaking = sum(1 for c in changes if c.severity == ChangeSeverity.CRITICAL)
        warnings = sum(1 for c in changes if c.severity == ChangeSeverity.WARNING)
        info = sum(1 for c in changes if c.severity == ChangeSeverity.INFO)

        # Generate summary
        has_drift = len(changes) > 0
        if not has_drift:
            summary = "No schema changes detected"
        elif breaking > 0:
            summary = f"BREAKING CHANGES: {breaking} critical, {warnings} warnings, {info} info"
        elif warnings > 0:
            summary = f"Schema changes detected: {warnings} warnings, {info} info"
        else:
            summary = f"Minor schema changes: {info} info-level changes"

        return DriftReport(
            table_name=table_name,
            has_drift=has_drift,
            changes=changes,
            old_version=baseline.version,
            new_version=baseline.version + 1 if has_drift else baseline.version,
            old_timestamp=baseline.timestamp,
            new_timestamp=datetime.now().isoformat(),
            breaking_changes=breaking,
            warning_changes=warnings,
            info_changes=info,
            summary=summary
        )

    def compare_versions(
        self,
        table_name: str,
        version_a: int,
        version_b: int
    ) -> DriftReport:
        """Compare two specific versions"""
        snapshots = self._snapshots.get(table_name, [])

        snap_a = next((s for s in snapshots if s.version == version_a), None)
        snap_b = next((s for s in snapshots if s.version == version_b), None)

        if not snap_a or not snap_b:
            return DriftReport(
                table_name=table_name,
                has_drift=False,
                changes=[],
                old_version=version_a,
                new_version=version_b,
                old_timestamp="",
                new_timestamp="",
                breaking_changes=0,
                warning_changes=0,
                info_changes=0,
                summary=f"Version {version_a} or {version_b} not found"
            )

        # Use detect_drift logic with snap_b schema
        self._snapshots[table_name] = [snap_a]  # Temporarily set baseline
        report = self.detect_drift(table_name, snap_b.columns)
        self._snapshots[table_name] = snapshots  # Restore

        return report

    def list_tables(self) -> List[str]:
        """List all tracked tables"""
        return list(self._snapshots.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "tables_tracked": len(self._snapshots),
            "total_snapshots": sum(len(s) for s in self._snapshots.values()),
            "tables": {
                name: {
                    "versions": len(snaps),
                    "latest_version": snaps[-1].version if snaps else 0,
                    "latest_timestamp": snaps[-1].timestamp if snaps else None
                }
                for name, snaps in self._snapshots.items()
            }
        }
