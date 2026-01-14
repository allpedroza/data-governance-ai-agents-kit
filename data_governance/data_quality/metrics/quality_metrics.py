# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Quality Metrics - Core data quality dimension measurements

Implements the standard data quality dimensions:
1. Completeness - Presence of required data
2. Uniqueness - Absence of duplicates
3. Validity - Conformance to formats/rules
4. Consistency - Agreement across data
5. Freshness - Timeliness of data
6. Accuracy - Correctness indicators
"""

import re
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class QualityStatus(Enum):
    """Quality check status"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    """Result of a quality metric evaluation"""
    metric_name: str
    dimension: str
    column: Optional[str]
    value: float  # 0.0 to 1.0 (percentage/score)
    status: QualityStatus
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passed(self) -> bool:
        return self.status == QualityStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "dimension": self.dimension,
            "column": self.column,
            "value": round(self.value, 4),
            "value_percent": f"{self.value:.2%}",
            "status": self.status.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "details": self.details,
            "message": self.message,
            "timestamp": self.timestamp
        }


class QualityMetric(ABC):
    """Abstract base class for quality metrics"""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Quality dimension name"""
        pass

    @abstractmethod
    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """Evaluate the metric on data"""
        pass


class CompletenessMetric(QualityMetric):
    """
    Completeness - Measures the presence of required data

    Checks:
    - Null/None values
    - Empty strings
    - Missing required fields
    """

    @property
    def dimension(self) -> str:
        return "completeness"

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        threshold: float = 0.95,
        treat_empty_as_null: bool = True,
        required_columns: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate completeness

        Args:
            data: List of row dictionaries
            column: Specific column to check (None = all columns)
            threshold: Minimum acceptable completeness (0-1)
            treat_empty_as_null: Treat empty strings as null
            required_columns: List of columns that must be present

        Returns:
            MetricResult with completeness score
        """
        if not data:
            return MetricResult(
                metric_name="completeness",
                dimension=self.dimension,
                column=column,
                value=0.0,
                status=QualityStatus.FAILED,
                threshold=threshold,
                message="No data to evaluate"
            )

        total_cells = 0
        null_cells = 0
        column_stats = {}

        columns_to_check = [column] if column else list(data[0].keys())

        for col in columns_to_check:
            col_nulls = 0
            col_total = 0

            for row in data:
                col_total += 1
                value = row.get(col)

                is_null = value is None
                if treat_empty_as_null and isinstance(value, str) and value.strip() == "":
                    is_null = True

                if is_null:
                    col_nulls += 1

            total_cells += col_total
            null_cells += col_nulls

            completeness = 1 - (col_nulls / col_total) if col_total > 0 else 0
            column_stats[col] = {
                "total": col_total,
                "nulls": col_nulls,
                "completeness": round(completeness, 4)
            }

        overall_completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0

        # Check required columns
        missing_required = []
        if required_columns:
            present_columns = set(data[0].keys()) if data else set()
            missing_required = [c for c in required_columns if c not in present_columns]

        # Determine status
        if missing_required:
            status = QualityStatus.FAILED
            message = f"Missing required columns: {missing_required}"
        elif overall_completeness >= threshold:
            status = QualityStatus.PASSED
            message = f"Completeness {overall_completeness:.2%} meets threshold {threshold:.2%}"
        elif overall_completeness >= threshold * 0.9:
            status = QualityStatus.WARNING
            message = f"Completeness {overall_completeness:.2%} below threshold {threshold:.2%}"
        else:
            status = QualityStatus.FAILED
            message = f"Completeness {overall_completeness:.2%} significantly below threshold {threshold:.2%}"

        # Find worst columns
        worst_columns = sorted(
            column_stats.items(),
            key=lambda x: x[1]["completeness"]
        )[:5]

        return MetricResult(
            metric_name="completeness",
            dimension=self.dimension,
            column=column,
            value=overall_completeness,
            status=status,
            threshold=threshold,
            message=message,
            details={
                "total_cells": total_cells,
                "null_cells": null_cells,
                "columns_checked": len(columns_to_check),
                "column_stats": column_stats,
                "worst_columns": [{"column": c, **s} for c, s in worst_columns],
                "missing_required": missing_required
            }
        )


class UniquenessMetric(QualityMetric):
    """
    Uniqueness - Measures absence of duplicates

    Checks:
    - Duplicate rows
    - Duplicate values in key columns
    - Near-duplicates (fuzzy matching)
    """

    @property
    def dimension(self) -> str:
        return "uniqueness"

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        threshold: float = 1.0,
        key_columns: Optional[List[str]] = None,
        case_sensitive: bool = True,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate uniqueness

        Args:
            data: List of row dictionaries
            column: Specific column to check for unique values
            threshold: Minimum acceptable uniqueness (0-1)
            key_columns: Columns that form a unique key
            case_sensitive: Whether string comparison is case-sensitive

        Returns:
            MetricResult with uniqueness score
        """
        if not data:
            return MetricResult(
                metric_name="uniqueness",
                dimension=self.dimension,
                column=column,
                value=1.0,
                status=QualityStatus.PASSED,
                threshold=threshold,
                message="No data to evaluate"
            )

        total_rows = len(data)
        duplicates = []
        seen_values: Dict[str, List[int]] = {}

        if column:
            # Check single column uniqueness
            for idx, row in enumerate(data):
                value = row.get(column)
                if value is not None:
                    key = str(value) if case_sensitive else str(value).lower()
                    if key in seen_values:
                        seen_values[key].append(idx)
                    else:
                        seen_values[key] = [idx]

            # Find duplicates
            for value, indices in seen_values.items():
                if len(indices) > 1:
                    duplicates.append({
                        "value": value,
                        "count": len(indices),
                        "row_indices": indices[:10]  # Limit indices
                    })

            unique_count = len(seen_values)
            duplicate_count = sum(len(d["row_indices"]) - 1 for d in duplicates)

        elif key_columns:
            # Check composite key uniqueness
            for idx, row in enumerate(data):
                key_parts = []
                for col in key_columns:
                    val = row.get(col, "")
                    key_parts.append(str(val) if case_sensitive else str(val).lower())
                key = "|".join(key_parts)

                if key in seen_values:
                    seen_values[key].append(idx)
                else:
                    seen_values[key] = [idx]

            for value, indices in seen_values.items():
                if len(indices) > 1:
                    duplicates.append({
                        "key": value,
                        "count": len(indices),
                        "row_indices": indices[:10]
                    })

            unique_count = len(seen_values)
            duplicate_count = sum(len(d["row_indices"]) - 1 for d in duplicates)

        else:
            # Check full row uniqueness
            for idx, row in enumerate(data):
                # Create hash of row
                row_str = "|".join(str(v) for v in row.values())
                if not case_sensitive:
                    row_str = row_str.lower()
                key = hashlib.md5(row_str.encode()).hexdigest()

                if key in seen_values:
                    seen_values[key].append(idx)
                else:
                    seen_values[key] = [idx]

            for key, indices in seen_values.items():
                if len(indices) > 1:
                    duplicates.append({
                        "row_hash": key[:8],
                        "count": len(indices),
                        "row_indices": indices[:10]
                    })

            unique_count = len(seen_values)
            duplicate_count = sum(len(d["row_indices"]) - 1 for d in duplicates)

        uniqueness = unique_count / total_rows if total_rows > 0 else 1.0

        # Determine status
        if uniqueness >= threshold:
            status = QualityStatus.PASSED
            message = f"Uniqueness {uniqueness:.2%} meets threshold {threshold:.2%}"
        elif uniqueness >= threshold * 0.95:
            status = QualityStatus.WARNING
            message = f"Uniqueness {uniqueness:.2%} slightly below threshold {threshold:.2%}"
        else:
            status = QualityStatus.FAILED
            message = f"Uniqueness {uniqueness:.2%} below threshold {threshold:.2%}"

        return MetricResult(
            metric_name="uniqueness",
            dimension=self.dimension,
            column=column,
            value=uniqueness,
            status=status,
            threshold=threshold,
            message=message,
            details={
                "total_rows": total_rows,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count,
                "duplicate_groups": len(duplicates),
                "top_duplicates": sorted(duplicates, key=lambda x: x["count"], reverse=True)[:10],
                "key_columns": key_columns
            }
        )


class ValidityMetric(QualityMetric):
    """
    Validity - Measures conformance to expected formats and rules

    Checks:
    - Data type conformance
    - Format patterns (email, phone, etc.)
    - Range validation
    - Enumeration validation
    """

    # Common validation patterns
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone_br": r'^[\+]?[(]?[0-9]{2,3}[)]?[-\s\.]?[0-9]{4,5}[-\s\.]?[0-9]{4}$',
        "cpf": r'^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$',
        "cnpj": r'^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$',
        "cep": r'^\d{5}-?\d{3}$',
        "date_iso": r'^\d{4}-\d{2}-\d{2}$',
        "date_br": r'^\d{2}/\d{2}/\d{4}$',
        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "url": r'^https?://[^\s]+$',
        "integer": r'^-?\d+$',
        "decimal": r'^-?\d+\.?\d*$',
        "boolean": r'^(true|false|0|1|yes|no|sim|não)$'
    }

    @property
    def dimension(self) -> str:
        return "validity"

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        threshold: float = 0.95,
        pattern: Optional[str] = None,
        pattern_name: Optional[str] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allowed_values: Optional[List[Any]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate validity

        Args:
            data: List of row dictionaries
            column: Column to validate (required)
            threshold: Minimum acceptable validity (0-1)
            pattern: Custom regex pattern
            pattern_name: Name of predefined pattern
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allowed_values: List of allowed values (enumeration)

        Returns:
            MetricResult with validity score
        """
        if not column:
            return MetricResult(
                metric_name="validity",
                dimension=self.dimension,
                column=column,
                value=0.0,
                status=QualityStatus.SKIPPED,
                threshold=threshold,
                message="Column parameter required for validity check"
            )

        if not data:
            return MetricResult(
                metric_name="validity",
                dimension=self.dimension,
                column=column,
                value=1.0,
                status=QualityStatus.PASSED,
                threshold=threshold,
                message="No data to evaluate"
            )

        # Get pattern
        regex_pattern = None
        if pattern:
            regex_pattern = pattern
        elif pattern_name and pattern_name in self.PATTERNS:
            regex_pattern = self.PATTERNS[pattern_name]

        total = 0
        valid = 0
        invalid_samples = []

        for row in data:
            value = row.get(column)
            if value is None:
                continue  # Skip nulls (handled by completeness)

            total += 1
            is_valid = True
            reason = None

            # Pattern validation
            if regex_pattern:
                if not re.match(regex_pattern, str(value), re.IGNORECASE):
                    is_valid = False
                    reason = f"Does not match pattern"

            # Range validation
            if is_valid and (min_value is not None or max_value is not None):
                try:
                    num_value = float(value)
                    if min_value is not None and num_value < min_value:
                        is_valid = False
                        reason = f"Below minimum {min_value}"
                    if max_value is not None and num_value > max_value:
                        is_valid = False
                        reason = f"Above maximum {max_value}"
                except (ValueError, TypeError):
                    is_valid = False
                    reason = "Not a valid number"

            # Enumeration validation
            if is_valid and allowed_values is not None:
                if value not in allowed_values:
                    is_valid = False
                    reason = f"Not in allowed values"

            if is_valid:
                valid += 1
            elif len(invalid_samples) < 10:
                invalid_samples.append({
                    "value": str(value)[:50],
                    "reason": reason
                })

        validity = valid / total if total > 0 else 1.0

        # Determine status
        if validity >= threshold:
            status = QualityStatus.PASSED
            message = f"Validity {validity:.2%} meets threshold {threshold:.2%}"
        elif validity >= threshold * 0.9:
            status = QualityStatus.WARNING
            message = f"Validity {validity:.2%} below threshold {threshold:.2%}"
        else:
            status = QualityStatus.FAILED
            message = f"Validity {validity:.2%} significantly below threshold {threshold:.2%}"

        return MetricResult(
            metric_name="validity",
            dimension=self.dimension,
            column=column,
            value=validity,
            status=status,
            threshold=threshold,
            message=message,
            details={
                "total_checked": total,
                "valid_count": valid,
                "invalid_count": total - valid,
                "pattern_name": pattern_name,
                "pattern": regex_pattern[:50] if regex_pattern else None,
                "min_value": min_value,
                "max_value": max_value,
                "allowed_values": allowed_values[:10] if allowed_values else None,
                "invalid_samples": invalid_samples
            }
        )


class ConsistencyMetric(QualityMetric):
    """
    Consistency - Measures agreement and coherence of data

    Checks:
    - Cross-field consistency (e.g., city matches state)
    - Referential integrity
    - Business rule compliance
    """

    @property
    def dimension(self) -> str:
        return "consistency"

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        threshold: float = 0.95,
        rule_type: str = "not_null_if",
        condition_column: Optional[str] = None,
        condition_value: Optional[Any] = None,
        expected_column: Optional[str] = None,
        lookup_data: Optional[List[Dict[str, Any]]] = None,
        lookup_key: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate consistency

        Args:
            data: List of row dictionaries
            column: Column to check
            threshold: Minimum acceptable consistency (0-1)
            rule_type: Type of consistency check
                - "not_null_if": Column must have value if condition met
                - "equals_if": Column must equal value if condition met
                - "referential": Column value must exist in lookup
            condition_column: Column for condition
            condition_value: Value that triggers condition
            expected_column: Column with expected value
            lookup_data: Reference data for referential checks
            lookup_key: Key column in lookup data

        Returns:
            MetricResult with consistency score
        """
        if not data:
            return MetricResult(
                metric_name="consistency",
                dimension=self.dimension,
                column=column,
                value=1.0,
                status=QualityStatus.PASSED,
                threshold=threshold,
                message="No data to evaluate"
            )

        total = 0
        consistent = 0
        inconsistent_samples = []

        if rule_type == "not_null_if":
            # Check: if condition_column == condition_value, then column must not be null
            for row in data:
                if condition_column and row.get(condition_column) == condition_value:
                    total += 1
                    if row.get(column) is not None:
                        consistent += 1
                    elif len(inconsistent_samples) < 10:
                        inconsistent_samples.append({
                            "condition": f"{condition_column}={condition_value}",
                            "column": column,
                            "value": "NULL"
                        })

        elif rule_type == "equals_if":
            # Check: if condition_column == condition_value, then column must equal expected
            for row in data:
                if condition_column and row.get(condition_column) == condition_value:
                    total += 1
                    if expected_column and row.get(column) == row.get(expected_column):
                        consistent += 1
                    elif len(inconsistent_samples) < 10:
                        inconsistent_samples.append({
                            "condition": f"{condition_column}={condition_value}",
                            "column": column,
                            "value": row.get(column),
                            "expected": row.get(expected_column)
                        })

        elif rule_type == "referential":
            # Check: column value must exist in lookup data
            if lookup_data and lookup_key:
                valid_keys = {str(row.get(lookup_key)) for row in lookup_data}
                for row in data:
                    value = row.get(column)
                    if value is not None:
                        total += 1
                        if str(value) in valid_keys:
                            consistent += 1
                        elif len(inconsistent_samples) < 10:
                            inconsistent_samples.append({
                                "column": column,
                                "value": value,
                                "message": "Not found in reference"
                            })

        else:
            return MetricResult(
                metric_name="consistency",
                dimension=self.dimension,
                column=column,
                value=0.0,
                status=QualityStatus.SKIPPED,
                threshold=threshold,
                message=f"Unknown rule type: {rule_type}"
            )

        consistency = consistent / total if total > 0 else 1.0

        # Determine status
        if consistency >= threshold:
            status = QualityStatus.PASSED
            message = f"Consistency {consistency:.2%} meets threshold {threshold:.2%}"
        elif consistency >= threshold * 0.9:
            status = QualityStatus.WARNING
            message = f"Consistency {consistency:.2%} below threshold {threshold:.2%}"
        else:
            status = QualityStatus.FAILED
            message = f"Consistency {consistency:.2%} significantly below threshold {threshold:.2%}"

        return MetricResult(
            metric_name="consistency",
            dimension=self.dimension,
            column=column,
            value=consistency,
            status=status,
            threshold=threshold,
            message=message,
            details={
                "rule_type": rule_type,
                "total_checked": total,
                "consistent_count": consistent,
                "inconsistent_count": total - consistent,
                "condition_column": condition_column,
                "condition_value": condition_value,
                "inconsistent_samples": inconsistent_samples
            }
        )


class FreshnessMetric(QualityMetric):
    """
    Freshness - Measures timeliness of data against SLA

    Checks:
    - Data age vs expected freshness
    - Update frequency
    - SLA compliance
    """

    @property
    def dimension(self) -> str:
        return "freshness"

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        threshold: float = 0.95,
        timestamp_column: Optional[str] = None,
        max_age_hours: float = 24.0,
        expected_update_time: Optional[str] = None,  # HH:MM format
        sla_hours: Optional[float] = None,
        reference_time: Optional[datetime] = None,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        **kwargs
    ) -> MetricResult:
        """
        Evaluate freshness

        Args:
            data: List of row dictionaries
            column: Column alias (uses timestamp_column)
            threshold: Minimum acceptable freshness (0-1)
            timestamp_column: Column containing timestamps
            max_age_hours: Maximum acceptable data age in hours
            expected_update_time: Expected daily update time (HH:MM)
            sla_hours: SLA window in hours
            reference_time: Reference time for comparison (default: now)
            date_format: Format of timestamps in data

        Returns:
            MetricResult with freshness score
        """
        if not timestamp_column:
            return MetricResult(
                metric_name="freshness",
                dimension=self.dimension,
                column=column,
                value=0.0,
                status=QualityStatus.SKIPPED,
                threshold=threshold,
                message="timestamp_column parameter required for freshness check"
            )

        if not data:
            return MetricResult(
                metric_name="freshness",
                dimension=self.dimension,
                column=timestamp_column,
                value=0.0,
                status=QualityStatus.WARNING,
                threshold=threshold,
                message="No data to evaluate freshness"
            )

        ref_time = reference_time or datetime.now()
        sla = sla_hours or max_age_hours

        # Find the most recent timestamp
        latest_timestamp = None
        oldest_timestamp = None
        valid_timestamps = 0
        stale_records = 0

        for row in data:
            ts_value = row.get(timestamp_column)
            if ts_value is None:
                continue

            try:
                if isinstance(ts_value, datetime):
                    ts = ts_value
                elif isinstance(ts_value, str):
                    # Try multiple formats
                    for fmt in [date_format, "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
                        try:
                            ts = datetime.strptime(ts_value[:26], fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue

                valid_timestamps += 1

                if latest_timestamp is None or ts > latest_timestamp:
                    latest_timestamp = ts
                if oldest_timestamp is None or ts < oldest_timestamp:
                    oldest_timestamp = ts

                # Check if record is stale
                age_hours = (ref_time - ts).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale_records += 1

            except Exception:
                continue

        if latest_timestamp is None:
            return MetricResult(
                metric_name="freshness",
                dimension=self.dimension,
                column=timestamp_column,
                value=0.0,
                status=QualityStatus.FAILED,
                threshold=threshold,
                message="No valid timestamps found"
            )

        # Calculate age
        age_hours = (ref_time - latest_timestamp).total_seconds() / 3600
        age_minutes = age_hours * 60

        # Calculate freshness score (1.0 = perfectly fresh, 0.0 = very stale)
        if age_hours <= sla:
            freshness = 1.0
        elif age_hours <= sla * 2:
            freshness = 1.0 - ((age_hours - sla) / sla)
        else:
            freshness = max(0.0, 1.0 - (age_hours / (sla * 4)))

        # SLA compliance
        sla_compliant = age_hours <= sla

        # Determine status
        if sla_compliant and freshness >= threshold:
            status = QualityStatus.PASSED
            message = f"Data is fresh (age: {age_hours:.1f}h, SLA: {sla:.1f}h)"
        elif age_hours <= sla * 1.5:
            status = QualityStatus.WARNING
            message = f"Data approaching SLA limit (age: {age_hours:.1f}h, SLA: {sla:.1f}h)"
        else:
            status = QualityStatus.FAILED
            message = f"Data exceeds SLA (age: {age_hours:.1f}h, SLA: {sla:.1f}h)"

        return MetricResult(
            metric_name="freshness",
            dimension=self.dimension,
            column=timestamp_column,
            value=freshness,
            status=status,
            threshold=threshold,
            message=message,
            details={
                "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
                "oldest_timestamp": oldest_timestamp.isoformat() if oldest_timestamp else None,
                "reference_time": ref_time.isoformat(),
                "age_hours": round(age_hours, 2),
                "age_minutes": round(age_minutes, 1),
                "sla_hours": sla,
                "sla_compliant": sla_compliant,
                "valid_timestamps": valid_timestamps,
                "stale_records": stale_records,
                "stale_percentage": round(stale_records / valid_timestamps, 4) if valid_timestamps > 0 else 0
            }
        )


class QualityMetrics:
    """
    Aggregator for all quality metrics

    Provides convenient methods to run multiple metrics
    """

    def __init__(self):
        self.completeness = CompletenessMetric()
        self.uniqueness = UniquenessMetric()
        self.validity = ValidityMetric()
        self.consistency = ConsistencyMetric()
        self.freshness = FreshnessMetric()

    def evaluate_all(
        self,
        data: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> List[MetricResult]:
        """
        Evaluate all quality dimensions

        Args:
            data: List of row dictionaries
            config: Configuration for each metric

        Returns:
            List of MetricResult
        """
        results = []
        config = config or {}

        # Completeness
        comp_config = config.get("completeness", {})
        results.append(self.completeness.evaluate(data, **comp_config))

        # Uniqueness
        uniq_config = config.get("uniqueness", {})
        results.append(self.uniqueness.evaluate(data, **uniq_config))

        # Validity (per column if configured)
        validity_configs = config.get("validity", [])
        if isinstance(validity_configs, dict):
            validity_configs = [validity_configs]
        for val_config in validity_configs:
            if val_config.get("column"):
                results.append(self.validity.evaluate(data, **val_config))

        # Consistency (per rule if configured)
        consistency_configs = config.get("consistency", [])
        if isinstance(consistency_configs, dict):
            consistency_configs = [consistency_configs]
        for cons_config in consistency_configs:
            results.append(self.consistency.evaluate(data, **cons_config))

        # Freshness
        fresh_config = config.get("freshness", {})
        if fresh_config.get("timestamp_column"):
            results.append(self.freshness.evaluate(data, **fresh_config))

        return results

    def get_summary(self, results: List[MetricResult]) -> Dict[str, Any]:
        """Generate summary from metric results"""
        passed = sum(1 for r in results if r.status == QualityStatus.PASSED)
        warnings = sum(1 for r in results if r.status == QualityStatus.WARNING)
        failed = sum(1 for r in results if r.status == QualityStatus.FAILED)
        skipped = sum(1 for r in results if r.status == QualityStatus.SKIPPED)

        # Overall score (average of non-skipped metrics)
        scored_results = [r for r in results if r.status != QualityStatus.SKIPPED]
        overall_score = sum(r.value for r in scored_results) / len(scored_results) if scored_results else 0

        # Group by dimension
        by_dimension = {}
        for r in results:
            if r.dimension not in by_dimension:
                by_dimension[r.dimension] = []
            by_dimension[r.dimension].append(r.to_dict())

        return {
            "total_checks": len(results),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "skipped": skipped,
            "overall_score": round(overall_score, 4),
            "overall_status": "passed" if failed == 0 and warnings == 0 else "warning" if failed == 0 else "failed",
            "by_dimension": by_dimension
        }
