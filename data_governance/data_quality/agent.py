"""
Data Quality Agent

Monitors data quality metrics including completeness, uniqueness,
validity, consistency, freshness, and schema drift detection.

Features:
- Multi-dimensional quality scoring
- SLA-based freshness monitoring
- Schema drift detection
- Configurable quality rules
- Alert generation
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .metrics.quality_metrics import (
    QualityMetrics,
    MetricResult,
    QualityStatus,
    CompletenessMetric,
    UniquenessMetric,
    ValidityMetric,
    ConsistencyMetric,
    FreshnessMetric
)
from .metrics.schema_drift import SchemaDriftDetector, DriftReport, SchemaSnapshot
from .rules.quality_rules import (
    RuleEvaluator,
    QualityRule,
    RuleSet,
    QualityAlert,
    AlertLevel
)
from .connectors.data_connector import (
    DataConnector,
    DataSource,
    create_connector
)


class QualityDimension(Enum):
    """Quality dimensions"""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    FRESHNESS = "freshness"
    SCHEMA = "schema"


@dataclass
class QualityReport:
    """Comprehensive quality report for a data source"""
    source_name: str
    source_type: str
    timestamp: str
    overall_score: float
    overall_status: str  # passed, warning, failed
    dimensions: Dict[str, Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    schema_drift: Optional[Dict[str, Any]]
    row_count: int
    columns_checked: int
    processing_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "overall_status": self.overall_status,
            "dimensions": self.dimensions,
            "metrics": self.metrics,
            "alerts": self.alerts,
            "schema_drift": self.schema_drift,
            "row_count": self.row_count,
            "columns_checked": self.columns_checked,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown report"""
        lines = [
            f"# Data Quality Report: {self.source_name}",
            f"*Generated: {self.timestamp}*",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Score | **{self.overall_score:.2%}** |",
            f"| Status | {self._status_emoji()} {self.overall_status.upper()} |",
            f"| Rows Analyzed | {self.row_count:,} |",
            f"| Columns Checked | {self.columns_checked} |",
            f"| Processing Time | {self.processing_time_ms}ms |",
            "",
            "## Dimensions",
            "",
            "| Dimension | Score | Status |",
            "|-----------|-------|--------|"
        ]

        for dim_name, dim_data in self.dimensions.items():
            score = dim_data.get("score", 0)
            status = dim_data.get("status", "unknown")
            emoji = "✅" if status == "passed" else "⚠️" if status == "warning" else "❌"
            lines.append(f"| {dim_name.capitalize()} | {score:.2%} | {emoji} {status} |")

        if self.alerts:
            lines.extend([
                "",
                "## Alerts",
                ""
            ])
            for alert in self.alerts:
                level = alert.get("level", "info")
                emoji = "🔴" if level == "critical" else "🟡" if level == "warning" else "🔵"
                lines.append(f"- {emoji} **{alert.get('rule_name')}**: {alert.get('message')}")

        if self.schema_drift and self.schema_drift.get("has_drift"):
            lines.extend([
                "",
                "## Schema Changes",
                ""
            ])
            for change in self.schema_drift.get("changes", [])[:10]:
                severity = change.get("severity", "info")
                emoji = "🔴" if severity == "critical" else "🟡"
                lines.append(f"- {emoji} {change.get('message')}")

        lines.extend([
            "",
            "---",
            f"*Report generated in {self.processing_time_ms}ms*"
        ])

        return "\n".join(lines)

    def _status_emoji(self) -> str:
        if self.overall_status == "passed":
            return "✅"
        elif self.overall_status == "warning":
            return "⚠️"
        else:
            return "❌"


class DataQualityAgent:
    """
    Data Quality Agent

    Monitors data quality across multiple dimensions with
    configurable rules and SLA-based alerting.

    Usage:
        agent = DataQualityAgent()

        # Evaluate quality from file
        report = agent.evaluate_file("data.parquet")

        # Evaluate with custom rules
        agent.add_rule(QualityRule(
            name="customer_email_required",
            dimension="completeness",
            table_name="customers",
            column="email",
            threshold=0.99
        ))
        report = agent.evaluate_file("customers.csv")

        # Check freshness with SLA
        report = agent.evaluate_file(
            "orders.parquet",
            freshness_config={
                "timestamp_column": "updated_at",
                "sla_hours": 4
            }
        )
    """

    def __init__(
        self,
        persist_dir: str = "./quality_data",
        enable_schema_tracking: bool = True
    ):
        """
        Initialize Data Quality Agent

        Args:
            persist_dir: Directory for persisting data
            enable_schema_tracking: Enable schema drift detection
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics = QualityMetrics()
        self.rule_evaluator = RuleEvaluator(
            persist_dir=str(self.persist_dir / "rules")
        )

        self.schema_detector = None
        if enable_schema_tracking:
            self.schema_detector = SchemaDriftDetector(
                persist_dir=str(self.persist_dir / "schemas")
            )

        # Report history
        self._reports: List[QualityReport] = []

        print("=" * 60)
        print("Data Quality Agent Initialized")
        print("=" * 60)
        print(f"  Persist Dir: {self.persist_dir}")
        print(f"  Schema Tracking: {'Enabled' if enable_schema_tracking else 'Disabled'}")
        print(f"  Rules Loaded: {self.rule_evaluator.get_statistics()['total_rules']}")
        print("=" * 60)

    def add_rule(self, rule: QualityRule, rule_set_name: str = "default") -> None:
        """Add a quality rule"""
        self.rule_evaluator.add_rule(rule, rule_set_name)

    def add_rule_set(self, rule_set: RuleSet) -> None:
        """Add a complete rule set"""
        self.rule_evaluator.save_rule_set(rule_set)

    def load_rules_from_file(self, file_path: str) -> int:
        """Load rules from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # List of rules
            for rule_data in data:
                rule = QualityRule.from_dict(rule_data)
                self.add_rule(rule)
            return len(data)
        else:
            # Rule set
            rule_set = RuleSet.from_dict(data)
            self.add_rule_set(rule_set)
            return len(rule_set.rules)

    def evaluate(
        self,
        connector: DataConnector,
        sample_size: int = 10000,
        completeness_config: Optional[Dict[str, Any]] = None,
        uniqueness_config: Optional[Dict[str, Any]] = None,
        validity_configs: Optional[List[Dict[str, Any]]] = None,
        consistency_configs: Optional[List[Dict[str, Any]]] = None,
        freshness_config: Optional[Dict[str, Any]] = None,
        check_schema_drift: bool = True
    ) -> QualityReport:
        """
        Evaluate data quality from a connector

        Args:
            connector: Data connector instance
            sample_size: Number of rows to sample
            completeness_config: Config for completeness check
            uniqueness_config: Config for uniqueness check
            validity_configs: List of validity check configs
            consistency_configs: List of consistency check configs
            freshness_config: Config for freshness/SLA check
            check_schema_drift: Whether to check for schema drift

        Returns:
            QualityReport with all metrics and alerts
        """
        start_time = time.time()

        # Get data info
        source_info = connector.get_info()
        schema = connector.get_schema()
        columns = list(schema.keys())

        # Read sample
        data = connector.read_sample(sample_size)

        # Build metrics config
        metrics_config = {}

        if completeness_config:
            metrics_config["completeness"] = completeness_config
        else:
            metrics_config["completeness"] = {"threshold": 0.95}

        if uniqueness_config:
            metrics_config["uniqueness"] = uniqueness_config

        if validity_configs:
            metrics_config["validity"] = validity_configs

        if consistency_configs:
            metrics_config["consistency"] = consistency_configs

        if freshness_config:
            metrics_config["freshness"] = freshness_config

        # Evaluate all metrics
        metric_results = self.metrics.evaluate_all(data, metrics_config)

        # Check schema drift
        drift_report = None
        if check_schema_drift and self.schema_detector:
            # Build schema dict for drift detection
            schema_dict = {}
            for col in columns:
                schema_dict[col] = {
                    "type": schema[col],
                    "nullable": True,  # Would need more info for accurate nullable
                    "position": columns.index(col)
                }

            drift_report = self.schema_detector.detect_drift(
                source_info.name,
                schema_dict
            )

            # Save new snapshot if changed
            if drift_report.has_drift:
                self.schema_detector.snapshot(source_info.name, schema_dict)

        # Evaluate rules and generate alerts
        alerts = self.rule_evaluator.evaluate_all(
            source_info.name,
            [r.to_dict() for r in metric_results]
        )

        # Calculate dimension scores
        dimensions = self._calculate_dimension_scores(metric_results)

        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_results)
        overall_status = self._determine_status(metric_results, alerts, drift_report)

        processing_time = int((time.time() - start_time) * 1000)

        report = QualityReport(
            source_name=source_info.name,
            source_type=source_info.source_type,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            overall_status=overall_status,
            dimensions=dimensions,
            metrics=[r.to_dict() for r in metric_results],
            alerts=[a.to_dict() for a in alerts],
            schema_drift=drift_report.to_dict() if drift_report else None,
            row_count=source_info.row_count or len(data),
            columns_checked=len(columns),
            processing_time_ms=processing_time,
            metadata={
                "sample_size": len(data),
                "schema": schema
            }
        )

        self._reports.append(report)
        return report

    def evaluate_file(
        self,
        file_path: str,
        file_type: Optional[str] = None,
        **kwargs
    ) -> QualityReport:
        """
        Evaluate quality from a file

        Args:
            file_path: Path to file
            file_type: Type of file (auto-detected if not provided)
            **kwargs: Additional arguments for evaluate()

        Returns:
            QualityReport
        """
        path = Path(file_path)

        if file_type is None:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                file_type = "parquet"
            elif suffix == ".csv":
                file_type = "csv"
            else:
                raise ValueError(f"Cannot auto-detect file type for: {suffix}")

        connector = create_connector(file_type, file_path)
        return self.evaluate(connector, **kwargs)

    def evaluate_sql(
        self,
        connection_string: str,
        table_name: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> QualityReport:
        """
        Evaluate quality from SQL table

        Args:
            connection_string: Database connection string
            table_name: Table name
            schema: Schema name (optional)
            **kwargs: Additional arguments for evaluate()

        Returns:
            QualityReport
        """
        connector = create_connector(
            "sql",
            connection_string,
            table_name=table_name,
            schema=schema
        )
        return self.evaluate(connector, **kwargs)

    def evaluate_delta(
        self,
        table_path: str,
        version: Optional[int] = None,
        **kwargs
    ) -> QualityReport:
        """
        Evaluate quality from Delta Lake table

        Args:
            table_path: Path to Delta table
            version: Specific version (optional)
            **kwargs: Additional arguments for evaluate()

        Returns:
            QualityReport
        """
        connector = create_connector(
            "delta",
            table_path,
            version=version
        )
        return self.evaluate(connector, **kwargs)

    def evaluate_freshness(
        self,
        connector: DataConnector,
        timestamp_column: str,
        sla_hours: float,
        max_age_hours: Optional[float] = None
    ) -> MetricResult:
        """
        Evaluate freshness only

        Args:
            connector: Data connector
            timestamp_column: Column with timestamps
            sla_hours: SLA in hours
            max_age_hours: Maximum acceptable age

        Returns:
            MetricResult for freshness
        """
        data = connector.read_sample(1000)

        return self.metrics.freshness.evaluate(
            data=data,
            timestamp_column=timestamp_column,
            sla_hours=sla_hours,
            max_age_hours=max_age_hours or sla_hours * 2
        )

    def check_schema_drift(
        self,
        connector: DataConnector,
        baseline_version: Optional[int] = None
    ) -> DriftReport:
        """
        Check for schema drift

        Args:
            connector: Data connector
            baseline_version: Compare against specific version

        Returns:
            DriftReport
        """
        if not self.schema_detector:
            raise RuntimeError("Schema tracking not enabled")

        source_info = connector.get_info()
        schema = connector.get_schema()
        columns = list(schema.keys())

        schema_dict = {}
        for col in columns:
            schema_dict[col] = {
                "type": schema[col],
                "nullable": True,
                "position": columns.index(col)
            }

        return self.schema_detector.detect_drift(
            source_info.name,
            schema_dict,
            baseline_version
        )

    def _calculate_dimension_scores(
        self,
        results: List[MetricResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate score per dimension"""
        dimensions = {}

        for result in results:
            dim = result.dimension
            if dim not in dimensions:
                dimensions[dim] = {
                    "scores": [],
                    "statuses": []
                }

            dimensions[dim]["scores"].append(result.value)
            dimensions[dim]["statuses"].append(result.status)

        # Aggregate
        for dim, data in dimensions.items():
            scores = data["scores"]
            statuses = data["statuses"]

            avg_score = sum(scores) / len(scores) if scores else 0

            if any(s == QualityStatus.FAILED for s in statuses):
                status = "failed"
            elif any(s == QualityStatus.WARNING for s in statuses):
                status = "warning"
            else:
                status = "passed"

            dimensions[dim] = {
                "score": avg_score,
                "status": status,
                "checks": len(scores)
            }

        return dimensions

    def _calculate_overall_score(self, results: List[MetricResult]) -> float:
        """Calculate overall quality score"""
        scored = [r for r in results if r.status != QualityStatus.SKIPPED]
        if not scored:
            return 1.0
        return sum(r.value for r in scored) / len(scored)

    def _determine_status(
        self,
        results: List[MetricResult],
        alerts: List[QualityAlert],
        drift_report: Optional[DriftReport]
    ) -> str:
        """Determine overall status"""
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            return "failed"

        # Check for failed metrics
        failed = [r for r in results if r.status == QualityStatus.FAILED]
        if failed:
            return "failed"

        # Check for breaking schema changes
        if drift_report and drift_report.breaking_changes > 0:
            return "failed"

        # Check for warnings
        warnings = [r for r in results if r.status == QualityStatus.WARNING]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        if warnings or warning_alerts:
            return "warning"

        return "passed"

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        table_name: Optional[str] = None
    ) -> List[QualityAlert]:
        """Get active (unacknowledged) alerts"""
        return self.rule_evaluator.get_active_alerts(level, table_name)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        return self.rule_evaluator.acknowledge_alert(alert_id)

    def get_report_history(
        self,
        source_name: Optional[str] = None,
        limit: int = 10
    ) -> List[QualityReport]:
        """Get report history"""
        reports = self._reports
        if source_name:
            reports = [r for r in reports if r.source_name == source_name]
        return reports[-limit:]

    def get_schema_history(
        self,
        table_name: str,
        limit: int = 10
    ) -> List[SchemaSnapshot]:
        """Get schema version history"""
        if not self.schema_detector:
            return []
        return self.schema_detector.get_snapshot_history(table_name, limit)

    def export_report(
        self,
        report: QualityReport,
        output_path: str,
        format: str = "json"
    ) -> str:
        """Export report to file"""
        path = Path(output_path)

        if format == "json":
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report.to_json())
        elif format == "markdown":
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report.to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")

        return str(path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "rules": self.rule_evaluator.get_statistics(),
            "reports_generated": len(self._reports)
        }

        if self.schema_detector:
            stats["schema_tracking"] = self.schema_detector.get_statistics()

        return stats

    def create_default_rules_for_table(
        self,
        connector: DataConnector
    ) -> RuleSet:
        """Create default rules for a table based on schema"""
        info = connector.get_info()
        columns = list(connector.get_schema().keys())
        return self.rule_evaluator.create_default_rules(info.name, columns)
