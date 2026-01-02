"""
Quality Rules - Configurable rules and alert system

Provides:
- Declarative quality rules
- Rule sets for tables
- Alert generation and routing
- SLA enforcement
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Generated quality alert"""
    alert_id: str
    level: AlertLevel
    rule_name: str
    table_name: str
    column_name: Optional[str]
    dimension: str
    message: str
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "rule_name": self.rule_name,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "dimension": self.dimension,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged
        }


@dataclass
class QualityRule:
    """
    A single quality rule

    Examples:
        # Completeness rule
        QualityRule(
            name="customer_email_required",
            dimension="completeness",
            table_name="customers",
            column="email",
            threshold=0.99,
            alert_level=AlertLevel.CRITICAL
        )

        # Freshness SLA rule
        QualityRule(
            name="orders_freshness_sla",
            dimension="freshness",
            table_name="orders",
            column="updated_at",
            threshold=0.95,
            params={"sla_hours": 4, "max_age_hours": 6}
        )
    """
    name: str
    dimension: str  # completeness, uniqueness, validity, consistency, freshness
    table_name: str
    column: Optional[str] = None
    threshold: float = 0.95
    alert_level: AlertLevel = AlertLevel.WARNING
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "table_name": self.table_name,
            "column": self.column,
            "threshold": self.threshold,
            "alert_level": self.alert_level.value,
            "params": self.params,
            "description": self.description,
            "enabled": self.enabled,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityRule":
        return cls(
            name=data.get("name", ""),
            dimension=data.get("dimension", ""),
            table_name=data.get("table_name", ""),
            column=data.get("column"),
            threshold=data.get("threshold", 0.95),
            alert_level=AlertLevel(data.get("alert_level", "warning")),
            params=data.get("params", {}),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            tags=data.get("tags", [])
        )


@dataclass
class RuleSet:
    """
    Collection of quality rules for a table or domain

    Features:
    - Group related rules
    - Enable/disable sets
    - Version control
    """
    name: str
    description: str
    rules: List[QualityRule]
    version: str = "1.0"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "version": self.version,
            "enabled": self.enabled,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleSet":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            rules=[QualityRule.from_dict(r) for r in data.get("rules", [])],
            version=data.get("version", "1.0"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", [])
        )

    def get_rules_for_table(self, table_name: str) -> List[QualityRule]:
        """Get enabled rules for a specific table"""
        return [
            r for r in self.rules
            if r.enabled and r.table_name == table_name
        ]

    def get_rules_by_dimension(self, dimension: str) -> List[QualityRule]:
        """Get enabled rules for a specific dimension"""
        return [
            r for r in self.rules
            if r.enabled and r.dimension == dimension
        ]


class RuleEvaluator:
    """
    Evaluates quality rules and generates alerts

    Usage:
        evaluator = RuleEvaluator()
        evaluator.load_rules_from_file("rules.json")

        # Evaluate rules for a table
        alerts = evaluator.evaluate(
            table_name="orders",
            data=order_data,
            metrics_results=quality_results
        )
    """

    def __init__(self, persist_dir: str = "./quality_rules"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.rule_sets: Dict[str, RuleSet] = {}
        self.alerts: List[QualityAlert] = []
        self._alert_counter = 0

        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from disk"""
        for file_path in self.persist_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                rule_set = RuleSet.from_dict(data)
                self.rule_sets[rule_set.name] = rule_set
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    def save_rule_set(self, rule_set: RuleSet) -> None:
        """Save a rule set to disk"""
        self.rule_sets[rule_set.name] = rule_set
        file_path = self.persist_dir / f"{rule_set.name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(rule_set.to_dict(), f, ensure_ascii=False, indent=2)

    def add_rule(self, rule: QualityRule, rule_set_name: str = "default") -> None:
        """Add a rule to a rule set"""
        if rule_set_name not in self.rule_sets:
            self.rule_sets[rule_set_name] = RuleSet(
                name=rule_set_name,
                description=f"Rule set: {rule_set_name}",
                rules=[]
            )

        self.rule_sets[rule_set_name].rules.append(rule)
        self.save_rule_set(self.rule_sets[rule_set_name])

    def get_rules_for_table(self, table_name: str) -> List[QualityRule]:
        """Get all enabled rules for a table"""
        rules = []
        for rule_set in self.rule_sets.values():
            if rule_set.enabled:
                rules.extend(rule_set.get_rules_for_table(table_name))
        return rules

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self._alert_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ALERT-{timestamp}-{self._alert_counter:04d}"

    def evaluate_rule(
        self,
        rule: QualityRule,
        metric_value: float,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[QualityAlert]:
        """
        Evaluate a single rule and generate alert if violated

        Args:
            rule: The rule to evaluate
            metric_value: The measured metric value
            details: Additional details from metric evaluation

        Returns:
            QualityAlert if rule violated, None otherwise
        """
        if not rule.enabled:
            return None

        # Check if threshold violated
        violated = metric_value < rule.threshold

        if not violated:
            return None

        # Generate alert
        alert = QualityAlert(
            alert_id=self._generate_alert_id(),
            level=rule.alert_level,
            rule_name=rule.name,
            table_name=rule.table_name,
            column_name=rule.column,
            dimension=rule.dimension,
            message=self._generate_alert_message(rule, metric_value),
            value=metric_value,
            threshold=rule.threshold,
            metadata={
                "rule_params": rule.params,
                "details": details or {}
            }
        )

        self.alerts.append(alert)
        return alert

    def _generate_alert_message(self, rule: QualityRule, value: float) -> str:
        """Generate human-readable alert message"""
        dimension = rule.dimension.capitalize()
        col_info = f" for column '{rule.column}'" if rule.column else ""

        if rule.dimension == "freshness":
            sla = rule.params.get("sla_hours", 24)
            return (
                f"{dimension} SLA violation on {rule.table_name}{col_info}. "
                f"Score: {value:.2%}, SLA: {sla}h"
            )
        else:
            return (
                f"{dimension} check failed on {rule.table_name}{col_info}. "
                f"Value: {value:.2%}, Threshold: {rule.threshold:.2%}"
            )

    def evaluate_all(
        self,
        table_name: str,
        metric_results: List[Dict[str, Any]]
    ) -> List[QualityAlert]:
        """
        Evaluate all rules for a table given metric results

        Args:
            table_name: Name of the table
            metric_results: List of metric result dictionaries

        Returns:
            List of generated alerts
        """
        rules = self.get_rules_for_table(table_name)
        new_alerts = []

        for rule in rules:
            # Find matching metric result
            for result in metric_results:
                if result.get("dimension") == rule.dimension:
                    # Check column match if specified
                    if rule.column and result.get("column") != rule.column:
                        continue

                    alert = self.evaluate_rule(
                        rule=rule,
                        metric_value=result.get("value", 0),
                        details=result.get("details")
                    )

                    if alert:
                        new_alerts.append(alert)
                    break

        return new_alerts

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        table_name: Optional[str] = None
    ) -> List[QualityAlert]:
        """Get unacknowledged alerts with optional filters"""
        alerts = [a for a in self.alerts if not a.acknowledged]

        if level:
            alerts = [a for a in alerts if a.level == level]
        if table_name:
            alerts = [a for a in alerts if a.table_name == table_name]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        return {
            "rule_sets": len(self.rule_sets),
            "total_rules": sum(len(rs.rules) for rs in self.rule_sets.values()),
            "enabled_rules": sum(
                len([r for r in rs.rules if r.enabled])
                for rs in self.rule_sets.values()
            ),
            "total_alerts": len(self.alerts),
            "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "alerts_by_level": {
                level.value: len([a for a in self.alerts if a.level == level])
                for level in AlertLevel
            }
        }

    def create_default_rules(self, table_name: str, columns: List[str]) -> RuleSet:
        """Create a default rule set for a table"""
        rules = []

        # Completeness rule for all columns
        rules.append(QualityRule(
            name=f"{table_name}_completeness",
            dimension="completeness",
            table_name=table_name,
            threshold=0.95,
            description="Overall completeness check",
            alert_level=AlertLevel.WARNING
        ))

        # Uniqueness for potential ID columns
        for col in columns:
            col_lower = col.lower()
            if "id" in col_lower or col_lower.endswith("_key"):
                rules.append(QualityRule(
                    name=f"{table_name}_{col}_uniqueness",
                    dimension="uniqueness",
                    table_name=table_name,
                    column=col,
                    threshold=1.0,
                    description=f"Uniqueness check for {col}",
                    alert_level=AlertLevel.CRITICAL
                ))

        # Create rule set
        rule_set = RuleSet(
            name=f"{table_name}_default",
            description=f"Default quality rules for {table_name}",
            rules=rules,
            tags=["auto-generated"]
        )

        self.save_rule_set(rule_set)
        return rule_set
