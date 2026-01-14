"""
Data Quality Metrics

Implements the core quality dimensions:
- Completeness: Missing/null values
- Uniqueness: Duplicate detection
- Validity: Format and range validation
- Consistency: Cross-field and cross-table checks
- Freshness: Data timeliness vs SLA
- Accuracy: Data correctness indicators
"""

from .quality_metrics import (
    QualityMetrics,
    CompletenessMetric,
    UniquenessMetric,
    ValidityMetric,
    ConsistencyMetric,
    FreshnessMetric,
    ClassBalanceMetric,
    LabelNoiseMetric,
    LeakageCheckMetric,
    GoldSetAgreementMetric,
    MetricResult
)

from .schema_drift import SchemaDriftDetector, SchemaChange, DriftReport

__all__ = [
    "QualityMetrics",
    "CompletenessMetric",
    "UniquenessMetric",
    "ValidityMetric",
    "ConsistencyMetric",
    "FreshnessMetric",
    "ClassBalanceMetric",
    "LabelNoiseMetric",
    "LeakageCheckMetric",
    "GoldSetAgreementMetric",
    "MetricResult",
    "SchemaDriftDetector",
    "SchemaChange",
    "DriftReport"
]
