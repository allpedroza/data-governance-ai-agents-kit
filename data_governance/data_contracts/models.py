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
Data Contracts models.

Defines schema, SLA, and quality rule structures used by the Data Contract Agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ContractField:
    """Schema definition for a field within a data contract."""

    name: str
    data_type: str
    required: bool = False
    description: str = ""
    pii: bool = False
    classification: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "required": self.required,
            "description": self.description,
            "pii": self.pii,
            "classification": self.classification,
            "constraints": self.constraints,
        }


@dataclass
class DataQualityRule:
    """Quality rule definition for a data contract."""

    name: str
    rule_type: str
    column: str
    severity: str = "warning"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rule_type": self.rule_type,
            "column": self.column,
            "severity": self.severity,
            "parameters": self.parameters,
        }


@dataclass
class DataContractSLA:
    """Service level agreements for the contract."""

    freshness_hours: Optional[int] = None
    latency_minutes: Optional[int] = None
    availability_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freshness_hours": self.freshness_hours,
            "latency_minutes": self.latency_minutes,
            "availability_pct": self.availability_pct,
        }


@dataclass
class DataContract:
    """Full data contract structure."""

    name: str
    version: str
    owner: str
    domain: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    fields: List[ContractField] = field(default_factory=list)
    quality_rules: List[DataQualityRule] = field(default_factory=list)
    sla: Optional[DataContractSLA] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "owner": self.owner,
            "domain": self.domain,
            "description": self.description,
            "source": self.source,
            "destination": self.destination,
            "fields": [field.to_dict() for field in self.fields],
            "quality_rules": [rule.to_dict() for rule in self.quality_rules],
            "sla": self.sla.to_dict() if self.sla else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Data Contract: {self.name}",
            f"*Version:* {self.version}",
            f"*Owner:* {self.owner}",
        ]

        if self.domain:
            lines.append(f"*Domain:* {self.domain}")
        if self.description:
            lines.append(f"*Description:* {self.description}")
        if self.source:
            lines.append(f"*Source:* {self.source}")
        if self.destination:
            lines.append(f"*Destination:* {self.destination}")
        if self.tags:
            lines.append(f"*Tags:* {', '.join(self.tags)}")

        lines.extend([
            "",
            "## Schema",
            "",
            "| Field | Type | Required | PII | Description |",
            "|---|---|---|---|---|",
        ])

        for field in self.fields:
            lines.append(
                f"| {field.name} | {field.data_type} | {'yes' if field.required else 'no'} | "
                f"{'yes' if field.pii else 'no'} | {field.description or '-'} |"
            )

        if self.quality_rules:
            lines.extend([
                "",
                "## Quality Rules",
                "",
                "| Rule | Type | Column | Severity | Parameters |",
                "|---|---|---|---|---|",
            ])
            for rule in self.quality_rules:
                params = json.dumps(rule.parameters, ensure_ascii=False)
                lines.append(
                    f"| {rule.name} | {rule.rule_type} | {rule.column} | {rule.severity} | {params} |"
                )

        if self.sla:
            lines.extend([
                "",
                "## SLA",
                "",
                f"- Freshness (hours): {self.sla.freshness_hours}",
                f"- Latency (minutes): {self.sla.latency_minutes}",
                f"- Availability (%): {self.sla.availability_pct}",
            ])

        if self.metadata:
            lines.extend([
                "",
                "## Metadata",
                "",
                "```json",
                json.dumps(self.metadata, ensure_ascii=False, indent=2),
                "```",
            ])

        lines.append("")
        lines.append(f"*Generated at {datetime.utcnow().isoformat()}Z*")

        return "\n".join(lines)


@dataclass
class ContractValidationFinding:
    """Single validation issue for a data contract."""

    level: str
    message: str
    column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "column": self.column,
        }


@dataclass
class ContractValidationReport:
    """Validation report for a contract applied to a dataset."""

    contract_name: str
    timestamp: str
    status: str
    row_count: int
    column_count: int
    findings: List[ContractValidationFinding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Contract Validation: {self.contract_name}",
            f"*Generated:* {self.timestamp}",
            f"*Status:* {self.status}",
            "",
            f"- Rows: {self.row_count}",
            f"- Columns: {self.column_count}",
        ]

        if self.findings:
            lines.extend(["", "## Findings", ""])
            for finding in self.findings:
                emoji = "❌" if finding.level == "error" else "⚠️"
                lines.append(f"- {emoji} {finding.message}")
        else:
            lines.extend(["", "✅ No issues found."])

        if self.metrics:
            lines.extend(["", "## Metrics", "", "```json", json.dumps(self.metrics, indent=2), "```"])

        return "\n".join(lines)
