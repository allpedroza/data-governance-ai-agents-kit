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
Data Steward Agent models.

Defines all data models used by the Data Steward Agent for issue triage,
glossary curation, quality rule drafting, impact analysis, approval workflows,
steward assignments, and activity logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from shared.serialization import SerializableMixin
except ImportError:
    import sys as _sys, pathlib as _pathlib
    _root = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "shared").is_dir())
    _sys.path.insert(0, str(_root))
    from shared.serialization import SerializableMixin


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IssueSeverity(Enum):
    """Severity levels for data issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):
    """Categories for data issues."""
    QUALITY = "quality"
    METADATA = "metadata"
    OWNERSHIP = "ownership"
    COMPLIANCE = "compliance"
    LINEAGE = "lineage"
    CHANGE_REQUEST = "change_request"


class IssueStatus(Enum):
    """Lifecycle statuses for data issues."""
    OPEN = "open"
    TRIAGED = "triaged"
    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ApprovalStatus(Enum):
    """Statuses for approval requests."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class GovernanceRole(Enum):
    """Governance roles within a data domain."""
    DATA_OWNER = "data_owner"
    DATA_STEWARD = "data_steward"
    DATA_CUSTODIAN = "data_custodian"


class GlossaryTermStatus(Enum):
    """Lifecycle statuses for glossary terms."""
    CANDIDATE = "candidate"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class QualityRuleStatus(Enum):
    """Lifecycle statuses for quality rules."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataIssue(SerializableMixin):
    """Issue de dados triada pelo agente."""

    issue_id: str
    title: str
    raw_description: str
    category: str
    severity: str
    status: str
    domain: Optional[str]
    dataset: Optional[str]
    attribute: Optional[str]
    probable_owner: Optional[str]
    probable_steward: Optional[str]
    root_cause_hypothesis: Optional[str]
    suggested_next_steps: List[str]
    sla_hours: Optional[int]
    source_agent_findings: Dict[str, Any]
    created_at: str
    updated_at: Optional[str]
    resolved_at: Optional[str]
    resolution_notes: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "raw_description": self.raw_description,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "domain": self.domain,
            "dataset": self.dataset,
            "attribute": self.attribute,
            "probable_owner": self.probable_owner,
            "probable_steward": self.probable_steward,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "suggested_next_steps": self.suggested_next_steps,
            "sla_hours": self.sla_hours,
            "source_agent_findings": self.source_agent_findings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        severity_indicator = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🔵 LOW",
        }
        sev_display = severity_indicator.get(self.severity, self.severity.upper())

        lines = [
            f"# Issue: {self.title}",
            f"*ID:* `{self.issue_id}`",
            "",
            "## Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Category | **{self.category.upper()}** |",
            f"| Severity | {sev_display} |",
            f"| Status | {self.status.upper()} |",
            f"| Domain | {self.domain or '-'} |",
            f"| Dataset | {self.dataset or '-'} |",
            f"| Attribute | {self.attribute or '-'} |",
            f"| Probable Owner | {self.probable_owner or '-'} |",
            f"| Probable Steward | {self.probable_steward or '-'} |",
            f"| SLA (hours) | {self.sla_hours or '-'} |",
            f"| Created | {self.created_at} |",
            f"| Updated | {self.updated_at or '-'} |",
            f"| Resolved | {self.resolved_at or '-'} |",
        ]

        lines.extend(["", "## Description", "", self.raw_description])

        if self.root_cause_hypothesis:
            lines.extend([
                "",
                "## Root Cause Hypothesis",
                "",
                self.root_cause_hypothesis,
            ])

        if self.suggested_next_steps:
            lines.extend(["", "## Suggested Next Steps", ""])
            for i, step in enumerate(self.suggested_next_steps, 1):
                lines.append(f"{i}. {step}")

        if self.resolution_notes:
            lines.extend(["", "## Resolution Notes", "", self.resolution_notes])

        if self.source_agent_findings:
            lines.extend([
                "",
                "## Source Agent Findings",
                "",
                "```json",
                json.dumps(self.source_agent_findings, ensure_ascii=False, indent=2),
                "```",
            ])

        return "\n".join(lines)


@dataclass
class GlossaryTerm(SerializableMixin):
    """Termo de glossario curado."""

    term_id: str
    term_name: str
    candidate_definitions: List[Dict[str, str]]
    proposed_definition: str
    proposed_definition_en: str
    business_description: str
    domain: str
    suggested_owner: Optional[str]
    suggested_steward: Optional[str]
    source_system: Optional[str]
    related_datasets: List[str]
    related_attributes: List[str]
    semantic_conflicts: List[Dict[str, str]]
    status: str
    approved_by: Optional[str]
    version: int
    created_at: str
    updated_at: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term_id": self.term_id,
            "term_name": self.term_name,
            "candidate_definitions": self.candidate_definitions,
            "proposed_definition": self.proposed_definition,
            "proposed_definition_en": self.proposed_definition_en,
            "business_description": self.business_description,
            "domain": self.domain,
            "suggested_owner": self.suggested_owner,
            "suggested_steward": self.suggested_steward,
            "source_system": self.source_system,
            "related_datasets": self.related_datasets,
            "related_attributes": self.related_attributes,
            "semantic_conflicts": self.semantic_conflicts,
            "status": self.status,
            "approved_by": self.approved_by,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        status_indicator = {
            "candidate": "📋 Candidate",
            "under_review": "🔍 Under Review",
            "approved": "✅ Approved",
            "deprecated": "🚫 Deprecated",
        }
        status_display = status_indicator.get(self.status, self.status.upper())

        lines = [
            f"# Glossary Term: {self.term_name}",
            f"*ID:* `{self.term_id}` | *Version:* {self.version} | *Status:* {status_display}",
            "",
            "## Proposed Definition",
            "",
            f"> {self.proposed_definition}",
            "",
            f"**English:** {self.proposed_definition_en}",
            "",
            "## Business Description",
            "",
            self.business_description,
            "",
            "## Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Domain | {self.domain} |",
            f"| Suggested Owner | {self.suggested_owner or '-'} |",
            f"| Suggested Steward | {self.suggested_steward or '-'} |",
            f"| Source System | {self.source_system or '-'} |",
            f"| Approved By | {self.approved_by or '-'} |",
            f"| Created | {self.created_at} |",
            f"| Updated | {self.updated_at or '-'} |",
        ]

        if self.candidate_definitions:
            lines.extend(["", "## Candidate Definitions", ""])
            for i, cdef in enumerate(self.candidate_definitions, 1):
                source = cdef.get("source", "unknown")
                definition = cdef.get("definition", "-")
                lines.append(f"{i}. **{source}:** {definition}")

        if self.semantic_conflicts:
            lines.extend(["", "## Semantic Conflicts", ""])
            for conflict in self.semantic_conflicts:
                conflict_type = conflict.get("type", "conflict")
                description = conflict.get("description", "-")
                lines.append(f"- ⚠️ **{conflict_type}:** {description}")

        if self.related_datasets:
            lines.extend(["", "## Related Datasets", ""])
            for ds in self.related_datasets:
                lines.append(f"- {ds}")

        if self.related_attributes:
            lines.extend(["", "## Related Attributes", ""])
            for attr in self.related_attributes:
                lines.append(f"- `{attr}`")

        return "\n".join(lines)


@dataclass
class QualityRuleDraft(SerializableMixin):
    """Regra de quality sugerida."""

    rule_id: str
    business_description: str
    technical_expression: str
    dimension: str
    dataset: str
    attribute: Optional[str]
    domain: str
    severity: str
    current_deviation: Optional[Dict[str, Any]]
    deviation_explanation: Optional[str]
    status: str
    approved_by: Optional[str]
    version: int
    created_at: str
    updated_at: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "business_description": self.business_description,
            "technical_expression": self.technical_expression,
            "dimension": self.dimension,
            "dataset": self.dataset,
            "attribute": self.attribute,
            "domain": self.domain,
            "severity": self.severity,
            "current_deviation": self.current_deviation,
            "deviation_explanation": self.deviation_explanation,
            "status": self.status,
            "approved_by": self.approved_by,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        severity_indicator = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🔵 LOW",
        }
        sev_display = severity_indicator.get(self.severity, self.severity.upper())

        status_indicator = {
            "draft": "📝 Draft",
            "pending_approval": "⏳ Pending Approval",
            "active": "✅ Active",
            "disabled": "🚫 Disabled",
        }
        status_display = status_indicator.get(self.status, self.status.upper())

        lines = [
            f"# Quality Rule: {self.rule_id}",
            f"*Version:* {self.version} | *Status:* {status_display} | *Severity:* {sev_display}",
            "",
            "## Business Description",
            "",
            self.business_description,
            "",
            "## Technical Expression",
            "",
            "```sql",
            self.technical_expression,
            "```",
            "",
            "## Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Dimension | {self.dimension} |",
            f"| Dataset | {self.dataset} |",
            f"| Attribute | {self.attribute or '-'} |",
            f"| Domain | {self.domain} |",
            f"| Approved By | {self.approved_by or '-'} |",
            f"| Created | {self.created_at} |",
            f"| Updated | {self.updated_at or '-'} |",
        ]

        if self.current_deviation:
            lines.extend([
                "",
                "## Current Deviation",
                "",
                "```json",
                json.dumps(self.current_deviation, ensure_ascii=False, indent=2),
                "```",
            ])

        if self.deviation_explanation:
            lines.extend([
                "",
                "## Deviation Explanation",
                "",
                self.deviation_explanation,
            ])

        return "\n".join(lines)


@dataclass
class ImpactReport(SerializableMixin):
    """Relatorio de impacto em linguagem humana."""

    report_id: str
    change_description: str
    dataset: str
    attribute: Optional[str]
    domain: str
    affected_reports: List[str]
    affected_teams: List[str]
    affected_rules: List[str]
    regulatory_exceptions: List[str]
    risk_level: str
    human_summary: str
    lineage_evidence: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "change_description": self.change_description,
            "dataset": self.dataset,
            "attribute": self.attribute,
            "domain": self.domain,
            "affected_reports": self.affected_reports,
            "affected_teams": self.affected_teams,
            "affected_rules": self.affected_rules,
            "regulatory_exceptions": self.regulatory_exceptions,
            "risk_level": self.risk_level,
            "human_summary": self.human_summary,
            "lineage_evidence": self.lineage_evidence,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        risk_indicator = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🔵 LOW",
        }
        risk_display = risk_indicator.get(self.risk_level, self.risk_level.upper())

        lines = [
            f"# Impact Report: {self.report_id}",
            f"*Generated:* {self.generated_at}",
            "",
            "## Change Description",
            "",
            self.change_description,
            "",
            "## Summary",
            "",
            self.human_summary,
            "",
            "## Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Dataset | {self.dataset} |",
            f"| Attribute | {self.attribute or '-'} |",
            f"| Domain | {self.domain} |",
            f"| Risk Level | {risk_display} |",
        ]

        if self.affected_reports:
            lines.extend(["", "## Affected Reports", ""])
            for report in self.affected_reports:
                lines.append(f"- {report}")

        if self.affected_teams:
            lines.extend(["", "## Affected Teams", ""])
            for team in self.affected_teams:
                lines.append(f"- {team}")

        if self.affected_rules:
            lines.extend(["", "## Affected Rules", ""])
            for rule in self.affected_rules:
                lines.append(f"- {rule}")

        if self.regulatory_exceptions:
            lines.extend(["", "## Regulatory Exceptions", ""])
            for exception in self.regulatory_exceptions:
                lines.append(f"- ⚠️ {exception}")

        if self.lineage_evidence:
            lines.extend([
                "",
                "## Lineage Evidence",
                "",
                "```json",
                json.dumps(self.lineage_evidence, ensure_ascii=False, indent=2),
                "```",
            ])

        return "\n".join(lines)


@dataclass
class ApprovalRequest(SerializableMixin):
    """Pedido de aprovacao."""

    request_id: str
    request_type: str
    item_id: str
    title: str
    summary: str
    evidence_package: Dict[str, Any] = field(default_factory=dict)
    proposed_approvers: List[str] = field(default_factory=list)
    status: str = ""
    submitted_by: str = ""
    submitted_at: str = ""
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    decision_notes: Optional[str] = None
    version_before: Optional[Dict[str, Any]] = None
    version_after: Optional[Dict[str, Any]] = None
    changelog_entry: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "item_id": self.item_id,
            "title": self.title,
            "summary": self.summary,
            "evidence_package": self.evidence_package,
            "proposed_approvers": self.proposed_approvers,
            "status": self.status,
            "submitted_by": self.submitted_by,
            "submitted_at": self.submitted_at,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "decision_notes": self.decision_notes,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "changelog_entry": self.changelog_entry,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        status_indicator = {
            "draft": "📝 Draft",
            "pending_review": "⏳ Pending Review",
            "approved": "✅ Approved",
            "rejected": "❌ Rejected",
            "revision_requested": "🔄 Revision Requested",
        }
        status_display = status_indicator.get(self.status, self.status.upper())

        lines = [
            f"# Approval Request: {self.title}",
            f"*ID:* `{self.request_id}` | *Status:* {status_display}",
            "",
            "## Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Request Type | {self.request_type} |",
            f"| Item ID | `{self.item_id}` |",
            f"| Submitted By | {self.submitted_by} |",
            f"| Submitted At | {self.submitted_at} |",
            f"| Reviewed By | {self.reviewed_by or '-'} |",
            f"| Reviewed At | {self.reviewed_at or '-'} |",
        ]

        lines.extend(["", "## Summary", "", self.summary])

        if self.proposed_approvers:
            lines.extend(["", "## Proposed Approvers", ""])
            for approver in self.proposed_approvers:
                lines.append(f"- {approver}")

        if self.evidence_package:
            lines.extend([
                "",
                "## Evidence Package",
                "",
                "```json",
                json.dumps(self.evidence_package, ensure_ascii=False, indent=2),
                "```",
            ])

        if self.decision_notes:
            lines.extend(["", "## Decision Notes", "", self.decision_notes])

        if self.changelog_entry:
            lines.extend(["", "## Changelog Entry", "", self.changelog_entry])

        if self.version_before or self.version_after:
            lines.extend(["", "## Version Diff", ""])
            if self.version_before:
                lines.extend([
                    "**Before:**",
                    "```json",
                    json.dumps(self.version_before, ensure_ascii=False, indent=2),
                    "```",
                ])
            if self.version_after:
                lines.extend([
                    "**After:**",
                    "```json",
                    json.dumps(self.version_after, ensure_ascii=False, indent=2),
                    "```",
                ])

        return "\n".join(lines)


@dataclass
class StewardAssignment(SerializableMixin):
    """Vincula pessoa a papel em dominio."""

    assignment_id: str
    person_name: str
    person_email: Optional[str]
    role: str
    domain: str
    datasets: List[str]
    assigned_at: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "person_name": self.person_name,
            "person_email": self.person_email,
            "role": self.role,
            "domain": self.domain,
            "datasets": self.datasets,
            "assigned_at": self.assigned_at,
            "is_active": self.is_active,
        }

    def to_markdown(self) -> str:
        active_display = "✅ Active" if self.is_active else "❌ Inactive"
        datasets_display = ", ".join(self.datasets) if self.datasets else "-"
        return (
            f"| `{self.assignment_id}` | {self.person_name} | "
            f"{self.person_email or '-'} | {self.role} | {self.domain} | "
            f"{datasets_display} | {self.assigned_at} | {active_display} |"
        )


@dataclass
class ActivityLogEntry(SerializableMixin):
    """Trilha de auditoria."""

    timestamp: str
    action: str
    actor: str
    domain: Optional[str] = None
    dataset: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "domain": self.domain,
            "dataset": self.dataset,
            "details": self.details,
        }

    def to_markdown(self) -> str:
        context_parts = []
        if self.domain:
            context_parts.append(f"domain={self.domain}")
        if self.dataset:
            context_parts.append(f"dataset={self.dataset}")
        context = f" [{', '.join(context_parts)}]" if context_parts else ""
        details_str = f" | {json.dumps(self.details, ensure_ascii=False)}" if self.details else ""
        return f"`{self.timestamp}` **{self.action}** by *{self.actor}*{context}{details_str}"
