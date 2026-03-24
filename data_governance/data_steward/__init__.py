"""
Data Steward Agent

Operational copilot for data stewards -- absorbs repetitive, operational
and documental work while keeping decisions, accountability and final
approval human.

Capabilities:
1. Issue intake & triage
2. Assisted business glossary curation
3. Quality rule drafting (business + technical)
4. Human-readable impact & lineage explanation
5. Approval workflow orchestration
"""

from .agent import DataStewardAgent
from .models import (
    ActivityLogEntry,
    ApprovalRequest,
    ApprovalStatus,
    DataIssue,
    GlossaryTerm,
    GlossaryTermStatus,
    GovernanceRole,
    ImpactReport,
    IssueCategory,
    IssueSeverity,
    IssueStatus,
    QualityRuleDraft,
    QualityRuleStatus,
    StewardAssignment,
)

__all__ = [
    "DataStewardAgent",
    "DataIssue",
    "GlossaryTerm",
    "QualityRuleDraft",
    "ImpactReport",
    "ApprovalRequest",
    "StewardAssignment",
    "ActivityLogEntry",
    "IssueSeverity",
    "IssueCategory",
    "IssueStatus",
    "ApprovalStatus",
    "GovernanceRole",
    "GlossaryTermStatus",
    "QualityRuleStatus",
]
