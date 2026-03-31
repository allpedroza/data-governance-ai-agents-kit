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
#   "spacy>=3.5.0; extra == \"spacy\"",
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
Data Steward Agent

Operational copilot for data stewards. Absorbs repetitive, operational
and documental work while keeping decisions, accountability and final
approval human.

Capabilities:
1. Issue intake & triage (classify, suggest severity, identify owner)
2. Assisted business glossary curation (consolidate definitions, detect conflicts)
3. Quality rule drafting (business language + technical expression)
4. Human-readable impact & lineage explanation
5. Approval workflow orchestration (evidence packages, changelogs)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from data_governance.rag_discovery.providers.base import LLMProvider
except ImportError:
    LLMProvider = None  # type: ignore[assignment,misc]

from data_governance.data_steward.models import (
    ActivityLogEntry,
    ApprovalRequest,
    DataIssue,
    GlossaryTerm,
    GlossaryTermStatus,
    ImpactReport,
    IssueStatus,
    QualityRuleDraft,
    QualityRuleStatus,
    StewardAssignment,
)
from data_governance.data_steward.glossary.glossary_curator import GlossaryCurator
from data_governance.data_steward.impact.impact_explainer import ImpactExplainer
from data_governance.data_steward.intake.issue_triager import IssueTriager
from data_governance.data_steward.quality_rules.rule_drafter import QualityRuleDrafter
from data_governance.data_steward.storage.steward_store import StewardStore
from data_governance.data_steward.workflow.approval_workflow import ApprovalWorkflow


class DataStewardAgent:
    """Operational copilot for data stewards.

    Composes five capabilities -- issue triage, glossary curation, quality
    rule drafting, impact explanation, and approval workflow -- into a
    single orchestration surface backed by JSON persistence.

    The agent **proposes**; the human steward **approves**.
    """

    def __init__(
        self,
        persist_dir: str = "./steward_data",
        llm_provider=None,
    ):
        self.store = StewardStore(persist_dir)
        self.triager = IssueTriager(llm_provider)
        self.curator = GlossaryCurator(llm_provider)
        self.rule_drafter = QualityRuleDrafter(llm_provider)
        self.impact_explainer = ImpactExplainer(llm_provider)
        self.workflow = ApprovalWorkflow(self.store)
        self.llm_provider = llm_provider

    # ==================================================================
    # 1. Issue Intake & Triage
    # ==================================================================

    def triage_issue(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DataIssue:
        """Receive a free-text data issue and return a triaged ``DataIssue``.

        Classifies the problem, suggests severity, identifies domain /
        dataset / attribute and proposes next steps.
        """
        issue = self.triager.triage(description, context)

        # Try to match owner from stored assignments
        assignments = self.store.list_assignments()
        if assignments:
            issue = self.triager.suggest_owner(issue, assignments)

        self.store.save_issue(issue)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=datetime.now().isoformat(),
                action="issue_triaged",
                actor="system",
                domain=issue.domain,
                dataset=issue.dataset,
                details={
                    "issue_id": issue.issue_id,
                    "category": issue.category,
                    "severity": issue.severity,
                },
            )
        )
        return issue

    def enrich_issue(
        self,
        issue_id: str,
        quality_report=None,
        classification_report=None,
    ) -> DataIssue:
        """Enrich an existing issue with evidence from other agents."""
        issue = self.store.load_issue(issue_id)
        if issue is None:
            raise ValueError(f"Issue '{issue_id}' not found")
        issue = self.triager.enrich_with_agent_findings(
            issue, quality_report, classification_report
        )
        self.store.save_issue(issue)
        return issue

    def list_issues(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[DataIssue]:
        """List issues with optional filters."""
        return self.store.list_issues(
            status=status, severity=severity, domain=domain
        )

    def resolve_issue(self, issue_id: str, notes: str) -> DataIssue:
        """Mark an issue as resolved."""
        issue = self.store.load_issue(issue_id)
        if issue is None:
            raise ValueError(f"Issue '{issue_id}' not found")
        now = datetime.now().isoformat()
        issue.status = IssueStatus.RESOLVED.value
        issue.resolved_at = now
        issue.resolution_notes = notes
        issue.updated_at = now
        self.store.save_issue(issue)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=now,
                action="issue_resolved",
                actor="steward",
                domain=issue.domain,
                dataset=issue.dataset,
                details={"issue_id": issue_id, "notes": notes},
            )
        )
        return issue

    # ==================================================================
    # 2. Business Glossary Curation
    # ==================================================================

    def curate_term(
        self,
        term_name: str,
        sources: List[Dict[str, str]],
        domain: str = "",
        related_datasets: Optional[List[str]] = None,
        related_attributes: Optional[List[str]] = None,
    ) -> GlossaryTerm:
        """Consolidate definitions from multiple sources into a proposed term."""
        # Detect conflicts with existing terms
        existing = self.store.list_terms()
        conflicts = self.curator.detect_conflicts(term_name, existing)

        term = self.curator.curate_term(
            term_name, sources, domain, related_datasets, related_attributes
        )
        if conflicts:
            term.semantic_conflicts = conflicts + term.semantic_conflicts

        self.store.save_term(term)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=datetime.now().isoformat(),
                action="term_curated",
                actor="system",
                domain=domain,
                dataset=None,
                details={
                    "term_id": term.term_id,
                    "term_name": term_name,
                    "conflicts": len(term.semantic_conflicts),
                },
            )
        )
        return term

    def detect_conflicts(self, term_name: str) -> List[Dict[str, str]]:
        """Check for semantic conflicts with existing glossary terms."""
        existing = self.store.list_terms()
        return self.curator.detect_conflicts(term_name, existing)

    def submit_term_for_approval(
        self, term_id: str, submitted_by: str = "steward"
    ) -> ApprovalRequest:
        """Submit a glossary term for formal approval."""
        term = self.store.load_term(term_id)
        if term is None:
            raise ValueError(f"Term '{term_id}' not found")

        term.status = GlossaryTermStatus.UNDER_REVIEW.value
        term.updated_at = datetime.now().isoformat()
        self.store.save_term(term)

        return self.workflow.create_request(
            request_type="glossary_term",
            item_id=term_id,
            title=f"Aprovacao de termo: {term.term_name}",
            summary=f"Definicao proposta: {term.proposed_definition}",
            evidence={"term": term.to_dict()},
            submitted_by=submitted_by,
            version_after=term.to_dict(),
        )

    def list_glossary(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[GlossaryTerm]:
        """List glossary terms with optional filters."""
        return self.store.list_terms(domain=domain, status=status)

    # ==================================================================
    # 3. Quality Rule Drafting
    # ==================================================================

    def draft_rules(
        self,
        dataset: str,
        domain: str,
        columns: Optional[List[Dict[str, Any]]] = None,
        quality_report=None,
        enrichment_result=None,
    ) -> List[QualityRuleDraft]:
        """Draft quality rules for a dataset.

        Returns rules in business language + technical expression for
        the steward to review and approve.
        """
        rules = self.rule_drafter.draft_rules(
            dataset, domain, columns, quality_report, enrichment_result
        )
        for rule in rules:
            self.store.save_rule(rule)

        self.store.append_activity(
            ActivityLogEntry(
                timestamp=datetime.now().isoformat(),
                action="rules_drafted",
                actor="system",
                domain=domain,
                dataset=dataset,
                details={"count": len(rules)},
            )
        )
        return rules

    def explain_deviation(
        self, rule_id: str, quality_report=None
    ) -> str:
        """Explain a quality rule deviation in business language."""
        rule = self.store.load_rule(rule_id)
        if rule is None:
            raise ValueError(f"Rule '{rule_id}' not found")
        return self.rule_drafter.explain_deviation(rule, quality_report)

    def submit_rule_for_approval(
        self, rule_id: str, submitted_by: str = "steward"
    ) -> ApprovalRequest:
        """Submit a quality rule for formal approval."""
        rule = self.store.load_rule(rule_id)
        if rule is None:
            raise ValueError(f"Rule '{rule_id}' not found")

        rule.status = QualityRuleStatus.PENDING_APPROVAL.value
        rule.updated_at = datetime.now().isoformat()
        self.store.save_rule(rule)

        return self.workflow.create_request(
            request_type="quality_rule",
            item_id=rule_id,
            title=f"Aprovacao de regra: {rule.business_description[:60]}",
            summary=(
                f"Regra: {rule.business_description}\n"
                f"Tecnica: {rule.technical_expression}"
            ),
            evidence={"rule": rule.to_dict()},
            submitted_by=submitted_by,
            version_after=rule.to_dict(),
        )

    def list_rules(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[QualityRuleDraft]:
        """List quality rules with optional filters."""
        return self.store.list_rules(domain=domain, status=status)

    # ==================================================================
    # 4. Impact Analysis
    # ==================================================================

    def explain_impact(
        self,
        change_description: str,
        dataset: str,
        attribute: Optional[str] = None,
        domain: str = "",
        lineage_data: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> ImpactReport:
        """Explain the impact of a proposed change in business language.

        Answers: which reports/KPIs may break, which teams are impacted,
        which rules depend on the attribute, and regulatory exceptions.
        """
        # Gather quality rules that may be affected
        all_rules = self.store.list_rules(domain=domain or None)
        relevant_rules = [
            r
            for r in all_rules
            if r.dataset == dataset
            or (attribute and r.attribute == attribute)
        ]

        report = self.impact_explainer.explain_impact(
            change_description=change_description,
            dataset=dataset,
            attribute=attribute,
            domain=domain,
            lineage_data=lineage_data,
            quality_rules=relevant_rules,
            contracts=contracts,
        )

        self.store.append_activity(
            ActivityLogEntry(
                timestamp=datetime.now().isoformat(),
                action="impact_analyzed",
                actor="system",
                domain=domain,
                dataset=dataset,
                details={
                    "report_id": report.report_id,
                    "risk_level": report.risk_level,
                },
            )
        )
        return report

    # ==================================================================
    # 5. Approval Workflow
    # ==================================================================

    def get_pending_approvals(
        self, approver: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """List pending approval requests."""
        return self.workflow.get_pending_approvals(approver)

    def submit_decision(
        self,
        request_id: str,
        status: str,
        reviewer: str,
        notes: str = "",
    ) -> ApprovalRequest:
        """Submit an approval decision (approved/rejected/revision_requested).

        When a glossary term or quality rule is approved, its status is
        updated accordingly.
        """
        request = self.workflow.submit_decision(
            request_id, status, reviewer, notes
        )

        # Side-effect: update the underlying item status
        if status == "approved":
            self._on_approval(request)
        elif status == "rejected":
            self._on_rejection(request)

        return request

    def _on_approval(self, request: ApprovalRequest) -> None:
        """Update item status after approval."""
        if request.request_type == "glossary_term":
            term = self.store.load_term(request.item_id)
            if term:
                term.status = GlossaryTermStatus.APPROVED.value
                term.approved_by = request.reviewed_by
                term.updated_at = datetime.now().isoformat()
                self.store.save_term(term)
        elif request.request_type == "quality_rule":
            rule = self.store.load_rule(request.item_id)
            if rule:
                rule.status = QualityRuleStatus.ACTIVE.value
                rule.approved_by = request.reviewed_by
                rule.updated_at = datetime.now().isoformat()
                self.store.save_rule(rule)

    def _on_rejection(self, request: ApprovalRequest) -> None:
        """Update item status after rejection."""
        if request.request_type == "glossary_term":
            term = self.store.load_term(request.item_id)
            if term:
                term.status = GlossaryTermStatus.CANDIDATE.value
                term.updated_at = datetime.now().isoformat()
                self.store.save_term(term)
        elif request.request_type == "quality_rule":
            rule = self.store.load_rule(request.item_id)
            if rule:
                rule.status = QualityRuleStatus.DRAFT.value
                rule.updated_at = datetime.now().isoformat()
                self.store.save_rule(rule)

    # ==================================================================
    # Ownership / Assignments
    # ==================================================================

    def assign_steward(
        self,
        domain: str,
        person_name: str,
        role: str = "data_steward",
        person_email: Optional[str] = None,
        datasets: Optional[List[str]] = None,
    ) -> StewardAssignment:
        """Assign a person to a governance role on a domain."""
        now = datetime.now().isoformat()
        assignment = StewardAssignment(
            assignment_id=str(uuid.uuid4())[:12],
            person_name=person_name,
            person_email=person_email,
            role=role,
            domain=domain,
            datasets=datasets or [],
            assigned_at=now,
            is_active=True,
        )
        self.store.save_assignment(assignment)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=now,
                action="steward_assigned",
                actor="admin",
                domain=domain,
                dataset=None,
                details={
                    "person": person_name,
                    "role": role,
                },
            )
        )
        return assignment

    def list_assignments(
        self, domain: Optional[str] = None
    ) -> List[StewardAssignment]:
        """List steward/owner assignments."""
        return self.store.list_assignments(domain=domain)

    # ==================================================================
    # Activity Log
    # ==================================================================

    def get_activity_log(
        self, limit: int = 50, domain: Optional[str] = None
    ) -> List[ActivityLogEntry]:
        """Get recent governance activity entries."""
        return self.store.load_activity_log(limit=limit, domain=domain)
