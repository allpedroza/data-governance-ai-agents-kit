"""Approval workflow orchestration for data governance stewardship."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from data_governance.data_steward.models import (
    ApprovalRequest,
    ApprovalStatus,
    ActivityLogEntry,
)


class ApprovalWorkflow:
    """Orchestrates approval workflows for governance artefacts.

    Assembles evidence packages, routes to approvers based on domain
    ownership, tracks decisions, and produces human-readable changelogs.
    """

    def __init__(self, store):
        """Args:
            store: ``StewardStore`` instance for persistence.
        """
        self.store = store

    # ------------------------------------------------------------------
    # Create & submit
    # ------------------------------------------------------------------

    def create_request(
        self,
        request_type: str,
        item_id: str,
        title: str,
        summary: str,
        evidence: Optional[Dict[str, Any]] = None,
        submitted_by: str = "system",
        version_before: Optional[Dict[str, Any]] = None,
        version_after: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Create an approval request with evidence package.

        Args:
            request_type: ``"glossary_term"``, ``"quality_rule"`` or
                ``"change_request"``.
            item_id: Identifier of the item under review.
            evidence: Compiled from quality reports, classification, etc.
        """
        proposed_approvers = self._find_approvers(request_type, item_id)

        now = datetime.now().isoformat()
        request = ApprovalRequest(
            request_id=str(uuid.uuid4())[:12],
            request_type=request_type,
            item_id=item_id,
            title=title,
            summary=summary,
            evidence_package=evidence or {},
            proposed_approvers=proposed_approvers,
            status=ApprovalStatus.PENDING_REVIEW.value,
            submitted_by=submitted_by,
            submitted_at=now,
            reviewed_by=None,
            reviewed_at=None,
            decision_notes=None,
            version_before=version_before,
            version_after=version_after,
            changelog_entry=None,
        )

        self.store.save_approval(request)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=now,
                action="approval_created",
                actor=submitted_by,
                domain=None,
                dataset=None,
                details={
                    "request_id": request.request_id,
                    "type": request_type,
                    "title": title,
                },
            )
        )
        return request

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def submit_decision(
        self,
        request_id: str,
        status: str,
        reviewer: str,
        notes: str = "",
    ) -> ApprovalRequest:
        """Record an approval decision.

        Args:
            status: ``"approved"``, ``"rejected"`` or
                ``"revision_requested"``.
        """
        request = self.store.load_approval(request_id)
        if request is None:
            raise ValueError(f"Approval request '{request_id}' not found")

        valid = {
            ApprovalStatus.APPROVED.value,
            ApprovalStatus.REJECTED.value,
            ApprovalStatus.REVISION_REQUESTED.value,
        }
        if status not in valid:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {valid}")

        now = datetime.now().isoformat()
        request.status = status
        request.reviewed_by = reviewer
        request.reviewed_at = now
        request.decision_notes = notes

        if status == ApprovalStatus.APPROVED.value:
            request.changelog_entry = self._generate_changelog(request, reviewer)

        self.store.save_approval(request)
        self.store.append_activity(
            ActivityLogEntry(
                timestamp=now,
                action=f"approval_{status}",
                actor=reviewer,
                domain=None,
                dataset=None,
                details={
                    "request_id": request_id,
                    "type": request.request_type,
                    "title": request.title,
                    "notes": notes,
                },
            )
        )
        return request

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pending_approvals(
        self, approver: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """List pending approval requests, optionally for a specific approver."""
        pending = self.store.list_approvals(
            status=ApprovalStatus.PENDING_REVIEW.value
        )
        if approver:
            return [r for r in pending if approver in r.proposed_approvers]
        return pending

    def get_request_history(self, item_id: str) -> List[ApprovalRequest]:
        """Get all approval requests for a specific item."""
        return [r for r in self.store.list_approvals() if r.item_id == item_id]

    # ------------------------------------------------------------------
    # Evidence compilation
    # ------------------------------------------------------------------

    def compile_evidence(
        self,
        quality_report=None,
        classification_report=None,
        lineage_data=None,
        contract_report=None,
        custom_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile evidence package from multiple agent outputs."""
        evidence: Dict[str, Any] = {}

        if quality_report is not None:
            evidence["quality"] = {
                "overall_score": getattr(quality_report, "overall_score", None),
                "overall_status": getattr(quality_report, "overall_status", None),
                "dimensions": getattr(quality_report, "dimensions", {}),
                "alerts_count": len(getattr(quality_report, "alerts", [])),
            }

        if classification_report is not None:
            evidence["classification"] = {
                "overall_sensitivity": getattr(
                    classification_report, "overall_sensitivity", None
                ),
                "categories_found": getattr(
                    classification_report, "categories_found", []
                ),
                "pii_columns_count": len(
                    getattr(classification_report, "pii_columns", [])
                ),
            }

        if lineage_data is not None:
            evidence["lineage"] = {
                "assets_count": len(lineage_data.get("assets", {})),
                "transformations_count": len(
                    lineage_data.get("transformations", [])
                ),
            }

        if contract_report is not None:
            evidence["contract"] = {
                "status": getattr(contract_report, "status", None),
                "findings_count": len(
                    getattr(contract_report, "findings", [])
                ),
            }

        if custom_evidence:
            evidence["custom"] = custom_evidence

        evidence["compiled_at"] = datetime.now().isoformat()
        return evidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_approvers(self, request_type: str, item_id: str) -> List[str]:
        assignments = self.store.list_assignments()
        owners = []
        stewards = []
        for a in assignments:
            if not a.is_active:
                continue
            if a.role == "data_owner":
                owners.append(a.person_name)
            elif a.role == "data_steward":
                stewards.append(a.person_name)
        # Owners first, then stewards, deduplicated
        return list(dict.fromkeys(owners + stewards))

    def _generate_changelog(
        self, request: ApprovalRequest, reviewer: str
    ) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"[{now_str}] {request.request_type.upper()} aprovado por {reviewer}",
            f"  Item: {request.title}",
        ]
        if request.decision_notes:
            lines.append(f"  Notas: {request.decision_notes}")
        if request.version_before and request.version_after:
            lines.append("  Versao anterior -> nova versao registrada")
        return "\n".join(lines)
