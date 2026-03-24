"""JSON file persistence for Data Steward entities.

Provides the StewardStore class which saves and loads steward data models
(issues, glossary terms, quality rules, approvals, assignments, and activity
log entries) as individual JSON files organized in subdirectories.
"""

import json
import logging
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from data_governance.data_steward.models import (
    ActivityLogEntry,
    ApprovalRequest,
    DataIssue,
    GlossaryTerm,
    ImpactReport,
    QualityRuleDraft,
    StewardAssignment,
)

logger = logging.getLogger(__name__)


def _dict_to_dataclass(cls: Type, data: Dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, handling Optional fields.

    Only passes keys that correspond to actual dataclass fields so that
    unexpected keys in the stored JSON do not cause TypeErrors.
    """
    valid_field_names = {f.name for f in dataclass_fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_field_names}
    return cls(**filtered)


class StewardStore:
    """JSON-file-backed persistence for Data Steward entities.

    Each entity type is stored in its own subdirectory as individual JSON
    files keyed by the entity's ID.  Activity log entries are appended to
    a single JSONL file for efficient sequential writes.
    """

    _SUBDIRS = (
        "issues",
        "glossary",
        "rules",
        "approvals",
        "assignments",
        "activity_log",
    )

    def __init__(self, persist_dir: str = "./steward_data") -> None:
        self._root = Path(persist_dir)
        for subdir in self._SUBDIRS:
            (self._root / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_json(self, directory: str, filename: str, data: Dict[str, Any]) -> None:
        """Write *data* as pretty-printed JSON to *directory/filename*."""
        path = self._root / directory / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_json(self, directory: str, filename: str) -> Optional[Dict[str, Any]]:
        """Read and return JSON from *directory/filename*, or None."""
        path = self._root / directory / filename
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _list_json_files(self, directory: str) -> List[Path]:
        """Return all *.json files in the given subdirectory."""
        return sorted((self._root / directory).glob("*.json"))

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def save_issue(self, issue: DataIssue) -> None:
        """Save a DataIssue to ``issues/{issue_id}.json``."""
        self._save_json("issues", f"{issue.issue_id}.json", issue.to_dict())

    def load_issue(self, issue_id: str) -> Optional[DataIssue]:
        """Load a DataIssue by its ID, or return ``None`` if not found."""
        data = self._load_json("issues", f"{issue_id}.json")
        if data is None:
            return None
        return _dict_to_dataclass(DataIssue, data)

    def list_issues(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[DataIssue]:
        """List all issues, optionally filtered by status, severity, or domain."""
        results: List[DataIssue] = []
        for path in self._list_json_files("issues"):
            with open(path, "r") as f:
                data = json.load(f)
            if status is not None and data.get("status") != status:
                continue
            if severity is not None and data.get("severity") != severity:
                continue
            if domain is not None and data.get("domain") != domain:
                continue
            results.append(_dict_to_dataclass(DataIssue, data))
        return results

    # ------------------------------------------------------------------
    # Glossary Terms
    # ------------------------------------------------------------------

    def save_term(self, term: GlossaryTerm) -> None:
        """Save a GlossaryTerm to ``glossary/{term_id}.json``."""
        self._save_json("glossary", f"{term.term_id}.json", term.to_dict())

    def load_term(self, term_id: str) -> Optional[GlossaryTerm]:
        """Load a GlossaryTerm by its ID, or return ``None`` if not found."""
        data = self._load_json("glossary", f"{term_id}.json")
        if data is None:
            return None
        return _dict_to_dataclass(GlossaryTerm, data)

    def list_terms(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[GlossaryTerm]:
        """List all glossary terms, optionally filtered by domain or status."""
        results: List[GlossaryTerm] = []
        for path in self._list_json_files("glossary"):
            with open(path, "r") as f:
                data = json.load(f)
            if domain is not None and data.get("domain") != domain:
                continue
            if status is not None and data.get("status") != status:
                continue
            results.append(_dict_to_dataclass(GlossaryTerm, data))
        return results

    # ------------------------------------------------------------------
    # Quality Rules
    # ------------------------------------------------------------------

    def save_rule(self, rule: QualityRuleDraft) -> None:
        """Save a QualityRuleDraft to ``rules/{rule_id}.json``."""
        self._save_json("rules", f"{rule.rule_id}.json", rule.to_dict())

    def load_rule(self, rule_id: str) -> Optional[QualityRuleDraft]:
        """Load a QualityRuleDraft by its ID, or return ``None`` if not found."""
        data = self._load_json("rules", f"{rule_id}.json")
        if data is None:
            return None
        return _dict_to_dataclass(QualityRuleDraft, data)

    def list_rules(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[QualityRuleDraft]:
        """List all quality rules, optionally filtered by domain or status."""
        results: List[QualityRuleDraft] = []
        for path in self._list_json_files("rules"):
            with open(path, "r") as f:
                data = json.load(f)
            if domain is not None and data.get("domain") != domain:
                continue
            if status is not None and data.get("status") != status:
                continue
            results.append(_dict_to_dataclass(QualityRuleDraft, data))
        return results

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------

    def save_approval(self, req: ApprovalRequest) -> None:
        """Save an ApprovalRequest to ``approvals/{request_id}.json``."""
        self._save_json("approvals", f"{req.request_id}.json", req.to_dict())

    def load_approval(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load an ApprovalRequest by its ID, or return ``None`` if not found."""
        data = self._load_json("approvals", f"{request_id}.json")
        if data is None:
            return None
        return _dict_to_dataclass(ApprovalRequest, data)

    def list_approvals(
        self,
        status: Optional[str] = None,
        approver: Optional[str] = None,
    ) -> List[ApprovalRequest]:
        """List all approval requests, optionally filtered by status or approver."""
        results: List[ApprovalRequest] = []
        for path in self._list_json_files("approvals"):
            with open(path, "r") as f:
                data = json.load(f)
            if status is not None and data.get("status") != status:
                continue
            if approver is not None and data.get("approver") != approver:
                continue
            results.append(_dict_to_dataclass(ApprovalRequest, data))
        return results

    # ------------------------------------------------------------------
    # Assignments
    # ------------------------------------------------------------------

    def save_assignment(self, assignment: StewardAssignment) -> None:
        """Save a StewardAssignment to ``assignments/{assignment_id}.json``."""
        self._save_json(
            "assignments",
            f"{assignment.assignment_id}.json",
            assignment.to_dict(),
        )

    def load_assignment(self, assignment_id: str) -> Optional[StewardAssignment]:
        """Load a StewardAssignment by its ID, or return ``None`` if not found."""
        data = self._load_json("assignments", f"{assignment_id}.json")
        if data is None:
            return None
        return _dict_to_dataclass(StewardAssignment, data)

    def list_assignments(
        self,
        domain: Optional[str] = None,
    ) -> List[StewardAssignment]:
        """List all steward assignments, optionally filtered by domain."""
        results: List[StewardAssignment] = []
        for path in self._list_json_files("assignments"):
            with open(path, "r") as f:
                data = json.load(f)
            if domain is not None and data.get("domain") != domain:
                continue
            results.append(_dict_to_dataclass(StewardAssignment, data))
        return results

    # ------------------------------------------------------------------
    # Activity Log
    # ------------------------------------------------------------------

    def append_activity(self, entry: ActivityLogEntry) -> None:
        """Append a single activity entry to the JSONL log file."""
        log_path = self._root / "activity_log" / "log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def load_activity_log(
        self,
        limit: int = 50,
        domain: Optional[str] = None,
    ) -> List[ActivityLogEntry]:
        """Read the last *limit* activity log entries, optionally filtered by domain.

        Reads the full JSONL file, applies the optional domain filter, and
        returns the most recent *limit* entries (newest last).
        """
        log_path = self._root / "activity_log" / "log.jsonl"
        if not log_path.exists():
            return []

        entries: List[ActivityLogEntry] = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if domain is not None and data.get("domain") != domain:
                    continue
                entries.append(_dict_to_dataclass(ActivityLogEntry, data))

        # Return the last `limit` entries (most recent).
        return entries[-limit:]
