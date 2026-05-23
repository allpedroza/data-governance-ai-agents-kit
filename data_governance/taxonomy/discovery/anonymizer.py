"""Anonymizer for column samples before they reach the LLM.

Wraps the existing ``SensitiveDataNERAgent`` (from ``ai_governance``) so that
``top_values`` / ``comment`` strings flowing into the LLM are scrubbed of
PII (CPF, CNPJ, emails, phones, names, addresses, etc.) and of any
content matching configured sensitive-term patterns.

When the NER agent is not available (optional dependency) the anonymizer
falls back to a conservative built-in regex set so PII never reaches the
LLM even in degraded mode.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .collector import ColumnProfile, SchemaInventory

logger = logging.getLogger(__name__)


_FALLBACK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("CPF",        re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNPJ",       re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("EMAIL",      re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("PHONE_BR",   re.compile(r"\+?\d{2,3}[\s-]?\(?\d{2}\)?[\s-]?\d{4,5}-?\d{4}")),
    ("CREDIT_CARD",re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IPV4",       re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("UUID",       re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                              r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
]


@dataclass
class _AnonReport:
    redactions: int = 0
    entities: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.entities is None:
            self.entities = {}

    def bump(self, label: str, count: int = 1) -> None:
        self.redactions += count
        self.entities[label] = self.entities.get(label, 0) + count


class SampleAnonymizer:
    """Anonymize free-text strings (top values, comments) before LLM ingestion.

    Parameters
    ----------
    ner_agent:
        Optional ``SensitiveDataNERAgent`` instance. When provided, used as
        the primary detector. When omitted, the conservative built-in regex
        set is used.
    placeholder_template:
        ``str.format`` template used for redactions. Defaults to
        ``"<{label}_REDACTED>"`` which keeps the LLM aware that something
        sensitive was removed without leaking the value.
    """

    def __init__(
        self,
        ner_agent: Optional[Any] = None,
        placeholder_template: str = "<{label}_REDACTED>",
    ) -> None:
        self.ner_agent = ner_agent
        self.placeholder_template = placeholder_template
        self._last_report = _AnonReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def anonymize_text(self, text: str) -> str:
        """Anonymize a single string."""
        if not text:
            return text
        self._last_report = _AnonReport()
        if self.ner_agent is not None:
            try:
                return self._ner_anonymize(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER anonymizer failed (%s); falling back to regex.", exc)
        return self._fallback_anonymize(text)

    def anonymize_inventory(self, inventory: SchemaInventory) -> SchemaInventory:
        """Mutate ``inventory`` in place, redacting ``top_values`` and ``comment``.

        Columns whose ``pattern_hints`` already flagged them as PII (cpf, email,
        cnpj) have their ``top_values`` dropped entirely — the hint is enough
        signal for the LLM.
        """
        for table in inventory.tables:
            for col in table.columns:
                self._anonymize_column(col)
        return inventory

    @property
    def last_report(self) -> Dict[str, Any]:
        return {
            "redactions": self._last_report.redactions,
            "entities": dict(self._last_report.entities),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _anonymize_column(self, col: ColumnProfile) -> None:
        pii_hints = {"cpf", "cnpj", "email", "uuid"}
        if any(h in pii_hints for h in col.pattern_hints):
            # Drop raw values — the hint is enough for the LLM to canonicalize.
            col.top_values = []
        else:
            col.top_values = [self.anonymize_text(v) for v in col.top_values]
        if col.comment:
            col.comment = self.anonymize_text(col.comment)

    def _ner_anonymize(self, text: str) -> str:
        entities = self.ner_agent.detect_entities(text)
        if not entities:
            return text
        # Process from end to start to keep indices stable
        sorted_ents = sorted(entities, key=lambda e: e.start, reverse=True)
        out = text
        for ent in sorted_ents:
            label = getattr(ent, "entity_type", "PII")
            placeholder = self.placeholder_template.format(label=label.upper())
            out = out[: ent.start] + placeholder + out[ent.end:]
            self._last_report.bump(label)
        return out

    def _fallback_anonymize(self, text: str) -> str:
        out = text
        for label, pattern in _FALLBACK_PATTERNS:
            def _sub(match: re.Match) -> str:
                self._last_report.bump(label)
                return self.placeholder_template.format(label=label)
            out = pattern.sub(_sub, out)
        return out
