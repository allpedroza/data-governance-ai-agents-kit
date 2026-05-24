"""Copilot interface for the Data Engineering Agent.

The copilot takes a short intent like ``"create staging table for orders"``
or ``"new dimension customers"`` and returns the complete bundle of
artifacts an engineer needs to materialize the asset *in compliance with
the taxonomy*: DDL, dbt model + schema.yml, and a CLI plan.

This is deliberately rule-based (no LLM call) because the heuristics are
narrow — zone keywords map to architectural prefixes, the cloud is
already known from the profile, and the column list is provided by the
caller. Keeping it deterministic also means every emitted command is
reproducible and reviewable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..taxonomy.context import ArchitectureProfile
from ..taxonomy.models import TaxonomyDocument
from .generators.cli import CLICommandGenerator, CLIPlan
from .generators.dbt import DbtAssets, DbtGenerator
from .generators.ddl import DDLGenerationOptions, DDLGenerator


# Map intent keywords to architectural prefixes per pattern.
_INTENT_KEYWORDS = {
    "dimension":  ("dim",  {"kimball"}),
    "dim":        ("dim",  {"kimball"}),
    "fact":       ("fct",  {"kimball"}),
    "fct":        ("fct",  {"kimball"}),
    "bridge":     ("bridge", {"kimball"}),
    "snapshot":   ("snapshot", {"kimball"}),
    "raw":        ("raw",  {"medallion"}),
    "staging":    ("stg",  {"medallion", "kimball", "data_vault"}),
    "stage":      ("stg",  {"medallion", "kimball", "data_vault"}),
    "stg":        ("stg",  {"medallion", "kimball", "data_vault"}),
    "bronze":     ("raw",  {"medallion"}),
    "silver":     ("stg",  {"medallion"}),
    "gold":       ("curated", {"medallion"}),
    "curated":    ("curated", {"medallion"}),
    "hub":        ("hub",  {"data_vault"}),
    "link":       ("lnk",  {"data_vault"}),
    "satellite":  ("sat",  {"data_vault"}),
    "sat":        ("sat",  {"data_vault"}),
}


_DBT_MATERIALIZATIONS = {
    "raw": "incremental",
    "stg": "view",
    "dim": "table",
    "fct": "incremental",
    "curated": "table",
    "hub": "table",
    "lnk": "table",
    "sat": "incremental",
    "bridge": "table",
    "snapshot": "snapshot",
}


@dataclass
class CopilotResponse:
    """Bundle of everything the engineer needs to create the asset."""
    intent: str
    detected_prefix: Optional[str]
    target_cloud: str
    target_pattern: str
    asset_name: str
    concepts_used: List[str] = field(default_factory=list)
    concepts_missing: List[str] = field(default_factory=list)
    ddl: Optional[str] = None
    dbt: Optional[DbtAssets] = None
    cli_plan: Optional[CLIPlan] = None
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "detected_prefix": self.detected_prefix,
            "target_cloud": self.target_cloud,
            "target_pattern": self.target_pattern,
            "asset_name": self.asset_name,
            "concepts_used": self.concepts_used,
            "concepts_missing": self.concepts_missing,
            "ddl": self.ddl,
            "dbt": {
                "model_sql": self.dbt.model_sql,
                "schema_yml": self.dbt.schema_yml,
                "materialization": self.dbt.materialization,
            } if self.dbt else None,
            "cli_plan": {
                "title": self.cli_plan.title,
                "commands": [c.to_dict() for c in self.cli_plan.commands],
                "notes": self.cli_plan.notes,
            } if self.cli_plan else None,
            "warnings": self.warnings,
            "explanation": self.explanation,
        }


class DataEngineeringCopilot:
    """Translate a natural intent into an asset-creation plan."""

    def __init__(
        self,
        taxonomy: TaxonomyDocument,
        profile: Optional[ArchitectureProfile] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.profile = profile or ArchitectureProfile.from_metadata(taxonomy.metadata)
        self._ddl = DDLGenerator(taxonomy, profile=self.profile)
        self._dbt = DbtGenerator(taxonomy, profile=self.profile)
        self._cli = CLICommandGenerator(taxonomy, profile=self.profile)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def suggest(
        self,
        intent: str,
        concept_names: Sequence[str],
        asset_name: Optional[str] = None,
        source_table: Optional[str] = None,
        source_schema: Optional[str] = None,
        primary_keys: Sequence[str] = (),
        emit_dbt: bool = True,
        emit_cli: bool = True,
        project: Optional[str] = None,
        warehouse: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CopilotResponse:
        """Build the asset-creation plan for a natural-language intent.

        Parameters
        ----------
        intent:
            Free-text such as ``"create dim table customers"``. The first
            architectural keyword recognized (dimension / staging / hub …)
            decides the prefix.
        concept_names:
            Concept identifiers from the taxonomy that should populate the
            asset.
        asset_name:
            Optional explicit name. When omitted, it is extracted from the
            tail of ``intent`` (last alphanumeric word).
        """
        prefix, prefix_warning = self._detect_prefix(intent)
        if not asset_name:
            asset_name = self._extract_asset_name(intent) or "asset"

        concepts_used: List[str] = []
        concepts_missing: List[str] = []
        for n in concept_names:
            if self.taxonomy.get_concept_by_name(n):
                concepts_used.append(n)
            else:
                concepts_missing.append(n)

        warnings: List[str] = []
        if prefix_warning:
            warnings.append(prefix_warning)
        if concepts_missing:
            warnings.append(
                "Concepts not found in taxonomy (will be omitted): "
                + ", ".join(concepts_missing)
            )

        ddl_options = DDLGenerationOptions(
            table_name=asset_name,
            concept_names=concepts_used,
            architectural_prefix=prefix,
            primary_keys=primary_keys,
            schema=source_schema,
            comment=f"Generated by DataEngineeringCopilot from intent: {intent}",
        )
        ddl = self._ddl.generate(ddl_options) if concepts_used else None

        dbt_assets: Optional[DbtAssets] = None
        if emit_dbt and concepts_used:
            materialization = _DBT_MATERIALIZATIONS.get(prefix or "", "table")
            dbt_assets = self._dbt.generate(
                model_name=self._cased_asset(asset_name, prefix),
                concept_names=concepts_used,
                source_table=source_table or asset_name,
                source_schema=source_schema,
                materialization=materialization,
                unique_key=primary_keys[0] if primary_keys else None,
            )

        cli_plan: Optional[CLIPlan] = None
        if emit_cli and concepts_used:
            cli_plan = self._cli.create_table_plan(
                ddl_options, project=project, warehouse=warehouse, location=location,
            )

        explanation = self._explain(prefix, asset_name, concepts_used)
        return CopilotResponse(
            intent=intent,
            detected_prefix=prefix,
            target_cloud=self.profile.cloud.cloud,
            target_pattern=self.profile.pattern.name,
            asset_name=self._cased_asset(asset_name, prefix),
            concepts_used=concepts_used,
            concepts_missing=concepts_missing,
            ddl=ddl, dbt=dbt_assets, cli_plan=cli_plan,
            warnings=warnings,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _detect_prefix(self, intent: str) -> tuple:
        lowered = intent.lower()
        for keyword, (prefix, patterns) in _INTENT_KEYWORDS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                if self.profile.pattern.name in patterns:
                    return prefix, None
                # Allow but warn — caller may know better than the inferred pattern
                return prefix, (
                    f"Intent suggested '{prefix}_' prefix, which is canonical "
                    f"for patterns {sorted(patterns)} — the active pattern is "
                    f"'{self.profile.pattern.name}'."
                )
        return None, None

    @staticmethod
    def _extract_asset_name(intent: str) -> Optional[str]:
        # Take the last alphanumeric token from the intent (e.g. "create dim customer" → "customer")
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", intent or "")
        if not tokens:
            return None
        for tok in reversed(tokens):
            if tok.lower() not in _INTENT_KEYWORDS and tok.lower() not in {
                "create", "build", "make", "table", "for", "the", "an", "a",
                "new", "asset", "model", "from", "of",
            }:
                return tok
        return tokens[-1]

    def _cased_asset(self, asset_name: str, prefix: Optional[str]) -> str:
        from .generators.ddl import _cased
        full = f"{prefix}_{asset_name}" if prefix else asset_name
        return _cased(full, self.profile.cloud.default_casing)

    def _explain(self, prefix: Optional[str], asset_name: str, concepts: List[str]) -> str:
        if not concepts:
            return (
                "No taxonomy concepts matched the request — the copilot did "
                "not emit DDL / dbt / CLI. Add the concepts to the taxonomy "
                "first or pass canonical names."
            )
        parts = [
            f"Detected prefix '{prefix}_'." if prefix else
            "No architectural prefix detected; the asset will be created without one."
        ]
        parts.append(
            f"Target cloud is {self.profile.cloud.cloud} (casing: "
            f"{self.profile.cloud.default_casing}); pattern is "
            f"{self.profile.pattern.name}."
        )
        parts.append(
            f"Columns populated from concepts: {', '.join(concepts)}."
        )
        return " ".join(parts)
