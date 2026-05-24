"""Data model for the canonical taxonomy document.

This module is the single source of truth consumed by the scorer, the HTML
generator, the Streamlit UI and any sibling agent that needs to align with
the data dictionary (naming conventions, validation rules, glossary, etc.).

Loading from YAML is hardened against schema drift: unknown keys in the
incoming dictionary are ignored, missing keys fall back to safe defaults,
and YAML loading is restricted to ``yaml.safe_load`` to avoid arbitrary
object construction.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional

import yaml


def _select_known(dc_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``data`` down to the fields declared on a dataclass."""
    if not isinstance(data, dict):
        return {}
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in data.items() if k in allowed}


@dataclass
class TaxonomyConcept:
    """Canonical data-dictionary concept."""
    name: str
    data_type: str
    definition: str
    accepted_types: List[str] = field(default_factory=list)
    entity_qualified_forms: Dict[str, str] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    group: str = ""
    domain: str = ""


@dataclass
class NamingConvention:
    casing_rules: Dict[str, Any] = field(default_factory=dict)
    canonical_acronyms: Dict[str, Any] = field(default_factory=dict)
    forbidden_forms: Dict[str, Any] = field(default_factory=dict)
    full_words_required: Dict[str, Any] = field(default_factory=dict)
    application_names: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBasedRule:
    """Context-sensitive rule (single-entity vs multi-entity tables)."""
    rule_type: str
    title: str = ""
    subtitle: str = ""
    definition: str = ""
    applicable_to: List[str] = field(default_factory=list)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    anti_patterns: Dict[str, List[str]] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class AmbiguousAlias:
    name: str
    description: str = ""
    resolution_rules: Dict[str, str] = field(default_factory=dict)


@dataclass
class LakeZone:
    name: str
    alias: str = ""
    description: str = ""
    naming_pattern: str = ""
    retention: str = ""
    types: List[str] = field(default_factory=list)


@dataclass
class LakeStandards:
    zones: List[LakeZone] = field(default_factory=list)
    project_structure: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    id: str
    name: str
    severity: str
    scope: str
    rule_type: str
    message: str
    pattern: Optional[str] = None
    forbidden: Optional[List[str]] = None
    canonical_ref: Optional[str] = None
    expected_ref: Optional[str] = None


@dataclass
class TaxonomyDocument:
    """Complete taxonomy document — the single source of truth."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    naming_conventions: NamingConvention = field(default_factory=NamingConvention)
    concept_groups: List[Dict[str, Any]] = field(default_factory=list)
    context_rules: List[ContextBasedRule] = field(default_factory=list)
    ambiguous_aliases: List[AmbiguousAlias] = field(default_factory=list)
    datetime_standards: Dict[str, Any] = field(default_factory=dict)
    glossary: Dict[str, str] = field(default_factory=dict)
    lake_standards: LakeStandards = field(default_factory=LakeStandards)
    ai_agent_instructions: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "TaxonomyDocument":
        """Load a taxonomy document from a YAML file using ``safe_load``."""
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Taxonomy YAML not found: {resolved}")
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(
                "Taxonomy YAML must be a mapping at the top level (got "
                f"{type(data).__name__})."
            )
        return cls.from_dict(data)

    @classmethod
    def from_yaml_string(cls, content: str) -> "TaxonomyDocument":
        """Load a taxonomy document from a YAML string."""
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ValueError("Taxonomy YAML must be a mapping at the top level.")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaxonomyDocument":
        doc = cls(
            metadata=data.get("metadata", {}) or {},
            datetime_standards=data.get("datetime_standards", {}) or {},
            glossary=data.get("glossary", {}) or {},
            ai_agent_instructions=data.get("ai_agent_instructions", {}) or {},
            concept_groups=list(data.get("concept_groups", []) or []),
        )

        nc_data = data.get("naming_conventions", {}) or {}
        doc.naming_conventions = NamingConvention(**_select_known(NamingConvention, nc_data))

        for rule in data.get("context_rules", []) or []:
            if isinstance(rule, dict) and rule.get("rule_type"):
                doc.context_rules.append(ContextBasedRule(**_select_known(ContextBasedRule, rule)))

        for alias in data.get("ambiguous_aliases", []) or []:
            if isinstance(alias, dict) and alias.get("name"):
                doc.ambiguous_aliases.append(AmbiguousAlias(**_select_known(AmbiguousAlias, alias)))

        ls_data = data.get("lake_standards", {}) or {}
        zones: List[LakeZone] = []
        for z in ls_data.get("zones", []) or []:
            if isinstance(z, dict) and z.get("name"):
                zones.append(LakeZone(**_select_known(LakeZone, z)))
        doc.lake_standards = LakeStandards(
            zones=zones,
            project_structure=ls_data.get("project_structure", {}) or {},
        )

        for rule in data.get("validation_rules", []) or []:
            if isinstance(rule, dict) and rule.get("id"):
                doc.validation_rules.append(ValidationRule(**_select_known(ValidationRule, rule)))

        return doc

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "naming_conventions": {
                "casing_rules": self.naming_conventions.casing_rules,
                "canonical_acronyms": self.naming_conventions.canonical_acronyms,
                "forbidden_forms": self.naming_conventions.forbidden_forms,
                "full_words_required": self.naming_conventions.full_words_required,
                "application_names": self.naming_conventions.application_names,
            },
            "concept_groups": self.concept_groups,
            "context_rules": [r.__dict__ for r in self.context_rules],
            "ambiguous_aliases": [a.__dict__ for a in self.ambiguous_aliases],
            "datetime_standards": self.datetime_standards,
            "glossary": self.glossary,
            "lake_standards": {
                "zones": [z.__dict__ for z in self.lake_standards.zones],
                "project_structure": self.lake_standards.project_structure,
            },
            "ai_agent_instructions": self.ai_agent_instructions,
            "validation_rules": [r.__dict__ for r in self.validation_rules],
        }

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Queries / integration helpers
    # ------------------------------------------------------------------
    def get_ai_instructions(self) -> Dict[str, Any]:
        return self.ai_agent_instructions

    def get_validation_rules(self) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self.validation_rules]

    def get_all_concepts(self) -> List[TaxonomyConcept]:
        concepts: List[TaxonomyConcept] = []
        for group in self.concept_groups:
            if not isinstance(group, dict):
                continue
            group_name = group.get("name", "")
            for concept_data in group.get("concepts", []) or []:
                if not isinstance(concept_data, dict) or not concept_data.get("name"):
                    continue
                concepts.append(TaxonomyConcept(
                    name=concept_data.get("name", ""),
                    data_type=concept_data.get("data_type", ""),
                    definition=concept_data.get("definition", ""),
                    accepted_types=list(concept_data.get("accepted_types", []) or []),
                    entity_qualified_forms=dict(concept_data.get("entity_qualified_forms", {}) or {}),
                    aliases=list(concept_data.get("aliases", []) or []),
                    group=group_name,
                ))
        return concepts

    def get_total_aliases(self) -> int:
        return sum(len(c.aliases) for c in self.get_all_concepts())

    def get_concept_by_name(self, name: str) -> Optional[TaxonomyConcept]:
        target = name.lower()
        for concept in self.get_all_concepts():
            if concept.name.lower() == target or target in (a.lower() for a in concept.aliases):
                return concept
        return None

    def get_canonical_acronyms(self) -> List[str]:
        items = self.naming_conventions.canonical_acronyms.get("items", []) or []
        return [str(i).upper() for i in items if i]

    def get_forbidden_abbreviations(self) -> List[str]:
        """Standalone abbreviation tokens that should never appear as words."""
        abbrevs: List[str] = []
        for rule in self.validation_rules:
            if rule.rule_type == "forbidden_substring" and rule.forbidden:
                abbrevs.extend(rule.forbidden)
        # Fall back to the full_words_required policy
        for entry in self.naming_conventions.full_words_required.get("rules", []) or []:
            if isinstance(entry, dict) and entry.get("forbidden"):
                abbrevs.append(entry["forbidden"])
        # Deduplicate, preserve order
        seen = set()
        ordered: List[str] = []
        for a in abbrevs:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(a)
        return ordered

    def export_for_metadata_enrichment(self) -> Dict[str, Any]:
        """Compact bundle other agents can consume to stay aligned.

        Consumed by Metadata Enrichment, Data Quality and Data Contracts agents
        so naming, glossary and validation policies are not duplicated.
        """
        return {
            "naming_conventions": {
                "casing": self.naming_conventions.casing_rules,
                "acronyms": self.get_canonical_acronyms(),
                "forbidden_abbreviations": self.get_forbidden_abbreviations(),
                "application_names": self.naming_conventions.application_names,
            },
            "glossary": self.glossary,
            "datetime_standards": self.datetime_standards,
            "concepts": [
                {
                    "name": c.name,
                    "definition": c.definition,
                    "data_type": c.data_type,
                    "accepted_types": c.accepted_types,
                    "aliases": c.aliases,
                    "group": c.group,
                }
                for c in self.get_all_concepts()
            ],
            "validation_rules": self.get_validation_rules(),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_name(
        self,
        name: str,
        scope: str,
        profile: Optional[Any] = None,
    ) -> List[str]:
        """Validate ``name`` against the rules whose ``scope`` matches.

        ``forbidden_substring`` rules match on token boundaries (PascalCase /
        snake_case / camelCase) so that, for example, "Cust" only flags
        "CustID" or "CustomerCust" — never "Customer".

        When ``profile`` (an :class:`ArchitectureProfile`) is provided, tokens
        that match its canonical prefixes / abbreviations (``dim``, ``fct``,
        ``stg``, ``hub``, ``id``, ``utc``…) are skipped during the
        ``forbidden_substring`` check — they are part of the architectural
        vocabulary, not abbreviations.
        """
        if not name:
            return []
        violations: List[str] = []
        tokens = _tokenize_identifier(name)
        token_set = {t.lower() for t in tokens}
        # Drop tokens that the architectural profile considers canonical
        if profile is not None and hasattr(profile, "is_canonical_token"):
            token_set = {t for t in token_set if not profile.is_canonical_token(t)}
        acronyms = {a.upper() for a in self.get_canonical_acronyms()}

        for rule in self.validation_rules:
            if rule.scope != scope:
                continue
            try:
                if rule.rule_type == "regex" and rule.pattern:
                    if not re.match(rule.pattern, name):
                        violations.append(rule.message.format(name=name))
                elif rule.rule_type == "forbidden_substring" and rule.forbidden:
                    matched = next(
                        (f for f in rule.forbidden if f.lower() in token_set),
                        None,
                    )
                    if matched:
                        violations.append(rule.message.format(name=name))
                elif rule.rule_type == "acronym_case" and acronyms:
                    # Each token that matches an acronym (case-insensitive)
                    # must be in upper-case form.
                    for token in tokens:
                        if token.upper() in acronyms and token != token.upper():
                            violations.append(rule.message.format(name=name))
                            break
            except (KeyError, IndexError, ValueError):
                # Malformed message template — skip silently instead of crashing.
                continue
        return violations


_TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _tokenize_identifier(name: str) -> List[str]:
    """Split an identifier into word tokens.

    Handles PascalCase, camelCase, snake_case and acronym runs:

        CustomerName -> ["Customer", "Name"]
        customer_id  -> ["customer", "id"]
        CustomerGUID -> ["Customer", "GUID"]
        VINNumber    -> ["VIN", "Number"]
    """
    if not name:
        return []
    parts: List[str] = []
    for chunk in re.split(r"[_\-\s]+", name):
        parts.extend(_TOKEN_PATTERN.findall(chunk))
    return parts
