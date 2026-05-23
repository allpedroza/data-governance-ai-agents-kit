import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class TaxonomyConcept:
    """Conceito canônico do dicionário de dados."""
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
    """Regras de nomenclatura."""
    casing_rules: Dict[str, Any] = field(default_factory=dict)
    canonical_acronyms: Dict[str, Any] = field(default_factory=dict)
    forbidden_forms: Dict[str, Any] = field(default_factory=dict)
    full_words_required: Dict[str, Any] = field(default_factory=dict)
    application_names: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBasedRule:
    """Regras baseadas em contexto (single-entity vs multi-entity)."""
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
    """Aliases ambíguos com regras de resolução."""
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
    """Documento completo de taxonomia — single source of truth."""
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

    @classmethod
    def from_yaml(cls, path: str) -> "TaxonomyDocument":
        """Carrega taxonomia de um arquivo YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaxonomyDocument":
        """Cria instância a partir de dicionário."""
        doc = cls(
            metadata=data.get("metadata", {}),
            datetime_standards=data.get("datetime_standards", {}),
            glossary=data.get("glossary", {}),
            ai_agent_instructions=data.get("ai_agent_instructions", {}),
            concept_groups=data.get("concept_groups", [])
        )

        nc_data = data.get("naming_conventions", {})
        doc.naming_conventions = NamingConvention(
            casing_rules=nc_data.get("casing_rules", {}),
            canonical_acronyms=nc_data.get("canonical_acronyms", {}),
            forbidden_forms=nc_data.get("forbidden_forms", {}),
            full_words_required=nc_data.get("full_words_required", {}),
            application_names=nc_data.get("application_names", {})
        )

        for rule in data.get("context_rules", []):
            doc.context_rules.append(ContextBasedRule(**rule))

        for alias in data.get("ambiguous_aliases", []):
            doc.ambiguous_aliases.append(AmbiguousAlias(**alias))

        ls_data = data.get("lake_standards", {})
        zones = [LakeZone(**z) for z in ls_data.get("zones", [])]
        doc.lake_standards = LakeStandards(
            zones=zones,
            project_structure=ls_data.get("project_structure", {})
        )

        for rule in data.get("validation_rules", []):
            doc.validation_rules.append(ValidationRule(**rule))

        return doc

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
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
                "project_structure": self.lake_standards.project_structure
            },
            "ai_agent_instructions": self.ai_agent_instructions,
            "validation_rules": [r.__dict__ for r in self.validation_rules]
        }

    def to_yaml(self) -> str:
        """Serializa para YAML."""
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def get_ai_instructions(self) -> Dict[str, Any]:
        return self.ai_agent_instructions

    def get_validation_rules(self) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self.validation_rules]

    def get_all_concepts(self) -> List[TaxonomyConcept]:
        """Retorna lista plana de todos os conceitos."""
        concepts = []
        for group in self.concept_groups:
            group_name = group.get("name", "")
            for concept_data in group.get("concepts", []):
                concept = TaxonomyConcept(
                    name=concept_data.get("name", ""),
                    data_type=concept_data.get("data_type", ""),
                    definition=concept_data.get("definition", ""),
                    accepted_types=concept_data.get("accepted_types", []),
                    entity_qualified_forms=concept_data.get("entity_qualified_forms", {}),
                    aliases=concept_data.get("aliases", []),
                    group=group_name
                )
                concepts.append(concept)
        return concepts

    def get_total_aliases(self) -> int:
        return sum(len(c.aliases) for c in self.get_all_concepts())

    def get_concept_by_name(self, name: str) -> Optional[TaxonomyConcept]:
        for concept in self.get_all_concepts():
            if concept.name.lower() == name.lower():
                return concept
        return None

    def validate_name(self, name: str, scope: str) -> List[str]:
        """Valida um nome contra as regras da taxonomia."""
        violations = []
        import re
        for rule in self.validation_rules:
            if rule.scope != scope:
                continue
            if rule.rule_type == "regex" and rule.pattern:
                if not re.match(rule.pattern, name):
                    violations.append(rule.message.format(name=name))
            elif rule.rule_type == "forbidden_substring" and rule.forbidden:
                for f in rule.forbidden:
                    if f in name:
                        violations.append(rule.message.format(name=name))
        return violations
