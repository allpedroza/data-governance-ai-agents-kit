"""dbt model + schema.yml generator from a TaxonomyDocument."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ...taxonomy.context import ArchitectureProfile
from ...taxonomy.models import TaxonomyConcept, TaxonomyDocument


@dataclass
class DbtAssets:
    model_sql: str
    schema_yml: str
    model_name: str
    materialization: str = "table"


_TESTS_BY_HINT: Dict[str, List[Dict[str, Any]]] = {
    # mapping logical patterns / domains to canonical dbt tests
    "id":      [{"unique": {}}, {"not_null": {}}],
    "email":   [{"not_null": {}}],
    "cpf":     [{"not_null": {}}],
    "cnpj":    [{"not_null": {}}],
}


class DbtGenerator:
    """Render dbt model SQL + schema.yml from a taxonomy."""

    def __init__(
        self,
        taxonomy: TaxonomyDocument,
        profile: Optional[ArchitectureProfile] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.profile = profile or ArchitectureProfile.from_metadata(taxonomy.metadata)

    def generate(
        self,
        model_name: str,
        concept_names: Sequence[str],
        source_table: str,
        source_schema: Optional[str] = None,
        materialization: str = "table",
        unique_key: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> DbtAssets:
        concepts: List[TaxonomyConcept] = []
        unknown: List[str] = []
        for name in concept_names:
            c = self.taxonomy.get_concept_by_name(name)
            if c:
                concepts.append(c)
            else:
                unknown.append(name)

        sql = self._render_sql(
            model_name=model_name,
            concepts=concepts,
            source_table=source_table,
            source_schema=source_schema,
            materialization=materialization,
            unique_key=unique_key,
            unknown=unknown,
        )
        schema = self._render_schema_yml(
            model_name=model_name,
            concepts=concepts,
            tags=tags,
            unique_key=unique_key,
        )
        return DbtAssets(
            model_sql=sql,
            schema_yml=schema,
            model_name=model_name,
            materialization=materialization,
        )

    # ------------------------------------------------------------------
    def _render_sql(
        self,
        model_name: str,
        concepts: List[TaxonomyConcept],
        source_table: str,
        source_schema: Optional[str],
        materialization: str,
        unique_key: Optional[str],
        unknown: List[str],
    ) -> str:
        config_args: Dict[str, Any] = {"materialized": materialization}
        if unique_key:
            config_args["unique_key"] = unique_key
        config_line = "{{ config(" + ", ".join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in config_args.items()
        ) + ") }}"

        cols_sql: List[str] = []
        for c in concepts:
            comment = c.definition.replace("\n", " ") if c.definition else ""
            inline_comment = f"  -- {comment}" if comment else ""
            cols_sql.append(f"    {c.name}{inline_comment}")

        # ref/source choice — dbt source() macro when schema is provided
        if source_schema:
            source_ref = f"{{{{ source('{source_schema}', '{source_table}') }}}}"
        else:
            source_ref = f"{{{{ ref('{source_table}') }}}}"

        warnings = "\n".join(f"-- WARNING: concept '{u}' not in taxonomy" for u in unknown)
        head = "\n".join(line for line in [
            f"-- Model: {model_name} (generated from taxonomy)",
            f"-- Source taxonomy: {self.taxonomy.metadata.get('title', 'taxonomy')}",
            warnings,
        ] if line)
        body = ",\n".join(cols_sql) if cols_sql else "    *"
        return (
            f"{head}\n{config_line}\n\n"
            f"select\n{body}\n"
            f"from {source_ref}\n"
        )

    def _render_schema_yml(
        self,
        model_name: str,
        concepts: List[TaxonomyConcept],
        tags: Optional[Sequence[str]],
        unique_key: Optional[str],
    ) -> str:
        cols: List[Dict[str, Any]] = []
        for c in concepts:
            entry: Dict[str, Any] = {
                "name": c.name,
                "description": c.definition or f"Concept canônico do grupo {c.group}.",
            }
            tests: List[Any] = []
            tokens = {t.lower() for t in c.name.replace("_", " ").split()}
            for hint, hint_tests in _TESTS_BY_HINT.items():
                if hint in tokens or hint in c.name.lower():
                    tests.extend(hint_tests)
            if c.name == unique_key:
                tests = [{"unique": {}}, {"not_null": {}}]
            if c.accepted_types:
                entry["meta"] = {"accepted_types": list(c.accepted_types)}
            if tests:
                entry["tests"] = tests
            cols.append(entry)

        model_entry: Dict[str, Any] = {
            "name": model_name,
            "description": f"Modelo gerado a partir da taxonomia '{self.taxonomy.metadata.get('title','taxonomy')}'.",
            "columns": cols,
        }
        if tags:
            model_entry["config"] = {"tags": list(tags)}

        document = {"version": 2, "models": [model_entry]}
        return yaml.dump(document, sort_keys=False, allow_unicode=True)
