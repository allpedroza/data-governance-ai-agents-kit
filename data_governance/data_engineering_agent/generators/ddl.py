"""DDL generator aligned to a TaxonomyDocument + ArchitectureProfile.

Given a list of concept names and a target zone, emits a ``CREATE TABLE``
statement in the dialect of the active cloud provider, with:

* table name cased per the cloud default (Snowflake → UPPER_CASE, BigQuery
  / Redshift / Databricks → snake_case), prefixed with the architectural
  pattern (e.g. ``dim_``, ``stg_``, ``hub_``) when ``architectural_prefix``
  matches the active pattern.
* column names cased per the cloud default.
* data types from each concept's ``accepted_types`` (first entry) or
  ``data_type``.
* ``NOT NULL`` flagged when the concept appears in a primary-key list.
* column / table comments populated from concept definitions.
* mandatory audit columns from ``datetime_standards`` (``CreatedAt``,
  ``UpdatedAt``) when ``include_audit=True``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from ...taxonomy.context import ArchitectureProfile, architectural_pattern, cloud_constraints
from ...taxonomy.models import TaxonomyConcept, TaxonomyDocument


# Mapping from logical taxonomy types to cloud-specific SQL types.
_TYPE_MAP = {
    "snowflake": {
        "STRING": "VARCHAR", "INTEGER": "NUMBER(38,0)", "INT64": "NUMBER(38,0)",
        "FLOAT": "FLOAT", "NUMERIC": "NUMBER(38,10)", "BOOLEAN": "BOOLEAN",
        "DATE": "DATE", "TIMESTAMP": "TIMESTAMP_NTZ", "TIME": "TIME",
        "JSON": "VARIANT", "BYTES": "BINARY",
    },
    "bigquery": {
        "STRING": "STRING", "INTEGER": "INT64", "INT64": "INT64",
        "FLOAT": "FLOAT64", "NUMERIC": "NUMERIC", "BOOLEAN": "BOOL",
        "DATE": "DATE", "TIMESTAMP": "TIMESTAMP", "TIME": "TIME",
        "JSON": "JSON", "BYTES": "BYTES",
    },
    "redshift": {
        "STRING": "VARCHAR(MAX)", "INTEGER": "BIGINT", "INT64": "BIGINT",
        "FLOAT": "DOUBLE PRECISION", "NUMERIC": "NUMERIC(38,10)", "BOOLEAN": "BOOLEAN",
        "DATE": "DATE", "TIMESTAMP": "TIMESTAMP", "TIME": "TIME",
        "JSON": "SUPER", "BYTES": "VARBYTE",
    },
    "databricks": {
        "STRING": "STRING", "INTEGER": "BIGINT", "INT64": "BIGINT",
        "FLOAT": "DOUBLE", "NUMERIC": "DECIMAL(38,10)", "BOOLEAN": "BOOLEAN",
        "DATE": "DATE", "TIMESTAMP": "TIMESTAMP", "TIME": "STRING",
        "JSON": "STRING", "BYTES": "BINARY",
    },
    "synapse": {
        "STRING": "NVARCHAR(MAX)", "INTEGER": "BIGINT", "INT64": "BIGINT",
        "FLOAT": "FLOAT", "NUMERIC": "DECIMAL(38,10)", "BOOLEAN": "BIT",
        "DATE": "DATE", "TIMESTAMP": "DATETIME2", "TIME": "TIME",
        "JSON": "NVARCHAR(MAX)", "BYTES": "VARBINARY(MAX)",
    },
}

_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DDLGenerationOptions:
    table_name: str
    concept_names: Sequence[str]
    schema: Optional[str] = None
    database: Optional[str] = None
    architectural_prefix: Optional[str] = None    # e.g. "dim_", "stg_", "hub_"
    primary_keys: Sequence[str] = field(default_factory=tuple)
    include_audit: bool = True
    if_not_exists: bool = True
    comment: Optional[str] = None


def _quote(ident: str, profile: ArchitectureProfile) -> str:
    if not _VALID_IDENT.match(ident):
        # Anything non-trivial gets quoted using the cloud's quote style.
        if profile.cloud.cloud == "bigquery":
            return f"`{ident}`"
        if profile.cloud.cloud in ("snowflake", "redshift", "databricks", "synapse"):
            return f'"{ident}"'
    return ident


def _cased(name: str, casing: str) -> str:
    if not name:
        return name
    if casing == "UPPER_CASE":
        # Convert CamelCase / snake_case → UPPER_SNAKE_CASE
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.upper().replace("__", "_")
    if casing == "snake_case":
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower().replace("__", "_")
    if casing == "PascalCase":
        parts = re.split(r"[_\s\-]+", name)
        return "".join(p[:1].upper() + p[1:] for p in parts if p)
    return name


def _audit_columns(taxonomy: TaxonomyDocument) -> List[tuple]:
    """Return (name, type) tuples for mandatory audit fields."""
    naming = (taxonomy.datetime_standards or {}).get("naming") or {}
    created = naming.get("created", "CreatedAt")
    updated = naming.get("updated", "UpdatedAt")
    return [(created, "TIMESTAMP"), (updated, "TIMESTAMP")]


class DDLGenerator:
    """Render a CREATE TABLE statement for a target cloud."""

    def __init__(
        self,
        taxonomy: TaxonomyDocument,
        profile: Optional[ArchitectureProfile] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.profile = profile or ArchitectureProfile.from_metadata(taxonomy.metadata)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, options: DDLGenerationOptions) -> str:
        casing = self.profile.cloud.default_casing
        cloud = self.profile.cloud.cloud
        type_map = _TYPE_MAP.get(cloud, _TYPE_MAP["bigquery"])

        # Resolve concepts (skip unknown but record them as warnings in comments)
        rows: List[str] = []
        unknown: List[str] = []
        for name in options.concept_names:
            concept = self.taxonomy.get_concept_by_name(name)
            if not concept:
                unknown.append(name)
                continue
            rows.append(self._render_column(concept, type_map, casing, options))

        if options.include_audit:
            for audit_name, logical_type in _audit_columns(self.taxonomy):
                rows.append(self._render_audit_column(audit_name, logical_type, type_map, casing))

        # Table name with architectural prefix when applicable
        prefix = options.architectural_prefix or ""
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        if prefix and prefix not in self.profile.pattern.canonical_prefixes:
            rows.append(
                f"  -- WARNING: prefix '{prefix}' is not canonical for "
                f"pattern '{self.profile.pattern.name}'"
            )

        table_name = _cased(f"{prefix}{options.table_name}".replace(" ", "_"), casing)
        if len(table_name) > self.profile.cloud.table_name_max:
            raise ValueError(
                f"Table name '{table_name}' ({len(table_name)} chars) exceeds "
                f"{cloud} limit of {self.profile.cloud.table_name_max} chars."
            )

        qualified = ".".join(
            part for part in (
                _cased(options.database, casing) if options.database else None,
                _cased(options.schema, casing) if options.schema else None,
                _quote(table_name, self.profile),
            ) if part
        )

        header = "CREATE TABLE"
        if options.if_not_exists:
            header += " IF NOT EXISTS"
        header += f" {qualified} ("

        # Primary key clause
        pk_cols = [
            _cased(self._resolve_column_name(p), casing)
            for p in options.primary_keys
        ]
        if pk_cols:
            rows.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

        body = ",\n".join(rows)
        footer = ")"
        if options.comment:
            if cloud == "bigquery":
                footer += f' OPTIONS(description="{options.comment}")'
            else:
                footer += f" COMMENT = '{options.comment.replace(chr(39), chr(39)*2)}'"
        statement = f"{header}\n{body}\n{footer};"

        prefix_lines: List[str] = [
            f"-- Generated for {cloud} (pattern: {self.profile.pattern.name})",
            f"-- Source taxonomy: {self.taxonomy.metadata.get('title', 'taxonomy')}",
        ]
        for u in unknown:
            prefix_lines.append(f"-- WARNING: concept '{u}' not found in taxonomy; skipped.")
        return "\n".join(prefix_lines + [statement])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_column_name(self, concept_name: str) -> str:
        concept = self.taxonomy.get_concept_by_name(concept_name)
        return concept.name if concept else concept_name

    def _logical_type(self, concept: TaxonomyConcept) -> str:
        if concept.accepted_types:
            return str(concept.accepted_types[0]).upper()
        return (concept.data_type or "STRING").upper()

    def _render_column(
        self,
        concept: TaxonomyConcept,
        type_map: dict,
        casing: str,
        options: DDLGenerationOptions,
    ) -> str:
        logical = self._logical_type(concept)
        sql_type = type_map.get(logical, type_map.get("STRING", "STRING"))
        name = _cased(concept.name, casing)
        if len(name) > self.profile.cloud.column_name_max:
            raise ValueError(
                f"Column name '{name}' ({len(name)} chars) exceeds "
                f"{self.profile.cloud.cloud} limit of "
                f"{self.profile.cloud.column_name_max} chars."
            )
        nullable = "" if concept.name in options.primary_keys else " NULL"
        if concept.name in options.primary_keys:
            nullable = " NOT NULL"
        comment_clause = self._format_comment(concept.definition)
        return f"  {_quote(name, self.profile)} {sql_type}{nullable}{comment_clause}"

    def _render_audit_column(
        self, name: str, logical_type: str, type_map: dict, casing: str,
    ) -> str:
        sql_type = type_map.get(logical_type, "TIMESTAMP")
        rendered = _cased(name, casing)
        default = ""
        if self.profile.cloud.cloud in ("snowflake", "bigquery", "redshift", "databricks", "synapse"):
            default = " DEFAULT CURRENT_TIMESTAMP"
        return f"  {_quote(rendered, self.profile)} {sql_type} NULL{default}"

    def _format_comment(self, text: str) -> str:
        if not text:
            return ""
        safe = text.replace("'", "''").strip()
        cloud = self.profile.cloud.cloud
        if cloud == "bigquery":
            # BigQuery does not support inline column comments in CREATE TABLE
            # for columns; use OPTIONS at column level
            return f' OPTIONS(description="{safe}")'
        return f" COMMENT '{safe}'"
