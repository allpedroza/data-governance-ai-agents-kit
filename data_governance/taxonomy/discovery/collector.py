"""Warehouse inventory collector.

Runs queries via an injected ``WarehouseConnector`` (BigQuery / Snowflake /
Redshift / Synapse) to assemble a structured, LLM-friendly inventory of the
schemas, tables and columns that exist in the data warehouse — plus
optional column profiles (top-K values, null ratio, distinct ratio) that
become *the* signal the LLM uses to canonicalize naming.

Profiling is opt-in (``profile_samples=True``) because it reads actual row
samples; never enable it without an ``anonymizer`` in front of the LLM.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # avoid hard dependency at import time
    from data_governance.warehouse.connectors.base import WarehouseConnector

logger = logging.getLogger(__name__)

# Regex hints used by deterministic pre-profiling. Cheap, run locally and
# do not require sending samples to the LLM at all when a match is unambiguous.
_PATTERN_HINTS = [
    ("cpf",          re.compile(r"^\d{11}$")),
    ("cnpj",         re.compile(r"^\d{14}$")),
    ("iso_country",  re.compile(r"^[A-Z]{2,3}$")),
    ("email",        re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
    ("uuid",         re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")),
    ("vin",          re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")),
    ("iso_datetime", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")),
]


@dataclass
class ColumnProfile:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    comment: str = ""
    null_ratio: Optional[float] = None
    distinct_ratio: Optional[float] = None
    top_values: List[str] = field(default_factory=list)
    pattern_hints: List[str] = field(default_factory=list)

    def to_llm_dict(self) -> Dict[str, Any]:
        """Compact projection sent to the LLM. Never includes raw samples."""
        payload: Dict[str, Any] = {
            "name": self.name,
            "type": self.data_type,
            "nullable": self.nullable,
        }
        if self.comment:
            payload["comment"] = self.comment[:240]
        if self.is_primary_key:
            payload["pk"] = True
        if self.is_foreign_key:
            payload["fk"] = True
        if self.null_ratio is not None:
            payload["null_ratio"] = round(self.null_ratio, 3)
        if self.distinct_ratio is not None:
            payload["distinct_ratio"] = round(self.distinct_ratio, 3)
        if self.pattern_hints:
            payload["pattern_hints"] = self.pattern_hints
        return payload


@dataclass
class TableInventory:
    name: str
    schema: str
    database: str
    table_type: str = "TABLE"
    row_count: Optional[int] = None
    columns: List[ColumnProfile] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        parts = [p for p in (self.database, self.schema, self.name) if p]
        return ".".join(parts)


@dataclass
class SchemaInventory:
    """Top-level container with everything needed to synthesize the taxonomy."""
    warehouse_type: str
    database: Optional[str]
    schema: Optional[str]
    tables: List[TableInventory] = field(default_factory=list)
    token_clusters: Dict[str, List[str]] = field(default_factory=dict)

    def column_count(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    def to_llm_dict(self, max_tables: int = 200) -> Dict[str, Any]:
        """Compact dict shipped to the LLM. Truncates very large inventories."""
        return {
            "warehouse_type": self.warehouse_type,
            "database": self.database,
            "schema": self.schema,
            "tables": [
                {
                    "fqn": t.fqn,
                    "table_type": t.table_type,
                    "row_count": t.row_count,
                    "columns": [c.to_llm_dict() for c in t.columns],
                }
                for t in self.tables[:max_tables]
            ],
            "token_clusters": self.token_clusters,
            "stats": {
                "total_tables": len(self.tables),
                "total_columns": self.column_count(),
            },
        }


_TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _tokenize(name: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r"[_\-\s]+", name or ""):
        parts.extend(_TOKEN_PATTERN.findall(chunk))
    return [p.lower() for p in parts if p]


class WarehouseInventoryCollector:
    """Collect schema + (optional) column profile information from a warehouse.

    The class is intentionally I/O-light: every query is either against the
    metadata catalog (``INFORMATION_SCHEMA``) or a *small* sample
    (``read_sample(n_rows=...)``) — never a full table scan.
    """

    def __init__(
        self,
        connector: "WarehouseConnector",
        max_tables: int = 50,
        profile_samples: bool = False,
        sample_rows: int = 50,
        max_columns_profiled_per_table: int = 30,
    ) -> None:
        self.connector = connector
        self.max_tables = max_tables
        self.profile_samples = profile_samples
        self.sample_rows = max(5, min(sample_rows, 500))
        self.max_columns_profiled_per_table = max_columns_profiled_per_table

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def collect(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        table_filter: Optional[Sequence[str]] = None,
    ) -> SchemaInventory:
        """Run the inventory queries and assemble a :class:`SchemaInventory`."""
        self._ensure_connected()
        tables = self._list_tables(schema=schema, database=database)
        if table_filter:
            wanted = {t.lower() for t in table_filter}
            tables = [t for t in tables if t.name.lower() in wanted]
        tables = tables[: self.max_tables]

        inventory_tables: List[TableInventory] = []
        for tbl in tables:
            try:
                columns_meta = self.connector.get_table_schema(
                    tbl.name, schema=tbl.schema, database=tbl.database
                )
            except Exception as exc:  # noqa: BLE001 — connector exceptions vary
                logger.warning("Failed to read schema of %s: %s", tbl.full_name, exc)
                continue

            columns = [self._column_from_meta(meta) for meta in columns_meta]

            if self.profile_samples and columns:
                self._enrich_with_samples(tbl, columns)

            inventory_tables.append(TableInventory(
                name=tbl.name,
                schema=tbl.schema or "",
                database=tbl.database or "",
                table_type=tbl.table_type or "TABLE",
                row_count=tbl.row_count,
                columns=columns,
            ))

        clusters = self._build_token_clusters(inventory_tables)
        warehouse_type = getattr(self.connector, "warehouse_type", "unknown")
        return SchemaInventory(
            warehouse_type=warehouse_type,
            database=database,
            schema=schema,
            tables=inventory_tables,
            token_clusters=clusters,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> None:
        if not getattr(self.connector, "_connected", False):
            try:
                self.connector.connect()
            except Exception as exc:  # noqa: BLE001 — surface a single, clear error
                raise RuntimeError(
                    f"Could not connect to warehouse: {exc}"
                ) from exc

    def _list_tables(self, schema, database):
        return self.connector.list_tables(
            schema=schema, database=database, include_views=True,
        )

    @staticmethod
    def _column_from_meta(meta: Dict[str, Any]) -> ColumnProfile:
        return ColumnProfile(
            name=str(meta.get("name", "")),
            data_type=str(meta.get("type", "") or meta.get("data_type", "")),
            nullable=bool(meta.get("nullable", True)),
            is_primary_key=bool(meta.get("primary_key", False)),
            is_foreign_key=bool(meta.get("foreign_key", False)),
            comment=str(meta.get("comment") or meta.get("description") or ""),
        )

    def _enrich_with_samples(self, tbl, columns: List[ColumnProfile]) -> None:
        """Profile up to ``max_columns_profiled_per_table`` columns per table."""
        try:
            sample = self.connector.read_sample(
                tbl.name,
                schema=tbl.schema,
                database=tbl.database,
                n_rows=self.sample_rows,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sampling failed for %s: %s", tbl.full_name, exc)
            return

        rows = sample.rows or []
        if not rows:
            return
        n = len(rows)
        for col in columns[: self.max_columns_profiled_per_table]:
            values = [r.get(col.name) for r in rows]
            non_null = [v for v in values if v is not None]
            col.null_ratio = round((n - len(non_null)) / n, 3) if n else None
            col.distinct_ratio = round(len(set(map(str, non_null))) / n, 3) if n else None
            # top-K values (raw strings, will be anonymized downstream)
            counter = Counter(str(v) for v in non_null)
            col.top_values = [v for v, _ in counter.most_common(5)]
            # local regex hints — no LLM, no sample exfiltration risk
            col.pattern_hints = self._infer_pattern_hints(non_null)

    @staticmethod
    def _infer_pattern_hints(values: Iterable[Any]) -> List[str]:
        hints: List[str] = []
        sample = [str(v) for v in values][:30]
        if not sample:
            return hints
        for label, pattern in _PATTERN_HINTS:
            matches = sum(1 for v in sample if pattern.match(v))
            if matches / max(len(sample), 1) >= 0.8:
                hints.append(label)
        return hints

    # Stop-tokens that, alone, never indicate a shared concept across tables.
    _STOP_TOKENS = {"id", "key", "code", "type", "name", "date", "at", "is", "the"}

    @classmethod
    def _build_token_clusters(cls, tables: List[TableInventory]) -> Dict[str, List[str]]:
        """Group ``table.column`` names by shared significant tokens.

        A column appears in every cluster whose token it contains (skipping
        stop-tokens like ``id``/``name`` which are too generic to be
        canonicalizing signals). A cluster is kept only when it pulls
        columns from **at least two different tables** — that is the
        signal of an alias candidate.

        Output example:
            {"national": ["CUSTOMER.NationalID", "AGENDA_CUSTOMERS.CustomerNationalID"], ...}
        """
        per_token: Dict[str, List[tuple]] = defaultdict(list)
        for table in tables:
            for col in table.columns:
                tokens = _tokenize(col.name)
                significant = [t for t in tokens if t not in cls._STOP_TOKENS]
                if not significant:
                    significant = tokens  # fall back so very short names still cluster
                for tok in set(significant):
                    per_token[tok].append((table.name, col.name))

        clusters: Dict[str, List[str]] = {}
        for tok, items in per_token.items():
            distinct_tables = {tbl for tbl, _ in items}
            if len(distinct_tables) >= 2:
                clusters[tok] = [f"{tbl}.{col}" for tbl, col in items]
        return clusters
