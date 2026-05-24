"""Architecture profile — cloud + architectural pattern context for scoring.

A taxonomy that scores 100/100 in a vacuum may be inadequate for the data
platform it lives on. ``ArchitectureProfile`` encodes the *context* the
scorer must respect: cloud-provider constraints (identifier length limits,
case sensitivity, default casing) and architectural pattern (Kimball,
Data Vault, Medallion, Inmon) so that prefixes like ``dim_``, ``fct_``,
``stg_``, ``hub_`` are recognized as **canonical**, not as forbidden
abbreviations.

The profile can be:

1. **Inferred** from a :class:`SchemaInventory` (``ArchitectureProfile.detect``).
2. **Declared** in ``TaxonomyDocument.metadata`` under the keys ``cloud``
   and ``architecture`` (``ArchitectureProfile.from_metadata``).
3. **Constructed manually** for tests / custom setups.

When both inference and declaration are available, the *declared* values
win and a finding is raised if they conflict with the observed inventory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery.collector import SchemaInventory

# ---------------------------------------------------------------------------
# Cloud constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CloudConstraints:
    cloud: str
    column_name_max: int
    table_name_max: int
    default_casing: str            # "snake_case" | "UPPER_CASE" | "PascalCase"
    case_sensitive: bool           # whether the cloud is case-sensitive by default
    quote_to_force_case: bool      # whether casing can be forced via quoting
    notes: str = ""


_CLOUD_REGISTRY: Dict[str, CloudConstraints] = {
    "bigquery": CloudConstraints(
        cloud="bigquery", column_name_max=300, table_name_max=1024,
        default_casing="snake_case", case_sensitive=False,
        quote_to_force_case=True,
        notes="BigQuery is case-insensitive for table/dataset names, case-sensitive for column names.",
    ),
    "snowflake": CloudConstraints(
        cloud="snowflake", column_name_max=255, table_name_max=255,
        default_casing="UPPER_CASE", case_sensitive=False,
        quote_to_force_case=True,
        notes="Snowflake folds unquoted identifiers to UPPER_CASE.",
    ),
    "redshift": CloudConstraints(
        cloud="redshift", column_name_max=127, table_name_max=127,
        default_casing="snake_case", case_sensitive=False,
        quote_to_force_case=False,
        notes="Tight 127-char limit; snake_case is dominant convention.",
    ),
    "databricks": CloudConstraints(
        cloud="databricks", column_name_max=256, table_name_max=256,
        default_casing="snake_case", case_sensitive=False,
        quote_to_force_case=True,
        notes="Unity Catalog is case-insensitive; snake_case dominant.",
    ),
    "synapse": CloudConstraints(
        cloud="synapse", column_name_max=128, table_name_max=128,
        default_casing="snake_case", case_sensitive=False,
        quote_to_force_case=True,
        notes="Synapse follows T-SQL conventions; 128-char identifier limit.",
    ),
    "generic": CloudConstraints(
        cloud="generic", column_name_max=128, table_name_max=128,
        default_casing="snake_case", case_sensitive=False,
        quote_to_force_case=True,
        notes="Unknown cloud — fall back to conservative limits.",
    ),
}


def cloud_constraints(cloud: str) -> CloudConstraints:
    return _CLOUD_REGISTRY.get((cloud or "generic").lower(), _CLOUD_REGISTRY["generic"])


# ---------------------------------------------------------------------------
# Architectural pattern
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchitecturalPattern:
    name: str                                  # "kimball" | "data_vault" | "medallion" | "inmon" | "undeclared"
    canonical_prefixes: tuple                  # ("dim_", "fct_", ...)
    expected_zones: tuple = ()                 # for medallion
    description: str = ""


_PATTERN_REGISTRY: Dict[str, ArchitecturalPattern] = {
    "kimball": ArchitecturalPattern(
        name="kimball",
        canonical_prefixes=("dim_", "fct_", "fact_", "bridge_", "snapshot_"),
        description="Dimensional modelling — facts and dimensions.",
    ),
    "data_vault": ArchitecturalPattern(
        name="data_vault",
        canonical_prefixes=("hub_", "lnk_", "link_", "sat_"),
        description="Data Vault 2.0 — hubs, links, satellites.",
    ),
    "medallion": ArchitecturalPattern(
        name="medallion",
        canonical_prefixes=("raw_", "stg_", "staged_", "bronze_", "silver_", "gold_", "curated_"),
        expected_zones=("raw", "staged", "curated"),
        description="Bronze / silver / gold zoning.",
    ),
    "inmon": ArchitecturalPattern(
        name="inmon",
        canonical_prefixes=(),
        description="3NF normalised model — no canonical prefixes.",
    ),
    "undeclared": ArchitecturalPattern(
        name="undeclared",
        canonical_prefixes=(),
        description="No discernible pattern.",
    ),
}


def architectural_pattern(name: str) -> ArchitecturalPattern:
    return _PATTERN_REGISTRY.get((name or "undeclared").lower(), _PATTERN_REGISTRY["undeclared"])


# Universal abbreviations that are *always* acceptable regardless of context.
# These are tokens whose use is so widespread that flagging them would be
# noise. Distinct from architectural prefixes (which are pattern-specific).
_UNIVERSAL_CANONICAL_ABBREVIATIONS = (
    "id", "pk", "fk", "uuid", "guid", "url", "uri", "ip", "ipv4", "ipv6",
    "iso", "utc", "api", "sql", "ddl", "dml", "cte", "etl", "elt",
    "vin", "cpf", "cnpj", "dni", "ein", "ssn",
)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass
class ArchitectureProfile:
    cloud: CloudConstraints
    pattern: ArchitecturalPattern
    canonical_abbreviations: tuple = _UNIVERSAL_CANONICAL_ABBREVIATIONS
    declared: bool = False                     # True when sourced from metadata
    detection_findings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "ArchitectureProfile":
        cloud = cloud_constraints(metadata.get("cloud") or metadata.get("platform") or "generic")
        pattern = architectural_pattern(metadata.get("architecture") or "undeclared")
        extra = tuple(str(a).lower() for a in (metadata.get("canonical_abbreviations") or []))
        return cls(
            cloud=cloud,
            pattern=pattern,
            canonical_abbreviations=_UNIVERSAL_CANONICAL_ABBREVIATIONS + extra,
            declared=True,
        )

    @classmethod
    def detect(
        cls,
        inventory: "SchemaInventory",
        declared: Optional["ArchitectureProfile"] = None,
        min_pattern_ratio: float = 0.35,
    ) -> "ArchitectureProfile":
        """Infer cloud + architectural pattern from an inventory.

        Cloud is taken from ``inventory.warehouse_type`` (always reliable).
        Pattern is inferred by counting which canonical prefix family
        dominates the observed table names. If no family clears
        ``min_pattern_ratio`` the pattern is ``undeclared`` and the scorer
        will treat that as a finding, not a penalty.

        When a ``declared`` profile is provided, declared values win — but
        any disagreement is captured in ``detection_findings`` so the
        scorer can surface it.
        """
        cloud = cloud_constraints(inventory.warehouse_type)
        votes: Counter = Counter()
        total = 0
        for table in inventory.tables:
            tname = (table.name or "").lower()
            if not tname:
                continue
            total += 1
            for pname, pat in _PATTERN_REGISTRY.items():
                if pname in ("undeclared", "inmon"):
                    continue
                if any(tname.startswith(p) for p in pat.canonical_prefixes):
                    votes[pname] += 1
                    break

        inferred_name = "undeclared"
        if total:
            top = votes.most_common(1)
            if top and top[0][1] / total >= min_pattern_ratio:
                inferred_name = top[0][0]
            elif not votes and total > 5:
                # Many tables without any architectural prefix — likely 3NF/Inmon
                inferred_name = "inmon"

        inferred = architectural_pattern(inferred_name)
        findings: List[str] = []

        if declared and declared.declared:
            # Use the declared profile but warn on conflict
            if declared.pattern.name != "undeclared" and declared.pattern.name != inferred_name and inferred_name != "undeclared":
                findings.append(
                    f"Declared architecture '{declared.pattern.name}' conflicts with "
                    f"observed prefixes (suggests '{inferred_name}')."
                )
            return ArchitectureProfile(
                cloud=cloud,                  # observed cloud always wins
                pattern=declared.pattern,
                canonical_abbreviations=declared.canonical_abbreviations,
                declared=True,
                detection_findings=findings,
            )

        if inferred_name == "undeclared" and total > 0:
            findings.append(
                "No dominant architectural pattern detected; declare "
                "`metadata.architecture` to anchor scoring."
            )
        return cls(
            cloud=cloud, pattern=inferred,
            declared=False, detection_findings=findings,
        )

    # ------------------------------------------------------------------
    # Helpers used by the scorer / validator
    # ------------------------------------------------------------------
    def is_canonical_token(self, token: str) -> bool:
        """True if a tokenised word is canonical in this context.

        Used by validators so ``dim`` in ``dim_customer`` (Kimball) is not
        flagged as an abbreviation, but ``cust`` in ``cust_id`` is.
        """
        if not token:
            return False
        tok = token.lower().rstrip("_")
        if tok in self.canonical_abbreviations:
            return True
        # strip trailing underscore from each registered prefix before compare
        for p in self.pattern.canonical_prefixes:
            if tok == p.rstrip("_"):
                return True
        return False

    def expected_column_casing(self) -> str:
        return self.cloud.default_casing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cloud": self.cloud.cloud,
            "cloud_column_max": self.cloud.column_name_max,
            "cloud_table_max": self.cloud.table_name_max,
            "cloud_default_casing": self.cloud.default_casing,
            "pattern": self.pattern.name,
            "canonical_prefixes": list(self.pattern.canonical_prefixes),
            "expected_zones": list(self.pattern.expected_zones),
            "declared": self.declared,
            "findings": self.detection_findings,
        }
