"""Tests for the ArchitectureProfile / context-aware scoring."""
from __future__ import annotations

import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance"); pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ]
    sys.modules["data_governance"] = pkg

from data_governance.taxonomy.context import (  # noqa: E402
    ArchitectureProfile, cloud_constraints, architectural_pattern,
)
from data_governance.taxonomy.discovery.collector import (  # noqa: E402
    ColumnProfile, SchemaInventory, TableInventory,
)
from data_governance.taxonomy.models import TaxonomyDocument  # noqa: E402
from data_governance.taxonomy.scorer import TaxonomyScorer  # noqa: E402


def _inv(warehouse_type: str, table_names):
    tables = [TableInventory(name=n, schema="public", database="d") for n in table_names]
    return SchemaInventory(warehouse_type=warehouse_type, database=None, schema=None, tables=tables)


def test_detect_kimball_pattern():
    inv = _inv("snowflake", ["dim_customer", "dim_product", "fct_sales", "fct_orders", "stg_raw"])
    p = ArchitectureProfile.detect(inv)
    assert p.pattern.name == "kimball"
    assert p.cloud.cloud == "snowflake"
    assert p.cloud.default_casing == "UPPER_CASE"


def test_detect_medallion_pattern():
    inv = _inv("bigquery", ["raw_orders", "raw_customers", "stg_orders", "gold_revenue"])
    p = ArchitectureProfile.detect(inv)
    assert p.pattern.name == "medallion"
    assert p.cloud.cloud == "bigquery"


def test_detect_data_vault_pattern():
    inv = _inv("redshift", ["hub_customer", "hub_product", "lnk_orders", "sat_customer_address"])
    p = ArchitectureProfile.detect(inv)
    assert p.pattern.name == "data_vault"
    assert p.cloud.column_name_max == 127


def test_detect_undeclared_with_no_prefixes():
    inv = _inv("bigquery", ["customers", "orders", "products"])
    p = ArchitectureProfile.detect(inv)
    assert p.pattern.name in ("inmon", "undeclared")


def test_declared_profile_wins_with_conflict_finding():
    inv = _inv("snowflake", ["dim_customer", "fct_orders"])
    declared = ArchitectureProfile.from_metadata({"cloud": "snowflake", "architecture": "data_vault"})
    p = ArchitectureProfile.detect(inv, declared=declared)
    assert p.pattern.name == "data_vault"      # declared wins
    assert any("conflict" in f.lower() for f in p.detection_findings)


def test_is_canonical_token_recognises_pattern_prefixes():
    p = ArchitectureProfile(
        cloud=cloud_constraints("snowflake"),
        pattern=architectural_pattern("kimball"),
    )
    assert p.is_canonical_token("dim")
    assert p.is_canonical_token("fct")
    assert p.is_canonical_token("id")          # universal
    assert not p.is_canonical_token("cust")    # not canonical anywhere


def test_validate_name_with_kimball_profile_does_not_flag_dim_prefix():
    """A table named 'dim_customer' must NOT be flagged for containing 'dim'.

    Build a minimal taxonomy with a forbidden_substring rule that includes
    'dim' (intentionally), then verify the profile suppresses the false
    positive.
    """
    yaml_doc = """
metadata: { title: t, version: '1', architecture: kimball, cloud: snowflake }
validation_rules:
  - id: VAL-X
    name: no abbreviations
    severity: warning
    scope: table
    rule_type: forbidden_substring
    forbidden: [dim, cust]
    message: "Tabela '{name}' contém abreviação"
"""
    tax = TaxonomyDocument.from_yaml_string(yaml_doc)
    profile = ArchitectureProfile.from_metadata(tax.metadata)
    # Without profile: dim_customer flagged twice (dim + cust)
    without = tax.validate_name("dim_customer", "table")
    assert without, "expected violations without profile"
    # With profile: 'dim' is canonical, only 'cust' remains a violation
    with_p = tax.validate_name("dim_customer", "table", profile=profile)
    # Only one violation message expected (for 'cust')
    assert len(with_p) <= len(without)


def test_scorer_rewards_casing_aligned_with_cloud_default():
    yaml_doc = """
metadata: { title: T, version: '1', cloud: snowflake }
naming_conventions:
  casing_rules: { columns: UPPER_CASE, examples: { column: [CUSTOMER_ID] } }
concept_groups:
  - name: g
    description: x
    concepts:
      - { name: CUSTOMER_ID, data_type: STRING, definition: 'identificador único do cliente', accepted_types: [STRING] }
"""
    tax = TaxonomyDocument.from_yaml_string(yaml_doc)
    s_aligned = TaxonomyScorer().score(tax).dimension_scores["naming_consistency"]
    # Same taxonomy but with declared PascalCase on Snowflake (mismatch)
    yaml2 = yaml_doc.replace("UPPER_CASE", "PascalCase")
    s_mis = TaxonomyScorer().score(TaxonomyDocument.from_yaml_string(yaml2)).dimension_scores["naming_consistency"]
    assert s_aligned > s_mis


def test_scorer_penalises_over_limit_identifiers_on_redshift():
    long_name = "X" * 200
    yaml_doc = f"""
metadata: {{ title: T, version: '1', cloud: redshift }}
naming_conventions: {{ casing_rules: {{ columns: snake_case }} }}
concept_groups:
  - name: g
    description: x
    concepts:
      - {{ name: {long_name}, data_type: STRING, definition: 'um conceito com nome muito longo para redshift', accepted_types: [STRING] }}
"""
    tax = TaxonomyDocument.from_yaml_string(yaml_doc)
    score = TaxonomyScorer().score(tax)
    msgs = " ".join(score.dimension_findings.get("naming_consistency", []))
    assert "exceed" in msgs and "redshift" in msgs


def test_medallion_pattern_requires_expected_zones():
    yaml_doc = """
metadata: { title: T, version: '1', cloud: bigquery, architecture: medallion }
lake_standards:
  zones:
    - { name: raw,    alias: bronze, naming_pattern: "raw_{x}", description: r }
    - { name: staged, alias: silver, naming_pattern: "stg_{x}", description: s }
    # missing curated/gold
concept_groups:
  - name: g
    description: x
    concepts:
      - { name: a, data_type: STRING, definition: 'definição válida com tamanho suficiente', accepted_types: [STRING] }
"""
    tax = TaxonomyDocument.from_yaml_string(yaml_doc)
    score = TaxonomyScorer().score(tax)
    msgs = " ".join(score.dimension_findings.get("lake_platform_alignment", []))
    assert "missing" in msgs.lower() or "Medallion" in msgs
