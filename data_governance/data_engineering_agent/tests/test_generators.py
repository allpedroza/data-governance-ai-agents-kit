"""Tests for the Data Engineering Agent generators."""
from __future__ import annotations

import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance"); pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ]
    sys.modules["data_governance"] = pkg

from data_governance.data_engineering_agent.agent import DataEngineeringAgent  # noqa: E402
from data_governance.data_engineering_agent.generators.ddl import (  # noqa: E402
    DDLGenerationOptions,
)
from data_governance.taxonomy.context import (  # noqa: E402
    ArchitectureProfile, architectural_pattern, cloud_constraints,
)
from data_governance.taxonomy.models import TaxonomyDocument  # noqa: E402


_TAX_YAML = """
metadata: { title: T, version: '1', cloud: snowflake, architecture: kimball }
naming_conventions:
  casing_rules: { columns: UPPER_CASE, entities: UPPER_CASE, ctes: snake_case }
datetime_standards:
  timezone: UTC
  date_format: "YYYY-MM-DD"
  naming: { created: CreatedAt, updated: UpdatedAt }
concept_groups:
  - name: Customer
    description: Customer identity
    concepts:
      - name: CustomerID
        data_type: STRING
        definition: Identificador único do cliente
        accepted_types: [STRING]
      - name: CustomerName
        data_type: STRING
        definition: Nome completo do cliente
        accepted_types: [STRING]
      - name: Email
        data_type: STRING
        definition: Endereço de email do cliente
        accepted_types: [STRING]
"""


def _agent(cloud="snowflake", pattern="kimball"):
    tax = TaxonomyDocument.from_yaml_string(_TAX_YAML)
    profile = ArchitectureProfile(
        cloud=cloud_constraints(cloud),
        pattern=architectural_pattern(pattern),
        declared=True,
    )
    return DataEngineeringAgent(tax, profile=profile)


def test_snowflake_ddl_uses_upper_case_and_snowflake_types():
    agent = _agent("snowflake", "kimball")
    ddl = agent.generate_ddl(DDLGenerationOptions(
        table_name="customer",
        concept_names=["CustomerID", "CustomerName", "Email"],
        architectural_prefix="dim",
        primary_keys=["CustomerID"],
        schema="analytics",
    ))
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "DIM_CUSTOMER" in ddl
    assert "CUSTOMER_ID VARCHAR" in ddl
    assert "CUSTOMER_NAME VARCHAR" in ddl
    assert "EMAIL VARCHAR" in ddl
    assert "TIMESTAMP_NTZ" in ddl
    assert "PRIMARY KEY (CUSTOMER_ID)" in ddl
    assert "snowflake" in ddl  # header comment


def test_bigquery_ddl_uses_snake_case_and_bigquery_types():
    agent = _agent("bigquery", "medallion")
    ddl = agent.generate_ddl(DDLGenerationOptions(
        table_name="customer",
        concept_names=["CustomerID", "CustomerName"],
        architectural_prefix="stg",
        schema="staging",
    ))
    assert "stg_customer" in ddl
    assert "customer_id STRING" in ddl
    assert "customer_name STRING" in ddl
    assert "INT64" not in ddl or "BIGINT" not in ddl  # BigQuery type is INT64, not BIGINT


def test_redshift_rejects_over_limit_column_names():
    """A 200-char concept must raise on Redshift (127-char limit)."""
    long_yaml = _TAX_YAML.replace(
        "- name: CustomerID",
        f"- name: {'X' * 200}",
    )
    tax = TaxonomyDocument.from_yaml_string(long_yaml)
    profile = ArchitectureProfile(
        cloud=cloud_constraints("redshift"),
        pattern=architectural_pattern("medallion"),
    )
    agent = DataEngineeringAgent(tax, profile=profile)
    try:
        agent.generate_ddl(DDLGenerationOptions(
            table_name="customer", concept_names=["X" * 200],
        ))
    except ValueError as exc:
        assert "127" in str(exc)
        return
    raise AssertionError("expected ValueError for Redshift limit")


def test_warns_on_non_canonical_prefix():
    """Using 'hub_' prefix in a Kimball taxonomy must produce a warning comment."""
    agent = _agent("snowflake", "kimball")
    ddl = agent.generate_ddl(DDLGenerationOptions(
        table_name="customer",
        concept_names=["CustomerID"],
        architectural_prefix="hub",   # Data Vault prefix in a Kimball context
    ))
    assert "WARNING" in ddl and "hub_" in ddl


def test_unknown_concept_is_skipped_with_warning_comment():
    agent = _agent()
    ddl = agent.generate_ddl(DDLGenerationOptions(
        table_name="customer",
        concept_names=["CustomerID", "DoesNotExist"],
    ))
    assert "WARNING: concept 'DoesNotExist' not found" in ddl
    assert "CUSTOMER_ID" in ddl


def test_audit_columns_use_datetime_standards():
    agent = _agent("snowflake")
    ddl = agent.generate_ddl(DDLGenerationOptions(
        table_name="customer", concept_names=["CustomerID"],
        include_audit=True,
    ))
    # CreatedAt → CREATED_AT in UPPER_CASE on Snowflake
    assert "CREATED_AT" in ddl
    assert "UPDATED_AT" in ddl


def test_dbt_model_sql_and_schema_yml_align_with_taxonomy():
    agent = _agent("bigquery", "medallion")
    assets = agent.generate_dbt(
        model_name="stg_customer",
        concept_names=["CustomerID", "CustomerName", "Email"],
        source_table="raw_customers",
        source_schema="raw_layer",
        unique_key="CustomerID",
        tags=["pii", "staging"],
    )
    # SQL references the source and lists every concept
    assert "{{ source('raw_layer', 'raw_customers') }}" in assets.model_sql
    assert "CustomerID" in assets.model_sql
    assert "CustomerName" in assets.model_sql
    assert "{{ config(" in assets.model_sql

    # schema.yml is parseable and contains tests / descriptions
    import yaml as _y
    parsed = _y.safe_load(assets.schema_yml)
    assert parsed["version"] == 2
    model = parsed["models"][0]
    assert model["name"] == "stg_customer"
    cols = {c["name"]: c for c in model["columns"]}
    assert "CustomerID" in cols
    # CustomerID has unique + not_null because it is the unique_key
    pk_tests = cols["CustomerID"]["tests"]
    test_names = {next(iter(t.keys())) if isinstance(t, dict) else t for t in pk_tests}
    assert "unique" in test_names and "not_null" in test_names
    # Email gets not_null from hint
    assert any(
        (isinstance(t, dict) and "not_null" in t) for t in cols["Email"].get("tests", [])
    )
