"""Tests for the CLI generator and the copilot."""
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


_TAX = """
metadata: { title: T, version: '1', cloud: bigquery, architecture: medallion }
naming_conventions:
  casing_rules: { columns: snake_case }
datetime_standards:
  naming: { created: CreatedAt, updated: UpdatedAt }
lake_standards:
  zones:
    - { name: raw,     alias: bronze, naming_pattern: "raw_{x}", description: r, retention: "90 days" }
    - { name: staged,  alias: silver, naming_pattern: "stg_{x}", description: s, retention: "1 year" }
    - { name: curated, alias: gold,   naming_pattern: "{type}_{x}", description: c, retention: "indefinite" }
concept_groups:
  - name: Orders
    description: Order facts
    concepts:
      - { name: OrderID,    data_type: STRING,  definition: 'Identificador único do pedido', accepted_types: [STRING] }
      - { name: OrderDate,  data_type: TIMESTAMP, definition: 'Data do pedido em UTC', accepted_types: [TIMESTAMP] }
      - { name: CustomerID, data_type: STRING,  definition: 'FK para o cliente', accepted_types: [STRING] }
      - { name: Amount,     data_type: FLOAT,   definition: 'Valor total do pedido', accepted_types: [FLOAT] }
"""


def _agent(cloud="bigquery", pattern="medallion"):
    tax = TaxonomyDocument.from_yaml_string(_TAX)
    profile = ArchitectureProfile(
        cloud=cloud_constraints(cloud),
        pattern=architectural_pattern(pattern),
        declared=True,
    )
    return DataEngineeringAgent(tax, profile=profile)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------
def test_bigquery_cli_plan_emits_bq_query_command():
    agent = _agent("bigquery", "medallion")
    plan = agent.generate_cli_plan(
        DDLGenerationOptions(
            table_name="orders",
            concept_names=["OrderID", "OrderDate", "Amount"],
            architectural_prefix="stg",
            schema="staging",
        ),
        project="my-proj", location="US",
    )
    assert plan.commands
    cmd = plan.commands[0]
    assert cmd.tool == "bq"
    assert "bq" in cmd.command
    assert "--location" in cmd.command and "US" in cmd.command
    assert "--project_id" in cmd.command and "my-proj" in cmd.command
    assert "stg_orders" in cmd.command
    assert plan.ddl and "stg_orders" in plan.ddl


def test_snowflake_cli_plan_uses_snowsql_with_session_preamble():
    agent = _agent("snowflake", "kimball")
    plan = agent.generate_cli_plan(
        DDLGenerationOptions(
            table_name="customer",
            concept_names=["CustomerID"],
            architectural_prefix="dim",
            schema="analytics", database="prod",
        ),
        warehouse="compute_wh",
    )
    cmd = plan.commands[0]
    assert cmd.tool == "snowsql"
    assert "USE WAREHOUSE COMPUTE_WH" in cmd.command
    assert "USE DATABASE PROD" in cmd.command
    assert "USE SCHEMA ANALYTICS" in cmd.command
    assert "DIM_CUSTOMER" in cmd.command


def test_cli_plan_shell_script_round_trip():
    agent = _agent()
    plan = agent.generate_cli_plan(
        DDLGenerationOptions(
            table_name="orders", concept_names=["OrderID"], architectural_prefix="stg",
        ),
    )
    script = plan.to_shell_script()
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    assert plan.commands[0].command in script


def test_cli_zone_plan_for_bigquery_creates_dataset():
    agent = _agent("bigquery", "medallion")
    plan = agent.generate_zone_plan("raw", project="my-proj", region="us-east1")
    assert plan.commands
    cmd = plan.commands[0]
    assert cmd.tool == "bq"
    assert "mk" in cmd.command and "--dataset" in cmd.command
    assert "raw_layer" in cmd.command


def test_cli_zone_plan_rejects_unknown_zone():
    agent = _agent()
    try:
        agent.generate_zone_plan("plutonium")
    except ValueError as exc:
        assert "plutonium" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown zone")


def test_cli_commands_quote_unsafe_arguments():
    """A concept definition containing shell metacharacters must NOT inject
    arguments into the emitted command — quoting must neutralize it."""
    nasty_yaml = _TAX.replace(
        "'Identificador único do pedido'",
        "'$(rm -rf /tmp/x) injection ;'",
    )
    tax = TaxonomyDocument.from_yaml_string(nasty_yaml)
    profile = ArchitectureProfile(
        cloud=cloud_constraints("bigquery"),
        pattern=architectural_pattern("medallion"),
        declared=True,
    )
    agent = DataEngineeringAgent(tax, profile=profile)
    plan = agent.generate_cli_plan(DDLGenerationOptions(
        table_name="orders", concept_names=["OrderID"], architectural_prefix="stg",
    ))
    cmd = plan.commands[0].command
    # The dangerous substring must be inside a single-quoted segment so the
    # shell cannot expand it. Easy check: presence of $(rm…) must coincide
    # with surrounding single quotes (shlex.quote uses single quotes).
    assert "$(rm -rf" in cmd            # the comment text survives literally
    # But it must be inside a quoted region
    assert "'$(rm -rf /tmp/x) injection ;'" in cmd or \
           "\"'$(rm -rf /tmp/x) injection ;'\"" in cmd or \
           "$(rm -rf /tmp/x)" not in cmd.split("'")[0]


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------
def test_copilot_detects_staging_prefix_for_medallion():
    agent = _agent("bigquery", "medallion")
    resp = agent.suggest(
        intent="create staging table for orders",
        concept_names=["OrderID", "OrderDate", "Amount"],
        source_table="raw_orders", source_schema="raw_layer",
        primary_keys=["OrderID"],
    )
    assert resp.detected_prefix == "stg"
    assert resp.asset_name == "stg_orders"
    assert resp.target_cloud == "bigquery"
    assert resp.target_pattern == "medallion"
    assert "OrderID" in resp.concepts_used
    assert resp.ddl and "stg_orders" in resp.ddl
    assert resp.cli_plan and resp.cli_plan.commands
    assert resp.dbt and resp.dbt.materialization == "view"


def test_copilot_detects_dimension_prefix_for_kimball():
    agent = _agent("snowflake", "kimball")
    resp = agent.suggest(
        intent="create new dimension customer",
        concept_names=["CustomerID"],
        primary_keys=["CustomerID"],
    )
    assert resp.detected_prefix == "dim"
    assert resp.asset_name == "DIM_CUSTOMER"


def test_copilot_warns_when_prefix_mismatches_pattern():
    """Asking for a Data Vault 'hub' table on a Kimball-shaped warehouse
    must succeed but emit a warning."""
    agent = _agent("snowflake", "kimball")
    resp = agent.suggest(
        intent="create hub customer",
        concept_names=["CustomerID"],
    )
    assert resp.detected_prefix == "hub"
    assert any("canonical" in w.lower() for w in resp.warnings), resp.warnings


def test_copilot_lists_missing_concepts_and_skips_them():
    agent = _agent("bigquery", "medallion")
    resp = agent.suggest(
        intent="staging table for orders",
        concept_names=["OrderID", "DoesNotExist", "Amount"],
    )
    assert resp.concepts_used == ["OrderID", "Amount"]
    assert resp.concepts_missing == ["DoesNotExist"]
    assert any("DoesNotExist" in w for w in resp.warnings)


def test_copilot_response_serializable():
    agent = _agent("bigquery", "medallion")
    resp = agent.suggest(
        intent="staging orders",
        concept_names=["OrderID"],
    )
    import json
    payload = json.dumps(resp.to_dict(), default=str)
    assert "stg_orders" in payload
    assert "concepts_used" in payload


def test_copilot_emits_no_artifacts_when_no_concepts_match():
    agent = _agent()
    resp = agent.suggest(
        intent="dim customer",
        concept_names=["Phantom1", "Phantom2"],
    )
    assert resp.ddl is None
    assert resp.dbt is None
    assert resp.cli_plan is None
    assert "did not emit" in resp.explanation.lower()
