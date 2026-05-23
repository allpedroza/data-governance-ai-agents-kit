"""Smoke tests for the taxonomy product end-to-end."""
from __future__ import annotations

import os
import sys
import types

# Stub the data_governance package to avoid triggering its heavy __init__
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance")
    pkg.__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]
    sys.modules["data_governance"] = pkg

from data_governance.taxonomy.agent import TaxonomyAgent  # noqa: E402
from data_governance.taxonomy.models import (  # noqa: E402
    TaxonomyDocument,
    _tokenize_identifier,
)


SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "taxonomy_schema.yaml"
)


def test_load_and_score_default_schema():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    score = agent.score_taxonomy(tax)
    assert 0 <= score.overall_score <= 100
    assert score.maturity_level in {1, 2, 3, 4, 5}
    assert set(score.dimension_scores) == set(score.benchmark_comparison)


def test_html_artifact_renders():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    score = agent.score_taxonomy(tax)
    html = agent.generate_html_artifact(tax, score)
    assert html.startswith("<!DOCTYPE html>")
    assert "Data Dictionary Taxonomy" in html


def test_improvement_plan_grouped_by_dimension():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    score = agent.score_taxonomy(tax)
    plan = agent.build_improvement_plan(score)
    assert plan.dimensions, "expected at least one dimension"
    dims = {d.dimension for d in plan.dimensions}
    assert dims == set(score.dimension_scores)
    for d in plan.dimensions:
        assert d.actions, f"dimension {d.dimension} has no actions"
    md = agent.generate_improvement_plan_markdown(plan)
    assert "# Plano de Melhoria de Taxonomia" in md
    assert "## Naming Consistency" in md


def test_as_is_markdown_report():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    score = agent.score_taxonomy(tax)
    md = agent.generate_current_state_report_markdown(tax, score)
    assert "Relatório AS-IS" in md
    assert "## Score por dimensão" in md


def test_validate_name_word_boundary_fix():
    """Regression test: 'Cust' must NOT flag 'Customer' (it's a substring)."""
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    assert tax.validate_name("CustomerName", "column") == []
    # But it must still flag the actual abbreviation
    assert tax.validate_name("CustName", "column") != []
    assert tax.validate_name("CustID", "column") != []


def test_validate_name_pascalcase_violation():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    violations = tax.validate_name("customer_id", "column")
    assert any("PascalCase" in v for v in violations)


def test_tokenizer():
    assert _tokenize_identifier("CustomerName") == ["Customer", "Name"]
    assert _tokenize_identifier("customer_id") == ["customer", "id"]
    assert _tokenize_identifier("CustomerGUID") == ["Customer", "GUID"]
    assert _tokenize_identifier("VINNumber") == ["VIN", "Number"]


def test_from_yaml_string_round_trip():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    yaml_str = tax.to_yaml()
    reloaded = TaxonomyDocument.from_yaml_string(yaml_str)
    assert len(reloaded.get_all_concepts()) == len(tax.get_all_concepts())


def test_from_yaml_ignores_unknown_keys():
    """The default schema has 'count' and 'pii_level' on groups, and 'description'
    on naming sub-sections that aren't fields of any dataclass — loading must
    still succeed without raising."""
    content = """
metadata: { title: "T", version: "1" }
naming_conventions:
  casing_rules: { columns: "PascalCase" }
  canonical_acronyms: { items: ["ID"], rule: "x", extra: "ignored" }
context_rules:
  - rule_type: single_entity
    title: t
    unknown_field: ignored
"""
    tax = TaxonomyDocument.from_yaml_string(content)
    assert tax.naming_conventions.canonical_acronyms["items"] == ["ID"]
    assert tax.context_rules[0].rule_type == "single_entity"


def test_integration_bundle_for_other_agents():
    agent = TaxonomyAgent()
    tax = agent.load_from_yaml(SCHEMA_PATH)
    bundle = agent.export_for_other_agents(tax)
    assert {"naming_conventions", "glossary", "concepts", "validation_rules"}.issubset(bundle)
    assert bundle["naming_conventions"]["acronyms"], "acronyms missing"
    assert bundle["concepts"], "no concepts exported"


def test_run_end_to_end():
    agent = TaxonomyAgent()
    out = agent.run(SCHEMA_PATH)
    for key in ("taxonomy", "score", "plan", "as_is_html", "as_is_markdown",
                "plan_markdown", "integration_bundle"):
        assert key in out, f"missing {key} in run() output"
