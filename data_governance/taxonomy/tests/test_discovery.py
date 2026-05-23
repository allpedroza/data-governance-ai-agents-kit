"""Tests for the discovery pipeline (mocked connector + mocked LLM)."""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance")
    pkg.__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]
    sys.modules["data_governance"] = pkg

from data_governance.taxonomy.discovery.anonymizer import SampleAnonymizer  # noqa: E402
from data_governance.taxonomy.discovery.collector import (  # noqa: E402
    ColumnProfile, SchemaInventory, TableInventory, WarehouseInventoryCollector,
)
from data_governance.taxonomy.discovery.evaluator import (  # noqa: E402
    TaxonomyEvaluationError, TaxonomyEvaluator,
)
from data_governance.taxonomy.discovery.synthesizer import (  # noqa: E402
    TaxonomySynthesisError, TaxonomySynthesizer,
)
from data_governance.taxonomy.discovery.pipeline import TaxonomyDiscoveryPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
@dataclass
class _Sample:
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int = 0
    execution_time_ms: float = 0.0

    def __post_init__(self) -> None:
        self.row_count = len(self.rows)


@dataclass
class _Table:
    name: str
    schema: str = "public"
    database: str = "demo"
    table_type: str = "TABLE"
    row_count: int = 100
    full_name: str = ""

    def __post_init__(self) -> None:
        self.full_name = f"{self.database}.{self.schema}.{self.name}"


class FakeConnector:
    """Mimics the WarehouseConnector contract enough for the collector."""
    warehouse_type = "fake_wh"

    def __init__(self, tables, schemas, samples) -> None:
        self._connected = False
        self._tables = tables
        self._schemas = schemas
        self._samples = samples

    def connect(self) -> None:
        self._connected = True

    def list_tables(self, schema=None, database=None, include_views=True):
        return self._tables

    def get_table_schema(self, table_name, schema=None, database=None):
        return self._schemas.get(table_name, [])

    def read_sample(self, table_name, schema=None, database=None, n_rows=50):
        return self._samples.get(table_name, _Sample(columns=[], rows=[]))


@dataclass
class _LLMResp:
    content: str
    tokens_used: Optional[int] = None


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: Optional[str] = None
        self.last_system: Optional[str] = None

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
        self.last_prompt = prompt
        self.last_system = system_prompt
        return _LLMResp(content=self.response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _build_warehouse():
    tables = [
        _Table(name="CUSTOMER"),
        _Table(name="AGENDA_CUSTOMERS"),
    ]
    schemas = {
        "CUSTOMER": [
            {"name": "NationalID", "type": "STRING", "nullable": False,
             "primary_key": True, "comment": "Documento nacional (CPF: 123.456.789-00)"},
            {"name": "Name", "type": "STRING", "nullable": True, "comment": ""},
            {"name": "Email", "type": "STRING", "nullable": True, "comment": ""},
        ],
        "AGENDA_CUSTOMERS": [
            {"name": "CustomerNationalID", "type": "STRING", "nullable": False, "comment": ""},
            {"name": "CustomerName", "type": "STRING", "nullable": True, "comment": ""},
            {"name": "CustomerEmail", "type": "STRING", "nullable": True, "comment": ""},
        ],
    }
    samples = {
        "CUSTOMER": _Sample(
            columns=["NationalID", "Name", "Email"],
            rows=[
                {"NationalID": "12345678901", "Name": "Alice Real",
                 "Email": "alice@example.com"},
                {"NationalID": "98765432100", "Name": "Bob Real",
                 "Email": "bob@corp.com"},
            ],
        ),
        "AGENDA_CUSTOMERS": _Sample(columns=[], rows=[]),
    }
    return FakeConnector(tables, schemas, samples)


_VALID_YAML_RESPONSE = """```yaml
metadata:
  title: Discovered Taxonomy
  version: "1.0"
  domain: Customer
  platform: fake_wh
  owner: Data Engineering
  steward: Analytics Engineering
naming_conventions:
  casing_rules:
    columns: PascalCase
    entities: UPPER_CASE
    ctes: snake_case
    examples:
      column: [CustomerName]
      entity: [CUSTOMER]
      cte: [stg_customer]
  canonical_acronyms: { items: [ID, CPF, GUID], rule: UPPER_CASE em compostos }
  forbidden_forms:
    items:
      - {pattern: customer_id, reason: snake_case em colunas}
concept_groups:
  - name: Customer Identity
    icon: 👤
    description: Identidade do cliente
    pii_level: high
    concepts:
      - name: NationalID
        data_type: STRING
        definition: Documento nacional do cliente (CPF/CNPJ).
        accepted_types: [STRING]
        entity_qualified_forms: { single_entity: NationalID, multi_entity: CustomerNationalID }
        aliases: [CustomerNationalID, CPF]
      - name: Email
        data_type: STRING
        definition: Endereço de email primário.
        accepted_types: [STRING]
        entity_qualified_forms: { single_entity: Email, multi_entity: CustomerEmail }
        aliases: [CustomerEmail]
context_rules:
  - rule_type: single_entity
    title: Single-Entity Tables
    subtitle: Omita prefixos redundantes
    definition: Tabelas com uma entidade omitem prefixo.
    examples: { CUSTOMER: [Name, NationalID] }
    anti_patterns: { CUSTOMER: [CustomerName] }
    rationale: Reduz redundância.
  - rule_type: multi_entity
    title: Multi-Entity Tables
    subtitle: Sempre prefixe
    definition: Tabelas com múltiplas entidades sempre prefixam.
    examples: { CONTRACT: [CustomerName, DealerName] }
    anti_patterns: { CONTRACT: [Name] }
    rationale: Desambigua.
datetime_standards: { timezone: UTC, date_format: "YYYY-MM-DD" }
glossary: { PII: "Personally Identifiable Information" }
lake_standards:
  zones:
    - { name: raw, alias: bronze, naming_pattern: "raw_{source}_{entity}", description: "raw" }
ai_agent_instructions:
  description: Use as regras canônicas.
  quick_rules: [Colunas em PascalCase, Tabelas em UPPER_CASE]
validation_rules:
  - {id: VAL-001, name: PascalCase, severity: error, scope: column,
     rule_type: regex, pattern: "^[A-Z][a-zA-Z0-9]*$",
     message: "Coluna '{name}' não segue PascalCase"}
```
"""

_VALID_PLAN_JSON = json.dumps({
    "executive_summary": "Taxonomia em nível Definido. Há lacunas em contexto e governança.",
    "strengths": ["Concepts canônicos definidos", "Casing rules claras"],
    "risks": ["Ausência de ambiguous_aliases", "Validation rules mínimas"],
    "quick_wins": ["Adicionar regra de validação UPPER_CASE para tabelas"],
    "dimensions": [
        {
            "dimension": "naming_consistency",
            "current_score": 80, "target_score": 95,
            "diagnosis": "Falta cobrir application_names.",
            "actions": [
                {"title": "Adicionar regra de validação UPPER_CASE para tabelas",
                 "detail": "Inserir VAL-003 com regex ^[A-Z][A-Z0-9_]*$",
                 "effort": "low", "expected_impact": "medium",
                 "owner_role": "data_engineer", "depends_on": []}
            ]
        },
        {
            "dimension": "governance_readiness",
            "current_score": 60, "target_score": 90,
            "diagnosis": "Sem versionamento explícito.",
            "actions": [
                {"title": "Versionar a taxonomia com SemVer",
                 "detail": "Adotar tag SemVer no metadata.version",
                 "effort": "low", "expected_impact": "high",
                 "owner_role": "steward", "depends_on": []}
            ]
        }
    ]
}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_collector_extracts_inventory_and_clusters():
    connector = _build_warehouse()
    collector = WarehouseInventoryCollector(connector, profile_samples=True, sample_rows=10)
    inv = collector.collect()
    assert inv.warehouse_type == "fake_wh"
    assert len(inv.tables) == 2
    assert inv.column_count() == 6
    # Token clustering must group customer+nationalid across the two tables
    clusters = inv.token_clusters
    assert any("national" in k for k in clusters), clusters
    # Profile should have inferred pattern hints
    customer = next(t for t in inv.tables if t.name == "CUSTOMER")
    nid = next(c for c in customer.columns if c.name == "NationalID")
    assert "cpf" in nid.pattern_hints
    email = next(c for c in customer.columns if c.name == "Email")
    assert "email" in email.pattern_hints


def test_anonymizer_redacts_pii_from_comments_and_drops_pii_values():
    inv = SchemaInventory(
        warehouse_type="x", database=None, schema=None,
        tables=[TableInventory(name="t", schema="s", database="d", columns=[
            ColumnProfile(
                name="email", data_type="STRING",
                comment="Contato em alice@example.com",
                top_values=["alice@example.com", "bob@example.com"],
                pattern_hints=["email"],
            ),
            ColumnProfile(
                name="city", data_type="STRING",
                comment="Cidade do cliente",
                top_values=["São Paulo", "alice@example.com"],
            ),
        ])],
    )
    SampleAnonymizer().anonymize_inventory(inv)
    cols = inv.tables[0].columns
    # email column: hint=email → top_values dropped, comment scrubbed
    assert cols[0].top_values == []
    assert "alice@example.com" not in cols[0].comment
    assert "REDACTED" in cols[0].comment
    # city: email inside top_values gets redacted, real city stays
    assert "alice@example.com" not in " ".join(cols[1].top_values)
    assert any("São Paulo" in v for v in cols[1].top_values)


def test_synthesizer_parses_yaml_response():
    syn = TaxonomySynthesizer(FakeLLM(_VALID_YAML_RESPONSE))
    inv = WarehouseInventoryCollector(_build_warehouse()).collect()
    result = syn.synthesize(inv)
    assert result.taxonomy.concept_groups
    assert len(result.taxonomy.get_all_concepts()) == 2


def test_synthesizer_rejects_invalid_yaml():
    syn = TaxonomySynthesizer(FakeLLM("```yaml\n:: not valid ::\n```"))
    inv = WarehouseInventoryCollector(_build_warehouse()).collect()
    try:
        syn.synthesize(inv)
    except TaxonomySynthesisError:
        return
    raise AssertionError("expected TaxonomySynthesisError")


def test_synthesizer_rejects_yaml_with_no_concept_groups():
    syn = TaxonomySynthesizer(FakeLLM("```yaml\nmetadata: { title: x }\n```"))
    inv = WarehouseInventoryCollector(_build_warehouse()).collect()
    try:
        syn.synthesize(inv)
    except TaxonomySynthesisError:
        return
    raise AssertionError("expected TaxonomySynthesisError")


def test_evaluator_renders_plan_html_safely():
    from data_governance.taxonomy.models import TaxonomyDocument
    from data_governance.taxonomy.scorer import TaxonomyScorer
    tax = TaxonomyDocument.from_yaml_string(
        _VALID_YAML_RESPONSE.replace("```yaml", "").replace("```", "")
    )
    score = TaxonomyScorer().score(tax)

    # Inject XSS payload via the LLM response
    poisoned = json.loads(_VALID_PLAN_JSON)
    poisoned["executive_summary"] = "<script>alert(1)</script>"
    poisoned["dimensions"][0]["actions"][0]["title"] = "<img src=x onerror=alert(1)>"
    evaluator = TaxonomyEvaluator(FakeLLM(json.dumps(poisoned)))
    result = evaluator.evaluate(tax, score)
    # The browser-executable forms must be absent — `<` and `>` come escaped
    assert "<script>alert(1)" not in result.plan_html
    assert "<img src=x" not in result.plan_html
    # Conversely, the escaped versions must be present (proves the payload
    # made it into the document but is inert text, not active markup)
    assert "&lt;script&gt;alert(1)" in result.plan_html
    assert "&lt;img src=x onerror=alert(1)&gt;" in result.plan_html


def test_evaluator_rejects_malformed_json():
    from data_governance.taxonomy.models import TaxonomyDocument
    from data_governance.taxonomy.scorer import TaxonomyScorer
    tax = TaxonomyDocument.from_yaml_string(
        _VALID_YAML_RESPONSE.replace("```yaml", "").replace("```", "")
    )
    score = TaxonomyScorer().score(tax)
    evaluator = TaxonomyEvaluator(FakeLLM("not json at all"))
    try:
        evaluator.evaluate(tax, score)
    except TaxonomyEvaluationError:
        return
    raise AssertionError("expected TaxonomyEvaluationError")


def test_pipeline_end_to_end():
    connector = _build_warehouse()
    pipeline = TaxonomyDiscoveryPipeline(
        llm=FakeLLM(_VALID_YAML_RESPONSE),
        evaluator_llm=FakeLLM(_VALID_PLAN_JSON),
    )
    result = pipeline.run(connector, profile_samples=True, sample_rows=5)
    # Inventory was collected
    assert result.inventory.column_count() == 6
    # Synthesis ran and produced concepts
    assert result.taxonomy.get_concept_by_name("NationalID")
    # Score was computed deterministically
    assert 0 <= result.score.overall_score <= 100
    # Improvement plan has html
    assert result.improvement_plan_html.startswith("<!DOCTYPE html>")
    # AS-IS HTML uses the canonical renderer
    assert "Data Dictionary Taxonomy" in result.as_is_html or "Discovered Taxonomy" in result.as_is_html


def test_pipeline_never_sends_raw_pii_to_llm():
    """Critical: even with profile_samples=True the LLM payload must NOT
    contain the raw CPF or email seen in the warehouse samples."""
    connector = _build_warehouse()
    llm = FakeLLM(_VALID_YAML_RESPONSE)
    pipeline = TaxonomyDiscoveryPipeline(llm=llm, evaluator_llm=FakeLLM(_VALID_PLAN_JSON))
    pipeline.run(connector, profile_samples=True, sample_rows=10)
    payload = llm.last_prompt or ""
    assert "12345678901" not in payload, "CPF leaked into LLM prompt"
    assert "alice@example.com" not in payload, "email leaked into LLM prompt"
    # But the structural signals must be there
    assert "cpf" in payload.lower() or "EMAIL_REDACTED" in payload or "email" in payload.lower()
