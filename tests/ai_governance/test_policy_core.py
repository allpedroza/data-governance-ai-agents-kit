"""Automated tests for the ai-governance-core.yaml policy pack.

These tests load the YAML policy definitions and simulate governance
events to verify that block / deny / allow decisions are deterministic
and aligned with the documented rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

# ---------------------------------------------------------------------------
# Path to the policy file under test
# ---------------------------------------------------------------------------
POLICY_FILE = (
    Path(__file__).resolve().parents[2]
    / "ai_governance"
    / "policy_engine"
    / "policy_packs"
    / "ai-governance-core.yaml"
)


# ---------------------------------------------------------------------------
# Lightweight policy evaluation engine (test-only)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationContext:
    """Simulated event context sent to the policy engine."""
    event: str = ""
    env: str = ""
    target: Dict[str, Any] = field(default_factory=dict)
    provider: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    # Evidence / metrics
    risk: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)


def _resolve(path: str, ctx: EvaluationContext) -> Any:
    """Resolve a dotted path like 'target.next_stage' against *ctx*."""
    # Flatten known top-level dicts into a single namespace
    ns: Dict[str, Any] = {
        "event": ctx.event,
        "env": ctx.env,
    }
    for group_name in ("target", "provider", "input", "risk", "validation", "compliance", "data"):
        group = getattr(ctx, group_name if group_name != "input" else "input_data", {})
        if isinstance(group, dict):
            for k, v in group.items():
                ns[f"{group_name}.{k}"] = v
                # Support deeper nesting: risk.assessment.status
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        ns[f"{group_name}.{k}.{k2}"] = v2

    return ns.get(path)


def _eval_simple_expr(expr: str, ctx: EvaluationContext) -> bool:
    """Evaluate a *very* simplified expression from the YAML policy.

    Supports patterns used in the policy file:
      - ``event == "..."``
      - ``target.next_stage == "..."``
      - ``event in [...]``
      - ``provider.type == "..."``
      - ``input.contains_pii == true``
      - ``risk.tier in ["low", "medium"]``
      - ``validation.AUC >= 0.85``
      - ``compliance.LGPD == "complete"``
      - ``data.contains_pii == false``
    """
    expr = expr.strip()

    # Handle && (AND)
    if "&&" in expr:
        parts = expr.split("&&")
        return all(_eval_simple_expr(p, ctx) for p in parts)

    # a in [x, y, z]
    m = re.match(r'(.+?)\s+in\s+\[(.+)]', expr)
    if m:
        lhs = _resolve(m.group(1).strip(), ctx)
        items_str = m.group(2)
        items = [s.strip().strip('"').strip("'") for s in items_str.split(",")]
        return str(lhs) in items

    # a == "b"  or  a == true/false  or  a == number
    m = re.match(r'(.+?)\s*==\s*(.+)', expr)
    if m:
        lhs = _resolve(m.group(1).strip(), ctx)
        rhs_raw = m.group(2).strip().strip('"').strip("'")
        if rhs_raw == "true":
            return bool(lhs) is True
        if rhs_raw == "false":
            return bool(lhs) is False
        return str(lhs) == rhs_raw

    # a >= number
    m = re.match(r'(.+?)\s*>=\s*(.+)', expr)
    if m:
        lhs = _resolve(m.group(1).strip(), ctx)
        rhs = float(m.group(2).strip())
        try:
            return float(lhs) >= rhs
        except (TypeError, ValueError):
            return False

    return False


def evaluate_policy(policy: Dict, ctx: EvaluationContext) -> str:
    """Return the decision: 'block', 'deny', or 'allow'."""
    when_expr = policy.get("when", "")
    if not _eval_simple_expr(when_expr, ctx):
        return "not_applicable"

    # Check 'require' clauses
    for req in policy.get("require", []):
        for key in ("evidence", "metric", "checklist", "flag"):
            clause = req.get(key)
            if clause and not _eval_simple_expr(clause, ctx):
                return policy.get("on_fail", "block")

    # Check 'deny_if' clauses
    for deny_clause in policy.get("deny_if", []):
        if _eval_simple_expr(deny_clause, ctx):
            return policy.get("on_fail", "deny")

    return "allow"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def policy_pack() -> Dict:
    """Load the ai-governance-core.yaml policy pack."""
    assert POLICY_FILE.exists(), f"Policy file not found: {POLICY_FILE}"
    with open(POLICY_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def policies(policy_pack: Dict) -> Dict[str, Dict]:
    """Index policies by id for convenient access."""
    return {p["id"]: p for p in policy_pack.get("policies", [])}


# ---------------------------------------------------------------------------
# Tests: gate-risk-approved
# ---------------------------------------------------------------------------

class TestGateRiskApproved:
    """Policy: gate-risk-approved — blocks prod transitions without risk approval."""

    def test_blocks_when_risk_not_approved(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            target={"next_stage": "prod"},
            risk={"assessment": {"status": "pending"}, "tier": "low"},
        )
        assert evaluate_policy(policies["gate-risk-approved"], ctx) == "block"

    def test_blocks_when_risk_tier_high(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            target={"next_stage": "prod"},
            risk={"assessment": {"status": "approved"}, "tier": "high"},
        )
        assert evaluate_policy(policies["gate-risk-approved"], ctx) == "block"

    def test_allows_when_approved_and_low_tier(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            target={"next_stage": "prod"},
            risk={"assessment": {"status": "approved"}, "tier": "low"},
        )
        assert evaluate_policy(policies["gate-risk-approved"], ctx) == "allow"

    def test_allows_when_approved_and_medium_tier(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            target={"next_stage": "prod"},
            risk={"assessment": {"status": "approved"}, "tier": "medium"},
        )
        assert evaluate_policy(policies["gate-risk-approved"], ctx) == "allow"

    def test_not_applicable_for_staging(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            target={"next_stage": "staging"},
        )
        assert evaluate_policy(policies["gate-risk-approved"], ctx) == "not_applicable"


# ---------------------------------------------------------------------------
# Tests: gate-validation-baseline
# ---------------------------------------------------------------------------

class TestGateValidationBaseline:
    """Policy: gate-validation-baseline — enforces model quality thresholds."""

    def test_blocks_low_auc(self, policies):
        ctx = EvaluationContext(
            event="cicd.deploy",
            env="prod",
            validation={"AUC": 0.80, "robustness_score": 0.75},
        )
        assert evaluate_policy(policies["gate-validation-baseline"], ctx) == "block"

    def test_blocks_low_robustness(self, policies):
        ctx = EvaluationContext(
            event="cicd.deploy",
            env="prod",
            validation={"AUC": 0.90, "robustness_score": 0.60},
        )
        assert evaluate_policy(policies["gate-validation-baseline"], ctx) == "block"

    def test_allows_above_thresholds(self, policies):
        ctx = EvaluationContext(
            event="cicd.deploy",
            env="prod",
            validation={"AUC": 0.92, "robustness_score": 0.80},
        )
        assert evaluate_policy(policies["gate-validation-baseline"], ctx) == "allow"

    def test_not_applicable_for_staging(self, policies):
        ctx = EvaluationContext(
            event="cicd.deploy",
            env="staging",
            validation={"AUC": 0.50, "robustness_score": 0.10},
        )
        assert evaluate_policy(policies["gate-validation-baseline"], ctx) == "not_applicable"


# ---------------------------------------------------------------------------
# Tests: gate-compliance-lgpd
# ---------------------------------------------------------------------------

class TestGateComplianceLGPD:
    """Policy: gate-compliance-lgpd — enforces LGPD checklist + PII flag."""

    def test_blocks_when_lgpd_incomplete(self, policies):
        ctx = EvaluationContext(
            event="train.start",
            compliance={"LGPD": "incomplete"},
            data={"contains_pii": False},
        )
        assert evaluate_policy(policies["gate-compliance-lgpd"], ctx) == "block"

    def test_blocks_when_pii_present(self, policies):
        ctx = EvaluationContext(
            event="train.start",
            compliance={"LGPD": "complete"},
            data={"contains_pii": True},
        )
        assert evaluate_policy(policies["gate-compliance-lgpd"], ctx) == "block"

    def test_allows_complete_and_no_pii(self, policies):
        ctx = EvaluationContext(
            event="registry.transition",
            compliance={"LGPD": "complete"},
            data={"contains_pii": False},
        )
        assert evaluate_policy(policies["gate-compliance-lgpd"], ctx) == "allow"


# ---------------------------------------------------------------------------
# Tests: runtime-no-external-llm-with-pii
# ---------------------------------------------------------------------------

class TestRuntimeNoExternalLLMWithPII:
    """Policy: runtime-no-external-llm-with-pii — denies PII in external LLM calls."""

    def test_denies_pii_in_external_llm(self, policies):
        ctx = EvaluationContext(
            event="inference.request",
            provider={"type": "external_llm"},
            input_data={"contains_pii": True},
        )
        assert evaluate_policy(policies["runtime-no-external-llm-with-pii"], ctx) == "deny"

    def test_allows_no_pii_in_external_llm(self, policies):
        ctx = EvaluationContext(
            event="inference.request",
            provider={"type": "external_llm"},
            input_data={"contains_pii": False},
        )
        assert evaluate_policy(policies["runtime-no-external-llm-with-pii"], ctx) == "allow"

    def test_not_applicable_for_internal_llm(self, policies):
        ctx = EvaluationContext(
            event="inference.request",
            provider={"type": "internal_llm"},
            input_data={"contains_pii": True},
        )
        assert evaluate_policy(policies["runtime-no-external-llm-with-pii"], ctx) == "not_applicable"


# ---------------------------------------------------------------------------
# Meta-test: all policy IDs are tested
# ---------------------------------------------------------------------------

class TestPolicyPackIntegrity:
    """Verify the YAML structure and completeness."""

    def test_pack_has_version(self, policy_pack):
        assert "version" in policy_pack

    def test_all_policies_have_id(self, policy_pack):
        for p in policy_pack.get("policies", []):
            assert "id" in p, f"Policy without id: {p}"

    def test_all_policies_have_when(self, policy_pack):
        for p in policy_pack.get("policies", []):
            assert "when" in p, f"Policy '{p.get('id')}' missing 'when' clause"

    def test_all_policies_have_on_fail(self, policy_pack):
        for p in policy_pack.get("policies", []):
            assert "on_fail" in p, f"Policy '{p.get('id')}' missing 'on_fail'"

    def test_known_policy_count(self, policy_pack):
        """Catch accidental additions/removals."""
        assert len(policy_pack.get("policies", [])) == 4
