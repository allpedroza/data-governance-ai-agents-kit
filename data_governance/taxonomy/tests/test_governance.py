"""Tests for the LLM governance gate."""
from __future__ import annotations

import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance"); pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ]
    sys.modules["data_governance"] = pkg

from dataclasses import dataclass
from typing import Optional

from data_governance.taxonomy.discovery.governance import (  # noqa: E402
    GovernanceConfig, LLMGovernanceGate, LocalGovernanceBackend,
    json_validator, yaml_fence_validator,
)


@dataclass
class _Resp:
    content: str
    tokens_used: Optional[int] = None


class _ScriptedLLM:
    """LLM whose responses are pre-scripted in order; raises after exhaustion."""
    model_name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
        self.calls.append({
            "prompt": prompt, "system_prompt": system_prompt,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        if not self._responses:
            raise RuntimeError("LLM exhausted")
        return _Resp(self._responses.pop(0))


# ---------------------------------------------------------------------------
# Local backend behavioural tests
# ---------------------------------------------------------------------------
def test_r2_retries_on_schema_failure_then_succeeds():
    llm = _ScriptedLLM(["this is not yaml", "```yaml\nmetadata: {title: T}\n```"])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(regime="r2", max_retries=2,
                                schema_validator=yaml_fence_validator),
    )
    resp = gate.generate("ignored", system_prompt="be terse")
    assert "```yaml" in resp.content
    assert len(llm.calls) == 2
    m = gate.metrics_history[0]
    assert "schema_invalid" in m.gates_triggered
    assert m.retries >= 1
    assert m.cdl == 0.0   # eventually passed → no compliance loss


def test_r2_blocks_when_all_retries_fail():
    llm = _ScriptedLLM(["bad", "still bad", "again bad"])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(regime="r2", max_retries=2,
                                schema_validator=yaml_fence_validator),
    )
    gate.generate("x")
    m = gate.metrics_history[0]
    assert m.cdl == 1.0
    assert "schema_invalid" in m.gates_triggered
    assert len(llm.calls) == 3   # 1 try + 2 retries


def test_r1_never_retries_only_logs():
    llm = _ScriptedLLM(["bad"])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(regime="r1", max_retries=5,
                                schema_validator=yaml_fence_validator),
    )
    gate.generate("x")
    assert len(llm.calls) == 1
    m = gate.metrics_history[0]
    assert "schema_invalid" in m.gates_triggered
    assert m.cdl == 0.0   # r1 never blocks


def test_empty_output_gate():
    llm = _ScriptedLLM(["", "```yaml\nmetadata: {}\n```"])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(regime="r2", max_retries=2,
                                schema_validator=yaml_fence_validator),
    )
    gate.generate("x")
    assert "empty_output" in gate.metrics_history[0].gates_triggered


def test_stability_check_detects_unstable_output():
    # 1st call passes schema, 2nd (stability sample) returns different content
    llm = _ScriptedLLM([
        "```yaml\nmetadata: {title: A}\n```",
        "```yaml\nmetadata: {title: B}\n```",
    ])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(
            regime="r2", max_retries=0, stability_samples=2,
            schema_validator=yaml_fence_validator,
        ),
    )
    gate.generate("x")
    m = gate.metrics_history[0]
    assert "unstable_output" in m.gates_triggered
    assert m.ipi < 1.0


def test_aggregate_metrics_combines_history():
    llm = _ScriptedLLM(["```yaml\nmetadata: {}\n```", "```yaml\nmetadata: {}\n```"])
    gate = LLMGovernanceGate(
        llm, backend=LocalGovernanceBackend(),
        config=GovernanceConfig(regime="r2", schema_validator=yaml_fence_validator),
    )
    gate.generate("a"); gate.generate("b")
    agg = gate.aggregate_metrics()
    assert agg["calls"] == 2
    assert agg["backend"] == "local"
    assert agg["regime"] == "r2"


def test_gate_is_drop_in_for_llmprovider_contract():
    llm = _ScriptedLLM(["```yaml\nmetadata: {}\n```"])
    gate = LLMGovernanceGate(llm)
    assert hasattr(gate, "generate") and callable(gate.generate)
    assert gate.model_name == "scripted"


def test_json_validator_accepts_only_parseable_json():
    assert json_validator('{"a": 1}')
    assert json_validator('```json\n{"a":1}\n```')
    assert not json_validator("not json")
    assert not json_validator("")
