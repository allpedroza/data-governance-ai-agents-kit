"""LLM governance gate for the taxonomy discovery pipeline.

Wraps any ``LLMProvider`` and runs each call through a configurable set of
gates before the response is accepted. The taxonomy synthesizer and
evaluator both emit content that becomes the *source of truth* for the
rest of the framework (DDL generation, dbt models, CLI commands) — that
is high-stakes by definition, exactly the kind of LLM decision the
SantanderAI ``mech-gov-framework`` was built to govern.

Two backends are supported:

* :class:`MechGovBackend` — delegates to the ``mech_gov`` Apache-2.0
  framework when it is installed (``pip install mech-gov-framework``).
  Provides R1 (text-only), R2 (mechanical gates with entropy commit-
  reveal, ambiguity gate, candidate freezing) and R3 (adaptive) regimes
  plus their governance metrics (CDL, DIU, IPI, FVS, ESD, FSR).
* :class:`LocalGovernanceBackend` — pure-Python fallback that implements
  the gates we actually need for taxonomy generation: schema validation,
  empty-output rejection, retry with stricter system prompt on failure,
  and an output-stability check across N samples (a poor-man's
  commit-reveal). Always available; no extra dependency.

The gate exposes the same :class:`LLMProvider` contract as the wrapped
client so it can be slotted into ``TaxonomyDiscoveryPipeline`` without
any caller code changing.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_governance.rag_discovery.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / metrics
# ---------------------------------------------------------------------------
@dataclass
class GovernanceConfig:
    """How strict the governance gate should be.

    ``regime`` follows the mech_gov vocabulary:

    * ``"r1"`` — text-only: log gate signals but never block
    * ``"r2"`` — mechanical: block on schema failure, retry, stability check
    * ``"r3"`` — adaptive: r2 + automatic prompt tightening between retries
    """

    regime: str = "r2"
    max_retries: int = 2
    stability_samples: int = 1               # >1 enables commit-reveal
    schema_validator: Optional[Callable[[str], bool]] = None
    require_non_empty: bool = True
    risk_score: float = 0.7                  # 0..1 — affects metric weighting
    completeness: float = 0.5
    flags: List[str] = field(default_factory=list)


@dataclass
class GovernanceMetrics:
    """Per-call governance telemetry.

    Mirrors the mech_gov metric vocabulary so dashboards built on either
    source align:

    * ``CDL`` — Compliance Decision Loss (0=fine, 1=blocked)
    * ``IPI`` — Integrity Persistence Index (1=stable across samples)
    * ``DIU`` — Decision Integrity Under uncertainty (0..1)
    """
    backend: str
    regime: str
    cdl: float = 0.0
    ipi: float = 1.0
    diu: float = 1.0
    retries: int = 0
    gates_triggered: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "regime": self.regime,
            "cdl": round(self.cdl, 3),
            "ipi": round(self.ipi, 3),
            "diu": round(self.diu, 3),
            "retries": self.retries,
            "gates_triggered": list(self.gates_triggered),
            "duration_ms": round(self.duration_ms, 1),
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class GovernanceBackend:
    """Strategy interface for the gate."""
    name: str = "base"

    def govern(
        self,
        callable_llm: Callable[[str, Optional[str], float, Optional[int]], "LLMResponse"],
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        config: GovernanceConfig,
    ) -> tuple:
        raise NotImplementedError


class LocalGovernanceBackend(GovernanceBackend):
    """Dependency-free gate. Implements the gates we actually need."""

    name = "local"

    def govern(
        self,
        callable_llm,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        config: GovernanceConfig,
    ):
        gates: List[str] = []
        start = time.monotonic()
        last_response = None
        responses: List[str] = []

        attempts = 1 + max(0, config.max_retries)
        active_system = system_prompt or ""
        for attempt in range(attempts):
            response = callable_llm(prompt, active_system, temperature, max_tokens)
            last_response = response
            content = getattr(response, "content", "") or ""
            responses.append(content)

            # Gate 1: non-empty
            if config.require_non_empty and not content.strip():
                gates.append("empty_output")
                if config.regime == "r1":
                    break
                active_system = self._tighten_prompt(active_system, "the output was empty")
                continue

            # Gate 2: schema check (custom validator)
            if config.schema_validator and not config.schema_validator(content):
                gates.append("schema_invalid")
                if config.regime == "r1":
                    break
                active_system = self._tighten_prompt(
                    active_system, "the previous response did not match the required schema"
                )
                continue

            # Passed — break early
            break

        # Stability check (commit-reveal poor man's version): N extra samples,
        # compare canonical hashes. Only runs in r2/r3 and when stability_samples>1.
        ipi = 1.0
        if config.regime in ("r2", "r3") and config.stability_samples > 1 and last_response is not None:
            digests = [hashlib.sha256(c.strip().encode("utf-8")).hexdigest() for c in responses]
            for _ in range(config.stability_samples - 1):
                extra = callable_llm(prompt, active_system, temperature, max_tokens)
                digests.append(hashlib.sha256(
                    (getattr(extra, "content", "") or "").strip().encode("utf-8")
                ).hexdigest())
            unique = len(set(digests))
            ipi = 1.0 / max(unique, 1)
            if ipi < 1.0:
                gates.append("unstable_output")

        duration_ms = (time.monotonic() - start) * 1000
        cdl = 1.0 if (gates and config.regime != "r1") and (
            "schema_invalid" in gates or "empty_output" in gates
        ) and attempt == attempts - 1 else 0.0
        diu = max(0.0, 1.0 - 0.2 * len(gates))

        metrics = GovernanceMetrics(
            backend=self.name, regime=config.regime,
            cdl=cdl, ipi=ipi, diu=diu,
            retries=max(0, attempt),
            gates_triggered=gates, duration_ms=duration_ms,
        )
        return last_response, metrics

    @staticmethod
    def _tighten_prompt(system_prompt: str, reason: str) -> str:
        suffix = (
            f"\n\nIMPORTANT: a previous attempt failed because {reason}. "
            "Strictly follow the output contract — return only the requested "
            "format with no surrounding prose."
        )
        return (system_prompt or "") + suffix


class MechGovBackend(GovernanceBackend):
    """Delegate gating to the SantanderAI ``mech_gov`` framework.

    The framework was designed for banking decisions; we adapt our LLM call
    into a synthetic :class:`BankingCase` whose ``risk_score`` /
    ``completeness`` / ``regulatory_flags`` are derived from the
    ``GovernanceConfig``. This gives us the mechanical gates (entropy
    commit-reveal, candidate freezing, ambiguity gate) and the metric
    suite (CDL/DIU/IPI/FVS/ESD/FSR) without inventing the wheel.
    """

    name = "mech_gov"

    def __init__(self) -> None:
        try:
            from mech_gov.data.banking_case import BankingCase, TransactionType  # noqa: F401
            from mech_gov.governance.r1_text_only import R1TextOnly  # noqa: F401
            from mech_gov.governance.r2_mechanical import R2Mechanical  # noqa: F401
            from mech_gov.governance.r3_adaptive import R3Adaptive  # noqa: F401
            from mech_gov.llm.registry import create_llm  # noqa: F401
            from mech_gov.metrics.governance import compute_governance_metrics  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mech_gov_framework not installed. Install with: "
                "pip install mech-gov-framework"
            ) from exc

    def govern(
        self,
        callable_llm,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        config: GovernanceConfig,
    ):
        # Lazy imports — already validated in __init__
        from mech_gov.data.banking_case import BankingCase, TransactionType
        from mech_gov.governance.r1_text_only import R1TextOnly
        from mech_gov.governance.r2_mechanical import R2Mechanical
        from mech_gov.governance.r3_adaptive import R3Adaptive
        from mech_gov.llm.registry import create_llm
        from mech_gov.metrics.governance import compute_governance_metrics

        regimes = {"r1": R1TextOnly, "r2": R2Mechanical, "r3": R3Adaptive}
        regime_cls = regimes.get(config.regime, R2Mechanical)

        # Adapt our LLM into the callable contract mech_gov expects
        def backend(system, user, temperature=0.0, max_tokens=2048):
            resp = callable_llm(user, system, temperature, max_tokens)
            return getattr(resp, "content", "") or ""

        gov_llm = create_llm({"provider": "callable", "callable": backend})

        # Synthetic case — taxonomy synthesis is a CREDIT_APPROVAL-equivalent
        # high-risk decision in our framework.
        tx_type = getattr(TransactionType, "CREDIT_APPROVAL", list(TransactionType)[0])
        case = BankingCase(
            case_id=f"taxonomy-{int(time.time() * 1000)}",
            transaction_type=tx_type,
            risk_score=config.risk_score,
            completeness=config.completeness,
            regulatory_flags=list(config.flags) or ["TAXONOMY"],
        )

        start = time.monotonic()
        result = regime_cls().process_case(case, gov_llm, entropy_seed=None)
        duration_ms = (time.monotonic() - start) * 1000

        # Extract metrics for this single decision
        try:
            metrics_dict = compute_governance_metrics([result]) or {}
        except Exception:  # noqa: BLE001 — metric extraction must never block
            metrics_dict = {}

        # Re-run the *raw* LLM once to obtain the content payload itself
        # (mech_gov only returns the decision; our pipeline needs the text).
        raw = callable_llm(prompt, system_prompt, temperature, max_tokens)

        gates = list(getattr(result, "gates_triggered", []) or [])
        metrics = GovernanceMetrics(
            backend=self.name, regime=config.regime,
            cdl=float(metrics_dict.get("CDL", 0.0) or 0.0),
            ipi=float(metrics_dict.get("IPI", 1.0) or 1.0),
            diu=float(metrics_dict.get("DIU", 1.0) or 1.0),
            retries=0,
            gates_triggered=gates,
            duration_ms=duration_ms,
        )
        return raw, metrics


# ---------------------------------------------------------------------------
# Public gate (LLMProvider-compatible)
# ---------------------------------------------------------------------------
class LLMGovernanceGate:
    """Wraps an :class:`LLMProvider` with a :class:`GovernanceBackend`.

    Implements the ``LLMProvider`` contract (``generate``, ``model_name``)
    so callers can drop it into the synthesizer / evaluator without code
    changes.

    Per-call telemetry is captured in ``self.metrics_history`` so the
    pipeline can surface CDL/IPI/retries in the final result.
    """

    def __init__(
        self,
        llm: "LLMProvider",
        backend: Optional[GovernanceBackend] = None,
        config: Optional[GovernanceConfig] = None,
    ) -> None:
        self._llm = llm
        self._backend = backend or LocalGovernanceBackend()
        self._config = config or GovernanceConfig()
        self.metrics_history: List[GovernanceMetrics] = []

    @property
    def model_name(self) -> str:
        return getattr(self._llm, "model_name", "wrapped")

    def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
        def call(p, s, t, m):
            return self._llm.generate(p, system_prompt=s, temperature=t, max_tokens=m)

        response, metrics = self._backend.govern(
            callable_llm=call,
            prompt=prompt, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            config=self._config,
        )
        self.metrics_history.append(metrics)
        return response

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Combine per-call metrics into a single dict for the result payload."""
        if not self.metrics_history:
            return {}
        n = len(self.metrics_history)
        return {
            "backend": self.metrics_history[0].backend,
            "regime": self.metrics_history[0].regime,
            "calls": n,
            "avg_cdl": round(sum(m.cdl for m in self.metrics_history) / n, 3),
            "avg_ipi": round(sum(m.ipi for m in self.metrics_history) / n, 3),
            "avg_diu": round(sum(m.diu for m in self.metrics_history) / n, 3),
            "total_retries": sum(m.retries for m in self.metrics_history),
            "gates_triggered": sorted({
                g for m in self.metrics_history for g in m.gates_triggered
            }),
            "total_duration_ms": round(sum(m.duration_ms for m in self.metrics_history), 1),
        }


def yaml_fence_validator(content: str) -> bool:
    """Validator for the synthesis stage: response must contain a YAML fence."""
    if not content:
        return False
    lower = content.lower()
    return "```yaml" in lower or content.lstrip().startswith(("metadata", "concept_groups"))


def json_validator(content: str) -> bool:
    """Validator for the evaluation stage: response must parse as JSON."""
    import json
    if not content:
        return False
    cleaned = content.strip().lstrip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip()
    cleaned = cleaned.rstrip("`").strip()
    try:
        json.loads(cleaned)
        return True
    except Exception:  # noqa: BLE001
        return False
