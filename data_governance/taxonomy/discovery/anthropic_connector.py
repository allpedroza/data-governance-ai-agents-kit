"""Anthropic Claude connector tuned for the taxonomy discovery pipeline.

Choosing the right model per stage matters a lot here because the two LLM
calls have very different cost/quality profiles:

================  ===========================  ======================
Stage             Task                         Model recommendation
================  ===========================  ======================
Synthesis (#1)    Parse a structured inventory Sonnet 4.6 — needs
                  into a 50-field YAML; must   reliable structured
                  follow a strict schema and   output and strong
                  detect aliases / clusters    instruction-following.
                  across many tables.          Haiku struggles with
                                               long YAML; Opus is
                                               overkill on cost.
Evaluation (#2)   Reason about taxonomy gaps   Sonnet 4.6 by default.
                  and emit a JSON plan; needs  Opus 4.7 for very large
                  judgment and grounded refs   YAMLs / regulated
                  to the YAML.                 contexts.
                                               Haiku 4.5 OK on small
                                               taxonomies < 30 concepts
                                               where speed matters.
================  ===========================  ======================

Cost-benefit conclusion: **Sonnet 4.6 is the default for both stages.**
Opus is reserved for evaluation when the user explicitly trades cost for
breadth of recommendations. Haiku is a budget mode for evaluation only.

The :class:`TaxonomyClaudeLLM` helper exposes class-method constructors
(``for_synthesis`` / ``for_evaluation``) that pick the right model
without leaking those choices into call sites.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from data_governance.rag_discovery.providers.llm.anthropic_llm import AnthropicLLM

logger = logging.getLogger(__name__)


# Canonical model identifiers (Claude 4.x family).
MODEL_OPUS_47 = "claude-opus-4-7"
MODEL_SONNET_46 = "claude-sonnet-4-6"
MODEL_HAIKU_45 = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ModelTier:
    """Indicative cost/latency tier metadata for human-facing UIs."""
    name: str
    short_label: str
    rel_cost: int       # 1=cheap, 5=expensive (input + output, rough USD ratio)
    rel_latency: int    # 1=fast, 5=slow
    notes: str


MODEL_TIERS = {
    MODEL_OPUS_47: ModelTier(
        "Opus 4.7", "opus", rel_cost=5, rel_latency=4,
        notes="Highest quality; reserve for high-stakes / regulated evaluations.",
    ),
    MODEL_SONNET_46: ModelTier(
        "Sonnet 4.6", "sonnet", rel_cost=2, rel_latency=2,
        notes="Best cost/quality balance; default for both pipeline stages.",
    ),
    MODEL_HAIKU_45: ModelTier(
        "Haiku 4.5", "haiku", rel_cost=1, rel_latency=1,
        notes="Lowest cost & latency; suitable only for evaluation on small taxonomies.",
    ),
}


# Recommended models per discovery stage and budget profile.
STAGE_RECOMMENDATIONS = {
    "synthesis":  {"balanced": MODEL_SONNET_46, "premium": MODEL_OPUS_47, "budget": MODEL_SONNET_46},
    "evaluation": {"balanced": MODEL_SONNET_46, "premium": MODEL_OPUS_47, "budget": MODEL_HAIKU_45},
}


class TaxonomyClaudeLLM(AnthropicLLM):
    """Claude LLM with stage-aware constructors for the discovery pipeline."""

    @classmethod
    def for_synthesis(
        cls,
        budget: str = "balanced",
        api_key: Optional[str] = None,
        max_tokens: int = 6000,
    ) -> "TaxonomyClaudeLLM":
        """Pick the recommended model for the synthesis (LLM #1) stage.

        Synthesis outputs a structured YAML document with many fields, so
        ``max_tokens`` defaults to a generous 6k to avoid truncation.
        """
        model = STAGE_RECOMMENDATIONS["synthesis"].get(
            budget, MODEL_SONNET_46
        )
        logger.info("Selecting %s for synthesis (budget=%s)", model, budget)
        return cls(
            model=model, api_key=api_key, default_max_tokens=max_tokens,
            default_temperature=0.0,
        )

    @classmethod
    def for_evaluation(
        cls,
        budget: str = "balanced",
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> "TaxonomyClaudeLLM":
        """Pick the recommended model for the evaluation (LLM #2) stage.

        Evaluation returns a JSON plan; 4k tokens is enough even for
        large taxonomies.
        """
        model = STAGE_RECOMMENDATIONS["evaluation"].get(
            budget, MODEL_SONNET_46
        )
        logger.info("Selecting %s for evaluation (budget=%s)", model, budget)
        return cls(
            model=model, api_key=api_key, default_max_tokens=max_tokens,
            default_temperature=0.0,
        )

    @classmethod
    def describe(cls, model: str) -> ModelTier:
        """Return tier metadata for a known model identifier."""
        return MODEL_TIERS.get(
            model,
            ModelTier(model, model, rel_cost=3, rel_latency=3, notes="Unknown model."),
        )
