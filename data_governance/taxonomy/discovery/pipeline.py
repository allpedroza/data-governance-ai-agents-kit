"""Top-level orchestrator for the taxonomy discovery pipeline.

    collect (DW queries) → anonymize (NER) → LLM #1 (synthesize YAML)
                                          → score (deterministic, no LLM)
                                          → LLM #2 (evaluate + plan HTML)

Each stage is replaceable via constructor injection; the default wiring
uses the project's own warehouse connectors, NER agent and LLM providers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from ..agent import TaxonomyAgent
from ..models import TaxonomyDocument
from ..scorer import TaxonomyScore, TaxonomyScorer
from .anonymizer import SampleAnonymizer
from .collector import SchemaInventory, WarehouseInventoryCollector
from .evaluator import EvaluationResult, TaxonomyEvaluator
from .governance import (
    GovernanceConfig, LLMGovernanceGate, LocalGovernanceBackend,
    json_validator, yaml_fence_validator,
)
from .synthesizer import SynthesisResult, TaxonomySynthesizer

if TYPE_CHECKING:
    from data_governance.rag_discovery.providers.base import LLMProvider
    from data_governance.warehouse.connectors.base import WarehouseConnector

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryRunResult:
    inventory: SchemaInventory
    synthesis: SynthesisResult
    score: TaxonomyScore
    evaluation: EvaluationResult
    anonymization_report: Dict[str, Any] = field(default_factory=dict)
    governance_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def taxonomy(self) -> TaxonomyDocument:
        return self.synthesis.taxonomy

    @property
    def yaml_text(self) -> str:
        return self.synthesis.yaml_text

    @property
    def as_is_html(self) -> str:
        """Render the AS-IS HTML report using the canonical generator."""
        agent = TaxonomyAgent()
        return agent.generate_html_artifact(self.taxonomy, self.score)

    @property
    def improvement_plan_html(self) -> str:
        return self.evaluation.plan_html

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory_stats": {
                "warehouse_type": self.inventory.warehouse_type,
                "tables": len(self.inventory.tables),
                "columns": self.inventory.column_count(),
                "token_clusters": len(self.inventory.token_clusters),
            },
            "anonymization": self.anonymization_report,
            "governance": self.governance_report,
            "synthesis_model": self.synthesis.model,
            "evaluation_model": self.evaluation.model,
            "score": self.score.to_dict(),
            "plan": self.evaluation.plan,
            "yaml": self.yaml_text,
        }


class TaxonomyDiscoveryPipeline:
    """Compose the four stages into a single ``run`` call.

    Parameters
    ----------
    llm:
        LLMProvider used for synthesis. If ``evaluator_llm`` is omitted, the
        same provider is reused for evaluation.
    evaluator_llm:
        Optional separate provider for the evaluation stage (useful when
        synthesis uses a cheap model and evaluation uses a stronger one).
    anonymizer:
        Optional :class:`SampleAnonymizer` instance. Created with no NER
        agent (fallback regex only) if omitted.
    scorer:
        Optional :class:`TaxonomyScorer`. Useful for tests / custom weights.
    """

    def __init__(
        self,
        llm: "LLMProvider",
        evaluator_llm: Optional["LLMProvider"] = None,
        anonymizer: Optional[SampleAnonymizer] = None,
        scorer: Optional[TaxonomyScorer] = None,
        synthesis_max_tokens: int = 4000,
        evaluation_max_tokens: int = 4000,
        governance: Optional[GovernanceConfig] = None,
        governance_backend: Optional[Any] = None,
    ) -> None:
        # Optionally wrap each LLM in a governance gate. Synthesis uses the
        # YAML-fence validator; evaluation uses the strict JSON validator.
        synth_llm = llm
        eval_llm = evaluator_llm or llm
        self._synth_gate: Optional[LLMGovernanceGate] = None
        self._eval_gate: Optional[LLMGovernanceGate] = None
        if governance is not None:
            backend = governance_backend or LocalGovernanceBackend()
            synth_cfg = GovernanceConfig(
                **{**governance.__dict__, "schema_validator": yaml_fence_validator}
            )
            eval_cfg = GovernanceConfig(
                **{**governance.__dict__, "schema_validator": json_validator}
            )
            self._synth_gate = LLMGovernanceGate(synth_llm, backend=backend, config=synth_cfg)
            self._eval_gate = LLMGovernanceGate(eval_llm, backend=backend, config=eval_cfg)
            synth_llm = self._synth_gate
            eval_llm = self._eval_gate

        self.synthesizer = TaxonomySynthesizer(synth_llm, max_tokens=synthesis_max_tokens)
        self.evaluator = TaxonomyEvaluator(eval_llm, max_tokens=evaluation_max_tokens)
        self.anonymizer = anonymizer or SampleAnonymizer()
        self.scorer = scorer or TaxonomyScorer()

    def run(
        self,
        connector: "WarehouseConnector",
        schema: Optional[str] = None,
        database: Optional[str] = None,
        table_filter: Optional[Sequence[str]] = None,
        max_tables: int = 50,
        profile_samples: bool = True,
        sample_rows: int = 50,
    ) -> DiscoveryRunResult:
        """End-to-end discovery run.

        ``profile_samples=True`` reads small samples for column profiling.
        Samples are *always* sent through ``self.anonymizer`` before any
        content reaches the LLM.
        """
        collector = WarehouseInventoryCollector(
            connector,
            max_tables=max_tables,
            profile_samples=profile_samples,
            sample_rows=sample_rows,
        )
        inventory = collector.collect(
            schema=schema, database=database, table_filter=table_filter,
        )
        logger.info(
            "Collected %d tables / %d columns from %s",
            len(inventory.tables), inventory.column_count(), inventory.warehouse_type,
        )

        # Always anonymize, even when profile_samples=False — comments can
        # carry sensitive info too.
        self.anonymizer.anonymize_inventory(inventory)
        anon_report = self.anonymizer.last_report

        synthesis = self.synthesizer.synthesize(inventory)
        logger.info(
            "Synthesized taxonomy with %d groups / %d concepts",
            len(synthesis.taxonomy.concept_groups),
            len(synthesis.taxonomy.get_all_concepts()),
        )

        # Resolve architecture profile from inventory + declared metadata
        from ..context import ArchitectureProfile
        declared = ArchitectureProfile.from_metadata(synthesis.taxonomy.metadata)
        profile = ArchitectureProfile.detect(inventory, declared=declared)
        logger.info(
            "Profile resolved: cloud=%s, pattern=%s, declared=%s",
            profile.cloud.cloud, profile.pattern.name, profile.declared,
        )

        score = self.scorer.score(synthesis.taxonomy, profile=profile)
        evaluation = self.evaluator.evaluate(synthesis.taxonomy, score)

        governance_report: Dict[str, Any] = {}
        if self._synth_gate is not None:
            governance_report["synthesis"] = self._synth_gate.aggregate_metrics()
        if self._eval_gate is not None:
            governance_report["evaluation"] = self._eval_gate.aggregate_metrics()

        return DiscoveryRunResult(
            inventory=inventory,
            synthesis=synthesis,
            score=score,
            evaluation=evaluation,
            anonymization_report=anon_report,
            governance_report=governance_report,
        )
