"""End-to-end taxonomy discovery pipeline.

Collects schema and column-profile information from a data warehouse,
anonymizes any PII/sensitive content present in the samples, and uses an
LLM to synthesize a candidate taxonomy YAML. A second LLM pass then
evaluates the candidate against the canonical best-practice framework and
returns an improvement plan rendered as HTML.

All external dependencies (warehouse connector, LLM provider, NER agent)
are injected via the public ``TaxonomyDiscoveryPipeline`` constructor so
each component can be mocked in tests and replaced in production.
"""
from .collector import (
    ColumnProfile,
    SchemaInventory,
    TableInventory,
    WarehouseInventoryCollector,
)
from .anonymizer import SampleAnonymizer
from .prompts import (
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from .synthesizer import TaxonomySynthesizer, SynthesisResult
from .evaluator import TaxonomyEvaluator, EvaluationResult
from .pipeline import TaxonomyDiscoveryPipeline, DiscoveryRunResult

__all__ = [
    "TaxonomyDiscoveryPipeline",
    "DiscoveryRunResult",
    "WarehouseInventoryCollector",
    "SchemaInventory",
    "TableInventory",
    "ColumnProfile",
    "SampleAnonymizer",
    "TaxonomySynthesizer",
    "SynthesisResult",
    "TaxonomyEvaluator",
    "EvaluationResult",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_USER_TEMPLATE",
    "EVALUATION_SYSTEM_PROMPT",
    "EVALUATION_USER_TEMPLATE",
]
