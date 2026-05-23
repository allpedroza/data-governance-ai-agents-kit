"""Taxonomy Evaluator, Scorer & Artifact Generator.

Lazy imports keep optional dependencies (PyYAML) from breaking sibling
modules when ``data_governance`` is imported for unrelated agents.
"""
try:
    from .models import (
        AmbiguousAlias,
        ContextBasedRule,
        LakeStandards,
        LakeZone,
        NamingConvention,
        TaxonomyConcept,
        TaxonomyDocument,
        ValidationRule,
    )
    from .scorer import (
        TaxonomyGap,
        TaxonomyRecommendation,
        TaxonomyScore,
        TaxonomyScorer,
    )
    from .agent import (
        DimensionPlan,
        ImprovementAction,
        ImprovementPlan,
        TaxonomyAgent,
    )
    from .html_generator import generate_taxonomy_html

    _taxonomy_available = True
except ImportError:  # PyYAML missing or similar
    _taxonomy_available = False

__all__ = [
    "TaxonomyAgent",
    "TaxonomyDocument",
    "TaxonomyScore",
    "TaxonomyScorer",
    "TaxonomyGap",
    "TaxonomyRecommendation",
    "TaxonomyConcept",
    "NamingConvention",
    "ContextBasedRule",
    "AmbiguousAlias",
    "LakeStandards",
    "LakeZone",
    "ValidationRule",
    "ImprovementAction",
    "DimensionPlan",
    "ImprovementPlan",
    "generate_taxonomy_html",
]
