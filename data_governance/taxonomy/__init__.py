try:
    from .models import TaxonomyDocument, TaxonomyConcept, NamingConvention, ContextBasedRule, AmbiguousAlias, LakeStandards, LakeZone, ValidationRule
    from .scorer import TaxonomyScorer, TaxonomyScore, TaxonomyGap, TaxonomyRecommendation
    from .agent import TaxonomyAgent
    from .html_generator import generate_taxonomy_html
    _taxonomy_available = True
except ImportError:
    _taxonomy_available = False
