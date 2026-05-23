from typing import Any, Dict, List
from .models import TaxonomyDocument
from .scorer import TaxonomyScorer, TaxonomyScore, TaxonomyRecommendation

class TaxonomyAgent:
    """Taxonomy Evaluator, Scorer & Artifact Generator."""
    
    def __init__(self, scorer=None):
        self.scorer = scorer or TaxonomyScorer()
    
    def load_from_yaml(self, path: str) -> TaxonomyDocument:
        """Load taxonomy from YAML file."""
        return TaxonomyDocument.from_yaml(path)
    
    def score_taxonomy(self, taxonomy: TaxonomyDocument) -> TaxonomyScore:
        """Score taxonomy against 8-dimension framework."""
        return self.scorer.score(taxonomy)
    
    def generate_delta(self, taxonomy: TaxonomyDocument, score: TaxonomyScore) -> List[TaxonomyRecommendation]:
        """Generate TO-BE recommendations based on the score."""
        return score.recommendations
    
    def export_yaml(self, taxonomy: TaxonomyDocument, path: str) -> str:
        """Export taxonomy as YAML to a file."""
        yaml_content = taxonomy.to_yaml()
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return yaml_content
    
    def generate_html_artifact(self, taxonomy: TaxonomyDocument, score: TaxonomyScore) -> str:
        """Generate HTML artifact. Delegates to html_generator."""
        from .html_generator import generate_taxonomy_html
        return generate_taxonomy_html(taxonomy.to_dict(), score.to_dict())
