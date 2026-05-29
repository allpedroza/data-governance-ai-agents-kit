from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from .models import TaxonomyDocument


@dataclass
class TaxonomyGap:
    dimension: str
    severity: str
    current_state: str
    target_state: str
    effort_estimate: str


@dataclass
class TaxonomyRecommendation:
    priority: int
    title: str
    description: str
    dimension: str
    expected_score_impact: float
    effort: str
    timeline: str


@dataclass
class TaxonomyScore:
    overall_score: float
    maturity_level: int
    maturity_label: str
    dimension_scores: Dict[str, float]
    dimension_findings: Dict[str, List[str]]
    gaps: List[TaxonomyGap]
    recommendations: List[TaxonomyRecommendation]
    benchmark_comparison: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "maturity_level": self.maturity_level,
            "maturity_label": self.maturity_label,
            "dimension_scores": self.dimension_scores,
            "dimension_findings": self.dimension_findings,
            "gaps": [g.__dict__ for g in self.gaps],
            "recommendations": [r.__dict__ for r in self.recommendations],
            "benchmark_comparison": self.benchmark_comparison,
        }


class TaxonomyScorer:
    """Scoring engine that evaluates a TaxonomyDocument against 8 dimensions."""

    WEIGHTS = {
        "naming_consistency": 0.15,
        "definition_completeness": 0.15,
        "alias_management": 0.10,
        "context_rules": 0.10,
        "taxonomy_structure": 0.15,
        "data_type_standardization": 0.10,
        "governance_readiness": 0.15,
        "lake_platform_alignment": 0.10,
    }

    def score(self, taxonomy: TaxonomyDocument) -> TaxonomyScore:
        findings: Dict[str, List[str]] = {k: [] for k in self.WEIGHTS.keys()}
        scores: Dict[str, float] = {}

        # 1. Naming Consistency
        nc_score = 0.0
        if taxonomy.naming_conventions.casing_rules:
            nc_score += 40
            findings["naming_consistency"].append("Casing rules defined.")
        if taxonomy.naming_conventions.canonical_acronyms:
            nc_score += 30
            findings["naming_consistency"].append("Canonical acronyms defined.")
        if taxonomy.naming_conventions.forbidden_forms:
            nc_score += 30
            findings["naming_consistency"].append("Forbidden forms defined.")
        scores["naming_consistency"] = nc_score

        # 2. Definition Completeness
        concepts = taxonomy.get_all_concepts()
        if not concepts:
            dc_score = 0.0
            findings["definition_completeness"].append("No concepts mapped.")
        else:
            with_def = sum(1 for c in concepts if c.definition)
            dc_score = (with_def / len(concepts)) * 100
            findings["definition_completeness"].append(f"{with_def}/{len(concepts)} concepts have definitions.")
        scores["definition_completeness"] = dc_score

        # 3. Alias Management
        if not concepts:
            am_score = 0.0
        else:
            with_alias = sum(1 for c in concepts if c.aliases)
            am_score = min(100.0, (with_alias / len(concepts)) * 100 * 1.5)  # Boost
            findings["alias_management"].append(f"{with_alias} concepts have mapped aliases.")
        if taxonomy.ambiguous_aliases:
            am_score = min(100.0, am_score + 20)
            findings["alias_management"].append("Ambiguous aliases resolution rules defined.")
        scores["alias_management"] = am_score

        # 4. Context Rules
        if taxonomy.context_rules:
            cr_score = min(100.0, len(taxonomy.context_rules) * 50)
            findings["context_rules"].append(f"{len(taxonomy.context_rules)} context rules defined.")
        else:
            cr_score = 0.0
            findings["context_rules"].append("No context rules defined.")
        scores["context_rules"] = cr_score

        # 5. Taxonomy Structure
        if taxonomy.concept_groups:
            ts_score = min(100.0, len(taxonomy.concept_groups) * 10)
            findings["taxonomy_structure"].append(f"{len(taxonomy.concept_groups)} concept groups defined.")
        else:
            ts_score = 0.0
            findings["taxonomy_structure"].append("No concept groups defined.")
        scores["taxonomy_structure"] = ts_score

        # 6. Data Type Standardization
        if not concepts:
            dt_score = 0.0
        else:
            with_types = sum(1 for c in concepts if c.accepted_types)
            dt_score = (with_types / len(concepts)) * 80
            findings["data_type_standardization"].append(f"{with_types}/{len(concepts)} concepts have standardized types.")
        if taxonomy.datetime_standards:
            dt_score += 20
            findings["data_type_standardization"].append("Datetime standards defined.")
        scores["data_type_standardization"] = dt_score

        # 7. Governance Readiness
        gr_score = 0.0
        if taxonomy.metadata.get("owner"):
            gr_score += 30
            findings["governance_readiness"].append("Data owner defined.")
        if taxonomy.metadata.get("steward"):
            gr_score += 30
            findings["governance_readiness"].append("Data steward defined.")
        if taxonomy.ai_agent_instructions and taxonomy.validation_rules:
            gr_score += 40
            findings["governance_readiness"].append("AI agent instructions and validation rules are present.")
        scores["governance_readiness"] = gr_score

        # 8. Lake/Platform Alignment
        la_score = 0.0
        if taxonomy.lake_standards and taxonomy.lake_standards.zones:
            la_score += 60
            findings["lake_platform_alignment"].append(f"{len(taxonomy.lake_standards.zones)} lake zones defined.")
        if taxonomy.lake_standards and taxonomy.lake_standards.project_structure:
            la_score += 40
            findings["lake_platform_alignment"].append("Project structure naming patterns defined.")
        scores["lake_platform_alignment"] = la_score

        overall_score = sum(scores[dim] * weight for dim, weight in self.WEIGHTS.items())
        mat_level, mat_label = self._calculate_maturity_level(overall_score)
        
        gaps = self._generate_gaps(scores)
        recommendations = self._generate_recommendations(gaps)

        return TaxonomyScore(
            overall_score=round(overall_score, 1),
            maturity_level=mat_level,
            maturity_label=mat_label,
            dimension_scores={k: round(v, 1) for k, v in scores.items()},
            dimension_findings=findings,
            gaps=gaps,
            recommendations=recommendations,
            benchmark_comparison=self._get_benchmark()
        )

    def _calculate_maturity_level(self, score: float) -> Tuple[int, str]:
        if score <= 20: return 1, "Initial"
        if score <= 40: return 2, "Managed"
        if score <= 60: return 3, "Defined"
        if score <= 80: return 4, "Measured"
        return 5, "Optimized"

    def _generate_gaps(self, scores: Dict[str, float]) -> List[TaxonomyGap]:
        gaps = []
        for dim, score in scores.items():
            if score < 50:
                gaps.append(TaxonomyGap(
                    dimension=dim,
                    severity="critical" if score < 20 else "major",
                    current_state=f"Score is low ({score:.1f}/100).",
                    target_state="Establish formalized rules and comprehensive mapping.",
                    effort_estimate="high"
                ))
            elif score < 80:
                gaps.append(TaxonomyGap(
                    dimension=dim,
                    severity="minor",
                    current_state=f"Score is acceptable ({score:.1f}/100) but lacks full coverage.",
                    target_state="Refine edge cases and ensure 100% metadata coverage.",
                    effort_estimate="medium"
                ))
        return gaps

    def _generate_recommendations(self, gaps: List[TaxonomyGap]) -> List[TaxonomyRecommendation]:
        recs = []
        priority = 1
        for gap in sorted(gaps, key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity]):
            recs.append(TaxonomyRecommendation(
                priority=priority,
                title=f"Improve {gap.dimension.replace('_', ' ').title()}",
                description=f"Move from current state to target: {gap.target_state}",
                dimension=gap.dimension,
                expected_score_impact=10.0 if gap.severity == "critical" else 5.0,
                effort=gap.effort_estimate,
                timeline="short" if gap.severity == "critical" else "medium"
            ))
            priority += 1
        return recs

    def _get_benchmark(self) -> Dict[str, float]:
        return {
            "naming_consistency": 65.0,
            "definition_completeness": 55.0,
            "alias_management": 40.0,
            "context_rules": 35.0,
            "taxonomy_structure": 60.0,
            "data_type_standardization": 70.0,
            "governance_readiness": 45.0,
            "lake_platform_alignment": 50.0,
        }
