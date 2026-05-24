"""Scoring engine for taxonomy maturity.

Evaluates a TaxonomyDocument against eight weighted dimensions. The scoring
favours coverage and depth over presence alone — e.g., having 11 concept
groups does not award a perfect structural score unless those groups are
populated, named and described.

When an :class:`ArchitectureProfile` is supplied (or declared in the
taxonomy metadata), the scorer becomes context-aware: cloud-specific
identifier limits influence ``naming_consistency``; architectural
prefixes (``dim_``, ``fct_``, ``stg_``, ``hub_``...) are treated as
canonical rather than forbidden abbreviations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .context import ArchitectureProfile
from .models import TaxonomyDocument

_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


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

    def score(
        self,
        taxonomy: TaxonomyDocument,
        profile: Optional[ArchitectureProfile] = None,
    ) -> TaxonomyScore:
        findings: Dict[str, List[str]] = {k: [] for k in self.WEIGHTS.keys()}
        scores: Dict[str, float] = {}

        # Resolve the architecture profile: explicit > metadata > undeclared/generic
        if profile is None:
            profile = ArchitectureProfile.from_metadata(taxonomy.metadata)
        for f_msg in profile.detection_findings:
            findings["naming_consistency"].append(f_msg)

        # 1. Naming Consistency — requires both definition and concrete examples
        nc = taxonomy.naming_conventions
        nc_score = 0.0
        if nc.casing_rules:
            nc_score += 25
            findings["naming_consistency"].append("Casing rules defined.")
            if nc.casing_rules.get("examples"):
                nc_score += 10
                findings["naming_consistency"].append("Casing rules backed by examples.")
            # Bonus when declared casing matches the cloud default
            declared_col_casing = str(nc.casing_rules.get("columns", "")).lower()
            expected = profile.expected_column_casing().lower()
            if declared_col_casing and declared_col_casing == expected:
                nc_score += 10
                findings["naming_consistency"].append(
                    f"Column casing aligned with {profile.cloud.cloud} default ({expected})."
                )
            elif declared_col_casing and declared_col_casing != expected:
                findings["naming_consistency"].append(
                    f"Column casing '{declared_col_casing}' differs from "
                    f"{profile.cloud.cloud} default '{expected}' — intentional?"
                )
        if nc.canonical_acronyms:
            nc_score += 15
            findings["naming_consistency"].append(
                f"{len(nc.canonical_acronyms.get('items', []))} canonical acronyms defined."
            )
        if nc.forbidden_forms:
            nc_score += 15
            findings["naming_consistency"].append(
                f"{len(nc.forbidden_forms.get('items', []))} forbidden patterns documented."
            )
        if nc.application_names:
            nc_score += 10
            findings["naming_consistency"].append("Application-name casing rules defined.")
        if nc.full_words_required:
            nc_score += 10
            findings["naming_consistency"].append("Full-word-required policy defined.")
        # Cloud identifier-limit awareness
        all_concepts_for_len = taxonomy.get_all_concepts()
        if all_concepts_for_len:
            over_limit = [
                c.name for c in all_concepts_for_len
                if len(c.name) > profile.cloud.column_name_max
            ]
            if over_limit:
                findings["naming_consistency"].append(
                    f"{len(over_limit)} concept name(s) exceed {profile.cloud.cloud} "
                    f"column-name limit of {profile.cloud.column_name_max} chars."
                )
                nc_score -= min(20, len(over_limit) * 4)
        scores["naming_consistency"] = max(0.0, min(100.0, nc_score))

        # 2. Definition Completeness — penalise very short definitions
        concepts = taxonomy.get_all_concepts()
        if not concepts:
            scores["definition_completeness"] = 0.0
            findings["definition_completeness"].append("No concepts mapped.")
        else:
            with_def = [c for c in concepts if c.definition and len(c.definition.strip()) >= 15]
            ratio = len(with_def) / len(concepts)
            scores["definition_completeness"] = round(ratio * 100, 1)
            findings["definition_completeness"].append(
                f"{len(with_def)}/{len(concepts)} concepts have meaningful definitions (≥15 chars)."
            )

        # 3. Alias Management
        if not concepts:
            am_score = 0.0
        else:
            with_alias = sum(1 for c in concepts if c.aliases)
            am_score = (with_alias / len(concepts)) * 80
            findings["alias_management"].append(
                f"{with_alias}/{len(concepts)} concepts have aliases mapped."
            )
        if taxonomy.ambiguous_aliases:
            am_score += 20
            findings["alias_management"].append(
                f"{len(taxonomy.ambiguous_aliases)} ambiguous-alias resolution rules defined."
            )
        scores["alias_management"] = round(min(100.0, am_score), 1)

        # 4. Context Rules — require both single- and multi-entity coverage
        if taxonomy.context_rules:
            seen_types = {r.rule_type for r in taxonomy.context_rules}
            cr_score = min(60.0, len(taxonomy.context_rules) * 30)
            findings["context_rules"].append(
                f"{len(taxonomy.context_rules)} context rules defined."
            )
            if {"single_entity", "multi_entity"}.issubset(seen_types):
                cr_score += 40
                findings["context_rules"].append("Both single- and multi-entity rules present.")
        else:
            cr_score = 0.0
            findings["context_rules"].append("No context rules defined.")
        scores["context_rules"] = round(min(100.0, cr_score), 1)

        # 5. Taxonomy Structure — coverage by populated groups, not raw count
        if taxonomy.concept_groups:
            populated = [g for g in taxonomy.concept_groups if g.get("concepts")]
            described = [g for g in populated if g.get("description")]
            named = [g for g in populated if g.get("name")]
            ts_score = 0.0
            ts_score += min(50.0, len(populated) * 8)  # cap at ~6 populated groups
            ts_score += min(25.0, (len(named) / max(len(populated), 1)) * 25)
            ts_score += min(25.0, (len(described) / max(len(populated), 1)) * 25)
            findings["taxonomy_structure"].append(
                f"{len(populated)}/{len(taxonomy.concept_groups)} groups have concepts; "
                f"{len(described)} have descriptions."
            )
        else:
            ts_score = 0.0
            findings["taxonomy_structure"].append("No concept groups defined.")
        scores["taxonomy_structure"] = round(min(100.0, ts_score), 1)

        # 6. Data Type Standardization
        if not concepts:
            dt_score = 0.0
        else:
            with_types = sum(1 for c in concepts if c.accepted_types)
            dt_score = (with_types / len(concepts)) * 70
            findings["data_type_standardization"].append(
                f"{with_types}/{len(concepts)} concepts declare accepted_types."
            )
        if taxonomy.datetime_standards:
            dt_score += 30
            findings["data_type_standardization"].append("Datetime standards defined.")
        scores["data_type_standardization"] = round(min(100.0, dt_score), 1)

        # 7. Governance Readiness
        gr_score = 0.0
        if taxonomy.metadata.get("owner"):
            gr_score += 20
            findings["governance_readiness"].append("Data owner defined.")
        if taxonomy.metadata.get("steward"):
            gr_score += 20
            findings["governance_readiness"].append("Data steward defined.")
        if taxonomy.metadata.get("version"):
            gr_score += 10
            findings["governance_readiness"].append("Versioning declared.")
        if taxonomy.ai_agent_instructions:
            gr_score += 20
            findings["governance_readiness"].append("AI agent instructions are present.")
        if taxonomy.validation_rules:
            gr_score += 30
            findings["governance_readiness"].append(
                f"{len(taxonomy.validation_rules)} machine-readable validation rules."
            )
        scores["governance_readiness"] = round(min(100.0, gr_score), 1)

        # 8. Lake/Platform Alignment — context-aware against architectural pattern
        la_score = 0.0
        zones = taxonomy.lake_standards.zones if taxonomy.lake_standards else []
        if zones:
            la_score += min(50.0, len(zones) * 18)
            findings["lake_platform_alignment"].append(
                f"{len(zones)} lake zones defined."
            )
            if all(z.naming_pattern for z in zones):
                la_score += 15
                findings["lake_platform_alignment"].append("All zones declare a naming pattern.")
            # Medallion pattern: expected zones must be present
            if profile.pattern.name == "medallion" and profile.pattern.expected_zones:
                declared = {(z.name or "").lower() for z in zones}
                missing = [z for z in profile.pattern.expected_zones if z not in declared]
                if missing:
                    findings["lake_platform_alignment"].append(
                        f"Medallion pattern requires zones {list(profile.pattern.expected_zones)}; "
                        f"missing: {missing}."
                    )
                    la_score -= min(15, len(missing) * 5)
                else:
                    la_score += 15
                    findings["lake_platform_alignment"].append("Medallion zones complete.")
        else:
            # No zones, but pattern is Inmon — that's intentional, not penalized hard
            if profile.pattern.name == "inmon":
                findings["lake_platform_alignment"].append(
                    "Inmon pattern detected — zones not expected; relying on schema design."
                )
                la_score = 60.0
        if taxonomy.lake_standards and taxonomy.lake_standards.project_structure:
            la_score += 20
            findings["lake_platform_alignment"].append("Project structure naming patterns defined.")
        scores["lake_platform_alignment"] = round(max(0.0, min(100.0, la_score)), 1)

        # Surface the resolved profile in the metadata of the score
        findings.setdefault("naming_consistency", []).append(
            f"Profile resolved: cloud={profile.cloud.cloud}, "
            f"pattern={profile.pattern.name}, declared={profile.declared}."
        )

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
            benchmark_comparison=self._get_benchmark(),
        )

    def _calculate_maturity_level(self, score: float) -> Tuple[int, str]:
        if score <= 20:
            return 1, "Initial"
        if score <= 40:
            return 2, "Managed"
        if score <= 60:
            return 3, "Defined"
        if score <= 80:
            return 4, "Measured"
        return 5, "Optimized"

    def _generate_gaps(self, scores: Dict[str, float]) -> List[TaxonomyGap]:
        gaps: List[TaxonomyGap] = []
        for dim, score in scores.items():
            if score < 50:
                gaps.append(TaxonomyGap(
                    dimension=dim,
                    severity="critical" if score < 20 else "major",
                    current_state=f"Score is low ({score:.1f}/100).",
                    target_state="Establish formalized rules and comprehensive mapping.",
                    effort_estimate="high",
                ))
            elif score < 80:
                gaps.append(TaxonomyGap(
                    dimension=dim,
                    severity="minor",
                    current_state=f"Score is acceptable ({score:.1f}/100) but lacks full coverage.",
                    target_state="Refine edge cases and ensure 100% metadata coverage.",
                    effort_estimate="medium",
                ))
        return gaps

    def _generate_recommendations(self, gaps: List[TaxonomyGap]) -> List[TaxonomyRecommendation]:
        recs: List[TaxonomyRecommendation] = []
        sorted_gaps = sorted(gaps, key=lambda g: _SEVERITY_ORDER.get(g.severity, 99))
        for priority, gap in enumerate(sorted_gaps, start=1):
            recs.append(TaxonomyRecommendation(
                priority=priority,
                title=f"Improve {gap.dimension.replace('_', ' ').title()}",
                description=f"Move from current state to target: {gap.target_state}",
                dimension=gap.dimension,
                expected_score_impact=10.0 if gap.severity == "critical" else 5.0,
                effort=gap.effort_estimate,
                timeline="short" if gap.severity == "critical" else "medium",
            ))
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
