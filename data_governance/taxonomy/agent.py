"""Taxonomy Evaluator, Scorer & Artifact Generator.

Thin orchestration layer exposing the operations a UI (Streamlit tab) or a
sibling agent (Metadata Enrichment, Data Quality, Data Contracts) needs to
consume the canonical taxonomy.

Product flow exposed here:

1. ``load_*`` — identify the *current* taxonomy from a YAML source.
2. ``score_taxonomy`` — evaluate it against the 8-dimension best-practice
   framework and produce a numeric score + maturity level.
3. ``generate_html_artifact`` — render an AS-IS visual report.
4. ``build_improvement_plan`` / ``generate_improvement_plan_markdown`` —
   produce a structured plan with concrete actions grouped by dimension.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .models import TaxonomyDocument
from .scorer import TaxonomyRecommendation, TaxonomyScore, TaxonomyScorer


# ---------------------------------------------------------------------------
# Improvement Plan model
# ---------------------------------------------------------------------------
@dataclass
class ImprovementAction:
    """A concrete, actionable step to lift one dimension's score."""
    title: str
    detail: str
    effort: str           # "low" | "medium" | "high"
    expected_impact: str  # "low" | "medium" | "high"


@dataclass
class DimensionPlan:
    dimension: str
    current_score: float
    target_score: float
    findings: List[str] = field(default_factory=list)
    actions: List[ImprovementAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "current_score": self.current_score,
            "target_score": self.target_score,
            "findings": self.findings,
            "actions": [a.__dict__ for a in self.actions],
        }


@dataclass
class ImprovementPlan:
    overall_score: float
    target_overall_score: float
    maturity_level: int
    maturity_label: str
    dimensions: List[DimensionPlan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "target_overall_score": self.target_overall_score,
            "maturity_level": self.maturity_level,
            "maturity_label": self.maturity_label,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


# Concrete action templates per dimension. Each tuple is (current<threshold, action)
# Actions only fire when the dimension is below the threshold listed.
_DIMENSION_ACTION_LIBRARY: Dict[str, List[tuple]] = {
    "naming_consistency": [
        (50, ImprovementAction(
            "Formalizar regras de casing",
            "Documente em YAML/Confluence o padrão por objeto (colunas, tabelas, CTEs) e publique exemplos canônicos para revisão automática.",
            effort="low", expected_impact="high")),
        (80, ImprovementAction(
            "Catalogar acrônimos canônicos",
            "Liste todos os acrônimos de negócio (VIN, GUID, CPF, etc.) com regra de UPPER_CASE preservado dentro de PascalCase.",
            effort="low", expected_impact="medium")),
        (80, ImprovementAction(
            "Bloquear forbidden_forms via linter",
            "Integre as regras de forbidden_forms ao pre-commit/SQLFluff/dbt-checks para barrar mixed-case na origem.",
            effort="medium", expected_impact="high")),
    ],
    "definition_completeness": [
        (60, ImprovementAction(
            "Completar definições faltantes",
            "Para cada concept sem definição (ou com <15 caracteres), preencher business definition revisada com data steward.",
            effort="medium", expected_impact="high")),
        (90, ImprovementAction(
            "Vincular definições ao glossário corporativo",
            "Cruze o glossary com OpenMetadata/Atlas para garantir uma única definição autoritativa por termo.",
            effort="medium", expected_impact="medium")),
    ],
    "alias_management": [
        (40, ImprovementAction(
            "Mapear aliases dos principais source systems",
            "Para cada concept top-20 (por uso), levantar nomes equivalentes em sistemas de origem (CRM, ERP, billing) e registrar como aliases.",
            effort="high", expected_impact="high")),
        (80, ImprovementAction(
            "Documentar regras de resolução para aliases ambíguos",
            "Para aliases compartilhados entre entidades (Name, Country, ID), publicar matriz de resolução por contexto de tabela.",
            effort="medium", expected_impact="medium")),
    ],
    "context_rules": [
        (50, ImprovementAction(
            "Definir regras single-entity vs multi-entity",
            "Estabeleça quando prefixar (multi-entity tables) vs omitir (single-entity), com exemplos e anti-patterns documentados.",
            effort="low", expected_impact="high")),
        (90, ImprovementAction(
            "Adicionar rationale e exemplos a cada regra",
            "Inclua justificativa (rationale) e exemplos positivos/negativos para reduzir ambiguidade na aplicação por humanos e LLMs.",
            effort="low", expected_impact="medium")),
    ],
    "taxonomy_structure": [
        (50, ImprovementAction(
            "Mapear domínios faltantes",
            "Identifique áreas de negócio sem grupo correspondente (ex.: Finance, Logistics) e criar concept_groups dedicados.",
            effort="high", expected_impact="high")),
        (90, ImprovementAction(
            "Descrever cada grupo com pii_level e owner",
            "Adicione metadata (pii_level, owner, description) a cada concept_group para ativar buscas e classificação automática.",
            effort="medium", expected_impact="medium")),
    ],
    "data_type_standardization": [
        (60, ImprovementAction(
            "Padronizar accepted_types por concept",
            "Defina o conjunto de tipos aceitos (STRING/NUMERIC/TIMESTAMP) para cada concept, alinhado ao warehouse (BigQuery/Snowflake).",
            effort="medium", expected_impact="high")),
        (90, ImprovementAction(
            "Formalizar datetime_standards",
            "Documente timezone (UTC), formato ISO 8601 e nomes canônicos (CreatedAt, UpdatedAt, DeletedAt).",
            effort="low", expected_impact="medium")),
    ],
    "governance_readiness": [
        (40, ImprovementAction(
            "Atribuir owner e steward",
            "Defina owner técnico (engenharia) e steward de negócio responsáveis por aprovar mudanças na taxonomia.",
            effort="low", expected_impact="high")),
        (80, ImprovementAction(
            "Publicar AI agent instructions e validation_rules",
            "Garanta que cada release da taxonomia contém ai_agent_instructions (machine-readable) e validation_rules (regex/forbidden) para CI.",
            effort="medium", expected_impact="high")),
        (95, ImprovementAction(
            "Versionar a taxonomia com SemVer",
            "Adote SemVer e changelog para a taxonomia, com aprovação obrigatória de owner+steward para mudanças breaking.",
            effort="low", expected_impact="medium")),
    ],
    "lake_platform_alignment": [
        (50, ImprovementAction(
            "Definir lake zones (bronze/silver/gold)",
            "Documente as zonas do lake, padrões de naming por zona e políticas de retenção.",
            effort="medium", expected_impact="high")),
        (80, ImprovementAction(
            "Padronizar project_structure",
            "Defina pattern de naming para projetos/datasets (ex.: `{org}-{env}-{domain}`) com exemplos.",
            effort="low", expected_impact="medium")),
    ],
}


class TaxonomyAgent:
    """Taxonomy Evaluator, Scorer & Artifact Generator."""

    TARGET_DIMENSION_SCORE = 90.0
    TARGET_OVERALL_SCORE = 85.0

    def __init__(self, scorer: Optional[TaxonomyScorer] = None) -> None:
        self.scorer = scorer or TaxonomyScorer()

    # ------------------------------------------------------------------
    # Loading / persisting
    # ------------------------------------------------------------------
    def load_from_yaml(self, path: str) -> TaxonomyDocument:
        return TaxonomyDocument.from_yaml(path)

    def load_from_yaml_string(self, content: str) -> TaxonomyDocument:
        return TaxonomyDocument.from_yaml_string(content)

    def export_yaml(self, taxonomy: TaxonomyDocument, path: str) -> str:
        yaml_content = taxonomy.to_yaml()
        target = os.path.abspath(path)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(yaml_content)
        return yaml_content

    # ------------------------------------------------------------------
    # Scoring & artifacts
    # ------------------------------------------------------------------
    def score_taxonomy(self, taxonomy: TaxonomyDocument) -> TaxonomyScore:
        return self.scorer.score(taxonomy)

    def generate_delta(
        self, taxonomy: TaxonomyDocument, score: TaxonomyScore,
    ) -> List[TaxonomyRecommendation]:
        return score.recommendations

    def generate_html_artifact(
        self, taxonomy: TaxonomyDocument, score: TaxonomyScore,
    ) -> str:
        from .html_generator import generate_taxonomy_html
        return generate_taxonomy_html(taxonomy.to_dict(), score.to_dict())

    # ------------------------------------------------------------------
    # Validation helpers — entry point for sibling agents
    # ------------------------------------------------------------------
    def validate_columns(
        self, taxonomy: TaxonomyDocument, columns: Iterable[str],
    ) -> Dict[str, List[str]]:
        """Validate column names against the taxonomy.

        Returns a mapping of ``column -> [violation messages]``; columns
        without violations are omitted.
        """
        report: Dict[str, List[str]] = {}
        for col in columns:
            issues = taxonomy.validate_name(col, scope="column")
            if issues:
                report[col] = issues
        return report

    def validate_table_name(
        self, taxonomy: TaxonomyDocument, table_name: str,
    ) -> List[str]:
        return taxonomy.validate_name(table_name, scope="table")

    def map_columns_to_concepts(
        self, taxonomy: TaxonomyDocument, columns: Iterable[str],
    ) -> Dict[str, Optional[str]]:
        """Best-effort match between a column name and a canonical concept.

        Used by Metadata Enrichment to align inferred labels with the canonical
        vocabulary, and by Data Discovery to enrich search results.
        """
        mapping: Dict[str, Optional[str]] = {}
        for col in columns:
            concept = taxonomy.get_concept_by_name(col)
            mapping[col] = concept.name if concept else None
        return mapping

    def export_for_other_agents(self, taxonomy: TaxonomyDocument) -> Dict[str, Any]:
        """Compact bundle to be consumed by other governance agents."""
        return taxonomy.export_for_metadata_enrichment()

    # ------------------------------------------------------------------
    # Improvement plan
    # ------------------------------------------------------------------
    def build_improvement_plan(self, score: TaxonomyScore) -> ImprovementPlan:
        """Translate a score into a structured plan grouped by dimension.

        Each dimension gets the subset of action templates whose threshold the
        current score is below. Dimensions already above
        ``TARGET_DIMENSION_SCORE`` get a single "maintain" action so the plan
        explicitly acknowledges their good shape.
        """
        dimensions: List[DimensionPlan] = []
        for dim, current in score.dimension_scores.items():
            actions: List[ImprovementAction] = []
            for threshold, action in _DIMENSION_ACTION_LIBRARY.get(dim, []):
                if current < threshold:
                    actions.append(action)
            if not actions:
                actions.append(ImprovementAction(
                    title="Manter a dimensão como referência",
                    detail=(
                        "Score acima do alvo. Mantenha revisões trimestrais e "
                        "publique como benchmark para outras áreas."
                    ),
                    effort="low",
                    expected_impact="low",
                ))
            dimensions.append(DimensionPlan(
                dimension=dim,
                current_score=current,
                target_score=max(self.TARGET_DIMENSION_SCORE, current),
                findings=list(score.dimension_findings.get(dim, [])),
                actions=actions,
            ))

        # Order: largest gap first so the plan reads as an attack list
        dimensions.sort(key=lambda d: d.target_score - d.current_score, reverse=True)

        return ImprovementPlan(
            overall_score=score.overall_score,
            target_overall_score=max(self.TARGET_OVERALL_SCORE, score.overall_score),
            maturity_level=score.maturity_level,
            maturity_label=score.maturity_label,
            dimensions=dimensions,
        )

    def generate_improvement_plan_markdown(self, plan: ImprovementPlan) -> str:
        """Render the improvement plan as a portable Markdown document."""
        lines: List[str] = []
        lines.append("# Plano de Melhoria de Taxonomia")
        lines.append("")
        lines.append(
            f"**Score atual:** {plan.overall_score:.1f}/100  ·  "
            f"**Maturidade:** Nível {plan.maturity_level} — {plan.maturity_label}  ·  "
            f"**Meta:** {plan.target_overall_score:.1f}/100"
        )
        lines.append("")
        lines.append(
            "Este plano lista as ações recomendadas para cada uma das oito "
            "dimensões do framework de avaliação. As dimensões aparecem na "
            "ordem da maior lacuna até a menor."
        )
        lines.append("")
        for d in plan.dimensions:
            gap = d.target_score - d.current_score
            lines.append(f"## {d.dimension.replace('_', ' ').title()}")
            lines.append("")
            lines.append(
                f"- **Score atual:** {d.current_score:.1f}/100  ·  "
                f"**Meta:** {d.target_score:.1f}/100  ·  **Gap:** {gap:.1f} pts"
            )
            if d.findings:
                lines.append("- **Achados atuais:**")
                for finding in d.findings:
                    lines.append(f"    - {finding}")
            lines.append("")
            lines.append("### Ações propostas")
            lines.append("")
            for idx, action in enumerate(d.actions, start=1):
                lines.append(f"{idx}. **{action.title}**")
                lines.append(f"   - {action.detail}")
                lines.append(
                    f"   - _Esforço:_ {action.effort}  ·  "
                    f"_Impacto esperado:_ {action.expected_impact}"
                )
            lines.append("")
        return "\n".join(lines)

    def generate_current_state_report_markdown(
        self, taxonomy: TaxonomyDocument, score: TaxonomyScore,
    ) -> str:
        """Render the AS-IS report as a portable Markdown document."""
        meta = taxonomy.metadata
        concepts = taxonomy.get_all_concepts()
        lines: List[str] = []
        lines.append(f"# Relatório AS-IS — {meta.get('title', 'Taxonomia')}")
        lines.append("")
        lines.append(
            f"**Versão:** {meta.get('version', 'n/d')}  ·  "
            f"**Domínio:** {meta.get('domain', 'n/d')}  ·  "
            f"**Plataforma:** {meta.get('platform', 'n/d')}"
        )
        lines.append(
            f"**Owner:** {meta.get('owner', 'não definido')}  ·  "
            f"**Steward:** {meta.get('steward', 'não definido')}"
        )
        lines.append("")
        lines.append("## Inventário")
        lines.append("")
        lines.append(f"- **Conceitos mapeados:** {len(concepts)}")
        lines.append(f"- **Grupos de conceitos:** {len(taxonomy.concept_groups)}")
        lines.append(f"- **Aliases catalogados:** {taxonomy.get_total_aliases()}")
        lines.append(f"- **Regras de contexto:** {len(taxonomy.context_rules)}")
        lines.append(f"- **Aliases ambíguos resolvidos:** {len(taxonomy.ambiguous_aliases)}")
        lines.append(f"- **Regras de validação machine-readable:** {len(taxonomy.validation_rules)}")
        lines.append(f"- **Lake zones definidas:** {len(taxonomy.lake_standards.zones)}")
        lines.append("")
        lines.append("## Score por dimensão")
        lines.append("")
        lines.append("| Dimensão | Score | Benchmark |")
        lines.append("|---|---|---|")
        for dim, sc in score.dimension_scores.items():
            bench = score.benchmark_comparison.get(dim, "—")
            lines.append(
                f"| {dim.replace('_', ' ').title()} | {sc:.1f}/100 | {bench} |"
            )
        lines.append("")
        lines.append(
            f"**Score global:** {score.overall_score:.1f}/100 — "
            f"Nível {score.maturity_level} ({score.maturity_label})"
        )
        lines.append("")
        lines.append("## Grupos de conceitos")
        lines.append("")
        for group in taxonomy.concept_groups:
            if not isinstance(group, dict):
                continue
            lines.append(
                f"### {group.get('icon', '📄')} {group.get('name', '—')} "
                f"({len(group.get('concepts', []) or [])} concepts)"
            )
            if group.get("description"):
                lines.append(group["description"])
                lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # End-to-end product entry point
    # ------------------------------------------------------------------
    def run(self, yaml_path: str) -> Dict[str, Any]:
        """End-to-end: load → score → produce both reports + integration bundle.

        Returns a dictionary with every artifact the UI / downstream agents
        need, so callers don't have to wire the pipeline manually.
        """
        taxonomy = self.load_from_yaml(yaml_path)
        score = self.score_taxonomy(taxonomy)
        plan = self.build_improvement_plan(score)
        return {
            "taxonomy": taxonomy,
            "score": score,
            "plan": plan,
            "as_is_html": self.generate_html_artifact(taxonomy, score),
            "as_is_markdown": self.generate_current_state_report_markdown(taxonomy, score),
            "plan_markdown": self.generate_improvement_plan_markdown(plan),
            "integration_bundle": self.export_for_other_agents(taxonomy),
        }
