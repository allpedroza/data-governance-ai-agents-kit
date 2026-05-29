# /// script
# dependencies = []
# ///
"""
Discovery Report models — Output structures for the 6-step guided discovery.

These models represent the structured output of a guided data discovery session,
covering business context, identified needs, viability analysis, and delivery plan.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiscoveryNeed:
    """Uma necessidade de dado identificada durante o discovery.

    Classifica cada necessidade como:
    - asset_existente: dado que já existe em uma tabela no catálogo
    - métrica_derivada: cálculo derivado de tabelas existentes (dbt metric, LookML, etc.)
    - gap: dado que não foi encontrado em nenhuma fonte conhecida
    """
    name: str                           # "Taxa de Conversão", "Região de Vendas"
    need_type: str                      # "asset_existente", "métrica_derivada", "gap"
    description: str = ""               # Descrição da necessidade
    matched_table: Optional[str] = None # "marts.fct_orders" ou None
    maturity_level: int = 0             # 0-4 (MATURITY_* constants)
    classification: str = ""            # "público", "confidencial", etc.
    quality_score: Optional[float] = None
    freshness_hours: Optional[float] = None
    lineage_depth: int = 0              # Quantos hops até a fonte
    relevance_score: float = 0.0        # Score de relevância da busca semântica

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'need_type': self.need_type,
            'description': self.description,
            'matched_table': self.matched_table,
            'maturity_level': self.maturity_level,
            'classification': self.classification,
            'quality_score': self.quality_score,
            'freshness_hours': self.freshness_hours,
            'lineage_depth': self.lineage_depth,
            'relevance_score': self.relevance_score,
        }


@dataclass
class DeliveryPlan:
    """Plano de entrega em versões (output da Etapa 6).

    Separa o que pode ser entregue rapidamente (dados com maturidade >= L3)
    do que precisa trabalho adicional de ingestão/modelagem.
    """
    v1_scope: str = ""                  # Descrição do scope V1
    v1_tables: List[str] = field(default_factory=list)  # Tabelas prontas
    v1_effort: str = "baixo"            # "baixo", "médio", "alto"
    v2_scope: str = ""                  # Descrição do scope V2
    v2_tables: List[str] = field(default_factory=list)  # Tabelas que precisam trabalho
    v2_effort: str = "médio"            # "baixo", "médio", "alto"
    gaps: List[str] = field(default_factory=list)  # Dados que não existem em nenhum lugar

    def to_dict(self) -> Dict[str, Any]:
        return {
            'v1_scope': self.v1_scope,
            'v1_tables': self.v1_tables,
            'v1_effort': self.v1_effort,
            'v2_scope': self.v2_scope,
            'v2_tables': self.v2_tables,
            'v2_effort': self.v2_effort,
            'gaps': self.gaps,
        }


@dataclass
class DiscoveryReport:
    """Relatório completo de Data Discovery (output do guided_discovery).

    Cobre as 6 etapas do framework refinado:
    1. business_context, pain_point, stakeholders
    2-3. needs (DiscoveryNeed list)
    4. risks, pii_warnings
    5. technical_summary
    6. delivery_plan
    """
    # Etapa 1: Contexto de Negócio
    business_context: str = ""
    pain_point: str = ""
    stakeholders: List[str] = field(default_factory=list)

    # Etapa 2-3: Necessidades e Inventário
    needs: List[DiscoveryNeed] = field(default_factory=list)

    # Etapa 4: Viabilidade e Riscos
    risks: List[str] = field(default_factory=list)
    pii_warnings: List[str] = field(default_factory=list)

    # Etapa 5: Assessment Técnico
    technical_summary: Dict[str, Any] = field(default_factory=dict)

    # Etapa 6: Plano de Entrega
    delivery_plan: Optional[DeliveryPlan] = None

    # Metadados do processo
    llm_answer: str = ""                # Resposta narrativa completa do LLM
    confidence: float = 0.0
    latency_ms: int = 0
    tables_found: int = 0
    tables_validated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'business_context': self.business_context,
            'pain_point': self.pain_point,
            'stakeholders': self.stakeholders,
            'needs': [n.to_dict() for n in self.needs],
            'risks': self.risks,
            'pii_warnings': self.pii_warnings,
            'technical_summary': self.technical_summary,
            'delivery_plan': self.delivery_plan.to_dict() if self.delivery_plan else None,
            'llm_answer': self.llm_answer,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'tables_found': self.tables_found,
            'tables_validated': self.tables_validated,
        }

    def to_markdown(self) -> str:
        """Gera relatório em formato markdown legível."""
        lines = [
            "# Relatório de Data Discovery",
            "",
            "## 1. Contexto de Negócio",
            f"**Processo:** {self.business_context}",
            f"**Dor central:** {self.pain_point}",
        ]

        if self.stakeholders:
            lines.append(f"**Stakeholders:** {', '.join(self.stakeholders)}")

        # Necessidades
        lines.extend(["", "## 2. Necessidades Identificadas", ""])

        if self.needs:
            lines.append("| Tipo | Nome | Tabela | Maturidade | Quality | Classificação |")
            lines.append("|------|------|--------|------------|---------|---------------|")
            for need in self.needs:
                from .models import MATURITY_LABELS
                mat_label = MATURITY_LABELS.get(need.maturity_level, "?")
                quality = f"{need.quality_score:.0%}" if need.quality_score is not None else "—"
                table = need.matched_table or "—"
                lines.append(
                    f"| {need.need_type} | {need.name} | {table} | "
                    f"{mat_label} | {quality} | {need.classification or '—'} |"
                )
        else:
            lines.append("_Nenhuma necessidade identificada._")

        # Riscos
        if self.risks or self.pii_warnings:
            lines.extend(["", "## 3. Viabilidade e Riscos", ""])
            for risk in self.risks:
                lines.append(f"- ⚠️ {risk}")
            for pii in self.pii_warnings:
                lines.append(f"- 🔒 {pii}")

        # Plano de entrega
        if self.delivery_plan:
            dp = self.delivery_plan
            lines.extend([
                "", "## 4. Plano de Entrega", "",
                f"### V1 — Entrega Rápida (esforço: {dp.v1_effort})",
                f"{dp.v1_scope}",
            ])
            if dp.v1_tables:
                lines.append(f"**Tabelas:** {', '.join(dp.v1_tables)}")

            lines.extend([
                "",
                f"### V2 — Entrega Completa (esforço: {dp.v2_effort})",
                f"{dp.v2_scope}",
            ])
            if dp.v2_tables:
                lines.append(f"**Tabelas:** {', '.join(dp.v2_tables)}")

            if dp.gaps:
                lines.extend(["", "### Gaps (dados inexistentes)"])
                for gap in dp.gaps:
                    lines.append(f"- ❌ {gap}")

        # Narrativa LLM
        if self.llm_answer:
            lines.extend(["", "---", "", "## Análise do Agente", "", self.llm_answer])

        # Metadados
        lines.extend([
            "", "---",
            f"_Confiança: {self.confidence:.0%} | "
            f"Tabelas encontradas: {self.tables_found} | "
            f"Validadas: {self.tables_validated} | "
            f"Latência: {self.latency_ms}ms_"
        ])

        return "\n".join(lines)
