"""Human-readable impact and lineage explanation for data governance."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from data_governance.rag_discovery.providers.base import LLMProvider
except ImportError:
    LLMProvider = None

from data_governance.data_steward.models import ImpactReport


class ImpactExplainer:
    """Explains data change impact in business language, not technical graphs.

    When there is a change to a field, rule or definition the explainer
    answers: which reports/KPIs may break, which teams are impacted, which
    quality rules depend on the attribute, and which regulatory exceptions
    exist -- all in human-readable language.
    """

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def explain_impact(
        self,
        change_description: str,
        dataset: str,
        attribute: Optional[str] = None,
        domain: str = "",
        lineage_data: Optional[Dict[str, Any]] = None,
        quality_rules: Optional[List] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> ImpactReport:
        """Generate human-readable impact report for a proposed change.

        Args:
            change_description: Free-text description of the change.
            dataset: Affected dataset.
            attribute: Affected column/attribute (optional).
            domain: Governance domain.
            lineage_data: Output from ``DataLineageAgent.analyze_pipeline()``.
            quality_rules: ``QualityRuleDraft`` instances that may be affected.
            contracts: Data contract dicts that may be affected.
        """
        if self.llm_provider is not None:
            return self._explain_with_llm(
                change_description,
                dataset,
                attribute,
                domain,
                lineage_data,
                quality_rules,
                contracts,
            )
        return self._explain_rule_based(
            change_description,
            dataset,
            attribute,
            domain,
            lineage_data,
            quality_rules,
            contracts,
        )

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _explain_with_llm(
        self,
        change_description,
        dataset,
        attribute,
        domain,
        lineage_data,
        quality_rules,
        contracts,
    ):
        system_prompt = (
            "Voce e um analista de impacto de governanca de dados.\n"
            "Dada uma mudanca proposta, analise o impacto e retorne JSON:\n"
            "{\n"
            '  "affected_reports": ["relatorios/KPIs que podem quebrar"],\n'
            '  "affected_teams": ["times/areas impactadas"],\n'
            '  "affected_rules": ["regras de qualidade dependentes"],\n'
            '  "regulatory_exceptions": ["excecoes regulatorias"],\n'
            '  "risk_level": "high|medium|low",\n'
            '  "human_summary": "resumo executivo em PT-BR, 3-5 frases"\n'
            "}\n"
            "Responda APENAS com JSON valido."
        )

        context_parts = [
            f"Mudanca proposta: {change_description}",
            f"Dataset: {dataset}",
        ]
        if attribute:
            context_parts.append(f"Atributo: {attribute}")
        if domain:
            context_parts.append(f"Dominio: {domain}")

        if lineage_data:
            downstream = self._extract_downstream(lineage_data, dataset)
            if downstream:
                context_parts.append(
                    f"Assets downstream:\n{json.dumps(downstream, ensure_ascii=False)}"
                )

        if quality_rules:
            rules_text = "\n".join(
                [
                    f"- {getattr(r, 'business_description', str(r))}"
                    for r in quality_rules[:10]
                ]
            )
            context_parts.append(f"Regras de qualidade afetadas:\n{rules_text}")

        if contracts:
            contracts_text = "\n".join(
                [
                    f"- Contrato: {c.get('name', 'N/A')}, SLA: {c.get('sla', 'N/A')}"
                    for c in contracts[:5]
                ]
            )
            context_parts.append(f"Contratos de dados afetados:\n{contracts_text}")

        try:
            response = self.llm_provider.generate(
                prompt="\n\n".join(context_parts),
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1500,
            )
            parsed = json.loads(response.content)
            now = datetime.now().isoformat()
            return ImpactReport(
                report_id=str(uuid.uuid4())[:12],
                change_description=change_description,
                dataset=dataset,
                attribute=attribute,
                domain=domain,
                affected_reports=parsed.get("affected_reports", []),
                affected_teams=parsed.get("affected_teams", []),
                affected_rules=parsed.get("affected_rules", []),
                regulatory_exceptions=parsed.get("regulatory_exceptions", []),
                risk_level=parsed.get("risk_level", "medium"),
                human_summary=parsed.get("human_summary", ""),
                lineage_evidence=lineage_data or {},
                generated_at=now,
            )
        except Exception:
            return self._explain_rule_based(
                change_description,
                dataset,
                attribute,
                domain,
                lineage_data,
                quality_rules,
                contracts,
            )

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _explain_rule_based(
        self,
        change_description,
        dataset,
        attribute,
        domain,
        lineage_data,
        quality_rules,
        contracts,
    ):
        affected_reports: List[str] = []
        affected_teams: List[str] = []
        affected_rules: List[str] = []
        regulatory_exceptions: List[str] = []

        # Downstream from lineage
        if lineage_data:
            downstream = self._extract_downstream(lineage_data, dataset)
            team_keywords = {
                "Financeiro": ["finance", "financ", "contab", "revenue"],
                "Comercial/CRM": ["customer", "client", "crm", "sales", "vend"],
                "RH": ["hr", "rh", "people", "employee"],
                "Marketing": ["market", "campaign", "lead"],
                "Produto": ["product", "catalog"],
            }
            for asset in downstream:
                name = asset.get("name", "").lower()
                if any(
                    kw in name
                    for kw in ("report", "dash", "kpi", "bi_", "analytics")
                ):
                    affected_reports.append(asset.get("name", ""))
                for team, keywords in team_keywords.items():
                    if any(kw in name for kw in keywords):
                        affected_teams.append(team)

        # Quality rules
        if quality_rules:
            for rule in quality_rules:
                rule_attr = getattr(rule, "attribute", None)
                rule_ds = getattr(rule, "dataset", None)
                if (rule_attr and attribute and rule_attr == attribute) or (
                    rule_ds == dataset
                ):
                    affected_rules.append(
                        getattr(rule, "business_description", str(rule))
                    )

        # Regulatory implications
        sensitive_keywords = [
            "cpf", "cnpj", "email", "telefone", "endereco",
            "nome", "pii", "lgpd", "gdpr", "sensivel",
        ]
        change_lower = change_description.lower()
        attr_lower = (attribute or "").lower()
        if any(kw in change_lower or kw in attr_lower for kw in sensitive_keywords):
            regulatory_exceptions.append(
                "Possivel implicacao LGPD/GDPR - verificar com DPO"
            )

        # Risk level
        if affected_reports or regulatory_exceptions:
            risk_level = "high"
        elif affected_rules or affected_teams:
            risk_level = "medium"
        else:
            risk_level = "low"

        affected_teams = list(dict.fromkeys(affected_teams))  # deduplicate

        # Human summary
        parts = [f"A mudanca '{change_description}' no dataset '{dataset}' "]
        if affected_reports:
            parts.append(
                f"pode impactar {len(affected_reports)} relatorio(s)/KPI(s). "
            )
        if affected_teams:
            parts.append(
                f"Times potencialmente afetados: {', '.join(affected_teams)}. "
            )
        if affected_rules:
            parts.append(
                f"{len(affected_rules)} regra(s) de qualidade podem ser impactadas. "
            )
        if regulatory_exceptions:
            parts.append(
                "Ha implicacoes regulatorias que devem ser avaliadas. "
            )
        if not any([affected_reports, affected_teams, affected_rules]):
            parts.append(
                "nao apresenta impactos downstream significativos identificados. "
                "Recomenda-se validacao manual."
            )

        now = datetime.now().isoformat()
        return ImpactReport(
            report_id=str(uuid.uuid4())[:12],
            change_description=change_description,
            dataset=dataset,
            attribute=attribute,
            domain=domain,
            affected_reports=affected_reports,
            affected_teams=affected_teams,
            affected_rules=affected_rules,
            regulatory_exceptions=regulatory_exceptions,
            risk_level=risk_level,
            human_summary="".join(parts),
            lineage_evidence=lineage_data or {},
            generated_at=now,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_downstream(
        self, lineage_data: Dict[str, Any], dataset: str
    ) -> List[Dict[str, str]]:
        """Extract downstream assets from lineage analysis output."""
        downstream: List[Dict[str, str]] = []
        transformations = lineage_data.get("transformations", [])
        assets = lineage_data.get("assets", {})

        visited: set = set()
        queue = [dataset]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for t in transformations:
                source = t.get("source", "")
                target = t.get("target", "")
                if isinstance(source, dict):
                    source = source.get("name", "")
                if isinstance(target, dict):
                    target = target.get("name", "")
                if source == current and target not in visited:
                    queue.append(target)
                    asset_info = {"name": target, "type": "unknown"}
                    if target in assets:
                        a = assets[target]
                        if isinstance(a, dict):
                            asset_info["type"] = a.get("type", "unknown")
                    downstream.append(asset_info)
        return downstream

    def summarize_for_business(self, impact: ImpactReport) -> str:
        """Generate executive summary. Uses LLM if available."""
        if self.llm_provider is not None:
            try:
                response = self.llm_provider.generate(
                    prompt=(
                        "Resumo de impacto para apresentar ao negocio:\n"
                        f"{impact.to_json()}"
                    ),
                    system_prompt=(
                        "Gere um resumo executivo de 3-5 frases em PT-BR, "
                        "focado em impacto ao negocio. Sem jargao tecnico."
                    ),
                    temperature=0.3,
                    max_tokens=300,
                )
                return response.content.strip()
            except Exception:
                pass
        return impact.human_summary
