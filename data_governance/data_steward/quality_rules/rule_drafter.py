"""Draft data quality rules in business language + technical expression."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from data_governance.rag_discovery.providers.base import LLMProvider
except ImportError:
    LLMProvider = None

from data_governance.data_steward.models import QualityRuleDraft, QualityRuleStatus


class QualityRuleDrafter:
    """Suggests quality rules in business language + technical expression.

    Transforms ambiguity into operational rules. The steward reviews and
    approves before any rule becomes active.
    """

    RULE_TEMPLATES = {
        "completeness": {
            "business": "{attribute} deve estar preenchido para todos os registros ativos",
            "technical": "{attribute} IS NOT NULL AND {attribute} != ''",
        },
        "uniqueness": {
            "business": "{attribute} deve ser unico por registro",
            "technical": "COUNT(*) = COUNT(DISTINCT {attribute})",
        },
        "validity": {
            "business": "{attribute} deve conter apenas valores validos do dominio permitido",
            "technical": "{attribute} IN (SELECT valid_value FROM domain_values WHERE domain = '{domain}')",
        },
        "consistency": {
            "business": "{attribute} deve ser consistente entre sistemas de origem e destino",
            "technical": "source.{attribute} = target.{attribute}",
        },
        "timeliness": {
            "business": "Dados devem ser atualizados dentro do SLA de {sla} horas",
            "technical": "DATEDIFF(hour, MAX(updated_at), CURRENT_TIMESTAMP) <= {sla}",
        },
        "referential_integrity": {
            "business": "{attribute} deve referenciar um registro valido na tabela mestre",
            "technical": "{attribute} IN (SELECT id FROM master_table)",
        },
    }

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def draft_rules(
        self,
        dataset: str,
        domain: str,
        columns: Optional[List[Dict[str, Any]]] = None,
        quality_report=None,
        enrichment_result=None,
    ) -> List[QualityRuleDraft]:
        """Suggest quality rules based on column profiles and quality reports.

        Args:
            dataset: Name of the dataset.
            domain: Governance domain (e.g. ``"finance"``).
            columns: Column metadata list, each with ``name``, ``type``,
                ``nullable``, and optionally ``sample_values``.
            quality_report: ``QualityReport`` from DataQualityAgent.
            enrichment_result: ``EnrichmentResult`` from MetadataEnrichmentAgent.
        """
        if self.llm_provider is not None:
            return self._draft_with_llm(
                dataset, domain, columns, quality_report, enrichment_result
            )
        return self._draft_rule_based(dataset, domain, columns, quality_report)

    def _draft_with_llm(
        self, dataset, domain, columns, quality_report, enrichment_result
    ):
        system_prompt = (
            "Voce e um especialista em regras de data quality.\n"
            "Para cada coluna relevante, sugira regras em formato JSON array:\n"
            "[\n"
            '  {\n'
            '    "business_description": "descricao em linguagem de negocio (PT-BR)",\n'
            '    "technical_expression": "expressao SQL/tecnica candidata",\n'
            '    "dimension": "completeness|uniqueness|validity|consistency|timeliness|referential_integrity",\n'
            '    "attribute": "nome_da_coluna ou null se regra de tabela",\n'
            '    "severity": "critical|high|medium|low"\n'
            "  }\n"
            "]\n"
            "Foque em regras praticas e acionaveis. Responda APENAS com JSON array valido."
        )

        context_parts = [f"Dataset: {dataset}", f"Dominio: {domain}"]

        if columns:
            cols_text = "\n".join(
                [
                    f"- {c.get('name')}: {c.get('type', 'unknown')}, nullable={c.get('nullable', True)}"
                    for c in columns[:30]
                ]
            )
            context_parts.append(f"Colunas:\n{cols_text}")

        if quality_report:
            alerts = getattr(quality_report, "alerts", [])
            if alerts:
                alerts_text = "\n".join(
                    [
                        f"- {a.get('rule_name', 'N/A')}: {a.get('message', 'N/A')}"
                        for a in alerts[:10]
                    ]
                )
                context_parts.append(f"Alertas de qualidade atuais:\n{alerts_text}")
            dims = getattr(quality_report, "dimensions", {})
            if dims:
                dims_text = "\n".join(
                    [f"- {k}: score={v.get('score', 'N/A')}" for k, v in dims.items()]
                )
                context_parts.append(f"Dimensoes de qualidade:\n{dims_text}")

        try:
            response = self.llm_provider.generate(
                prompt="\n\n".join(context_parts),
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000,
            )
            parsed = json.loads(response.content)
            now = datetime.now().isoformat()

            rules = []
            for r in parsed:
                rules.append(
                    QualityRuleDraft(
                        rule_id=str(uuid.uuid4())[:12],
                        business_description=r.get("business_description", ""),
                        technical_expression=r.get("technical_expression", ""),
                        dimension=r.get("dimension", "validity"),
                        dataset=dataset,
                        attribute=r.get("attribute"),
                        domain=domain,
                        severity=r.get("severity", "medium"),
                        current_deviation=None,
                        deviation_explanation=None,
                        status=QualityRuleStatus.DRAFT.value,
                        approved_by=None,
                        version=1,
                        created_at=now,
                        updated_at=None,
                    )
                )
            return rules
        except Exception:
            return self._draft_rule_based(dataset, domain, columns, quality_report)

    def _draft_rule_based(self, dataset, domain, columns, quality_report):
        """Generate basic rules from templates."""
        rules: List[QualityRuleDraft] = []
        now = datetime.now().isoformat()
        cols = columns or []

        for col in cols:
            col_name = col.get("name", "unknown")
            nullable = col.get("nullable", True)

            # Completeness rule for non-nullable columns
            if not nullable:
                tpl = self.RULE_TEMPLATES["completeness"]
                rules.append(
                    QualityRuleDraft(
                        rule_id=str(uuid.uuid4())[:12],
                        business_description=tpl["business"].format(
                            attribute=col_name
                        ),
                        technical_expression=tpl["technical"].format(
                            attribute=col_name
                        ),
                        dimension="completeness",
                        dataset=dataset,
                        attribute=col_name,
                        domain=domain,
                        severity="high",
                        current_deviation=None,
                        deviation_explanation=None,
                        status=QualityRuleStatus.DRAFT.value,
                        approved_by=None,
                        version=1,
                        created_at=now,
                        updated_at=None,
                    )
                )

            # Uniqueness rule for ID-like columns
            id_patterns = ["_id", "id_", "cpf", "cnpj", "email", "code", "codigo"]
            if any(p in col_name.lower() for p in id_patterns):
                tpl = self.RULE_TEMPLATES["uniqueness"]
                sev = (
                    "critical"
                    if any(
                        k in col_name.lower() for k in ("cpf", "cnpj", "email")
                    )
                    else "high"
                )
                rules.append(
                    QualityRuleDraft(
                        rule_id=str(uuid.uuid4())[:12],
                        business_description=tpl["business"].format(
                            attribute=col_name
                        ),
                        technical_expression=tpl["technical"].format(
                            attribute=col_name
                        ),
                        dimension="uniqueness",
                        dataset=dataset,
                        attribute=col_name,
                        domain=domain,
                        severity=sev,
                        current_deviation=None,
                        deviation_explanation=None,
                        status=QualityRuleStatus.DRAFT.value,
                        approved_by=None,
                        version=1,
                        created_at=now,
                        updated_at=None,
                    )
                )

        # Rules from quality report alerts
        if quality_report:
            for alert in getattr(quality_report, "alerts", [])[:5]:
                rules.append(
                    QualityRuleDraft(
                        rule_id=str(uuid.uuid4())[:12],
                        business_description=(
                            f"Investigar alerta: {alert.get('message', 'Desvio detectado')}"
                        ),
                        technical_expression=(
                            f"-- Baseado em alerta: {alert.get('rule_name', 'N/A')}"
                        ),
                        dimension=alert.get("dimension", "validity"),
                        dataset=dataset,
                        attribute=alert.get("column"),
                        domain=domain,
                        severity=alert.get("level", "medium"),
                        current_deviation=alert,
                        deviation_explanation=alert.get("message"),
                        status=QualityRuleStatus.DRAFT.value,
                        approved_by=None,
                        version=1,
                        created_at=now,
                        updated_at=None,
                    )
                )

        return rules

    def explain_deviation(self, rule: QualityRuleDraft, quality_report=None) -> str:
        """Explain a quality deviation in business language."""
        if self.llm_provider is not None and rule.current_deviation:
            try:
                response = self.llm_provider.generate(
                    prompt=(
                        f"Regra: {rule.business_description}\n"
                        f"Desvio encontrado: {json.dumps(rule.current_deviation, ensure_ascii=False)}\n"
                        f"Dataset: {rule.dataset}, Atributo: {rule.attribute}"
                    ),
                    system_prompt=(
                        "Explique o desvio de qualidade em linguagem simples de "
                        "negocio (PT-BR), em 2-3 frases. Inclua o impacto "
                        "potencial e sugestao de correcao."
                    ),
                    temperature=0.3,
                    max_tokens=300,
                )
                return response.content.strip()
            except Exception:
                pass

        if rule.current_deviation:
            msg = rule.current_deviation.get("message", "Desvio detectado")
            return (
                f"Foi detectado um desvio na regra '{rule.business_description}': "
                f"{msg}. Recomenda-se investigacao pelo steward do dominio "
                f"{rule.domain}."
            )
        return (
            f"Nenhum desvio registrado para a regra '{rule.business_description}'."
        )

    def suggest_dimensions(
        self, dataset: str, columns: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest relevant quality dimensions for a dataset."""
        dims = {"completeness", "validity"}
        for col in columns:
            name = col.get("name", "").lower()
            if any(p in name for p in ("_id", "id_", "cpf", "cnpj", "email")):
                dims.add("uniqueness")
            if any(p in name for p in ("date", "data", "timestamp", "updated")):
                dims.add("timeliness")
            if any(p in name for p in ("fk_", "ref_", "_ref")):
                dims.add("referential_integrity")
        return sorted(dims)
