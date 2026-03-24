"""Issue intake and triage for data governance stewardship."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from data_governance.rag_discovery.providers.base import LLMProvider, LLMResponse
except ImportError:
    LLMProvider = None
    LLMResponse = None

from data_governance.data_steward.models import (
    DataIssue, IssueCategory, IssueSeverity, IssueStatus, StewardAssignment
)


class IssueTriager:
    """Classifies and triages data issues using LLM with rule-based fallback."""

    # Keyword map for rule-based fallback
    CATEGORY_KEYWORDS = {
        "quality": ["vazio", "nulo", "null", "duplicado", "invalido", "errado", "inconsistente", "empty", "duplicate", "invalid", "wrong", "broken", "missing value"],
        "metadata": ["definicao", "significado", "descricao", "glossario", "definition", "meaning", "what is", "what does", "describe"],
        "ownership": ["responsavel", "dono", "owner", "quem aprova", "who owns", "who approves", "accountable"],
        "compliance": ["lgpd", "gdpr", "sensivel", "pii", "vazamento", "sensitive", "leak", "exposed", "regulat"],
        "lineage": ["origem", "fonte", "sistema correto", "de onde vem", "source", "lineage", "upstream", "downstream", "pipeline"],
        "change_request": ["alterar", "mudar", "atualizar", "remover", "adicionar", "change", "update", "modify", "remove", "add", "deprecate"],
    }

    SEVERITY_KEYWORDS = {
        "critical": ["urgente", "critico", "producao", "urgent", "critical", "production", "outage", "break", "down"],
        "high": ["importante", "impacto", "kpi", "receita", "important", "impact", "revenue", "financial"],
        "medium": ["moderado", "medium", "moderate", "review"],
        "low": ["menor", "cosmetic", "minor", "low", "nice to have"],
    }

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def triage(self, raw_description: str, context: Optional[Dict[str, Any]] = None) -> DataIssue:
        """
        Receive free-text description, return triaged DataIssue.
        Uses LLM when available, falls back to keyword matching.
        """
        if self.llm_provider is not None:
            return self._triage_with_llm(raw_description, context)
        return self._triage_rule_based(raw_description, context)

    def _triage_with_llm(self, raw_description, context):
        """LLM-powered triage."""
        system_prompt = """Voce e um assistente de triagem de issues de governanca de dados.
Analise a descricao do problema e retorne um JSON com:
{
    "title": "titulo curto e descritivo",
    "category": "quality|metadata|ownership|compliance|lineage|change_request",
    "severity": "critical|high|medium|low",
    "domain": "dominio provavel ou null",
    "dataset": "dataset provavel ou null",
    "attribute": "atributo provavel ou null",
    "root_cause_hypothesis": "hipotese de causa raiz",
    "suggested_next_steps": ["passo 1", "passo 2", "passo 3"],
    "sla_hours": numero_de_horas_sugerido
}
Responda APENAS com JSON valido, sem markdown."""

        user_prompt = f"Issue reportada:\n{raw_description}"
        if context:
            user_prompt += f"\n\nContexto adicional:\n{json.dumps(context, ensure_ascii=False)}"

        try:
            response = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000,
            )
            parsed = json.loads(response.content)
            # Build DataIssue from LLM response
            now = datetime.now().isoformat()
            return DataIssue(
                issue_id=str(uuid.uuid4())[:12],
                title=parsed.get("title", raw_description[:80]),
                raw_description=raw_description,
                category=parsed.get("category", IssueCategory.QUALITY.value),
                severity=parsed.get("severity", IssueSeverity.MEDIUM.value),
                status=IssueStatus.TRIAGED.value,
                domain=parsed.get("domain"),
                dataset=parsed.get("dataset"),
                attribute=parsed.get("attribute"),
                probable_owner=None,
                probable_steward=None,
                root_cause_hypothesis=parsed.get("root_cause_hypothesis"),
                suggested_next_steps=parsed.get("suggested_next_steps", []),
                sla_hours=parsed.get("sla_hours"),
                source_agent_findings={},
                created_at=now,
                updated_at=None,
                resolved_at=None,
                resolution_notes=None,
            )
        except Exception:
            # Fallback to rule-based on any LLM failure
            return self._triage_rule_based(raw_description, context)

    def _triage_rule_based(self, raw_description, context):
        """Rule-based fallback using keyword matching."""
        text = raw_description.lower()

        # Classify category
        category = IssueCategory.QUALITY.value  # default
        best_score = 0
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                category = cat

        # Classify severity
        severity = IssueSeverity.MEDIUM.value
        best_score = 0
        for sev, keywords in self.SEVERITY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                severity = sev

        # Generate title (first 80 chars)
        title = raw_description.strip().split('\n')[0][:80]

        # Basic next steps based on category
        next_steps_map = {
            "quality": ["Verificar metricas de qualidade do dataset", "Consultar Quality Agent para diagnostico", "Contatar steward do dominio"],
            "metadata": ["Consultar glossario de negocio", "Verificar catalogo de metadados", "Alinhar definicao com owner do dominio"],
            "ownership": ["Consultar matriz de responsabilidades", "Verificar assignments do dominio", "Escalar para Data Governance Office"],
            "compliance": ["Executar classificacao de sensibilidade", "Verificar politicas de acesso", "Contatar DPO/Compliance Officer"],
            "lineage": ["Mapear linhagem do dataset", "Identificar sistema-fonte oficial", "Consultar Lineage Agent"],
            "change_request": ["Documentar mudanca solicitada", "Avaliar impacto downstream", "Submeter para aprovacao"],
        }

        sla_map = {"critical": 4, "high": 24, "medium": 72, "low": 168}

        now = datetime.now().isoformat()
        return DataIssue(
            issue_id=str(uuid.uuid4())[:12],
            title=title,
            raw_description=raw_description,
            category=category,
            severity=severity,
            status=IssueStatus.TRIAGED.value,
            domain=None,
            dataset=None,
            attribute=None,
            probable_owner=None,
            probable_steward=None,
            root_cause_hypothesis=None,
            suggested_next_steps=next_steps_map.get(category, []),
            sla_hours=sla_map.get(severity, 72),
            source_agent_findings={},
            created_at=now,
            updated_at=None,
            resolved_at=None,
            resolution_notes=None,
        )

    def enrich_with_agent_findings(self, issue: DataIssue,
                                    quality_report=None,
                                    classification_report=None) -> DataIssue:
        """Enrich issue with evidence from other agents."""
        findings = dict(issue.source_agent_findings)

        if quality_report is not None:
            findings["quality"] = {
                "overall_score": getattr(quality_report, "overall_score", None),
                "overall_status": getattr(quality_report, "overall_status", None),
                "alerts": getattr(quality_report, "alerts", []),
            }
            # Auto-elevate severity if quality is critical
            if getattr(quality_report, "overall_status", "") == "failed" and issue.severity != IssueSeverity.CRITICAL.value:
                issue.severity = IssueSeverity.HIGH.value

        if classification_report is not None:
            findings["classification"] = {
                "overall_sensitivity": getattr(classification_report, "overall_sensitivity", None),
                "pii_columns": getattr(classification_report, "pii_columns", []),
            }
            # Auto-elevate if PII is involved
            pii = getattr(classification_report, "pii_columns", [])
            if pii and issue.category == IssueCategory.QUALITY.value:
                issue.severity = IssueSeverity.HIGH.value

        issue.source_agent_findings = findings
        issue.updated_at = datetime.now().isoformat()
        return issue

    def suggest_owner(self, issue: DataIssue,
                      assignments: List[StewardAssignment]) -> DataIssue:
        """Suggest probable owner/steward based on domain/dataset from assignments."""
        for a in assignments:
            if not a.is_active:
                continue
            domain_match = issue.domain and a.domain.lower() == issue.domain.lower()
            dataset_match = issue.dataset and issue.dataset in a.datasets
            if domain_match or dataset_match:
                if a.role == "data_owner":
                    issue.probable_owner = a.person_name
                elif a.role == "data_steward":
                    issue.probable_steward = a.person_name
        issue.updated_at = datetime.now().isoformat()
        return issue
