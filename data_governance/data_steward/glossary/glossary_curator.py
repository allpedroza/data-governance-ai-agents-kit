"""Assisted business glossary curation for data governance."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from data_governance.rag_discovery.providers.base import LLMProvider
except ImportError:
    LLMProvider = None

from data_governance.data_steward.models import GlossaryTerm, GlossaryTermStatus


class GlossaryCurator:
    """Curates business glossary terms - consolidates definitions, detects conflicts, proposes standards."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def curate_term(self, term_name: str,
                    sources: List[Dict[str, str]],
                    domain: str = "",
                    related_datasets: Optional[List[str]] = None,
                    related_attributes: Optional[List[str]] = None) -> GlossaryTerm:
        """
        Consolidate definitions from multiple sources and propose a standard definition.

        sources: [{"source": "SQL view comment", "definition": "..."}, {"source": "Dashboard label", "definition": "..."}]

        With LLM: generates proposed definition, business description, detects conflicts.
        Without LLM: picks first definition as proposed, lists all as candidates.
        """
        if self.llm_provider is not None:
            return self._curate_with_llm(term_name, sources, domain, related_datasets, related_attributes)
        return self._curate_rule_based(term_name, sources, domain, related_datasets, related_attributes)

    def _curate_with_llm(self, term_name, sources, domain, related_datasets, related_attributes):
        system_prompt = """Voce e um curador de glossario de negocio para governanca de dados.
Analise as definicoes candidatas de um termo e retorne JSON com:
{
    "proposed_definition": "definicao padrao consolidada em PT-BR",
    "proposed_definition_en": "standard consolidated definition in EN",
    "business_description": "descricao amigavel para stakeholders de negocio em PT-BR",
    "semantic_conflicts": [{"conflict": "descricao do conflito", "sources": "fonte1 vs fonte2"}],
    "suggested_owner": "perfil de quem deveria ser owner (ex: 'Gerente Financeiro') ou null",
    "suggested_source_system": "sistema que deveria ser a referencia oficial ou null"
}
Responda APENAS com JSON valido."""

        sources_text = "\n".join([f"- Fonte: {s.get('source', 'N/A')}: {s.get('definition', 'N/A')}" for s in sources])
        user_prompt = f"Termo: {term_name}\nDominio: {domain or 'nao especificado'}\n\nDefinicoes candidatas:\n{sources_text}"

        try:
            response = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000,
            )
            parsed = json.loads(response.content)
            now = datetime.now().isoformat()
            return GlossaryTerm(
                term_id=str(uuid.uuid4())[:12],
                term_name=term_name,
                candidate_definitions=sources,
                proposed_definition=parsed.get("proposed_definition", sources[0].get("definition", "") if sources else ""),
                proposed_definition_en=parsed.get("proposed_definition_en", ""),
                business_description=parsed.get("business_description", ""),
                domain=domain,
                suggested_owner=parsed.get("suggested_owner"),
                suggested_steward=None,
                source_system=parsed.get("suggested_source_system"),
                related_datasets=related_datasets or [],
                related_attributes=related_attributes or [],
                semantic_conflicts=parsed.get("semantic_conflicts", []),
                status=GlossaryTermStatus.CANDIDATE.value,
                approved_by=None,
                version=1,
                created_at=now,
                updated_at=None,
            )
        except Exception:
            return self._curate_rule_based(term_name, sources, domain, related_datasets, related_attributes)

    def _curate_rule_based(self, term_name, sources, domain, related_datasets, related_attributes):
        """Fallback: use first source as proposed, detect simple conflicts."""
        proposed = sources[0].get("definition", "") if sources else ""
        conflicts = []

        # Simple conflict detection: if definitions differ significantly
        definitions = [s.get("definition", "").strip().lower() for s in sources]
        seen = set()
        for i, d1 in enumerate(definitions):
            for j, d2 in enumerate(definitions):
                if i < j and d1 != d2:
                    key = (min(i,j), max(i,j))
                    if key not in seen:
                        seen.add(key)
                        conflicts.append({
                            "conflict": f"Definicoes divergentes para '{term_name}'",
                            "sources": f"{sources[i].get('source', f'Fonte {i+1}')} vs {sources[j].get('source', f'Fonte {j+1}')}"
                        })

        now = datetime.now().isoformat()
        return GlossaryTerm(
            term_id=str(uuid.uuid4())[:12],
            term_name=term_name,
            candidate_definitions=sources,
            proposed_definition=proposed,
            proposed_definition_en="",
            business_description=proposed,
            domain=domain,
            suggested_owner=None,
            suggested_steward=None,
            source_system=None,
            related_datasets=related_datasets or [],
            related_attributes=related_attributes or [],
            semantic_conflicts=conflicts,
            status=GlossaryTermStatus.CANDIDATE.value,
            approved_by=None,
            version=1,
            created_at=now,
            updated_at=None,
        )

    def detect_conflicts(self, term_name: str,
                         existing_terms: List[GlossaryTerm]) -> List[Dict[str, str]]:
        """Detect semantic conflicts between a term name and existing terms."""
        conflicts = []
        name_lower = term_name.lower().replace("_", " ").replace("-", " ")

        for term in existing_terms:
            existing_lower = term.term_name.lower().replace("_", " ").replace("-", " ")
            # Check for similar names with different definitions
            if name_lower == existing_lower and term.status != GlossaryTermStatus.DEPRECATED.value:
                conflicts.append({
                    "conflict": f"Termo '{term_name}' ja existe com definicao diferente",
                    "existing_term_id": term.term_id,
                    "existing_definition": term.proposed_definition,
                })
            # Check for partial matches (substring)
            elif (name_lower in existing_lower or existing_lower in name_lower) and len(name_lower) > 3:
                conflicts.append({
                    "conflict": f"Possivel duplicata: '{term_name}' similar a '{term.term_name}'",
                    "existing_term_id": term.term_id,
                    "existing_definition": term.proposed_definition,
                })
        return conflicts

    def generate_business_description(self, term: GlossaryTerm) -> str:
        """Generate a business-friendly description. Uses LLM if available."""
        if self.llm_provider is not None:
            try:
                response = self.llm_provider.generate(
                    prompt=f"Gere uma descricao curta e amigavel para negocio do termo '{term.term_name}' com definicao: {term.proposed_definition}",
                    system_prompt="Voce gera descricoes curtas (1-2 frases) de termos de dados para stakeholders de negocio. Responda apenas com a descricao, sem formatacao.",
                    temperature=0.3,
                    max_tokens=200,
                )
                return response.content.strip()
            except Exception:
                pass
        return term.proposed_definition
