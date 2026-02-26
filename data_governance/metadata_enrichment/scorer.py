"""
Metadata Quality Evaluator

Two-stage evaluation of existing metadata quality:
  Stage 1 — Structural triage (no LLM): detect absent/trivial values
  Stage 2 — LLM clarity evaluation: assess whether content is clear and useful

Usage:
    evaluator = MetadataQualityEvaluator(llm_provider)
    diag = evaluator.evaluate({
        "name": "tb_clientes",
        "schema": "public",
        "database": "sales_db",
        "description": "Dados dos clientes",
        "owner": "",
        "tags": [],
        "classification": "",
        "columns": [
            {"name": "cpf", "description": ""},
            {"name": "nome", "description": "Nome completo do cliente cadastrado no sistema"},
        ],
        "source": "warehouse",
    })
    print(diag.status, diag.quality_score)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from rag_discovery.providers.base import LLMProvider
except ImportError:
    from data_governance.rag_discovery.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Weights for composite score
# ---------------------------------------------------------------------------

_W_DESCRIPTION = 0.30
_W_OWNER       = 0.10
_W_TAGS        = 0.10
_W_CLASSIF     = 0.10
_W_COLUMNS     = 0.40

_MIN_TRIVIAL_LEN = 10  # strings shorter than this → structurally absent


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ColumnQuality:
    """Quality assessment for a single column's existing metadata."""
    name: str
    quality: str          # "absent" | "poor" | "sufficient"
    score: float          # 0.0 (absent), 0.5 (poor), 1.0 (sufficient)
    existing_value: str   # original description text
    llm_reasoning: str    # LLM justification shown in UI tooltip


@dataclass
class TableMetadataDiagnosis:
    """Full quality diagnosis for one table."""
    table_name: str
    schema: str
    database: str
    quality_score: float          # 0.0–1.0 composite weighted score
    status: str                   # "absent" | "poor" | "sufficient"
    description_quality: str      # "absent" | "poor" | "sufficient"
    description_reasoning: str    # LLM justification
    existing_description: str
    existing_owner: str
    existing_tags: List[str]
    existing_classification: str
    column_qualities: List[ColumnQuality] = field(default_factory=list)
    columns_to_enrich: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    last_modified: Optional[str] = None
    source: str = "warehouse"     # "openmetadata" | "warehouse" | "file"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MetadataQualityEvaluator:
    """
    Evaluates the semantic quality of existing metadata.

    Two-stage approach:
      1. Structural triage (no LLM) — catches empty / trivial values fast
      2. LLM clarity evaluation — one API call per table evaluates the
         table description + all column descriptions together
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    # ------------------------------------------------------------------ public

    def evaluate(self, table_meta: Dict[str, Any]) -> TableMetadataDiagnosis:
        """
        Evaluate a single table dict.

        Expected keys: name, schema, database, description, owner, tags,
                       classification, columns (list of {name, description}),
                       row_count, last_modified, source.
        """
        table_name = table_meta.get("name", "") or ""
        description = table_meta.get("description", "") or ""
        owner = table_meta.get("owner", "") or ""
        tags = list(table_meta.get("tags") or [])
        classification = table_meta.get("classification", "") or ""
        columns = list(table_meta.get("columns") or [])
        row_count = table_meta.get("row_count")
        last_modified = table_meta.get("last_modified")
        source = table_meta.get("source", "warehouse")

        # Stage 1 — structural triage on the table description
        desc_trivial = self._structural_triage(description, table_name)

        column_qualities: List[ColumnQuality] = []

        if desc_trivial is not None:
            # Table description is structurally absent — no LLM needed for it
            desc_quality = desc_trivial
            desc_reasoning = "Campo vazio, nulo ou trivial (igual ao nome ou muito curto)."

            # Still evaluate columns that have some content via LLM
            cols_with_content = [
                c for c in columns
                if self._structural_triage(
                    c.get("description", "") or "", c.get("name", "")
                ) is None
            ]
            cols_trivial = [
                c for c in columns
                if self._structural_triage(
                    c.get("description", "") or "", c.get("name", "")
                ) is not None
            ]

            # Trivial columns — mark directly without LLM
            for c in cols_trivial:
                column_qualities.append(ColumnQuality(
                    name=c.get("name", ""),
                    quality="absent",
                    score=0.0,
                    existing_value=c.get("description", "") or "",
                    llm_reasoning="Campo vazio, nulo ou trivial.",
                ))

            # Columns with content — evaluate via LLM
            if cols_with_content:
                llm_result = self._llm_evaluate_clarity(
                    table_name=table_name,
                    table_desc="",  # table description is absent, skip it
                    columns=[
                        {"name": c.get("name", ""), "description": c.get("description", "") or ""}
                        for c in cols_with_content
                    ],
                )
                for c in cols_with_content:
                    cname = c.get("name", "")
                    cr = llm_result.get("columns", {}).get(cname, {})
                    q = cr.get("quality", "poor")
                    column_qualities.append(ColumnQuality(
                        name=cname,
                        quality=q,
                        score=1.0 if q == "sufficient" else 0.5,
                        existing_value=c.get("description", "") or "",
                        llm_reasoning=cr.get("reasoning", ""),
                    ))

        else:
            # Table description has content — evaluate everything together via LLM
            llm_result = self._llm_evaluate_clarity(
                table_name=table_name,
                table_desc=description,
                columns=[
                    {"name": c.get("name", ""), "description": c.get("description", "") or ""}
                    for c in columns
                ],
            )
            tr = llm_result.get("table", {})
            desc_quality = tr.get("quality", "poor")
            desc_reasoning = tr.get("reasoning", "")

            for c in columns:
                cname = c.get("name", "")
                triage = self._structural_triage(c.get("description", "") or "", cname)
                if triage is not None:
                    column_qualities.append(ColumnQuality(
                        name=cname,
                        quality=triage,
                        score=0.0,
                        existing_value=c.get("description", "") or "",
                        llm_reasoning="Campo vazio, nulo ou trivial.",
                    ))
                else:
                    cr = llm_result.get("columns", {}).get(cname, {})
                    q = cr.get("quality", "poor")
                    column_qualities.append(ColumnQuality(
                        name=cname,
                        quality=q,
                        score=1.0 if q == "sufficient" else 0.5,
                        existing_value=c.get("description", "") or "",
                        llm_reasoning=cr.get("reasoning", ""),
                    ))

        columns_to_enrich = [
            cq.name for cq in column_qualities if cq.quality != "sufficient"
        ]

        score = self._compute_score(
            desc_quality=desc_quality,
            has_owner=bool(owner.strip()),
            has_tags=len(tags) > 0,
            has_classification=bool(classification.strip()),
            column_qualities=column_qualities,
        )
        status = self._determine_status(score)

        return TableMetadataDiagnosis(
            table_name=table_name,
            schema=table_meta.get("schema", "") or "",
            database=table_meta.get("database", "") or "",
            quality_score=score,
            status=status,
            description_quality=desc_quality,
            description_reasoning=desc_reasoning,
            existing_description=description,
            existing_owner=owner,
            existing_tags=tags,
            existing_classification=classification,
            column_qualities=column_qualities,
            columns_to_enrich=columns_to_enrich,
            row_count=row_count,
            last_modified=str(last_modified) if last_modified else None,
            source=source,
        )

    def evaluate_batch(
        self,
        tables: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[TableMetadataDiagnosis]:
        """
        Evaluate a list of table dicts.

        Args:
            tables: List of table dicts (same format as evaluate())
            on_progress: Optional callback(current, total) for progress tracking

        Returns:
            List of TableMetadataDiagnosis, one per table
        """
        results: List[TableMetadataDiagnosis] = []
        total = len(tables)
        for i, tbl in enumerate(tables):
            results.append(self.evaluate(tbl))
            if on_progress:
                on_progress(i + 1, total)
        return results

    # ----------------------------------------------------------------- private

    def _structural_triage(self, value: str, name: str) -> Optional[str]:
        """
        Returns "absent" if value is clearly empty or trivial (no LLM needed).
        Returns None if value should proceed to LLM evaluation.
        """
        if not value or not value.strip():
            return "absent"
        v = value.strip()
        if len(v) < _MIN_TRIVIAL_LEN:
            return "absent"
        # Equals the field/table name (normalised — ignores case, underscores, spaces)
        norm_val = re.sub(r"[_\s]", "", v).lower()
        norm_name = re.sub(r"[_\s]", "", name).lower()
        if norm_val == norm_name:
            return "absent"
        return None

    def _llm_evaluate_clarity(
        self,
        table_name: str,
        table_desc: str,
        columns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Single LLM call per table. Evaluates the table description and all
        column descriptions. Returns structured JSON.
        """
        system_prompt = (
            "Você é um auditor de qualidade de metadados de dados.\n\n"
            "Para cada item, classifique como:\n"
            '- "sufficient": a descrição explica claramente o propósito de negócio, '
            "é específica, e um analista entenderia o conteúdo sem ver os dados.\n"
            '- "poor": existe conteúdo mas é vago, genérico, puramente técnico sem '
            'contexto de negócio, ou óbvio demais (ex: "Tabela de dados", '
            '"ID do registro", "Campo boolean", "Dados do sistema").\n\n'
            "Retorne APENAS JSON válido no formato:\n"
            "{\n"
            '  "table": {"quality": "sufficient|poor", "reasoning": "..."},\n'
            '  "columns": {\n'
            '    "nome_coluna": {"quality": "sufficient|poor", "reasoning": "..."},\n'
            "    ...\n"
            "  }\n"
            "}"
        )

        cols_text = "\n".join(
            f'  - {c["name"]}: "{c["description"]}"'
            for c in columns
        ) or "  (nenhuma coluna com descrição)"

        prompt = (
            f"Tabela: {table_name}\n"
            f'Descrição da tabela: "{table_desc}"\n\n'
            f"Colunas com descrição:\n{cols_text}\n\n"
            "Avalie cada item e retorne o JSON:"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1500,
            )
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception:
            # Fallback: mark everything poor so it will be suggested for enrichment
            fallback_cols = {
                c["name"]: {
                    "quality": "poor",
                    "reasoning": "Avaliação automática não disponível.",
                }
                for c in columns
            }
            return {
                "table": {
                    "quality": "poor" if table_desc else "absent",
                    "reasoning": "Avaliação automática não disponível.",
                },
                "columns": fallback_cols,
            }

    def _compute_score(
        self,
        desc_quality: str,
        has_owner: bool,
        has_tags: bool,
        has_classification: bool,
        column_qualities: List[ColumnQuality],
    ) -> float:
        """Compute the weighted composite quality score (0.0–1.0)."""
        desc_score = {"absent": 0.0, "poor": 0.5, "sufficient": 1.0}.get(desc_quality, 0.0)
        owner_score = 1.0 if has_owner else 0.0
        tags_score = 1.0 if has_tags else 0.0
        classif_score = 1.0 if has_classification else 0.0

        if column_qualities:
            col_score = sum(cq.score for cq in column_qualities) / len(column_qualities)
        else:
            col_score = 0.0

        return (
            desc_score  * _W_DESCRIPTION
            + owner_score   * _W_OWNER
            + tags_score    * _W_TAGS
            + classif_score * _W_CLASSIF
            + col_score     * _W_COLUMNS
        )

    def _determine_status(self, score: float) -> str:
        if score < 0.30:
            return "absent"
        elif score < 0.70:
            return "poor"
        return "sufficient"
