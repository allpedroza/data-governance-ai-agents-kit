# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""Agent for classifying sensitive data using only schema and metadata.

The DataClassificationAgent evaluates table and column metadata to flag
PII (Personally Identifiable Information), PHI (Protected Health Information)
and financial data, providing LGPD/GDPR-oriented recommendations without
accessing raw data. An optional LLM step can review the same metadata to
validate the sensitivity decision when the user provides an :class:`LLMProvider`.

Exposes LGPD-specific properties on the result objects:
  - ``has_pii`` / ``risk_level`` / ``lgpd_sensitive_columns`` on TableClassification
  - ``column_name`` / ``pii_labels`` / ``lgpd_categories`` / ``evidence`` on ColumnClassification

Also supports DataSampler-based classification via ``classify_from_sample()``.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Path setup for flexible import contexts
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from rag_discovery.providers.base import LLMProvider
except ImportError:
    try:
        from data_governance.rag_discovery.providers.base import LLMProvider
    except ImportError:
        LLMProvider = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LGPD category mapping
# ---------------------------------------------------------------------------

#: Maps heuristic categories to LGPD data classification buckets.
_LGPD_CATEGORY_MAP: Dict[str, str] = {
    "PII":       "pessoal_ordinario",   # Art. 5 I — identifies a natural person
    "PHI":       "pessoal_sensivel",    # Art. 5 IX — health / biometric data
    "FINANCIAL": "pessoal_sensivel",    # Art. 5 IX — financial / payment data
}

#: Human-readable labels for each heuristic category.
_PII_LABEL_MAP: Dict[str, str] = {
    "PII":       "PII Pessoal",
    "PHI":       "PHI / Saúde",
    "FINANCIAL": "Financeiro / Pagamento",
}


# ---------------------------------------------------------------------------
# DataSampler pattern → heuristic category
# ---------------------------------------------------------------------------

_SAMPLER_PATTERN_TO_CATEGORY: Dict[str, str] = {
    "cpf":         "PII",
    "email":       "PII",
    "phone":       "PII",
    "ip_address":  "PII",
    "cep":         "PII",
    "cnpj":        "FINANCIAL",
    "credit_card": "FINANCIAL",
}

_SEMANTIC_TYPE_TO_CATEGORY: Dict[str, str] = {
    "pii":     "PII",
    "email":   "PII",
    "phone":   "PII",
    "name":    "PII",
    "address": "PII",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ColumnMetadata:
    """Minimal representation of a column used for classification."""

    name: str
    type: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def normalized(self) -> Dict[str, str]:
        """Return lowercase versions of name and description for matching."""
        return {
            "name": self.name.lower(),
            "description": self.description.lower(),
            "type": self.type.lower(),
        }


@dataclass
class TableSchema:
    """Structured table metadata passed to the classifier."""

    name: str
    database: str = ""
    schema: str = ""
    description: str = ""
    columns: List[ColumnMetadata] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    owner: str = ""

    def full_name(self) -> str:
        """Return a dotted identifier when database/schema are present."""
        parts = [p for p in [self.database, self.schema, self.name] if p]
        return ".".join(parts) if parts else self.name


@dataclass
class SensitiveDataRule:
    """Rule used to detect sensitive information via metadata."""

    category: str
    keywords: Sequence[str]
    types: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    description: str = ""
    weight: float = 1.0

    def match(self, column: ColumnMetadata) -> Tuple[float, List[str]]:
        """Return a score and reasons for a given column."""
        score = 0.0
        reasons: List[str] = []
        normalized = column.normalized()

        # Tokenize column name to avoid partial matches (e.g. 'conta' in 'contato')
        name_tokens = set(re.split(r"[_\s]+", normalized["name"]))

        # Keyword match on name/description
        for keyword in self.keywords:
            if "_" in keyword or " " in keyword:
                # Compound keyword (e.g. "credit_card"): use full substring match
                name_match = keyword in normalized["name"]
            else:
                # Single-word keyword: require whole-token match to avoid false positives
                name_match = keyword in name_tokens
            if name_match:
                score += 0.5 * self.weight
                reasons.append(f"nome contém '{keyword}'")
            elif keyword in normalized["description"]:
                score += 0.35 * self.weight
                reasons.append(f"descrição menciona '{keyword}'")

        # Type hints
        for expected_type in self.types:
            if expected_type in normalized["type"]:
                score += 0.25 * self.weight
                reasons.append(f"tipo '{column.type}' indica {self.category}")

        # Tags
        column_tags = {t.lower() for t in column.tags}
        for tag in self.tags:
            if tag in column_tags:
                score += 0.2 * self.weight
                reasons.append(f"tag '{tag}' presente")

        return score, reasons


@dataclass
class ColumnClassification:
    """Classification output for a column."""

    column: ColumnMetadata
    categories: List[str]
    confidence: float
    reasons: List[str]
    suggested_controls: List[str]
    _detection_method: str = field(default="name_only", repr=False)

    # ---------------------------------------------------------------- LGPD helpers

    @property
    def column_name(self) -> str:
        """Convenience accessor for ``column.name``."""
        return self.column.name

    @property
    def pii_labels(self) -> List[str]:
        """Human-readable labels for each detected category."""
        return [_PII_LABEL_MAP.get(cat, cat) for cat in self.categories]

    @property
    def lgpd_categories(self) -> List[str]:
        """LGPD data type per detected category (deduplicated)."""
        return list(dict.fromkeys(
            _LGPD_CATEGORY_MAP.get(cat, "nao_pessoal") for cat in self.categories
        ))

    @property
    def risk_level(self) -> str:
        """Risk level derived from category and confidence."""
        if "PHI" in self.categories:
            return "high"
        if "FINANCIAL" in self.categories:
            return "high"
        if "PII" in self.categories:
            # Any PII that passed the detection threshold (≥0.45) is high risk
            return "high" if self.confidence >= 0.45 else "medium"
        return "none"

    @property
    def evidence(self) -> str:
        """Human-readable detection evidence."""
        return "; ".join(self.reasons) if self.reasons else "Regra heurística aplicada."

    @property
    def detection_method(self) -> str:
        return self._detection_method


@dataclass
class TableClassification:
    """Aggregated classification for a table."""

    table: TableSchema
    sensitivity_level: str
    detected_categories: List[str]
    columns: List[ColumnClassification]
    compliance_flags: Dict[str, Any]
    recommended_actions: List[str]
    rationale: str
    llm_assessment: Optional["LLMAssessment"] = None
    detection_method: str = "name_only"  # "name_only" | "data_sample" | "combined"

    # ---------------------------------------------------------------- LGPD helpers

    @property
    def table_name(self) -> str:
        """Convenience accessor for ``table.name``."""
        return self.table.name

    @property
    def has_pii(self) -> bool:
        """True when any sensitive category was detected."""
        return len(self.detected_categories) > 0

    @property
    def risk_level(self) -> str:
        """Normalised risk level (high / medium / low / none)."""
        level = self.sensitivity_level
        if level == "CRITICAL":
            return "high"
        if level == "HIGH":
            return "high"
        if level == "MEDIUM":
            return "medium"
        if level == "LOW" and self.detected_categories:
            return "low"
        return "none"

    @property
    def lgpd_sensitive_columns(self) -> List[str]:
        """Column names whose data falls under LGPD Art. 5 IX (pessoal_sensivel)."""
        result = []
        for cc in self.columns:
            if "pessoal_sensivel" in cc.lgpd_categories:
                result.append(cc.column.name)
        return result

    @property
    def recommended_classification(self) -> str:
        """Recommended data access level: restricted / confidential / internal."""
        level = self.sensitivity_level
        if level in ("HIGH", "CRITICAL"):
            return "restricted"
        if level == "MEDIUM":
            return "confidential"
        return "internal"

    @property
    def pii_columns(self) -> List[ColumnClassification]:
        """Alias for ``columns`` — exposes the same interface expected by the wizard UI."""
        return self.columns


@dataclass
class LLMAssessment:
    """Result of an LLM review using only metadata/schema context."""

    is_sensitive: bool
    categories: List[str]
    confidence: float
    explanation: str
    raw_response: str
    prompt: str
    model: str


class DataClassificationAgent:
    """Classify sensitive data using schema-level signals only."""

    DEFAULT_RULES: Sequence[SensitiveDataRule] = (
        SensitiveDataRule(
            category="PII",
            keywords=(
                # Document identifiers
                "cpf", "rg", "ssn", "tax_id", "passport", "passaporte",
                "identidade", "documento", "document",
                # Name
                "nome", "name", "razao_social", "nomecomp",
                # Contact
                "email", "e-mail", "e_mail",
                "phone", "telefone", "celular", "fone",
                # Address
                "address", "endereco", "logradouro", "rua",
                "zipcode", "cep", "zip_code",
                # Date of birth
                "birth", "nascimento", "birthday", "birth_date",
                "dt_nasc", "data_nasc",
                # Network
                "ip_address", "ip_addr", "endereco_ip",
            ),
            types=("string", "varchar", "email", "phone", "uuid"),
            tags=("pii", "personal"),
            description="Identificadores pessoais protegidos por LGPD/GDPR.",
        ),
        SensitiveDataRule(
            category="PHI",
            keywords=(
                "patient", "paciente",
                "diagnosis", "diagnostico", "cid",
                "medical", "clinico",
                "health", "saude", "doenca",
                "lab", "exame",
                "prescription", "receita",
                "insurance",
                "biometria", "biometric", "fingerprint",
                "foto", "photo",
                "senha", "password", "pwd", "hash_senha",
            ),
            types=("clinical", "disease", "health", "json"),
            tags=("phi", "health", "biometric"),
            weight=1.1,
            description="Informações de saúde e biometria (PHI) com maiores restrições.",
        ),
        SensitiveDataRule(
            category="FINANCIAL",
            keywords=(
                "credit_card", "card", "cartao", "card_number", "num_cartao",
                "cvv", "iban", "swift", "routing",
                "account", "conta", "agencia", "banco",
                "transaction", "transacao",
                "invoice", "fatura",
                "salary", "salario", "renda", "income",
            ),
            types=("decimal", "double", "money", "currency", "card"),
            tags=("financial", "pci"),
            weight=1.05,
            description="Dados financeiros e de pagamento sujeitos a PCI/LGPD.",
        ),
    )

    def __init__(
        self,
        rules: Optional[Sequence[SensitiveDataRule]] = None,
        lgpd_requirements: Optional[Sequence[str]] = None,
        gdpr_requirements: Optional[Sequence[str]] = None,
        llm_provider: Optional[Any] = None,
    ) -> None:
        self.rules = rules or list(self.DEFAULT_RULES)
        self.lgpd_requirements = lgpd_requirements or [
            "Minimização de dados",
            "Registro de consentimento",
            "Controle de acesso baseado em papéis",
            "Mascaramento ou tokenização"
        ]
        self.gdpr_requirements = gdpr_requirements or [
            "Avaliação de impacto de proteção de dados (DPIA)",
            "Direito ao esquecimento e portabilidade",
            "Retenção limitada e auditoria",
            "Pseudonimização para processamento analítico"
        ]
        self.llm_provider = llm_provider

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def classify_table(self, table: TableSchema) -> TableClassification:
        """Classify a single table based on its metadata."""
        column_results: List[ColumnClassification] = []
        detected_categories: List[str] = []
        table_reasons: List[str] = []

        for column in table.columns:
            classifications, reasons = self._classify_column(column)
            if classifications:
                detected_categories.extend(classifications)
                column_results.append(
                    ColumnClassification(
                        column=column,
                        categories=classifications,
                        confidence=min(1.0, sum(reason[1] for reason in reasons)),
                        reasons=[r for r, _ in reasons],
                        suggested_controls=self._suggest_controls(classifications),
                    )
                )

        unique_categories = sorted(set(detected_categories))
        sensitivity_level = self._derive_sensitivity_level(unique_categories)

        table_reasons.extend(self._table_level_reasons(table, unique_categories))

        return TableClassification(
            table=table,
            sensitivity_level=sensitivity_level,
            detected_categories=unique_categories,
            columns=column_results,
            compliance_flags={
                "lgpd": bool(unique_categories),
                "gdpr": bool(unique_categories),
            },
            recommended_actions=self._recommended_actions(sensitivity_level),
            rationale="; ".join(table_reasons) if table_reasons else "Metadados analisados sem achados críticos.",
        )

    def classify_table_with_llm(self, table: TableSchema) -> TableClassification:
        """Classify a table and optionally validate sensitivity with an LLM.

        The method first runs the heuristic classification and, when a configured
        LLM provider is available, asks the model to decide if the table contains
        PII ou dados sensíveis com base apenas em metadados/schema. The final
        sensitivity level merges both signals.
        """

        classification = self.classify_table(table)
        if not self.llm_provider:
            return classification

        assessment = self._run_llm_assessment(table, classification)
        classification.llm_assessment = assessment

        merged_categories = sorted(
            set(classification.detected_categories + assessment.categories)
        )
        classification.detected_categories = merged_categories
        classification.sensitivity_level = self._derive_sensitivity_level(merged_categories)
        classification.compliance_flags["llm_sensitive"] = assessment.is_sensitive
        classification.recommended_actions = self._recommended_actions(
            classification.sensitivity_level
        )
        rationale_parts = [classification.rationale]
        if assessment.explanation:
            rationale_parts.append(f"LLM: {assessment.explanation}")
        classification.rationale = " | ".join(rationale_parts)
        return classification

    def classify_catalog(self, tables: Iterable[TableSchema]) -> List[TableClassification]:
        """Run classification across multiple tables."""
        return [self.classify_table(table) for table in tables]

    def classify_batch_from_dicts(
        self,
        table_dicts: List[Dict[str, Any]],
        use_llm: bool = False,
    ) -> List[TableClassification]:
        """Classify a list of table dicts (wizard format).

        Args:
            table_dicts: Each dict has keys: name, schema, database, description,
                         owner, tags, columns (list of {name, type, description, tags}).
            use_llm: If True and an LLM provider is configured, validates with LLM.

        Returns:
            List of TableClassification, one per input dict.
        """
        results: List[TableClassification] = []
        for td in table_dicts:
            table = TableSchema(
                name=td.get("name", ""),
                database=td.get("database", "") or "",
                schema=td.get("schema", "") or "",
                description=td.get("description", "") or "",
                columns=[
                    ColumnMetadata(
                        name=c.get("name", "") if isinstance(c, dict) else str(c),
                        type=c.get("type", "") or "" if isinstance(c, dict) else "",
                        description=c.get("description", "") or "" if isinstance(c, dict) else "",
                        tags=c.get("tags", []) or [] if isinstance(c, dict) else [],
                    )
                    for c in (td.get("columns") or [])
                ],
                tags=td.get("tags", []) or [],
                owner=td.get("owner", "") or "",
            )
            tc = self.classify_table_with_llm(table) if (use_llm and self.llm_provider) else self.classify_table(table)
            results.append(tc)
        return results

    def classify_from_sample(self, sample_result: Any) -> "TableClassification":
        """Classify PII from a DataSampler SampleResult (patterns on real data).

        Accepts any SampleResult duck-type with:
          - ``.table_name: str``
          - ``.columns: List`` where each column has ``.name``, ``.patterns``,
            ``.inferred_semantic_type``, ``.dtype``

        Returns:
            TableClassification with ``detection_method = "data_sample"``.
        """
        columns: List[ColumnMetadata] = []
        for col_profile in (sample_result.columns or []):
            # Tags derived from DataSampler pattern detections
            tags: List[str] = []
            for pattern in (getattr(col_profile, "patterns", None) or []):
                cat = _SAMPLER_PATTERN_TO_CATEGORY.get(pattern)
                if cat:
                    tags.append(cat.lower())

            # Also check inferred semantic type
            sem = getattr(col_profile, "inferred_semantic_type", None)
            if sem:
                cat = _SEMANTIC_TYPE_TO_CATEGORY.get(sem)
                if cat and cat.lower() not in tags:
                    tags.append(cat.lower())

            columns.append(ColumnMetadata(
                name=getattr(col_profile, "name", ""),
                type=getattr(col_profile, "dtype", "") or "",
                description="",
                tags=tags,
            ))

        table = TableSchema(
            name=getattr(sample_result, "table_name", ""),
            columns=columns,
        )
        tc = self.classify_table(table)
        tc.detection_method = "data_sample"
        # Propagate detection method to columns
        for cc in tc.columns:
            cc._detection_method = "data_sample"
        return tc

    def merge_with_sample(
        self,
        name_classification: "TableClassification",
        sample_classification: "TableClassification",
    ) -> "TableClassification":
        """Merge name-based and data-based classifications.

        Data evidence takes precedence; name-based fills gaps for columns
        not found in the sample classification.

        Returns:
            A new TableClassification with ``detection_method = "combined"``.
        """
        # Index data-based results by column name
        sample_by_col: Dict[str, ColumnClassification] = {
            cc.column.name: cc for cc in sample_classification.columns
        }
        name_by_col: Dict[str, ColumnClassification] = {
            cc.column.name: cc for cc in name_classification.columns
        }

        merged_cols: List[ColumnClassification] = []

        all_col_names = list(dict.fromkeys(
            list(name_by_col.keys()) + list(sample_by_col.keys())
        ))

        for col_name in all_col_names:
            name_cc = name_by_col.get(col_name)
            sample_cc = sample_by_col.get(col_name)

            if name_cc and sample_cc:
                # Merge: combine categories and reasons; data takes precedence
                merged_categories = sorted(set(name_cc.categories + sample_cc.categories))
                merged_reasons = list(dict.fromkeys(
                    [f"[nome] {r}" for r in name_cc.reasons]
                    + [f"[dados] {r}" for r in sample_cc.reasons]
                ))
                merged_cols.append(ColumnClassification(
                    column=sample_cc.column,
                    categories=merged_categories,
                    confidence=max(name_cc.confidence, sample_cc.confidence),
                    reasons=merged_reasons,
                    suggested_controls=self._suggest_controls(merged_categories),
                    _detection_method="combined",
                ))
            elif sample_cc:
                sample_cc._detection_method = "combined"
                merged_cols.append(sample_cc)
            elif name_cc:
                merged_cols.append(name_cc)

        merged_categories = sorted(set(
            name_classification.detected_categories
            + sample_classification.detected_categories
        ))
        sensitivity_level = self._derive_sensitivity_level(merged_categories)
        rationale_parts = []
        if name_classification.rationale:
            rationale_parts.append(f"[nome] {name_classification.rationale}")
        if sample_classification.rationale:
            rationale_parts.append(f"[dados] {sample_classification.rationale}")

        return TableClassification(
            table=name_classification.table,
            sensitivity_level=sensitivity_level,
            detected_categories=merged_categories,
            columns=merged_cols,
            compliance_flags={
                "lgpd": bool(merged_categories),
                "gdpr": bool(merged_categories),
            },
            recommended_actions=self._recommended_actions(sensitivity_level),
            rationale=" | ".join(rationale_parts) or "Metadados analisados sem achados.",
            detection_method="combined",
        )

    def _classify_column(self, column: ColumnMetadata) -> Tuple[List[str], List[Tuple[str, float]]]:
        matches: List[str] = []
        reasons: List[Tuple[str, float]] = []

        for rule in self.rules:
            score, rule_reasons = rule.match(column)
            if score >= 0.45:  # threshold tuned for metadata-only context
                matches.append(rule.category)
                reasons.extend([(reason, min(1.0, score)) for reason in rule_reasons])

        return sorted(set(matches)), reasons

    def _derive_sensitivity_level(self, categories: Sequence[str]) -> str:
        if not categories:
            return "LOW"
        if "PHI" in categories:
            return "CRITICAL"
        if "PII" in categories and "FINANCIAL" in categories:
            return "HIGH"
        if "FINANCIAL" in categories:
            return "HIGH"
        return "MEDIUM"

    def _format_table_metadata(self, table: TableSchema, categories: Sequence[str]) -> str:
        """Serialize table metadata into a prompt-friendly text block."""

        lines = [f"Tabela: {table.full_name()}"]
        if table.description:
            lines.append(f"Descrição: {table.description}")
        if table.tags:
            lines.append(f"Tags: {', '.join(table.tags)}")
        if categories:
            lines.append(f"Categorias heurísticas: {', '.join(categories)}")

        lines.append("Colunas:")
        for column in table.columns:
            details = [f"nome={column.name}"]
            if column.type:
                details.append(f"tipo={column.type}")
            if column.description:
                details.append(f"descrição={column.description}")
            if column.tags:
                details.append(f"tags={', '.join(column.tags)}")
            lines.append("- " + "; ".join(details))

        return "\n".join(lines)

    def _table_level_reasons(self, table: TableSchema, categories: Sequence[str]) -> List[str]:
        reasons: List[str] = []
        normalized_desc = self._normalize_text(table.description)
        normalized_tags = {t.lower() for t in table.tags}

        if "pii" in normalized_desc:
            reasons.append("descrição da tabela indica PII")
        if "phi" in normalized_desc or "health" in normalized_desc:
            reasons.append("descrição da tabela indica PHI")
        if "financial" in normalized_desc or "financeiro" in normalized_desc:
            reasons.append("descrição da tabela indica dados financeiros")

        if "pii" in normalized_tags:
            reasons.append("tag de tabela: pii")
        if "phi" in normalized_tags:
            reasons.append("tag de tabela: phi")
        if "financial" in normalized_tags or "pci" in normalized_tags:
            reasons.append("tag de tabela: financial/pci")

        if categories:
            reasons.append(f"categorias detectadas nas colunas: {', '.join(categories)}")

        return reasons

    def _suggest_controls(self, categories: Sequence[str]) -> List[str]:
        controls = set()
        if "PII" in categories or "PHI" in categories:
            controls.update({"mascaramento", "tokenização", "registro de consentimento"})
        if "FINANCIAL" in categories:
            controls.update({"criptografia forte", "segregação de funções", "monitoramento de acesso"})
        if "PHI" in categories:
            controls.update({"acesso mínimo necessário", "trilhas de auditoria"})
        return sorted(controls)

    def _recommended_actions(self, sensitivity_level: str) -> List[str]:
        if sensitivity_level == "LOW":
            return ["Nenhuma ação obrigatória além de monitoramento padrão."]
        if sensitivity_level == "MEDIUM":
            return [
                "Aplicar controles básicos de acesso e retenção",
                "Validar se consentimento é necessário",
            ]
        if sensitivity_level == "HIGH":
            return [
                "Implementar mascaramento/tokenização nos ambientes compartilhados",
                "Habilitar auditoria de acesso e retenção mínima",
                *self.lgpd_requirements,
                *self.gdpr_requirements,
            ]
        return [
            "Exigir DPIA e avaliação de risco de segurança",
            "Isolar dados em zonas com controles reforçados",
            "Requerer aprovação de DPO para novas finalidades",
            *self.lgpd_requirements,
            *self.gdpr_requirements,
        ]

    def _run_llm_assessment(
        self, table: TableSchema, classification: TableClassification
    ) -> "LLMAssessment":
        """Ask the configured LLM to validate if the table is sensível/PII."""

        if not self.llm_provider:
            raise ValueError("Nenhum LLM provider configurado para avaliação.")

        prompt = self._build_llm_prompt(table, classification)
        system_prompt = (
            "Você é um assistente de governança de dados. Analise apenas metadados "
            "(nomes, descrições, tipos e tags) e responda se a tabela envolve PII "
            "ou dados sensíveis conforme LGPD/GDPR. Não invente colunas ou dados."
        )

        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=320,
        )

        return self._parse_llm_response(prompt, response)

    def _build_llm_prompt(
        self, table: TableSchema, classification: TableClassification
    ) -> str:
        metadata_block = self._format_table_metadata(
            table, classification.detected_categories
        )

        return (
            "Avalie se a tabela abaixo contém PII ou dados sensíveis a partir "
            "dos metadados (sem ver dados brutos). Responda apenas em JSON com "
            "as chaves: sensitive_table (true/false), main_categories (lista "
            "como ['PII','PHI','FINANCIAL'] quando aplicável), confidence (0-1) "
            "e explanation (frase curta).\n\n"
            f"Metadados:\n{metadata_block}"
        )

    def _parse_llm_response(self, prompt: str, response: Any) -> "LLMAssessment":
        content = response.content if hasattr(response, "content") else str(response)
        parsed: Dict[str, Any]
        try:
            json_start = content.find("{")
            json_end = content.rfind("}")
            parsed = json.loads(content[json_start : json_end + 1])
        except Exception:
            parsed = {}

        is_sensitive = bool(parsed.get("sensitive_table"))
        raw_categories = parsed.get("main_categories") or []
        categories = [str(cat).upper() for cat in raw_categories if str(cat).strip()]
        confidence = (
            float(parsed.get("confidence"))
            if parsed.get("confidence") is not None
            else 0.5
        )
        explanation = str(parsed.get("explanation") or content).strip()

        return LLMAssessment(
            is_sensitive=is_sensitive,
            categories=categories,
            confidence=max(0.0, min(confidence, 1.0)),
            explanation=explanation,
            raw_response=content,
            prompt=prompt,
            model=getattr(response, "model", "unknown"),
        )
