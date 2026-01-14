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
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag_discovery.providers.base import LLMProvider


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

        # Keyword match on name/description
        for keyword in self.keywords:
            if keyword in normalized["name"]:
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
                "cpf",
                "cnpj",
                "ssn",
                "tax_id",
                "passport",
                "rg",
                "nome",
                "name",
                "email",
                "e-mail",
                "phone",
                "telefone",
                "celular",
                "address",
                "endereco",
                "zipcode",
                "cep",
                "birth",
                "nascimento",
            ),
            types=("string", "varchar", "email", "phone", "uuid"),
            tags=("pii", "personal"),
            description="Identificadores pessoais protegidos por LGPD/GDPR.",
        ),
        SensitiveDataRule(
            category="PHI",
            keywords=(
                "patient",
                "paciente",
                "diagnosis",
                "diagnostico",
                "medical",
                "clinico",
                "health",
                "saude",
                "lab",
                "exame",
                "prescription",
                "receita",
                "insurance",
            ),
            types=("clinical", "disease", "health", "json"),
            tags=("phi", "health"),
            weight=1.1,
            description="Informações de saúde (PHI) com maiores restrições.",
        ),
        SensitiveDataRule(
            category="FINANCIAL",
            keywords=(
                "credit_card",
                "card",
                "cvv",
                "iban",
                "swift",
                "routing",
                "account",
                "conta",
                "agencia",
                "banco",
                "transaction",
                "transacao",
                "invoice",
                "fatura",
                "salary",
                "salario",
            ),
            types=("decimal", "double", "money", "currency", "card"),
            tags=("financial", "pci"),
            weight=1.05,
            description="Dados financeiros e de pagamento sujeitos a PCI/GDPR.",
        ),
    )

    def __init__(
        self,
        rules: Optional[Sequence[SensitiveDataRule]] = None,
        lgpd_requirements: Optional[Sequence[str]] = None,
        gdpr_requirements: Optional[Sequence[str]] = None,
        llm_provider: Optional[LLMProvider] = None,
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
