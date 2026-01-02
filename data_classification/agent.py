"""
Data Classification Agent

Classifies data based on sensitivity levels (PII, PHI, Financial, etc.)
using pattern matching, metadata analysis, and configurable rules.

Features:
- PII detection (CPF, CNPJ, SSN, email, phone, etc.)
- PHI detection (medical records, health data)
- Financial data detection (credit cards, bank accounts)
- Custom classification rules
- Confidence scoring
- Multi-format support (CSV, Parquet, SQL, Delta)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SensitivityLevel(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataCategory(Enum):
    """Data categories for classification"""
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry
    FINANCIAL = "financial"
    PROPRIETARY = "proprietary"
    PUBLIC = "public"


@dataclass
class ColumnClassification:
    """Classification result for a single column"""
    name: str
    original_type: str
    categories: List[str] = field(default_factory=list)
    sensitivity_level: str = "public"
    pii_type: Optional[str] = None
    confidence: float = 0.0
    detected_patterns: List[str] = field(default_factory=list)
    sample_matches: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_type": self.original_type,
            "categories": self.categories,
            "sensitivity_level": self.sensitivity_level,
            "pii_type": self.pii_type,
            "confidence": self.confidence,
            "detected_patterns": self.detected_patterns,
            "sample_matches": self.sample_matches[:3],  # Limit samples
            "recommendations": self.recommendations
        }


@dataclass
class ClassificationReport:
    """Complete classification report for a data source"""
    source_name: str
    source_type: str
    classification_timestamp: str
    overall_sensitivity: str
    categories_found: List[str]
    columns: List[ColumnClassification]
    pii_columns: List[str] = field(default_factory=list)
    phi_columns: List[str] = field(default_factory=list)
    pci_columns: List[str] = field(default_factory=list)
    financial_columns: List[str] = field(default_factory=list)
    row_count: int = 0
    columns_analyzed: int = 0
    high_risk_count: int = 0
    recommendations: List[str] = field(default_factory=list)
    compliance_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "classification_timestamp": self.classification_timestamp,
            "overall_sensitivity": self.overall_sensitivity,
            "categories_found": self.categories_found,
            "summary": {
                "row_count": self.row_count,
                "columns_analyzed": self.columns_analyzed,
                "pii_columns": len(self.pii_columns),
                "phi_columns": len(self.phi_columns),
                "pci_columns": len(self.pci_columns),
                "financial_columns": len(self.financial_columns),
                "high_risk_count": self.high_risk_count
            },
            "columns": [c.to_dict() for c in self.columns],
            "pii_columns": self.pii_columns,
            "phi_columns": self.phi_columns,
            "pci_columns": self.pci_columns,
            "financial_columns": self.financial_columns,
            "recommendations": self.recommendations,
            "compliance_flags": self.compliance_flags
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# Classification Report: {self.source_name}",
            "",
            f"**Generated:** {self.classification_timestamp}",
            f"**Overall Sensitivity:** {self.overall_sensitivity.upper()}",
            "",
            "## Summary",
            f"- Rows analyzed: {self.row_count:,}",
            f"- Columns analyzed: {self.columns_analyzed}",
            f"- High-risk columns: {self.high_risk_count}",
            "",
            "## Categories Detected",
        ]

        if self.pii_columns:
            lines.append(f"- **PII:** {', '.join(self.pii_columns)}")
        if self.phi_columns:
            lines.append(f"- **PHI:** {', '.join(self.phi_columns)}")
        if self.pci_columns:
            lines.append(f"- **PCI:** {', '.join(self.pci_columns)}")
        if self.financial_columns:
            lines.append(f"- **Financial:** {', '.join(self.financial_columns)}")

        lines.extend(["", "## Column Details", ""])
        lines.append("| Column | Sensitivity | Categories | Confidence |")
        lines.append("|--------|------------|------------|------------|")

        for col in self.columns:
            cats = ", ".join(col.categories) if col.categories else "-"
            lines.append(f"| {col.name} | {col.sensitivity_level} | {cats} | {col.confidence:.0%} |")

        if self.recommendations:
            lines.extend(["", "## Recommendations", ""])
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        if self.compliance_flags:
            lines.extend(["", "## Compliance Flags", ""])
            for flag in self.compliance_flags:
                lines.append(f"- {flag}")

        return "\n".join(lines)


class DataClassificationAgent:
    """
    Agent for classifying data based on sensitivity and compliance requirements.

    Detects:
    - PII (Personally Identifiable Information)
    - PHI (Protected Health Information)
    - PCI (Payment Card Industry data)
    - Financial data
    """

    # PII patterns (Brazilian and international)
    PII_PATTERNS = {
        "cpf": r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",
        "cnpj": r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
        "rg": r"\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_br": r"\(?\d{2}\)?[\s.-]?\d{4,5}[-.]?\d{4}",
        "phone_intl": r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}",
        "passport": r"[A-Z]{2}\d{6,7}",
        "drivers_license": r"[A-Z]{2}\d{7,10}",
        "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "mac_address": r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}",
        "nis_pis": r"\d{3}\.?\d{5}\.?\d{2}-?\d",
        "titulo_eleitor": r"\d{4}\s?\d{4}\s?\d{4}",
        "cep": r"\d{5}-?\d{3}",
    }

    # PHI patterns
    PHI_PATTERNS = {
        "cid10": r"[A-Z]\d{2}\.?\d{0,2}",  # ICD-10 codes
        "cns": r"\d{15}",  # Cartão Nacional de Saúde
        "crm": r"CRM[\s/-]?\d{4,6}[\s/-]?[A-Z]{2}",
        "medical_record": r"MR[\s-]?\d{6,10}",
        "blood_type": r"\b(A|B|AB|O)[+-]\b",
    }

    # PCI/Financial patterns
    FINANCIAL_PATTERNS = {
        "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "bank_account_br": r"\d{4,5}-?\d{1}",
        "agency": r"\d{4}-?\d{1}",
        "iban": r"[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}",
        "swift": r"[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?",
        "cvv": r"\b\d{3,4}\b",
    }

    # Column name hints
    PII_COLUMN_HINTS = {
        "cpf", "cnpj", "rg", "ssn", "social_security", "email", "e_mail",
        "telefone", "phone", "celular", "mobile", "endereco", "address",
        "nome", "name", "sobrenome", "surname", "nascimento", "birth",
        "documento", "document", "identidade", "identity", "passaporte",
        "passport", "cep", "zip", "postal", "ip", "mac", "cnh", "license"
    }

    PHI_COLUMN_HINTS = {
        "diagnostico", "diagnosis", "cid", "icd", "prontuario", "medical_record",
        "prescricao", "prescription", "medicamento", "medication", "alergia",
        "allergy", "doenca", "disease", "tratamento", "treatment", "exame",
        "exam", "resultado", "result", "hospital", "clinica", "clinic",
        "medico", "doctor", "crm", "cns", "cartao_saude", "health_card",
        "sangue", "blood", "tipo_sanguineo", "blood_type", "peso", "weight",
        "altura", "height", "pressao", "pressure", "glicemia", "glucose"
    }

    FINANCIAL_COLUMN_HINTS = {
        "cartao", "card", "credito", "credit", "debito", "debit", "conta",
        "account", "banco", "bank", "agencia", "branch", "saldo", "balance",
        "transacao", "transaction", "pagamento", "payment", "fatura",
        "invoice", "cvv", "cvc", "iban", "swift", "pix", "boleto",
        "valor", "amount", "preco", "price", "receita", "revenue",
        "despesa", "expense", "lucro", "profit", "salario", "salary",
        "renda", "income"
    }

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, str]] = None,
        sensitivity_rules: Optional[Dict[str, str]] = None,
        sample_size: int = 1000
    ):
        """
        Initialize the Data Classification Agent.

        Args:
            custom_patterns: Additional regex patterns for detection
            sensitivity_rules: Custom rules for sensitivity assignment
            sample_size: Number of rows to sample for pattern detection
        """
        self.sample_size = sample_size
        self.custom_patterns = custom_patterns or {}
        self.sensitivity_rules = sensitivity_rules or {}

        # Compile all patterns
        self._compiled_pii = {k: re.compile(v, re.IGNORECASE) for k, v in self.PII_PATTERNS.items()}
        self._compiled_phi = {k: re.compile(v, re.IGNORECASE) for k, v in self.PHI_PATTERNS.items()}
        self._compiled_financial = {k: re.compile(v, re.IGNORECASE) for k, v in self.FINANCIAL_PATTERNS.items()}

        if custom_patterns:
            self._compiled_custom = {k: re.compile(v, re.IGNORECASE) for k, v in custom_patterns.items()}
        else:
            self._compiled_custom = {}

    def classify_from_csv(
        self,
        file_path: str,
        encoding: str = "utf-8",
        separator: str = ",",
        sample_size: Optional[int] = None
    ) -> ClassificationReport:
        """Classify data from a CSV file."""
        import pandas as pd

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=separator,
            nrows=sample_size or self.sample_size
        )

        return self._classify_dataframe(df, Path(file_path).name, "csv")

    def classify_from_parquet(
        self,
        file_path: str,
        sample_size: Optional[int] = None
    ) -> ClassificationReport:
        """Classify data from a Parquet file."""
        import pandas as pd

        df = pd.read_parquet(file_path)
        if sample_size or self.sample_size:
            df = df.head(sample_size or self.sample_size)

        return self._classify_dataframe(df, Path(file_path).name, "parquet")

    def classify_from_dataframe(
        self,
        df: Any,
        source_name: str = "dataframe"
    ) -> ClassificationReport:
        """Classify data from a pandas DataFrame."""
        return self._classify_dataframe(df, source_name, "dataframe")

    def classify_from_sql(
        self,
        connection_string: str,
        query: str,
        table_name: str,
        sample_size: Optional[int] = None
    ) -> ClassificationReport:
        """Classify data from a SQL query."""
        import pandas as pd
        from sqlalchemy import create_engine

        engine = create_engine(connection_string)
        limit = sample_size or self.sample_size
        limited_query = f"SELECT * FROM ({query}) AS subq LIMIT {limit}"

        df = pd.read_sql(limited_query, engine)
        return self._classify_dataframe(df, table_name, "sql")

    def classify_from_delta(
        self,
        path: str,
        sample_size: Optional[int] = None
    ) -> ClassificationReport:
        """Classify data from a Delta Lake table."""
        from deltalake import DeltaTable

        dt = DeltaTable(path)
        df = dt.to_pandas()

        if sample_size or self.sample_size:
            df = df.head(sample_size or self.sample_size)

        return self._classify_dataframe(df, Path(path).name, "delta")

    def _classify_dataframe(
        self,
        df: Any,
        source_name: str,
        source_type: str
    ) -> ClassificationReport:
        """Core classification logic for a DataFrame."""
        columns: List[ColumnClassification] = []
        pii_columns: List[str] = []
        phi_columns: List[str] = []
        pci_columns: List[str] = []
        financial_columns: List[str] = []
        all_categories: Set[str] = set()
        high_risk_count = 0

        for col_name in df.columns:
            col_classification = self._classify_column(df, col_name)
            columns.append(col_classification)

            # Track categories
            for cat in col_classification.categories:
                all_categories.add(cat)
                if cat == "pii":
                    pii_columns.append(col_name)
                elif cat == "phi":
                    phi_columns.append(col_name)
                elif cat == "pci":
                    pci_columns.append(col_name)
                elif cat == "financial":
                    financial_columns.append(col_name)

            # Count high-risk columns
            if col_classification.sensitivity_level in ("confidential", "restricted"):
                high_risk_count += 1

        # Determine overall sensitivity
        overall_sensitivity = self._determine_overall_sensitivity(columns)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pii_columns, phi_columns, pci_columns, financial_columns, overall_sensitivity
        )

        # Check compliance flags
        compliance_flags = self._check_compliance(
            pii_columns, phi_columns, pci_columns, financial_columns
        )

        return ClassificationReport(
            source_name=source_name,
            source_type=source_type,
            classification_timestamp=datetime.now().isoformat(),
            overall_sensitivity=overall_sensitivity,
            categories_found=list(all_categories),
            columns=columns,
            pii_columns=pii_columns,
            phi_columns=phi_columns,
            pci_columns=pci_columns,
            financial_columns=financial_columns,
            row_count=len(df),
            columns_analyzed=len(df.columns),
            high_risk_count=high_risk_count,
            recommendations=recommendations,
            compliance_flags=compliance_flags
        )

    def _classify_column(self, df: Any, col_name: str) -> ColumnClassification:
        """Classify a single column."""
        categories: List[str] = []
        detected_patterns: List[str] = []
        sample_matches: List[str] = []
        pii_type: Optional[str] = None
        max_confidence = 0.0

        col_lower = col_name.lower()
        col_data = df[col_name].dropna().astype(str)

        # Check column name hints
        name_hint_score = 0.0
        if any(hint in col_lower for hint in self.PII_COLUMN_HINTS):
            name_hint_score = 0.3
            categories.append("pii")
        if any(hint in col_lower for hint in self.PHI_COLUMN_HINTS):
            name_hint_score = max(name_hint_score, 0.3)
            if "phi" not in categories:
                categories.append("phi")
        if any(hint in col_lower for hint in self.FINANCIAL_COLUMN_HINTS):
            name_hint_score = max(name_hint_score, 0.3)
            if "financial" not in categories:
                categories.append("financial")

        # Check patterns in data
        sample = col_data.head(min(100, len(col_data)))

        # PII patterns
        for pattern_name, pattern in self._compiled_pii.items():
            matches = sample[sample.str.contains(pattern, na=False)]
            if len(matches) > 0:
                match_rate = len(matches) / len(sample)
                if match_rate > 0.1:  # At least 10% match
                    confidence = min(0.5 + match_rate * 0.5, 1.0)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        pii_type = pattern_name
                    detected_patterns.append(f"pii:{pattern_name}")
                    sample_matches.extend(matches.head(3).tolist())
                    if "pii" not in categories:
                        categories.append("pii")

        # PHI patterns
        for pattern_name, pattern in self._compiled_phi.items():
            matches = sample[sample.str.contains(pattern, na=False)]
            if len(matches) > 0:
                match_rate = len(matches) / len(sample)
                if match_rate > 0.1:
                    confidence = min(0.5 + match_rate * 0.5, 1.0)
                    max_confidence = max(max_confidence, confidence)
                    detected_patterns.append(f"phi:{pattern_name}")
                    sample_matches.extend(matches.head(3).tolist())
                    if "phi" not in categories:
                        categories.append("phi")

        # Financial patterns
        for pattern_name, pattern in self._compiled_financial.items():
            matches = sample[sample.str.contains(pattern, na=False)]
            if len(matches) > 0:
                match_rate = len(matches) / len(sample)
                if match_rate > 0.1:
                    confidence = min(0.5 + match_rate * 0.5, 1.0)
                    max_confidence = max(max_confidence, confidence)
                    detected_patterns.append(f"financial:{pattern_name}")
                    sample_matches.extend(matches.head(3).tolist())
                    if "financial" not in categories:
                        categories.append("financial")
                    if pattern_name == "credit_card":
                        if "pci" not in categories:
                            categories.append("pci")

        # Custom patterns
        for pattern_name, pattern in self._compiled_custom.items():
            matches = sample[sample.str.contains(pattern, na=False)]
            if len(matches) > 0:
                match_rate = len(matches) / len(sample)
                if match_rate > 0.1:
                    detected_patterns.append(f"custom:{pattern_name}")
                    sample_matches.extend(matches.head(3).tolist())

        # Combine confidence from name hints and pattern matching
        final_confidence = max(max_confidence, name_hint_score)
        if max_confidence > 0 and name_hint_score > 0:
            final_confidence = min(max_confidence + name_hint_score * 0.5, 1.0)

        # Determine sensitivity level
        sensitivity_level = self._determine_column_sensitivity(categories, final_confidence)

        # Generate recommendations
        recommendations = self._generate_column_recommendations(
            col_name, categories, sensitivity_level, pii_type
        )

        return ColumnClassification(
            name=col_name,
            original_type=str(df[col_name].dtype),
            categories=categories,
            sensitivity_level=sensitivity_level,
            pii_type=pii_type,
            confidence=final_confidence,
            detected_patterns=detected_patterns,
            sample_matches=sample_matches[:3],
            recommendations=recommendations
        )

    def _determine_column_sensitivity(
        self,
        categories: List[str],
        confidence: float
    ) -> str:
        """Determine sensitivity level based on categories."""
        if not categories:
            return "public"

        if "phi" in categories or "pci" in categories:
            return "restricted"
        if "pii" in categories and confidence > 0.7:
            return "restricted"
        if "pii" in categories or "financial" in categories:
            return "confidential"

        return "internal"

    def _determine_overall_sensitivity(
        self,
        columns: List[ColumnClassification]
    ) -> str:
        """Determine overall sensitivity for the data source."""
        levels = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        max_level = 0

        for col in columns:
            level = levels.get(col.sensitivity_level, 0)
            max_level = max(max_level, level)

        reverse_levels = {v: k for k, v in levels.items()}
        return reverse_levels[max_level]

    def _generate_column_recommendations(
        self,
        col_name: str,
        categories: List[str],
        sensitivity: str,
        pii_type: Optional[str]
    ) -> List[str]:
        """Generate recommendations for a column."""
        recommendations = []

        if "pii" in categories:
            recommendations.append(f"Consider masking or encrypting '{col_name}'")
            if pii_type in ("cpf", "cnpj", "ssn"):
                recommendations.append(f"Implement tokenization for {pii_type.upper()}")

        if "phi" in categories:
            recommendations.append(f"Apply HIPAA/LGPD health data controls to '{col_name}'")
            recommendations.append("Restrict access to authorized healthcare personnel")

        if "pci" in categories:
            recommendations.append(f"Apply PCI-DSS controls to '{col_name}'")
            recommendations.append("Do not store CVV/CVC values")

        if sensitivity == "restricted":
            recommendations.append("Implement field-level encryption")
            recommendations.append("Enable audit logging for access")

        return recommendations

    def _generate_recommendations(
        self,
        pii_columns: List[str],
        phi_columns: List[str],
        pci_columns: List[str],
        financial_columns: List[str],
        overall_sensitivity: str
    ) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []

        if pii_columns:
            recommendations.append(
                f"LGPD/GDPR: {len(pii_columns)} columns contain PII - implement data protection controls"
            )

        if phi_columns:
            recommendations.append(
                f"HIPAA: {len(phi_columns)} columns contain PHI - apply healthcare data regulations"
            )

        if pci_columns:
            recommendations.append(
                f"PCI-DSS: {len(pci_columns)} columns contain payment data - ensure compliance"
            )

        if financial_columns:
            recommendations.append(
                f"Financial data: {len(financial_columns)} columns require financial controls"
            )

        if overall_sensitivity in ("confidential", "restricted"):
            recommendations.extend([
                "Implement role-based access control (RBAC)",
                "Enable encryption at rest and in transit",
                "Configure data retention policies",
                "Set up access audit logging"
            ])

        return recommendations

    def _check_compliance(
        self,
        pii_columns: List[str],
        phi_columns: List[str],
        pci_columns: List[str],
        financial_columns: List[str]
    ) -> List[str]:
        """Check and flag compliance requirements."""
        flags = []

        if pii_columns:
            flags.append("LGPD - Lei Geral de Proteção de Dados (Brazil)")
            flags.append("GDPR - General Data Protection Regulation (EU)")

        if phi_columns:
            flags.append("HIPAA - Health Insurance Portability and Accountability Act")
            flags.append("CFM - Conselho Federal de Medicina (Brazil)")

        if pci_columns:
            flags.append("PCI-DSS - Payment Card Industry Data Security Standard")

        if financial_columns:
            flags.append("SOX - Sarbanes-Oxley Act")
            flags.append("BACEN - Banco Central do Brasil regulations")

        return flags

    def add_custom_pattern(self, name: str, pattern: str) -> None:
        """Add a custom pattern for detection."""
        self.custom_patterns[name] = pattern
        self._compiled_custom[name] = re.compile(pattern, re.IGNORECASE)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "pii_patterns": len(self.PII_PATTERNS),
            "phi_patterns": len(self.PHI_PATTERNS),
            "financial_patterns": len(self.FINANCIAL_PATTERNS),
            "custom_patterns": len(self.custom_patterns),
            "sample_size": self.sample_size
        }
