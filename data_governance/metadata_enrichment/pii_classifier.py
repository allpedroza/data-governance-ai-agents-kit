"""
PII Classifier for Data Governance

Classifies columns as PII (Personally Identifiable Information) using two approaches:

  1. Name-based estimation (no data access):
     Infers likely PII from column names using keyword matching.
     Fast — runs during the catalog scan without sampling data.

  2. Data-based classification (requires SampleResult):
     Uses DataSampler pattern detection results (regex matches on actual values).
     Runs after sampling, during enrichment — provides evidence and match ratios.

LGPD mapping (Lei Geral de Proteção de Dados, Lei 13.709/2018):
  - pessoal_ordinario: data that identifies a natural person (name, CPF, email, phone...)
  - pessoal_sensivel:  sensitive personal data (Art. 5 IX): health, biometric, financial,
                        racial/ethnic origin, religious/political conviction
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LGPD category metadata
# ---------------------------------------------------------------------------

_PII_TYPES: Dict[str, Dict] = {
    "cpf":         {"label": "CPF",                   "lgpd": "pessoal_ordinario",  "risk": "high"},
    "rg":          {"label": "RG / Identidade",       "lgpd": "pessoal_ordinario",  "risk": "high"},
    "passport":    {"label": "Passaporte",             "lgpd": "pessoal_ordinario",  "risk": "high"},
    "document_id": {"label": "Documento identificação","lgpd": "pessoal_ordinario",  "risk": "high"},
    "credit_card": {"label": "Cartão de crédito",     "lgpd": "pessoal_sensivel",   "risk": "high"},
    "email":       {"label": "E-mail",                "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "phone":       {"label": "Telefone",              "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "name":        {"label": "Nome",                  "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "address":     {"label": "Endereço",              "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "birthdate":   {"label": "Data de nascimento",    "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "ip_address":  {"label": "Endereço IP",           "lgpd": "pessoal_ordinario",  "risk": "medium"},
    "salary":      {"label": "Salário / Renda",       "lgpd": "pessoal_sensivel",   "risk": "high"},
    "health":      {"label": "Dado de saúde",         "lgpd": "pessoal_sensivel",   "risk": "high"},
    "biometric":   {"label": "Dado biométrico",       "lgpd": "pessoal_sensivel",   "risk": "high"},
    "password":    {"label": "Senha / Credencial",    "lgpd": "pessoal_sensivel",   "risk": "high"},
    "cep":         {"label": "CEP",                   "lgpd": "pessoal_ordinario",  "risk": "low"},
    "cnpj":        {"label": "CNPJ",                  "lgpd": "nao_pessoal",        "risk": "low"},
}

# Mapping from DataSampler pattern names to PII types
_SAMPLER_PATTERN_TO_PII: Dict[str, str] = {
    "cpf":         "cpf",
    "email":       "email",
    "phone":       "phone",
    "credit_card": "credit_card",
    "ip_address":  "ip_address",
    "cep":         "cep",
    "cnpj":        "cnpj",
}

# Mapping from DataSampler inferred_semantic_type to PII types
_SEMANTIC_TYPE_TO_PII: Dict[str, str] = {
    "pii":     "cpf",        # generic pii — will be refined by pattern
    "email":   "email",
    "phone":   "phone",
    "name":    "name",
    "address": "address",
}

# Column name keywords → PII type (checked with word boundary / substring)
_NAME_SIGNALS: List[Tuple[str, str]] = [
    # High confidence exact keywords
    ("cpf",              "cpf"),
    ("cnpj",             "cnpj"),
    ("rg",               "rg"),
    ("identidade",       "rg"),
    ("passaporte",       "passport"),
    ("passport",         "passport"),
    ("documento",        "document_id"),
    ("document",         "document_id"),
    ("credit_card",      "credit_card"),
    ("cartao",           "credit_card"),
    ("card_number",      "credit_card"),
    ("num_cartao",       "credit_card"),
    ("email",            "email"),
    ("e_mail",           "email"),
    ("phone",            "phone"),
    ("telefone",         "phone"),
    ("celular",          "phone"),
    ("fone",             "phone"),
    # Medium confidence — match as word parts
    ("nome",             "name"),
    ("name",             "name"),
    ("razao_social",     "name"),
    ("nomecomp",         "name"),
    ("endereco",         "address"),
    ("address",          "address"),
    ("logradouro",       "address"),
    ("rua",              "address"),
    ("cep",              "cep"),
    ("zipcode",          "cep"),
    ("zip_code",         "cep"),
    ("nascimento",       "birthdate"),
    ("birthday",         "birthdate"),
    ("birth_date",       "birthdate"),
    ("dt_nasc",          "birthdate"),
    ("data_nasc",        "birthdate"),
    ("salario",          "salary"),
    ("salary",           "salary"),
    ("renda",            "salary"),
    ("income",           "salary"),
    ("senha",            "password"),
    ("password",         "password"),
    ("pwd",              "password"),
    ("hash_senha",       "password"),
    ("saude",            "health"),
    ("health",           "health"),
    ("diagnostico",      "health"),
    ("cid",              "health"),
    ("doenca",           "health"),
    ("biometria",        "biometric"),
    ("biometric",        "biometric"),
    ("fingerprint",      "biometric"),
    ("foto",             "biometric"),
    ("photo",            "biometric"),
    ("ip_address",       "ip_address"),
    ("ip_addr",          "ip_address"),
    ("endereco_ip",      "ip_address"),
]

_RISK_ORDER = {"high": 3, "medium": 2, "low": 1, "none": 0}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PIIColumnDiagnosis:
    """PII classification result for a single column."""
    column_name: str
    pii_types: List[str]              # e.g. ["cpf", "email"]
    pii_labels: List[str]             # e.g. ["CPF", "E-mail"]
    lgpd_categories: List[str]        # "pessoal_ordinario" | "pessoal_sensivel" | "nao_pessoal"
    risk_level: str                   # "high" | "medium" | "low"
    detection_method: str             # "name_pattern" | "data_pattern" | "both"
    match_ratio: Optional[float]      # % of sampled values matching (data-based only)
    evidence: str                     # human-readable explanation


@dataclass
class PIITableDiagnosis:
    """PII classification result for a table."""
    table_name: str
    schema: str
    database: str
    has_pii: bool
    risk_level: str                   # "high" | "medium" | "low" | "none"
    pii_columns: List[PIIColumnDiagnosis]
    detection_method: str             # "name_only" | "data_sample" | "combined"
    lgpd_sensitive_columns: List[str] # column names with pessoal_sensivel data
    recommended_classification: str   # "restricted" | "confidential" | "internal"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PIIClassifier:
    """
    Classifies table columns as PII using name-based or data-based detection.

    Name-based (fast, no DB access): uses column names to estimate PII likelihood.
    Data-based (accurate): uses DataSampler SampleResult pattern matches.
    """

    # -------------------------------------------------------------- public

    def estimate_from_column_names(
        self,
        table_name: str,
        columns: List[str],
        schema: str = "",
        database: str = "",
    ) -> PIITableDiagnosis:
        """
        Estimate PII risk from column names alone (no data sampling needed).

        Args:
            table_name: Table name
            columns: List of column name strings
            schema, database: Optional for metadata

        Returns:
            PIITableDiagnosis with name-based detection
        """
        pii_cols: List[PIIColumnDiagnosis] = []
        for col_name in columns:
            diag = self._check_name(col_name)
            if diag:
                pii_cols.append(diag)
        return self._build_table_diagnosis(
            table_name, schema, database, pii_cols, "name_only"
        )

    def classify_from_sample(self, sample_result) -> PIITableDiagnosis:
        """
        Classify PII from a DataSampler SampleResult (pattern matches on real data).

        Args:
            sample_result: SampleResult from DataSampler (has .columns with ColumnProfile)

        Returns:
            PIITableDiagnosis with data-based evidence
        """
        pii_cols: List[PIIColumnDiagnosis] = []

        for col_profile in sample_result.columns:
            diag = self._check_profile(col_profile)
            if diag:
                pii_cols.append(diag)

        return self._build_table_diagnosis(
            sample_result.table_name, "", "", pii_cols, "data_sample"
        )

    def merge_diagnoses(
        self,
        name_diag: PIITableDiagnosis,
        data_diag: PIITableDiagnosis,
    ) -> PIITableDiagnosis:
        """
        Merge name-based and data-based diagnoses.
        Data-based evidence takes precedence; name-based fills gaps.
        """
        by_col: Dict[str, PIIColumnDiagnosis] = {}

        for cd in name_diag.pii_columns:
            by_col[cd.column_name] = cd

        for cd in data_diag.pii_columns:
            if cd.column_name in by_col:
                existing = by_col[cd.column_name]
                merged_types = list(dict.fromkeys(existing.pii_types + cd.pii_types))
                by_col[cd.column_name] = PIIColumnDiagnosis(
                    column_name=cd.column_name,
                    pii_types=merged_types,
                    pii_labels=[_PII_TYPES[t]["label"] for t in merged_types if t in _PII_TYPES],
                    lgpd_categories=list(dict.fromkeys(
                        existing.lgpd_categories + cd.lgpd_categories
                    )),
                    risk_level=max(
                        existing.risk_level, cd.risk_level,
                        key=lambda r: _RISK_ORDER.get(r, 0)
                    ),
                    detection_method="both",
                    match_ratio=cd.match_ratio,
                    evidence=f"[Nome] {existing.evidence}  |  [Dados] {cd.evidence}",
                )
            else:
                by_col[cd.column_name] = cd

        merged_cols = list(by_col.values())
        return self._build_table_diagnosis(
            name_diag.table_name, name_diag.schema, name_diag.database,
            merged_cols, "combined",
        )

    def classify_batch_from_names(
        self,
        tables: List[Dict],
    ) -> List[PIITableDiagnosis]:
        """
        Estimate PII for a batch of tables using column names only.

        Args:
            tables: List of dicts with keys: name, schema, database,
                    columns (list of {name: str, ...})

        Returns:
            List of PIITableDiagnosis
        """
        results = []
        for tbl in tables:
            col_names = [
                c.get("name", "") if isinstance(c, dict) else str(c)
                for c in (tbl.get("columns") or [])
            ]
            results.append(self.estimate_from_column_names(
                table_name=tbl.get("name", ""),
                columns=col_names,
                schema=tbl.get("schema", ""),
                database=tbl.get("database", ""),
            ))
        return results

    # -------------------------------------------------------------- private

    def _check_name(self, col_name: str) -> Optional[PIIColumnDiagnosis]:
        """Check a column name against PII name signals. Returns None if no PII detected."""
        norm = col_name.lower().replace(" ", "_")
        matched: List[str] = []
        for signal, pii_type in _NAME_SIGNALS:
            if signal in norm:
                if pii_type not in matched:
                    matched.append(pii_type)

        if not matched:
            return None

        # Filter: skip cnpj (not personal) unless also paired with personal PII
        personal = [t for t in matched if _PII_TYPES.get(t, {}).get("lgpd") != "nao_pessoal"]
        if not personal:
            return None

        matched = personal
        labels = [_PII_TYPES[t]["label"] for t in matched if t in _PII_TYPES]
        lgpd = list(dict.fromkeys(
            _PII_TYPES[t]["lgpd"] for t in matched if t in _PII_TYPES
        ))
        risk = max(
            (_PII_TYPES[t].get("risk", "low") for t in matched if t in _PII_TYPES),
            key=lambda r: _RISK_ORDER.get(r, 0),
            default="low",
        )
        evidence = f"Nome da coluna contém indicador(es) de PII: {', '.join(matched)}"
        return PIIColumnDiagnosis(
            column_name=col_name,
            pii_types=matched,
            pii_labels=labels,
            lgpd_categories=lgpd,
            risk_level=risk,
            detection_method="name_pattern",
            match_ratio=None,
            evidence=evidence,
        )

    def _check_profile(self, col_profile) -> Optional[PIIColumnDiagnosis]:
        """Check a DataSampler ColumnProfile for PII patterns. Returns None if no PII."""
        matched_types: List[str] = []
        match_ratios: Dict[str, float] = {}

        # From data patterns (regex matches on actual values)
        for pattern_name in (col_profile.patterns or []):
            pii_type = _SAMPLER_PATTERN_TO_PII.get(pattern_name)
            if pii_type and pii_type not in matched_types:
                matched_types.append(pii_type)
                match_ratios[pii_type] = 1.0  # pattern match means ≥50% threshold

        # From inferred_semantic_type
        sem = col_profile.inferred_semantic_type
        if sem:
            pii_type = _SEMANTIC_TYPE_TO_PII.get(sem)
            if pii_type and pii_type not in matched_types:
                matched_types.append(pii_type)

        # Also check column name for remaining signals
        name_diag = self._check_name(col_profile.name)
        if name_diag:
            for t in name_diag.pii_types:
                if t not in matched_types:
                    matched_types.append(t)

        if not matched_types:
            return None

        personal = [t for t in matched_types if _PII_TYPES.get(t, {}).get("lgpd") != "nao_pessoal"]
        if not personal:
            return None

        matched_types = personal
        labels = [_PII_TYPES[t]["label"] for t in matched_types if t in _PII_TYPES]
        lgpd = list(dict.fromkeys(
            _PII_TYPES[t]["lgpd"] for t in matched_types if t in _PII_TYPES
        ))
        risk = max(
            (_PII_TYPES[t].get("risk", "low") for t in matched_types if t in _PII_TYPES),
            key=lambda r: _RISK_ORDER.get(r, 0),
            default="low",
        )

        evidence_parts = []
        if col_profile.patterns:
            pii_patterns = [p for p in col_profile.patterns if p in _SAMPLER_PATTERN_TO_PII]
            if pii_patterns:
                evidence_parts.append(
                    f"Padrão detectado em dados: {', '.join(pii_patterns)}"
                )
        if sem and sem in _SEMANTIC_TYPE_TO_PII:
            evidence_parts.append(f"Tipo semântico inferido: {sem}")
        if name_diag:
            evidence_parts.append(f"Nome da coluna: {col_profile.name}")
        evidence = "  |  ".join(evidence_parts) if evidence_parts else col_profile.name

        # Best match ratio from data patterns
        best_ratio = max(match_ratios.values()) if match_ratios else None

        return PIIColumnDiagnosis(
            column_name=col_profile.name,
            pii_types=matched_types,
            pii_labels=labels,
            lgpd_categories=lgpd,
            risk_level=risk,
            detection_method="data_pattern" if col_profile.patterns else "name_pattern",
            match_ratio=best_ratio,
            evidence=evidence,
        )

    @staticmethod
    def _build_table_diagnosis(
        table_name: str,
        schema: str,
        database: str,
        pii_cols: List[PIIColumnDiagnosis],
        method: str,
    ) -> PIITableDiagnosis:
        has_pii = len(pii_cols) > 0

        if not has_pii:
            risk = "none"
        else:
            risk = max(
                (c.risk_level for c in pii_cols),
                key=lambda r: _RISK_ORDER.get(r, 0),
                default="low",
            )

        sensitive = [
            c.column_name for c in pii_cols
            if "pessoal_sensivel" in c.lgpd_categories
        ]

        if risk == "high" or sensitive:
            recommended = "restricted"
        elif risk == "medium":
            recommended = "confidential"
        elif risk == "low":
            recommended = "internal"
        else:
            recommended = "internal"

        return PIITableDiagnosis(
            table_name=table_name,
            schema=schema,
            database=database,
            has_pii=has_pii,
            risk_level=risk,
            pii_columns=pii_cols,
            detection_method=method,
            lgpd_sensitive_columns=sensitive,
            recommended_classification=recommended,
        )
