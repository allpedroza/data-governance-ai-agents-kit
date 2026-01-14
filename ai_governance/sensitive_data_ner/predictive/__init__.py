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
"""
Predictive Detection Module

Contains validation functions and heuristics for improving
entity detection accuracy beyond simple regex matching.

Features:
- Checksum validation (CPF, CNPJ, credit cards, IBAN, etc.)
- Context analysis and keyword detection
- SpaCy-based NER enhancement (when available)
"""

from .validators import (
    validate_cpf,
    validate_cnpj,
    validate_credit_card,
    validate_cns,
    validate_iban,
    validate_ssn,
    validate_ip_address,
    validate_person_name,
    validate_swift_bic,
    get_validator,
)

from .heuristics import (
    PredictiveDetector,
    ContextAnalyzer,
    calculate_entity_confidence,
)

# SpaCy helper (optional - gracefully handles missing SpaCy)
try:
    from .spacy_helper import (
        is_spacy_available,
        load_spacy_model,
        validate_person_name_with_spacy,
        contains_verb,
        extract_entities,
        get_pos_tags,
    )
    _SPACY_EXPORTS = [
        "is_spacy_available",
        "load_spacy_model",
        "validate_person_name_with_spacy",
        "contains_verb",
        "extract_entities",
        "get_pos_tags",
    ]
except ImportError:
    _SPACY_EXPORTS = []

__all__ = [
    "validate_cpf",
    "validate_cnpj",
    "validate_credit_card",
    "validate_cns",
    "validate_iban",
    "validate_ssn",
    "validate_ip_address",
    "validate_person_name",
    "validate_swift_bic",
    "get_validator",
    "PredictiveDetector",
    "ContextAnalyzer",
    "calculate_entity_confidence",
] + _SPACY_EXPORTS
