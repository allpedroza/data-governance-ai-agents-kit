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
Sensitive Data NER Agent

Named Entity Recognition for detecting and anonymizing sensitive data
including PII, PHI, PCI, Financial, Business-sensitive information,
and Credentials (API keys, tokens, secrets).

This module serves as a protective filter for LLM requests, preventing
sensitive data leakage to third-party AI services.

Features:
- Deterministic detection (regex patterns for 50+ entity types)
- Predictive detection (heuristics, checksum validation, context analysis)
- Multiple anonymization strategies (REDACT, MASK, HASH, PARTIAL, ENCRYPT)
- Secure vault for storing original/anonymized mappings with:
  - AES-256 encryption
  - Key rotation
  - Role-based access control
  - Audit logging
"""

from .agent import (
    SensitiveDataNERAgent,
    NERResult,
    DetectedEntity,
    FilterAction,
    FilterPolicy,
    AnonymizationConfig,
    EntityCategory,
    analyze_text,
)

from .anonymizers import (
    AnonymizationStrategy,
    TextAnonymizer,
    AnonymizedEntity,
    anonymize_text,
)

# Optional vault imports (requires cryptography package)
try:
    from .vault import (
        SecureVault,
        VaultConfig,
        VaultSession,
        AccessLevel,
        AccessController,
        KeyManager,
        AuditLogger,
        AuditEventType,
        RetentionPolicy,
    )
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False

__all__ = [
    # Agent
    "SensitiveDataNERAgent",
    "NERResult",
    "DetectedEntity",
    "FilterAction",
    "FilterPolicy",
    "EntityCategory",
    "analyze_text",
    # Anonymization
    "AnonymizationConfig",
    "AnonymizationStrategy",
    "TextAnonymizer",
    "AnonymizedEntity",
    "anonymize_text",
    # Vault (optional)
    "SecureVault",
    "VaultConfig",
    "VaultSession",
    "AccessLevel",
    "AccessController",
    "KeyManager",
    "AuditLogger",
    "AuditEventType",
    "RetentionPolicy",
    "HAS_VAULT",
]

__version__ = "1.2.0"
