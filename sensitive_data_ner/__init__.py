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
    "HAS_VAULT",
]

__version__ = "1.1.0"
