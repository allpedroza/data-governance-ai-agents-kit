"""
Sensitive Data NER Agent

Named Entity Recognition for detecting and anonymizing sensitive data
including PII, PHI, PCI, Financial, and Business-sensitive information.

This module serves as a protective filter for LLM requests, preventing
sensitive data leakage to third-party AI services.
"""

from .agent import (
    SensitiveDataNERAgent,
    NERResult,
    DetectedEntity,
    AnonymizationConfig,
    EntityCategory,
)

__all__ = [
    "SensitiveDataNERAgent",
    "NERResult",
    "DetectedEntity",
    "AnonymizationConfig",
    "EntityCategory",
]

__version__ = "1.0.0"
