"""
Entity Patterns Module

Contains regex patterns and context keywords for detecting
sensitive data entities across multiple categories.
"""

from .entity_patterns import (
    PII_PATTERNS,
    PHI_PATTERNS,
    PCI_PATTERNS,
    FINANCIAL_PATTERNS,
    CONTEXT_KEYWORDS,
    EntityPatternConfig,
    get_all_patterns,
)

__all__ = [
    "PII_PATTERNS",
    "PHI_PATTERNS",
    "PCI_PATTERNS",
    "FINANCIAL_PATTERNS",
    "CONTEXT_KEYWORDS",
    "EntityPatternConfig",
    "get_all_patterns",
]
