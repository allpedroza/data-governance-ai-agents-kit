"""
Data Classification Agent

Classifies data based on sensitivity levels (PII, PHI, Financial, etc.)
using pattern matching, metadata analysis, and configurable rules.
"""

from .agent import (
    DataClassificationAgent,
    ClassificationReport,
    ColumnClassification,
    SensitivityLevel,
    DataCategory
)

__all__ = [
    "DataClassificationAgent",
    "ClassificationReport",
    "ColumnClassification",
    "SensitivityLevel",
    "DataCategory"
]
