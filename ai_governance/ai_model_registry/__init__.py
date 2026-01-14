"""
AI Model Registry Agent

Inventário vivo de modelos/LLMs com rastreamento de versões, stages, owners e propósito.
Objetivo: eliminar "shadow AI" e servir como base para todos os controles de AI Governance.
"""

from .agent import (
    # Main Agent
    AIModelRegistryAgent,
    # Enums
    ModelType,
    ModelStage,
    ModelStatus,
    RiskLevel,
    DataSensitivity,
    LicenseType,
    ChangeType,
    # Data Classes
    ModelOwner,
    ModelMetrics,
    ModelVersion,
    ModelDependency,
    DataSource,
    AuditEntry,
    RegisteredModel,
    RegistryStatistics,
    SearchResult,
)

__all__ = [
    # Main Agent
    "AIModelRegistryAgent",
    # Enums
    "ModelType",
    "ModelStage",
    "ModelStatus",
    "RiskLevel",
    "DataSensitivity",
    "LicenseType",
    "ChangeType",
    # Data Classes
    "ModelOwner",
    "ModelMetrics",
    "ModelVersion",
    "ModelDependency",
    "DataSource",
    "AuditEntry",
    "RegisteredModel",
    "RegistryStatistics",
    "SearchResult",
]

__version__ = "1.0.0"
