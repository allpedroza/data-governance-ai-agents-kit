"""
AI Governance Agents

This package contains agents focused on AI governance, including:
- Sensitive Data NER: Detection and anonymization of sensitive data in text
- AI Business Value: Business value assessment for AI initiatives

These agents help organizations manage AI-related risks and maximize
the value of AI investments.
"""

from .ai_business_value import (
    AIBusinessValueAgent,
    AIInitiativeReport,
    AIInitiativeScore,
    CostBreakdown,
    BenefitProjection,
    RiskAssessment,
    ValueCalculator,
    ROICalculator,
    ValueCategory,
    InitiativeType,
    RiskLevel,
    MaturityLevel,
)

# Import NER agent when available
try:
    from .sensitive_data_ner import (
        SensitiveDataNERAgent,
        NERResult,
        DetectedEntity,
        FilterPolicy,
        FilterAction,
    )
    _ner_available = True
except ImportError:
    _ner_available = False

__all__ = [
    # AI Business Value
    'AIBusinessValueAgent',
    'AIInitiativeReport',
    'AIInitiativeScore',
    'CostBreakdown',
    'BenefitProjection',
    'RiskAssessment',
    'ValueCalculator',
    'ROICalculator',
    'ValueCategory',
    'InitiativeType',
    'RiskLevel',
    'MaturityLevel',
]

if _ner_available:
    __all__.extend([
        'SensitiveDataNERAgent',
        'NERResult',
        'DetectedEntity',
        'FilterPolicy',
        'FilterAction',
    ])

__version__ = '1.0.0'
