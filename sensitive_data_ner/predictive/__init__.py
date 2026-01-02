"""
Predictive Detection Module

Contains validation functions and heuristics for improving
entity detection accuracy beyond simple regex matching.
"""

from .validators import (
    validate_cpf,
    validate_cnpj,
    validate_credit_card,
    validate_cns,
    validate_iban,
    validate_ssn,
    validate_ip_address,
    get_validator,
)

from .heuristics import (
    PredictiveDetector,
    ContextAnalyzer,
    calculate_entity_confidence,
)

__all__ = [
    "validate_cpf",
    "validate_cnpj",
    "validate_credit_card",
    "validate_cns",
    "validate_iban",
    "validate_ssn",
    "validate_ip_address",
    "get_validator",
    "PredictiveDetector",
    "ContextAnalyzer",
    "calculate_entity_confidence",
]
