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
