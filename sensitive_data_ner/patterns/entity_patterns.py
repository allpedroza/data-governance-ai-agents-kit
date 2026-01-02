"""
Entity Patterns for Sensitive Data Detection

This module contains all regex patterns and context keywords used
for detecting sensitive data entities in text.

Categories:
- PII: Personally Identifiable Information
- PHI: Protected Health Information
- PCI: Payment Card Industry Data
- FINANCIAL: Financial and banking information
- BUSINESS: Strategic/proprietary business information
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Pattern
from enum import Enum
import re


class EntityCategory(Enum):
    """Categories of sensitive data entities."""
    PII = "pii"
    PHI = "phi"
    PCI = "pci"
    FINANCIAL = "financial"
    BUSINESS = "business"


@dataclass
class EntityPatternConfig:
    """Configuration for an entity pattern."""
    name: str
    category: EntityCategory
    pattern: str
    description: str
    has_validation: bool = False  # Whether a validation function exists
    context_boost: float = 0.2  # Confidence boost when context keywords found
    priority: int = 1  # Higher priority patterns are checked first
    locale: str = "universal"  # Country/region specific (e.g., "br", "us", "eu")

    def __post_init__(self):
        self._compiled: Pattern = re.compile(self.pattern, re.IGNORECASE)

    @property
    def compiled(self) -> Pattern:
        return self._compiled


# =============================================================================
# PII PATTERNS - Personally Identifiable Information
# =============================================================================

PII_PATTERNS: Dict[str, EntityPatternConfig] = {
    # Brazilian Documents
    "cpf": EntityPatternConfig(
        name="cpf",
        category=EntityCategory.PII,
        pattern=r"\b\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2}\b",
        description="Brazilian CPF (Individual Taxpayer ID)",
        has_validation=True,
        locale="br",
        priority=2
    ),
    "cnpj": EntityPatternConfig(
        name="cnpj",
        category=EntityCategory.PII,
        pattern=r"\b\d{2}\.?\d{3}\.?\d{3}/?0001[-.]?\d{2}\b",
        description="Brazilian CNPJ (Company Tax ID)",
        has_validation=True,
        locale="br",
        priority=2
    ),
    "rg": EntityPatternConfig(
        name="rg",
        category=EntityCategory.PII,
        pattern=r"\b\d{1,2}\.?\d{3}\.?\d{3}[-.]?[\dxX]\b",
        description="Brazilian RG (Identity Card)",
        locale="br"
    ),

    # US Documents
    "ssn": EntityPatternConfig(
        name="ssn",
        category=EntityCategory.PII,
        pattern=r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        description="US Social Security Number",
        has_validation=True,
        locale="us",
        priority=2
    ),
    "ein": EntityPatternConfig(
        name="ein",
        category=EntityCategory.PII,
        pattern=r"\b\d{2}[-]?\d{7}\b",
        description="US Employer Identification Number",
        locale="us"
    ),

    # European Documents
    "nif_es": EntityPatternConfig(
        name="nif_es",
        category=EntityCategory.PII,
        pattern=r"\b[A-Z]?\d{7,8}[A-Z]\b",
        description="Spanish NIF/NIE",
        locale="es"
    ),
    "nif_pt": EntityPatternConfig(
        name="nif_pt",
        category=EntityCategory.PII,
        pattern=r"\b[1-9]\d{8}\b",
        description="Portuguese NIF",
        locale="pt"
    ),

    # Universal Identifiers
    "email": EntityPatternConfig(
        name="email",
        category=EntityCategory.PII,
        pattern=r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        description="Email Address",
        priority=3
    ),
    "phone_intl": EntityPatternConfig(
        name="phone_intl",
        category=EntityCategory.PII,
        pattern=r"\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
        description="International Phone Number"
    ),
    "phone_br": EntityPatternConfig(
        name="phone_br",
        category=EntityCategory.PII,
        pattern=r"\b\(?\d{2}\)?[-.\s]?\d{4,5}[-.]?\d{4}\b",
        description="Brazilian Phone Number",
        locale="br"
    ),
    "passport": EntityPatternConfig(
        name="passport",
        category=EntityCategory.PII,
        pattern=r"\b[A-Z]{1,2}\d{6,9}\b",
        description="Passport Number"
    ),
    "drivers_license": EntityPatternConfig(
        name="drivers_license",
        category=EntityCategory.PII,
        pattern=r"\b[A-Z]{0,2}\d{5,12}\b",
        description="Driver's License Number"
    ),

    # Network Identifiers
    "ip_address": EntityPatternConfig(
        name="ip_address",
        category=EntityCategory.PII,
        pattern=r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        description="IPv4 Address",
        has_validation=True
    ),
    "ipv6_address": EntityPatternConfig(
        name="ipv6_address",
        category=EntityCategory.PII,
        pattern=r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        description="IPv6 Address"
    ),
    "mac_address": EntityPatternConfig(
        name="mac_address",
        category=EntityCategory.PII,
        pattern=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        description="MAC Address"
    ),

    # Personal Names (predictive - context-dependent)
    "person_name": EntityPatternConfig(
        name="person_name",
        category=EntityCategory.PII,
        pattern=r"\b[A-Z][a-záàâãéèêíïóôõöúçñ]+(?:\s+(?:de|da|do|dos|das|e|van|von|der|del|la|el))?\s+[A-Z][a-záàâãéèêíïóôõöúçñ]+(?:\s+[A-Z][a-záàâãéèêíïóôõöúçñ]+)*\b",
        description="Person Full Name (Portuguese/Spanish)",
        context_boost=0.3
    ),

    # Address Components
    "cep": EntityPatternConfig(
        name="cep",
        category=EntityCategory.PII,
        pattern=r"\b\d{5}[-.]?\d{3}\b",
        description="Brazilian CEP (Postal Code)",
        locale="br"
    ),
    "zipcode_us": EntityPatternConfig(
        name="zipcode_us",
        category=EntityCategory.PII,
        pattern=r"\b\d{5}(?:[-]?\d{4})?\b",
        description="US ZIP Code",
        locale="us"
    ),

    # Dates (potential DOB)
    "date_br": EntityPatternConfig(
        name="date_br",
        category=EntityCategory.PII,
        pattern=r"\b(?:0[1-9]|[12]\d|3[01])[/.-](?:0[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b",
        description="Date (DD/MM/YYYY - Brazilian format)",
        locale="br"
    ),
    "date_us": EntityPatternConfig(
        name="date_us",
        category=EntityCategory.PII,
        pattern=r"\b(?:0[1-9]|1[0-2])[/.-](?:0[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b",
        description="Date (MM/DD/YYYY - US format)",
        locale="us"
    ),
}


# =============================================================================
# PHI PATTERNS - Protected Health Information
# =============================================================================

PHI_PATTERNS: Dict[str, EntityPatternConfig] = {
    # Medical Record Numbers
    "medical_record": EntityPatternConfig(
        name="medical_record",
        category=EntityCategory.PHI,
        pattern=r"\b(?:MRN|PRN|RH|HC|PRONT)[-:\s]?\d{5,12}\b",
        description="Medical Record Number",
        priority=2
    ),

    # Health Cards
    "cns": EntityPatternConfig(
        name="cns",
        category=EntityCategory.PHI,
        pattern=r"\b[1-2]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b",
        description="Brazilian CNS (National Health Card)",
        has_validation=True,
        locale="br",
        priority=2
    ),
    "sus_card": EntityPatternConfig(
        name="sus_card",
        category=EntityCategory.PHI,
        pattern=r"\b\d{15}\b",
        description="SUS Card Number (15 digits)",
        locale="br"
    ),

    # Medical Codes
    "icd10": EntityPatternConfig(
        name="icd10",
        category=EntityCategory.PHI,
        pattern=r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b",
        description="ICD-10 Diagnosis Code",
        priority=2
    ),
    "icd11": EntityPatternConfig(
        name="icd11",
        category=EntityCategory.PHI,
        pattern=r"\b[A-Z]{2}\d{2}(?:\.\d{1,2})?(?:/[A-Z0-9]+)?\b",
        description="ICD-11 Diagnosis Code"
    ),
    "cpt_code": EntityPatternConfig(
        name="cpt_code",
        category=EntityCategory.PHI,
        pattern=r"\b\d{5}[A-Z]?\b",
        description="CPT Procedure Code",
        context_boost=0.3
    ),

    # Medical License Numbers
    "crm": EntityPatternConfig(
        name="crm",
        category=EntityCategory.PHI,
        pattern=r"\bCRM[-/\s]?[A-Z]{2}[-/\s]?\d{4,6}\b",
        description="Brazilian CRM (Medical License)",
        locale="br"
    ),
    "coren": EntityPatternConfig(
        name="coren",
        category=EntityCategory.PHI,
        pattern=r"\bCOREN[-/\s]?[A-Z]{2}[-/\s]?\d{4,6}\b",
        description="Brazilian COREN (Nursing License)",
        locale="br"
    ),

    # Lab Results (patterns that may indicate PHI context)
    "blood_type": EntityPatternConfig(
        name="blood_type",
        category=EntityCategory.PHI,
        pattern=r"\b(?:tipo\s+sangu[íi]neo|blood\s+type)?:?\s*[ABO]{1,2}[+-]\b",
        description="Blood Type",
        priority=2
    ),

    # Insurance
    "health_insurance": EntityPatternConfig(
        name="health_insurance",
        category=EntityCategory.PHI,
        pattern=r"\b(?:ANS|UNIMED|BRADESCO\s+SA[UÚ]DE|SULAM[EÉ]RICA|AMIL)[-:\s]?\d{8,16}\b",
        description="Health Insurance ID (Brazil)",
        locale="br"
    ),

    # NPI (US)
    "npi": EntityPatternConfig(
        name="npi",
        category=EntityCategory.PHI,
        pattern=r"\b\d{10}\b",
        description="National Provider Identifier (US)",
        locale="us",
        context_boost=0.4
    ),

    # DEA Number (US)
    "dea_number": EntityPatternConfig(
        name="dea_number",
        category=EntityCategory.PHI,
        pattern=r"\b[A-Z]{2}\d{7}\b",
        description="DEA Registration Number",
        locale="us",
        context_boost=0.3
    ),
}


# =============================================================================
# PCI PATTERNS - Payment Card Industry Data
# =============================================================================

PCI_PATTERNS: Dict[str, EntityPatternConfig] = {
    # Credit/Debit Cards
    "credit_card": EntityPatternConfig(
        name="credit_card",
        category=EntityCategory.PCI,
        pattern=r"\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2})|3[47]\d{2}|3(?:0[0-5]|[68]\d)\d)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,4}\b",
        description="Credit/Debit Card Number",
        has_validation=True,
        priority=3
    ),
    "card_visa": EntityPatternConfig(
        name="card_visa",
        category=EntityCategory.PCI,
        pattern=r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        description="Visa Card Number",
        has_validation=True,
        priority=3
    ),
    "card_mastercard": EntityPatternConfig(
        name="card_mastercard",
        category=EntityCategory.PCI,
        pattern=r"\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        description="Mastercard Number",
        has_validation=True,
        priority=3
    ),
    "card_amex": EntityPatternConfig(
        name="card_amex",
        category=EntityCategory.PCI,
        pattern=r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",
        description="American Express Card Number",
        has_validation=True,
        priority=3
    ),

    # Card Security
    "cvv": EntityPatternConfig(
        name="cvv",
        category=EntityCategory.PCI,
        pattern=r"\b(?:CVV|CVC|CID|CSC)[-:\s]?\d{3,4}\b",
        description="Card Security Code",
        priority=3
    ),
    "card_expiry": EntityPatternConfig(
        name="card_expiry",
        category=EntityCategory.PCI,
        pattern=r"\b(?:0[1-9]|1[0-2])[/.-]?(?:2[0-9]|[3-9]\d)\b",
        description="Card Expiration Date (MM/YY)",
        context_boost=0.4
    ),

    # Magnetic Stripe Data
    "track_data": EntityPatternConfig(
        name="track_data",
        category=EntityCategory.PCI,
        pattern=r"\%?[Bb]\d{13,19}\^[A-Za-z\s/]+\^\d{4}",
        description="Magnetic Stripe Track Data",
        priority=3
    ),
}


# =============================================================================
# FINANCIAL PATTERNS - Banking and Financial Information
# =============================================================================

FINANCIAL_PATTERNS: Dict[str, EntityPatternConfig] = {
    # Bank Accounts
    "iban": EntityPatternConfig(
        name="iban",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b",
        description="International Bank Account Number",
        has_validation=True,
        priority=2
    ),
    "swift_bic": EntityPatternConfig(
        name="swift_bic",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
        description="SWIFT/BIC Code",
        priority=2
    ),
    "bank_account_br": EntityPatternConfig(
        name="bank_account_br",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b(?:AG|AGENCIA)[-:\s]?\d{4}[-\s]?\d?[-\s]?(?:CC|CONTA)[-:\s]?\d{5,12}[-]?\d?\b",
        description="Brazilian Bank Account (Agency + Account)",
        locale="br"
    ),
    "routing_number": EntityPatternConfig(
        name="routing_number",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b\d{9}\b",
        description="US Bank Routing Number (ABA)",
        locale="us",
        context_boost=0.4
    ),

    # PIX (Brazilian Instant Payment)
    "pix_key_phone": EntityPatternConfig(
        name="pix_key_phone",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b\+55\s?\d{2}\s?\d{5}[-]?\d{4}\b",
        description="PIX Key (Phone)",
        locale="br"
    ),
    "pix_key_random": EntityPatternConfig(
        name="pix_key_random",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
        description="PIX Random Key (UUID)",
        locale="br"
    ),

    # Cryptocurrency
    "bitcoin_address": EntityPatternConfig(
        name="bitcoin_address",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,59}\b",
        description="Bitcoin Address"
    ),
    "ethereum_address": EntityPatternConfig(
        name="ethereum_address",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b0x[a-fA-F0-9]{40}\b",
        description="Ethereum Address"
    ),

    # Tax Related
    "tax_id": EntityPatternConfig(
        name="tax_id",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b(?:TAX\s*ID|TIN|VAT)[-:\s]?[A-Z0-9]{8,15}\b",
        description="Tax Identification Number"
    ),

    # Financial Values (for context detection)
    "currency_value": EntityPatternConfig(
        name="currency_value",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b(?:R\$|US\$|\$|€|£)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b",
        description="Currency Value",
        context_boost=0.2
    ),

    # Investment Accounts
    "brokerage_account": EntityPatternConfig(
        name="brokerage_account",
        category=EntityCategory.FINANCIAL,
        pattern=r"\b(?:CONTA|ACCOUNT)[-:\s]?\d{6,12}\b",
        description="Brokerage Account Number",
        context_boost=0.3
    ),
}


# =============================================================================
# CONTEXT KEYWORDS - For Predictive Detection
# =============================================================================

CONTEXT_KEYWORDS: Dict[EntityCategory, Dict[str, Set[str]]] = {
    EntityCategory.PII: {
        "high_confidence": {
            "cpf", "cnpj", "rg", "ssn", "social security", "passport",
            "carteira de identidade", "documento", "identidade",
            "nome completo", "full name", "data de nascimento", "birth date",
            "endereço", "address", "telefone", "phone", "celular", "mobile",
            "email", "e-mail", "correio eletrônico"
        },
        "medium_confidence": {
            "cliente", "customer", "usuário", "user", "pessoa", "person",
            "titular", "holder", "proprietário", "owner", "contato", "contact",
            "cadastro", "registration", "matrícula", "enrollment"
        },
        "low_confidence": {
            "nome", "name", "sobrenome", "surname", "idade", "age",
            "sexo", "gender", "nacionalidade", "nationality"
        }
    },
    EntityCategory.PHI: {
        "high_confidence": {
            "diagnóstico", "diagnosis", "paciente", "patient",
            "prontuário", "medical record", "receita", "prescription",
            "medicamento", "medication", "tratamento", "treatment",
            "cid", "icd", "exame", "exam", "resultado", "result",
            "cirurgia", "surgery", "hospital", "clínica", "clinic",
            "médico", "doctor", "enfermeiro", "nurse"
        },
        "medium_confidence": {
            "saúde", "health", "doença", "disease", "sintoma", "symptom",
            "alergia", "allergy", "histórico médico", "medical history",
            "consulta", "appointment", "internação", "hospitalization",
            "laboratório", "laboratory", "sangue", "blood"
        },
        "low_confidence": {
            "peso", "weight", "altura", "height", "pressão", "pressure",
            "temperatura", "temperature", "frequência", "frequency"
        }
    },
    EntityCategory.PCI: {
        "high_confidence": {
            "cartão", "card", "crédito", "credit", "débito", "debit",
            "visa", "mastercard", "amex", "american express",
            "cvv", "cvc", "validade", "expiry", "expiration",
            "bandeira", "brand", "número do cartão", "card number"
        },
        "medium_confidence": {
            "pagamento", "payment", "transação", "transaction",
            "compra", "purchase", "fatura", "invoice", "bill"
        },
        "low_confidence": {
            "loja", "store", "venda", "sale", "valor", "amount"
        }
    },
    EntityCategory.FINANCIAL: {
        "high_confidence": {
            "banco", "bank", "conta", "account", "agência", "branch",
            "iban", "swift", "bic", "transferência", "transfer",
            "pix", "saldo", "balance", "extrato", "statement",
            "investimento", "investment", "aplicação", "deposit"
        },
        "medium_confidence": {
            "financeiro", "financial", "dinheiro", "money",
            "empréstimo", "loan", "crédito", "credit", "débito", "debit",
            "juros", "interest", "parcela", "installment"
        },
        "low_confidence": {
            "valor", "value", "preço", "price", "custo", "cost",
            "receita", "revenue", "despesa", "expense"
        }
    },
    EntityCategory.BUSINESS: {
        "high_confidence": {
            "confidencial", "confidential", "restrito", "restricted",
            "estratégico", "strategic", "proprietário", "proprietary",
            "segredo", "secret", "interno", "internal",
            "projeto", "project", "iniciativa", "initiative"
        },
        "medium_confidence": {
            "aquisição", "acquisition", "fusão", "merger",
            "parceria", "partnership", "contrato", "contract",
            "negociação", "negotiation", "proposta", "proposal",
            "roadmap", "pipeline", "forecast", "previsão"
        },
        "low_confidence": {
            "meta", "goal", "objetivo", "objective", "plano", "plan",
            "orçamento", "budget", "resultado", "result"
        }
    }
}


def get_all_patterns() -> Dict[str, EntityPatternConfig]:
    """Return all patterns merged into a single dictionary."""
    all_patterns = {}
    all_patterns.update(PII_PATTERNS)
    all_patterns.update(PHI_PATTERNS)
    all_patterns.update(PCI_PATTERNS)
    all_patterns.update(FINANCIAL_PATTERNS)
    return all_patterns
