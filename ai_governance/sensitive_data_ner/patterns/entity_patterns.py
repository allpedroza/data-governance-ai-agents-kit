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
    CREDENTIALS = "credentials"


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
    case_sensitive: bool = False  # Whether pattern should be case-sensitive

    def __post_init__(self):
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled: Pattern = re.compile(self.pattern, flags)

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
        # Brazilian RG: XX.XXX.XXX-X or variations
        # Higher priority to match before generic patterns
        pattern=r"\b(?:RG[-:\s]*)?(\d{1,2}\.?\d{3}\.?\d{3}[-.]?[\dxX])\b",
        description="Brazilian RG (Identity Card)",
        locale="br",
        priority=3,  # High priority
        context_boost=0.3
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
        # Matches various Brazilian phone formats:
        # (11) 98765-4321, 11 98765-4321, 11987654321, +55 11 98765-4321
        pattern=r"\b(?:\+55[-.\s]?)?\(?\d{2}\)?[-.\s]?\d{4,5}[-.\s]?\d{4}\b",
        description="Brazilian Phone Number",
        locale="br",
        priority=2,  # Higher priority to match before generic patterns
        context_boost=0.3  # Boost when "tel", "telefone", etc. nearby
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
    # Case-sensitive to ensure proper noun capitalization
    "person_name": EntityPatternConfig(
        name="person_name",
        category=EntityCategory.PII,
        pattern=r"\b[A-Z][a-záàâãéèêíïóôõöúçñ]{2,}(?:\s+(?:de|da|do|dos|das|e|van|von|der|del|la|el|Di|De|Da|Van|Von))?\s+[A-Z][a-záàâãéèêíïóôõöúçñ]{2,}(?:\s+(?:de|da|do|dos|das|e)?\s*[A-Z][a-záàâãéèêíïóôõöúçñ]{2,})*\b",
        description="Person Full Name (Portuguese/Spanish)",
        context_boost=0.3,
        case_sensitive=True,  # Critical: names must have proper capitalization
        has_validation=True,  # Enable validation to filter false positives
        priority=2  # Higher priority to check before other patterns
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
        # CPT codes: 5 digits, often with letter suffix (e.g., 99213, 99214F)
        # More specific pattern to avoid matching generic 5-digit numbers
        # Require either: letter suffix OR CPT prefix OR specific ranges
        pattern=r"\b(?:CPT[-:\s]*)?(?:(?:[0-8]\d{4}|9[0-8]\d{3}|990\d{2}|991\d{2}|992\d{2}|993\d{2}|994\d{2})[A-Z]|CPT[-:\s]*\d{5})\b",
        description="CPT Procedure Code",
        context_boost=0.4,
        priority=1  # Lower priority than more specific patterns
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
        # SWIFT/BIC: 4 letters (bank) + 2 letters (country ISO) + 2 alphanumeric (location) + optional 3 (branch)
        # Must be case-sensitive to avoid matching common words
        pattern=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
        description="SWIFT/BIC Code",
        priority=2,
        case_sensitive=True,  # Avoid matching words like "CONSEGUE", "CORRENTE"
        has_validation=True  # Enable validation for country code check
    ),
    "bank_account_br": EntityPatternConfig(
        name="bank_account_br",
        category=EntityCategory.FINANCIAL,
        # Flexible pattern for Brazilian bank accounts
        # Matches: "Ag 1234 CC 56789", "AG: 0001 / CC: 98765-4", "Agencia 1234 Conta 12345-6", etc.
        pattern=r"\b(?:AG(?:[EÊ]NCIA)?|[Aa]g(?:[eê]ncia)?)[-:\s]*\d{3,5}[-\s/]*(?:\d[-\s/]*)?(?:C/?C|CONTA|[Cc]onta)[-:\s]*\d{4,12}[-]?\d?\b",
        description="Brazilian Bank Account (Agency + Account)",
        locale="br",
        priority=3  # Higher priority to match before generic patterns
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
# CREDENTIALS PATTERNS - API Keys, Tokens, Secrets, Passwords
# =============================================================================

CREDENTIALS_PATTERNS: Dict[str, EntityPatternConfig] = {
    # OpenAI API Keys
    "openai_api_key": EntityPatternConfig(
        name="openai_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bsk-[a-zA-Z0-9]{20,}T3BlbkFJ[a-zA-Z0-9]{20,}\b",
        description="OpenAI API Key",
        priority=3
    ),
    "openai_api_key_v2": EntityPatternConfig(
        name="openai_api_key_v2",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bsk-proj-[a-zA-Z0-9_-]{80,}\b",
        description="OpenAI Project API Key",
        priority=3
    ),

    # Anthropic API Keys
    "anthropic_api_key": EntityPatternConfig(
        name="anthropic_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bsk-ant-api[a-zA-Z0-9_-]{80,}\b",
        description="Anthropic API Key",
        priority=3
    ),

    # AWS Credentials
    "aws_access_key": EntityPatternConfig(
        name="aws_access_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b",
        description="AWS Access Key ID",
        priority=3
    ),
    "aws_secret_key": EntityPatternConfig(
        name="aws_secret_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[A-Za-z0-9/+=]{40}\b",
        description="AWS Secret Access Key",
        context_boost=0.4
    ),

    # Azure Credentials
    "azure_storage_key": EntityPatternConfig(
        name="azure_storage_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[A-Za-z0-9+/]{86}==\b",
        description="Azure Storage Account Key",
        priority=3
    ),
    "azure_connection_string": EntityPatternConfig(
        name="azure_connection_string",
        category=EntityCategory.CREDENTIALS,
        pattern=r"DefaultEndpointsProtocol=https?;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]+;?",
        description="Azure Storage Connection String",
        priority=3
    ),
    "azure_sas_token": EntityPatternConfig(
        name="azure_sas_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[?&]sig=[A-Za-z0-9%]+(&[a-z]+=[\w%-]+)+\b",
        description="Azure SAS Token",
        priority=2
    ),

    # Google Cloud
    "gcp_api_key": EntityPatternConfig(
        name="gcp_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bAIza[0-9A-Za-z_-]{35}\b",
        description="Google Cloud API Key",
        priority=3
    ),
    "gcp_service_account": EntityPatternConfig(
        name="gcp_service_account",
        category=EntityCategory.CREDENTIALS,
        pattern=r'"type"\s*:\s*"service_account".*"private_key"\s*:\s*"-----BEGIN',
        description="GCP Service Account JSON",
        priority=3
    ),

    # GitHub Tokens
    "github_token_classic": EntityPatternConfig(
        name="github_token_classic",
        category=EntityCategory.CREDENTIALS,
        # GitHub classic PATs: ghp_ followed by 20-50 alphanumeric chars
        pattern=r"\bghp_[a-zA-Z0-9]{20,50}\b",
        description="GitHub Personal Access Token (Classic)",
        priority=3
    ),
    "github_token_fine": EntityPatternConfig(
        name="github_token_fine",
        category=EntityCategory.CREDENTIALS,
        # Fine-grained tokens: github_pat_ followed by variable length
        pattern=r"\bgithub_pat_[a-zA-Z0-9_]{20,100}\b",
        description="GitHub Fine-Grained Token",
        priority=3
    ),
    "github_oauth": EntityPatternConfig(
        name="github_oauth",
        category=EntityCategory.CREDENTIALS,
        # OAuth tokens: gho_ followed by 20-50 alphanumeric chars
        pattern=r"\bgho_[a-zA-Z0-9]{20,50}\b",
        description="GitHub OAuth Access Token",
        priority=3
    ),
    "github_app_token": EntityPatternConfig(
        name="github_app_token",
        category=EntityCategory.CREDENTIALS,
        # App tokens: ghu_ or ghs_ followed by 20-50 alphanumeric chars
        pattern=r"\b(?:ghu|ghs)_[a-zA-Z0-9]{20,50}\b",
        description="GitHub App Token",
        priority=3
    ),

    # GitLab Tokens
    "gitlab_token": EntityPatternConfig(
        name="gitlab_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bglpat-[a-zA-Z0-9_-]{20,}\b",
        description="GitLab Personal Access Token",
        priority=3
    ),

    # Stripe
    "stripe_secret_key": EntityPatternConfig(
        name="stripe_secret_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bsk_(?:live|test)_[a-zA-Z0-9]{24,}\b",
        description="Stripe Secret Key",
        priority=3
    ),
    "stripe_publishable_key": EntityPatternConfig(
        name="stripe_publishable_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bpk_(?:live|test)_[a-zA-Z0-9]{24,}\b",
        description="Stripe Publishable Key",
        priority=2
    ),

    # Slack
    "slack_token": EntityPatternConfig(
        name="slack_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*\b",
        description="Slack Token",
        priority=3
    ),
    "slack_webhook": EntityPatternConfig(
        name="slack_webhook",
        category=EntityCategory.CREDENTIALS,
        pattern=r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+",
        description="Slack Webhook URL",
        priority=3
    ),

    # Twilio
    "twilio_api_key": EntityPatternConfig(
        name="twilio_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bSK[a-f0-9]{32}\b",
        description="Twilio API Key",
        priority=3
    ),

    # SendGrid
    "sendgrid_api_key": EntityPatternConfig(
        name="sendgrid_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bSG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}\b",
        description="SendGrid API Key",
        priority=3
    ),

    # Mailchimp
    "mailchimp_api_key": EntityPatternConfig(
        name="mailchimp_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[a-f0-9]{32}-us\d{1,2}\b",
        description="Mailchimp API Key",
        priority=3
    ),

    # NPM Token
    "npm_token": EntityPatternConfig(
        name="npm_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bnpm_[a-zA-Z0-9]{36}\b",
        description="NPM Access Token",
        priority=3
    ),

    # PyPI Token
    "pypi_token": EntityPatternConfig(
        name="pypi_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bpypi-AgEIcHlwaS5vcmc[a-zA-Z0-9_-]{50,}\b",
        description="PyPI API Token",
        priority=3
    ),

    # Docker Hub Token
    "docker_hub_token": EntityPatternConfig(
        name="docker_hub_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bdckr_pat_[a-zA-Z0-9_-]{27}\b",
        description="Docker Hub Personal Access Token",
        priority=3
    ),

    # Hugging Face Token
    "huggingface_token": EntityPatternConfig(
        name="huggingface_token",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bhf_[a-zA-Z0-9]{34}\b",
        description="Hugging Face API Token",
        priority=3
    ),

    # Generic API Keys (context-dependent)
    "generic_api_key": EntityPatternConfig(
        name="generic_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b(?:api[_-]?key|apikey|api[_-]?token|access[_-]?token|auth[_-]?token|secret[_-]?key|private[_-]?key)[\s:=]+['\"]?[a-zA-Z0-9_-]{20,}['\"]?\b",
        description="Generic API Key",
        priority=2,
        context_boost=0.3
    ),

    # JWT Tokens
    "jwt_token": EntityPatternConfig(
        name="jwt_token",
        category=EntityCategory.CREDENTIALS,
        # JWT format: base64url.base64url.base64url (header.payload.signature)
        # Also matches incomplete JWTs (2 parts) which may appear in logs
        pattern=r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)?\b",
        description="JWT Token",
        priority=3,
        context_boost=0.3  # Boost when "jwt", "token", "authorization" nearby
    ),

    # Bearer Tokens (including JWTs)
    "bearer_token": EntityPatternConfig(
        name="bearer_token",
        category=EntityCategory.CREDENTIALS,
        # Bearer followed by token (may contain dots for JWTs)
        pattern=r"\b[Bb]earer\s+[a-zA-Z0-9_.-]{20,}\b",
        description="Bearer Token",
        priority=3,  # High priority - credentials are critical
        context_boost=0.3  # Boost when "authorization", "header", etc. nearby
    ),

    # Basic Auth (base64)
    "basic_auth": EntityPatternConfig(
        name="basic_auth",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[Bb]asic\s+[A-Za-z0-9+/=]{20,}\b",
        description="Basic Auth Credentials",
        priority=2
    ),

    # Private Keys
    "private_key_rsa": EntityPatternConfig(
        name="private_key_rsa",
        category=EntityCategory.CREDENTIALS,
        pattern=r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        description="Private Key (RSA/EC/DSA)",
        priority=3
    ),
    "private_key_pgp": EntityPatternConfig(
        name="private_key_pgp",
        category=EntityCategory.CREDENTIALS,
        pattern=r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
        description="PGP Private Key",
        priority=3
    ),

    # Database Connection Strings
    "database_url": EntityPatternConfig(
        name="database_url",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|mssql)://[^\s]+:[^\s]+@[^\s]+\b",
        description="Database Connection URL with Credentials",
        priority=3
    ),

    # Password Patterns (in config files)
    "password_config": EntityPatternConfig(
        name="password_config",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b(?:password|passwd|pwd|secret|senha)[\s:=]+['\"]?[^\s'\"]{8,}['\"]?\b",
        description="Password in Configuration",
        priority=2,
        context_boost=0.3
    ),

    # Encryption Keys (hex)
    "encryption_key_hex": EntityPatternConfig(
        name="encryption_key_hex",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b(?:encryption[_-]?key|aes[_-]?key|secret[_-]?key)[\s:=]+['\"]?[a-fA-F0-9]{32,64}['\"]?\b",
        description="Encryption Key (Hex)",
        priority=3
    ),

    # Webhook URLs (generic)
    "webhook_url": EntityPatternConfig(
        name="webhook_url",
        category=EntityCategory.CREDENTIALS,
        pattern=r"https?://[^\s]+/webhook[s]?/[a-zA-Z0-9_/-]+",
        description="Webhook URL",
        priority=2
    ),

    # Datadog API Key
    "datadog_api_key": EntityPatternConfig(
        name="datadog_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[a-f0-9]{32}\b",
        description="Datadog API Key",
        context_boost=0.4
    ),

    # Heroku API Key
    "heroku_api_key": EntityPatternConfig(
        name="heroku_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
        description="Heroku API Key (UUID format)",
        context_boost=0.3
    ),

    # Firebase
    "firebase_api_key": EntityPatternConfig(
        name="firebase_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\bAIza[0-9A-Za-z_-]{35}\b",
        description="Firebase API Key",
        priority=3
    ),

    # Algolia
    "algolia_api_key": EntityPatternConfig(
        name="algolia_api_key",
        category=EntityCategory.CREDENTIALS,
        pattern=r"\b[a-f0-9]{32}\b",
        description="Algolia API Key",
        context_boost=0.4
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
            "email", "e-mail", "correio eletrônico",
            # Abbreviated forms
            "tel", "fone", "cel", "whatsapp", "whats", "zap",
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
            "laboratório", "laboratory", "sangue", "blood",
            "diabetes", "hipertensão", "asma", "câncer", "cancer"  # Common diseases
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
            "banco", "bank", "conta", "account", "agência", "agencia", "branch",
            "ag:", "cc:", "c/c",  # Common Brazilian bank shortcuts
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
    },
    EntityCategory.CREDENTIALS: {
        "high_confidence": {
            "api_key", "api key", "apikey", "secret", "token",
            "password", "senha", "credential", "credencial",
            "private key", "chave privada", "access key", "secret key",
            "auth token", "bearer", "jwt", "oauth",
            "aws_access", "aws_secret", "openai", "anthropic",
            "github_token", "gitlab", "stripe", "twilio",
            # Portuguese terms
            "autorização", "autorizacao", "chave de api", "chave api",
        },
        "medium_confidence": {
            "authorization", "autenticação", "authentication",
            "connection string", "database url", "redis url",
            "mongodb", "postgres", "mysql", "webhook",
            "slack", "discord", "sendgrid", "mailchimp",
            # Portuguese terms
            "header", "cabeçalho", "permissão", "permissao",
        },
        "low_confidence": {
            "config", "configuração", "environment", "env",
            "settings", ".env", "secrets", "vault"
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
    all_patterns.update(CREDENTIALS_PATTERNS)
    return all_patterns
