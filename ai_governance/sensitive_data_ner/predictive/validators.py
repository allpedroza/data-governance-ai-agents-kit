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
Entity Validators

Checksum and format validation functions for verifying
that detected patterns are actually valid entities.

These validators provide a higher confidence level than
regex matching alone.
"""

import re
from typing import Callable, Optional


def validate_cpf(cpf: str) -> bool:
    """
    Validate Brazilian CPF using check digits.

    The CPF has 11 digits where the last 2 are check digits
    calculated using a specific algorithm (mod 11).

    Args:
        cpf: CPF string (with or without formatting)

    Returns:
        True if valid, False otherwise
    """
    # Remove non-digits
    cpf = re.sub(r'\D', '', cpf)

    # Must be exactly 11 digits
    if len(cpf) != 11:
        return False

    # Check for known invalid patterns (all same digit)
    if cpf == cpf[0] * 11:
        return False

    # Calculate first check digit
    total = sum(int(cpf[i]) * (10 - i) for i in range(9))
    remainder = (total * 10) % 11
    if remainder == 10:
        remainder = 0
    if remainder != int(cpf[9]):
        return False

    # Calculate second check digit
    total = sum(int(cpf[i]) * (11 - i) for i in range(10))
    remainder = (total * 10) % 11
    if remainder == 10:
        remainder = 0
    if remainder != int(cpf[10]):
        return False

    return True


def validate_cnpj(cnpj: str) -> bool:
    """
    Validate Brazilian CNPJ using check digits.

    The CNPJ has 14 digits where the last 2 are check digits.

    Args:
        cnpj: CNPJ string (with or without formatting)

    Returns:
        True if valid, False otherwise
    """
    # Remove non-digits
    cnpj = re.sub(r'\D', '', cnpj)

    # Must be exactly 14 digits
    if len(cnpj) != 14:
        return False

    # Check for known invalid patterns
    if cnpj == cnpj[0] * 14:
        return False

    # Weights for check digit calculation
    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

    # Calculate first check digit
    total = sum(int(cnpj[i]) * weights1[i] for i in range(12))
    remainder = total % 11
    check1 = 0 if remainder < 2 else 11 - remainder
    if check1 != int(cnpj[12]):
        return False

    # Calculate second check digit
    total = sum(int(cnpj[i]) * weights2[i] for i in range(13))
    remainder = total % 11
    check2 = 0 if remainder < 2 else 11 - remainder
    if check2 != int(cnpj[13]):
        return False

    return True


def validate_credit_card(card_number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.

    The Luhn algorithm (mod 10) is used by most credit cards
    to prevent typos and simple errors.

    Args:
        card_number: Card number string (with or without separators)

    Returns:
        True if valid according to Luhn, False otherwise
    """
    # Remove non-digits
    card = re.sub(r'\D', '', card_number)

    # Valid lengths: 13-19 digits
    if not 13 <= len(card) <= 19:
        return False

    # Luhn algorithm
    total = 0
    is_second = False

    for digit in reversed(card):
        d = int(digit)
        if is_second:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        is_second = not is_second

    return total % 10 == 0


def validate_cns(cns: str) -> bool:
    """
    Validate Brazilian CNS (Cartão Nacional de Saúde).

    CNS has 15 digits and uses specific validation rules
    based on the first digit.

    Args:
        cns: CNS string (with or without spaces)

    Returns:
        True if valid, False otherwise
    """
    # Remove non-digits
    cns = re.sub(r'\D', '', cns)

    # Must be exactly 15 digits
    if len(cns) != 15:
        return False

    # First digit determines validation method
    first_digit = int(cns[0])

    if first_digit in [1, 2]:
        # Definitive CNS - mod 11 validation
        total = sum(int(cns[i]) * (15 - i) for i in range(15))
        return total % 11 == 0

    elif first_digit in [7, 8, 9]:
        # Provisional CNS - mod 11 with adjustment
        total = sum(int(cns[i]) * (15 - i) for i in range(15))
        remainder = total % 11
        return remainder == 0

    return False


def validate_ssn(ssn: str) -> bool:
    """
    Validate US Social Security Number format.

    SSN validation is primarily format-based as there's no
    public checksum algorithm. We check for invalid ranges.

    Args:
        ssn: SSN string

    Returns:
        True if valid format, False otherwise
    """
    # Remove non-digits
    ssn = re.sub(r'\D', '', ssn)

    # Must be exactly 9 digits
    if len(ssn) != 9:
        return False

    # Area number (first 3 digits) cannot be:
    # - 000
    # - 666
    # - 900-999
    area = int(ssn[:3])
    if area == 0 or area == 666 or 900 <= area <= 999:
        return False

    # Group number (middle 2 digits) cannot be 00
    group = int(ssn[3:5])
    if group == 0:
        return False

    # Serial number (last 4 digits) cannot be 0000
    serial = int(ssn[5:])
    if serial == 0:
        return False

    return True


def validate_iban(iban: str) -> bool:
    """
    Validate International Bank Account Number (IBAN).

    Uses mod-97 algorithm as per ISO 13616.

    Args:
        iban: IBAN string (with or without spaces)

    Returns:
        True if valid, False otherwise
    """
    # Remove spaces and convert to uppercase
    iban = re.sub(r'\s', '', iban.upper())

    # Minimum length is 15, maximum is 34
    if not 15 <= len(iban) <= 34:
        return False

    # First two characters must be letters (country code)
    if not iban[:2].isalpha():
        return False

    # Characters 3-4 must be check digits
    if not iban[2:4].isdigit():
        return False

    # Rearrange: move first 4 characters to end
    rearranged = iban[4:] + iban[:4]

    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    numeric = ''
    for char in rearranged:
        if char.isalpha():
            numeric += str(ord(char) - ord('A') + 10)
        else:
            numeric += char

    # Check mod 97
    return int(numeric) % 97 == 1


def validate_ip_address(ip: str) -> bool:
    """
    Validate IPv4 address.

    Checks that each octet is in valid range (0-255).

    Args:
        ip: IP address string

    Returns:
        True if valid IPv4, False otherwise
    """
    parts = ip.split('.')

    if len(parts) != 4:
        return False

    for part in parts:
        try:
            num = int(part)
            if not 0 <= num <= 255:
                return False
            # No leading zeros allowed (except for "0")
            if len(part) > 1 and part[0] == '0':
                return False
        except ValueError:
            return False

    return True


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Basic validation checking structure and domain.

    Args:
        email: Email address string

    Returns:
        True if valid format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False

    # Additional checks
    local, domain = email.rsplit('@', 1)

    # Local part shouldn't start/end with dot
    if local.startswith('.') or local.endswith('.'):
        return False

    # No consecutive dots
    if '..' in email:
        return False

    # Domain must have at least one dot
    if '.' not in domain:
        return False

    return True


# Common Portuguese/Spanish/English words that are NOT names (false positive exclusions)
_NOT_PERSON_NAMES = {
    # Common nouns and verbs that may start with capital (sentence start)
    "erro", "token", "acesso", "usuario", "usuário", "cliente", "sistema",
    "cartao", "cartão", "pagamento", "autenticacao", "autenticação", "falha",
    "falhou", "portador", "titular", "conta", "senha", "login", "logout",
    "sessao", "sessão", "servico", "serviço", "endpoint", "api", "request",
    "response", "codigo", "código", "chave", "valor", "dados", "arquivo",
    "processo", "processamento", "transacao", "transação", "compra", "venda",
    "produto", "item", "pedido", "ordem", "status", "estado", "tipo",
    # English technical terms that may appear capitalized
    "critical", "error", "warning", "info", "debug", "fatal", "success",
    "failed", "failure", "exception", "message", "alert", "notice",
    "invalid", "valid", "null", "undefined", "unknown", "default",
    "system", "server", "client", "user", "admin", "root", "guest",
    "public", "private", "internal", "external", "local", "remote",
    "input", "output", "result", "return", "value", "data", "response",
    "request", "query", "update", "delete", "insert", "select", "create",
    # Technical terms
    "string", "integer", "float", "boolean", "array", "object", "null",
    "true", "false", "undefined", "function", "class", "method", "variable",
    # Common adjectives/adverbs
    "final", "inicial", "primeiro", "ultimo", "último", "novo", "antigo",
    "grande", "pequeno", "alto", "baixo", "bom", "mau", "melhor", "pior",
    # Common verbs (conjugated forms that might match)
    "errou", "falhou", "sucesso", "processou", "validou",
}

# Common phrase patterns that are NOT names
_NOT_NAME_PATTERNS = [
    r"^erro\s",  # "Erro ao..."
    r"^error\s",  # "Error in..."
    r"^critical\s",  # "Critical Error..."
    r"^warning\s",  # "Warning:..."
    r"^falha\s",  # "Falha na..."
    r"^token\s",  # "Token de..."
    r"^cartao\s",  # "Cartão de..."
    r"^cartão\s",
    r"\s+(de|do|da)\s+(acesso|pagamento|autenticacao|autenticação|sistema|cliente|usuario|usuário)$",
    r"^(usuario|usuário)\s+\w+$",  # "Usuário X" where X is not a name
    r"^(system|server|client|user)\s+\w+$",  # "System Error", "Server Fault"
]


def _validate_person_name_regex(name: str) -> bool:
    """
    Validate person name using regex-based rules.

    This is the fallback validation when SpaCy is not available.
    """
    if not name or len(name) < 5:
        return False

    # Split into parts
    parts = name.split()
    if len(parts) < 2:
        return False

    # Connectors that are allowed in lowercase
    connectors = {"de", "da", "do", "dos", "das", "e", "van", "von", "der", "del", "la", "el"}

    # Check each word
    name_parts = 0  # Count actual name parts (not connectors)
    for part in parts:
        part_lower = part.lower()

        # Skip connectors
        if part_lower in connectors:
            continue

        # Check if first word is in exclusion list (case-insensitive)
        if name_parts == 0 and part_lower in _NOT_PERSON_NAMES:
            return False

        # Name parts must start with uppercase
        if not part[0].isupper():
            return False

        # Rest of the name part should be lowercase (except for compound names like "McDonald")
        # Allow flexibility but ensure it's not ALL CAPS
        if part.isupper() and len(part) > 2:
            return False

        name_parts += 1

    # Must have at least 2 proper name parts (first + last name)
    if name_parts < 2:
        return False

    # Check against common false positive patterns
    name_lower = name.lower()
    for pattern in _NOT_NAME_PATTERNS:
        if re.search(pattern, name_lower):
            return False

    return True


def validate_person_name(name: str) -> bool:
    """
    Validate that a detected string is actually a person's name.

    Uses a combination of:
    1. SpaCy NER (PER entity classification) - when available
    2. SpaCy POS tagging (PROPN vs VERB/NOUN) - when available
    3. Regex-based rules (capitalization, exclusion lists)

    The validation is conservative: if SpaCy confidently says it's NOT
    a person name (e.g., contains verbs, classified as ORG/LOC), we reject.
    If SpaCy is uncertain or unavailable, we fall back to regex rules.

    Args:
        name: Detected name string

    Returns:
        True if appears to be a valid person name, False otherwise
    """
    # First apply regex rules (fast, always available)
    if not _validate_person_name_regex(name):
        return False

    # Try SpaCy validation for additional accuracy
    try:
        from .spacy_helper import (
            is_spacy_available,
            validate_person_name_with_spacy,
            contains_verb
        )

        if is_spacy_available():
            # Check for verbs - key indicator of false positive
            has_verb, verbs = contains_verb(name)
            if has_verb:
                # If detected as containing verbs, likely not a name
                # But check if SpaCy also thinks it's a person
                is_valid, confidence, reason = validate_person_name_with_spacy(name)

                if not is_valid and confidence >= 0.7:
                    return False

                # Even if uncertain, if it contains verbs, be conservative
                if has_verb and confidence < 0.8:
                    return False

            # Full SpaCy validation
            is_valid, confidence, reason = validate_person_name_with_spacy(name)

            # If SpaCy confidently rejects, trust it
            if not is_valid and confidence >= 0.75:
                return False

            # If SpaCy confidently confirms, accept
            if is_valid and confidence >= 0.85:
                return True

    except ImportError:
        # SpaCy helper not available, continue with regex result
        pass
    except Exception as e:
        # Log error but don't fail
        import logging
        logging.getLogger(__name__).debug(f"SpaCy validation error: {str(e)}")

    # Default: regex validation passed
    return True


# Common Portuguese/Spanish words that accidentally match SWIFT/BIC pattern
# These words have 8 characters, all uppercase letters, and happen to contain
# what looks like a valid country code in positions 4-5
_SWIFT_FALSE_POSITIVES = {
    "CONSEGUE",  # Portuguese verb "consegue" - contains EG (Egypt)
    "CONSIGNA",  # Spanish/Portuguese - contains IG (not valid but similar pattern)
    "CONSELHE",  # Portuguese - contains EL (not valid)
    "CONSEJOS",  # Spanish - contains EJ (not valid)
    "CONSULTE",  # Portuguese - contains UL (not valid)
    "CONTINUE",  # English/Portuguese - contains IN (India)
    "CONTEXTO",  # Portuguese - contains EX (not valid)
    "CONTRATE",  # Portuguese - contains RA (not valid)
    "CONTROLE",  # Portuguese - contains RO (Romania)
    "CONVERSE",  # English/Portuguese - contains VE (Venezuela)
    "COMPLETE",  # English - contains PL (Poland)
    "COMPARTE",  # Spanish - contains PA (Panama)
    "COMPROVE",  # Portuguese - contains RO (Romania)
}

# Valid ISO 3166-1 alpha-2 country codes for SWIFT/BIC validation
_VALID_COUNTRY_CODES = {
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS",
    "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE",
    "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM",
    "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC",
    "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG",
    "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO",
    "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW"
}


def validate_swift_bic(code: str) -> bool:
    """
    Validate SWIFT/BIC code structure.

    SWIFT/BIC format:
    - 4 letters: Bank code
    - 2 letters: ISO 3166-1 alpha-2 country code
    - 2 alphanumeric: Location code
    - 3 alphanumeric (optional): Branch code

    Args:
        code: SWIFT/BIC code string

    Returns:
        True if valid structure, False otherwise
    """
    if not code:
        return False

    code = code.upper().strip()

    # Check against known false positives (common words)
    if code in _SWIFT_FALSE_POSITIVES:
        return False

    # Must be 8 or 11 characters
    if len(code) not in (8, 11):
        return False

    # First 4 characters must be letters (bank code)
    if not code[:4].isalpha():
        return False

    # Characters 5-6 must be a valid country code
    country_code = code[4:6]
    if country_code not in _VALID_COUNTRY_CODES:
        return False

    # Characters 7-8 must be alphanumeric (location)
    if not code[6:8].isalnum():
        return False

    # If 11 characters, last 3 must be alphanumeric (branch)
    if len(code) == 11 and not code[8:11].isalnum():
        return False

    return True


# Validator registry
_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "cpf": validate_cpf,
    "cnpj": validate_cnpj,
    "credit_card": validate_credit_card,
    "card_visa": validate_credit_card,
    "card_mastercard": validate_credit_card,
    "card_amex": validate_credit_card,
    "cns": validate_cns,
    "ssn": validate_ssn,
    "iban": validate_iban,
    "ip_address": validate_ip_address,
    "email": validate_email,
    "person_name": validate_person_name,
    "swift_bic": validate_swift_bic,
}


def get_validator(pattern_name: str) -> Optional[Callable[[str], bool]]:
    """
    Get the validator function for a pattern.

    Args:
        pattern_name: Name of the pattern (e.g., "cpf", "credit_card")

    Returns:
        Validator function or None if no validator exists
    """
    return _VALIDATORS.get(pattern_name)


def has_validator(pattern_name: str) -> bool:
    """Check if a validator exists for the given pattern."""
    return pattern_name in _VALIDATORS
