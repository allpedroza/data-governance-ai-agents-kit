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


# Common Portuguese/Spanish words that are NOT names (false positive exclusions)
_NOT_PERSON_NAMES = {
    # Common nouns and verbs that may start with capital (sentence start)
    "erro", "erro", "token", "acesso", "usuario", "usuário", "cliente", "sistema",
    "cartao", "cartão", "pagamento", "autenticacao", "autenticação", "falha",
    "falhou", "portador", "titular", "conta", "senha", "login", "logout",
    "sessao", "sessão", "servico", "serviço", "endpoint", "api", "request",
    "response", "codigo", "código", "chave", "valor", "dados", "arquivo",
    "processo", "processamento", "transacao", "transação", "compra", "venda",
    "produto", "item", "pedido", "ordem", "status", "estado", "tipo",
    # Technical terms
    "string", "integer", "float", "boolean", "array", "object", "null",
    "true", "false", "undefined", "function", "class", "method", "variable",
    # Common adjectives/adverbs
    "final", "inicial", "primeiro", "ultimo", "último", "novo", "antigo",
    "grande", "pequeno", "alto", "baixo", "bom", "mau", "melhor", "pior",
    # Common verbs (conjugated forms that might match)
    "erro", "errou", "falha", "falhou", "sucesso", "processou", "validou",
}

# Common phrase patterns that are NOT names
_NOT_NAME_PATTERNS = [
    r"^erro\s",  # "Erro ao..."
    r"^falha\s",  # "Falha na..."
    r"^token\s",  # "Token de..."
    r"^cartao\s",  # "Cartão de..."
    r"^cartão\s",
    r"\s+(de|do|da)\s+(acesso|pagamento|autenticacao|autenticação|sistema|cliente|usuario|usuário)$",
    r"^(usuario|usuário)\s+\w+$",  # "Usuário X" where X is not a name
]


def validate_person_name(name: str) -> bool:
    """
    Validate that a detected string is actually a person's name.

    Checks:
    1. Proper capitalization (each name part starts with uppercase)
    2. Not in the exclusion list of common words
    3. Minimum length requirements
    4. Has at least two proper name parts

    Args:
        name: Detected name string

    Returns:
        True if appears to be a valid person name, False otherwise
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
