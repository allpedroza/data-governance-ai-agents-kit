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
