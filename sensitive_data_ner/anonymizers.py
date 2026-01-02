"""
Anonymization Module

Provides multiple strategies for anonymizing detected sensitive entities
while preserving text structure and meaning where possible.

Strategies:
- MASK: Replace with asterisks/X characters
- REDACT: Replace with category label
- HASH: Replace with deterministic hash
- PSEUDONYMIZE: Replace with consistent fake data
- ENCRYPT: Reversible encryption (requires key)
- PARTIAL: Mask partial content, keep some characters visible
"""

import re
import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from base64 import b64encode, b64decode

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class AnonymizationStrategy(Enum):
    """Available anonymization strategies."""
    MASK = "mask"
    REDACT = "redact"
    HASH = "hash"
    PSEUDONYMIZE = "pseudonymize"
    ENCRYPT = "encrypt"
    PARTIAL = "partial"


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization behavior."""
    strategy: AnonymizationStrategy = AnonymizationStrategy.REDACT
    mask_char: str = "*"
    hash_length: int = 8
    preserve_format: bool = True
    partial_visible_start: int = 2
    partial_visible_end: int = 2
    encryption_key: Optional[str] = None
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "pii": "[PII]",
        "phi": "[PHI]",
        "pci": "[CARTÃO]",
        "financial": "[FINANCEIRO]",
        "business": "[CONFIDENCIAL]",
    })
    entity_labels: Dict[str, str] = field(default_factory=lambda: {
        "cpf": "[CPF]",
        "cnpj": "[CNPJ]",
        "email": "[EMAIL]",
        "phone": "[TELEFONE]",
        "credit_card": "[CARTÃO]",
        "ssn": "[SSN]",
        "name": "[NOME]",
        "address": "[ENDEREÇO]",
        "iban": "[IBAN]",
        "icd10": "[CID]",
    })


@dataclass
class AnonymizedEntity:
    """Represents an anonymized entity."""
    original: str
    anonymized: str
    entity_type: str
    category: str
    start: int
    end: int
    strategy_used: AnonymizationStrategy
    is_reversible: bool = False
    reversal_key: Optional[str] = None


class Anonymizer(ABC):
    """Abstract base class for anonymizers."""

    @abstractmethod
    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Anonymize a value.

        Args:
            value: Original value to anonymize
            entity_type: Type of entity (e.g., "cpf", "email")
            category: Category (e.g., "pii", "phi")

        Returns:
            Tuple of (anonymized_value, is_reversible, reversal_key)
        """
        pass


class MaskAnonymizer(Anonymizer):
    """Replace value with mask characters while optionally preserving format."""

    def __init__(self, config: AnonymizationConfig):
        self.config = config

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        if self.config.preserve_format:
            # Preserve separators, mask only alphanumeric
            result = ""
            for char in value:
                if char.isalnum():
                    result += self.config.mask_char
                else:
                    result += char
            return result, False, None
        else:
            return self.config.mask_char * len(value), False, None


class RedactAnonymizer(Anonymizer):
    """Replace value with a category/entity label."""

    def __init__(self, config: AnonymizationConfig):
        self.config = config

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        # Try entity-specific label first
        if entity_type in self.config.entity_labels:
            label = self.config.entity_labels[entity_type]
        elif category in self.config.category_labels:
            label = self.config.category_labels[category]
        else:
            label = f"[{category.upper()}]"

        return label, False, None


class HashAnonymizer(Anonymizer):
    """Replace value with deterministic hash (consistent per value)."""

    def __init__(self, config: AnonymizationConfig, salt: Optional[str] = None):
        self.config = config
        self.salt = salt or secrets.token_hex(16)

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        # Create salted hash
        hash_input = f"{self.salt}:{value}".encode()
        full_hash = hashlib.sha256(hash_input).hexdigest()

        # Truncate to configured length
        short_hash = full_hash[:self.config.hash_length]

        # Format with entity type prefix for clarity
        anonymized = f"{entity_type.upper()}_{short_hash}"

        return anonymized, False, None


class PartialAnonymizer(Anonymizer):
    """Partially mask value, keeping some characters visible."""

    def __init__(self, config: AnonymizationConfig):
        self.config = config

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        # Get alphanumeric characters only for counting
        alphanum = re.sub(r'\W', '', value)
        total_len = len(alphanum)

        # Minimum masked characters
        min_masked = max(4, total_len - self.config.partial_visible_start - self.config.partial_visible_end)

        if total_len <= 4:
            # Too short, mask entirely
            return self.config.mask_char * len(value), False, None

        # Build masked version preserving format
        result = []
        alphanum_index = 0

        for char in value:
            if char.isalnum():
                if alphanum_index < self.config.partial_visible_start:
                    result.append(char)
                elif alphanum_index >= total_len - self.config.partial_visible_end:
                    result.append(char)
                else:
                    result.append(self.config.mask_char)
                alphanum_index += 1
            else:
                result.append(char)

        return ''.join(result), False, None


class PseudonymAnonymizer(Anonymizer):
    """Replace with consistent fake data (same input = same output)."""

    # Fake data pools
    FAKE_NAMES = [
        "João Silva", "Maria Santos", "Pedro Oliveira", "Ana Costa",
        "Carlos Souza", "Julia Lima", "Lucas Pereira", "Fernanda Alves",
        "John Doe", "Jane Smith", "Bob Johnson", "Alice Williams"
    ]
    FAKE_EMAILS = [
        "usuario1@exemplo.com", "contato@empresa.com", "info@dominio.com",
        "user@example.com", "contact@company.com", "hello@domain.com"
    ]
    FAKE_PHONES = [
        "(11) 9999-9999", "(21) 8888-8888", "(31) 7777-7777",
        "(555) 123-4567", "(555) 987-6543"
    ]

    def __init__(self, config: AnonymizationConfig, seed: Optional[str] = None):
        self.config = config
        self.seed = seed or secrets.token_hex(8)
        self._mapping: Dict[str, str] = {}

    def _get_fake(self, value: str, pool: List[str]) -> str:
        """Get consistent fake value from pool."""
        if value not in self._mapping:
            # Use hash to deterministically select from pool
            hash_val = int(hashlib.md5(
                f"{self.seed}:{value}".encode()
            ).hexdigest(), 16)
            self._mapping[value] = pool[hash_val % len(pool)]
        return self._mapping[value]

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        # Select appropriate pool
        if entity_type in ("person_name", "name"):
            fake = self._get_fake(value, self.FAKE_NAMES)
        elif entity_type == "email":
            fake = self._get_fake(value, self.FAKE_EMAILS)
        elif entity_type in ("phone", "phone_br", "phone_intl"):
            fake = self._get_fake(value, self.FAKE_PHONES)
        else:
            # Generate pseudo-random replacement maintaining length
            hash_val = hashlib.md5(f"{self.seed}:{value}".encode()).hexdigest()
            if value[0].isdigit():
                # Numeric replacement
                fake = hash_val[:len(value)]
                fake = ''.join(c if c.isdigit() else str(ord(c) % 10) for c in fake)
            else:
                fake = f"PSEUDO_{hash_val[:8]}"

        return fake, False, None


class EncryptAnonymizer(Anonymizer):
    """Encrypt value with Fernet (reversible with key)."""

    def __init__(self, config: AnonymizationConfig):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )
        self.config = config
        if config.encryption_key:
            self.key = config.encryption_key.encode()
        else:
            self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)

    def anonymize(
        self,
        value: str,
        entity_type: str,
        category: str
    ) -> Tuple[str, bool, Optional[str]]:
        encrypted = self.fernet.encrypt(value.encode())
        # Return as base64 string with prefix
        anonymized = f"ENC:{encrypted.decode()}"
        return anonymized, True, self.key.decode()

    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt a previously encrypted value."""
        if encrypted_value.startswith("ENC:"):
            encrypted_value = encrypted_value[4:]
        return self.fernet.decrypt(encrypted_value.encode()).decode()


class AnonymizerFactory:
    """Factory for creating anonymizers based on strategy."""

    _anonymizers: Dict[AnonymizationStrategy, type] = {
        AnonymizationStrategy.MASK: MaskAnonymizer,
        AnonymizationStrategy.REDACT: RedactAnonymizer,
        AnonymizationStrategy.HASH: HashAnonymizer,
        AnonymizationStrategy.PSEUDONYMIZE: PseudonymAnonymizer,
        AnonymizationStrategy.PARTIAL: PartialAnonymizer,
        AnonymizationStrategy.ENCRYPT: EncryptAnonymizer,
    }

    @classmethod
    def create(
        cls,
        config: AnonymizationConfig,
        **kwargs
    ) -> Anonymizer:
        """
        Create an anonymizer based on configuration.

        Args:
            config: Anonymization configuration
            **kwargs: Additional arguments for specific anonymizers

        Returns:
            Anonymizer instance
        """
        anonymizer_class = cls._anonymizers.get(config.strategy)
        if not anonymizer_class:
            raise ValueError(f"Unknown strategy: {config.strategy}")

        if config.strategy == AnonymizationStrategy.HASH:
            return anonymizer_class(config, salt=kwargs.get("salt"))
        elif config.strategy == AnonymizationStrategy.PSEUDONYMIZE:
            return anonymizer_class(config, seed=kwargs.get("seed"))
        else:
            return anonymizer_class(config)


class TextAnonymizer:
    """
    High-level text anonymizer that processes text with detected entities.
    """

    def __init__(
        self,
        config: Optional[AnonymizationConfig] = None,
        **kwargs
    ):
        """
        Initialize text anonymizer.

        Args:
            config: Anonymization configuration
            **kwargs: Additional arguments passed to anonymizer factory
        """
        self.config = config or AnonymizationConfig()
        self.anonymizer = AnonymizerFactory.create(self.config, **kwargs)
        self._reversal_keys: Dict[str, str] = {}

    def anonymize_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Tuple[str, List[AnonymizedEntity]]:
        """
        Anonymize all detected entities in text.

        Args:
            text: Original text
            entities: List of detected entities with positions

        Returns:
            Tuple of (anonymized text, list of anonymization records)
        """
        if not entities:
            return text, []

        # Sort entities by position (reverse order for replacement)
        sorted_entities = sorted(
            entities,
            key=lambda e: e.get("start", 0),
            reverse=True
        )

        result = text
        anonymized_records = []

        for entity in sorted_entities:
            original = entity.get("value", "")
            entity_type = entity.get("entity_type", "unknown")
            category = entity.get("category", "unknown")
            start = entity.get("start", 0)
            end = entity.get("end", len(original))

            # Anonymize
            anonymized, is_reversible, key = self.anonymizer.anonymize(
                original, entity_type, category
            )

            # Replace in text
            result = result[:start] + anonymized + result[end:]

            # Record
            record = AnonymizedEntity(
                original=original,
                anonymized=anonymized,
                entity_type=entity_type,
                category=category,
                start=start,
                end=start + len(anonymized),
                strategy_used=self.config.strategy,
                is_reversible=is_reversible,
                reversal_key=key
            )
            anonymized_records.append(record)

            if is_reversible and key:
                self._reversal_keys[anonymized] = key

        return result, list(reversed(anonymized_records))

    def get_reversal_keys(self) -> Dict[str, str]:
        """Get all reversal keys for encrypted values."""
        return self._reversal_keys.copy()


def anonymize_text(
    text: str,
    entities: List[Dict[str, Any]],
    strategy: AnonymizationStrategy = AnonymizationStrategy.REDACT,
    **kwargs
) -> Tuple[str, List[AnonymizedEntity]]:
    """
    Convenience function to anonymize text.

    Args:
        text: Original text
        entities: List of detected entities
        strategy: Anonymization strategy to use
        **kwargs: Additional configuration options

    Returns:
        Tuple of (anonymized text, list of anonymization records)
    """
    config = AnonymizationConfig(strategy=strategy, **kwargs)
    anonymizer = TextAnonymizer(config)
    return anonymizer.anonymize_entities(text, entities)
