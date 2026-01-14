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
Sensitive Data NER Agent

Named Entity Recognition agent for detecting and anonymizing sensitive data
in text before sending to LLMs or other external services.

This agent serves as a protective filter to prevent data leakage of:
- PII (Personally Identifiable Information)
- PHI (Protected Health Information)
- PCI (Payment Card Industry Data)
- Financial Information
- Business-sensitive/Strategic Information
- Credentials (API Keys, Tokens, Secrets, Passwords)

Features:
- Deterministic detection (regex patterns)
- Predictive detection (heuristics, validation, context analysis)
- Multiple anonymization strategies
- Integration with Classification Agent business terms
- LLM request filtering capability
- Configurable sensitivity thresholds
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Set, Tuple, Any,
    Callable, Union, Iterator
)
from enum import Enum
from datetime import datetime
import hashlib

from .patterns.entity_patterns import (
    EntityCategory,
    EntityPatternConfig,
    PII_PATTERNS,
    PHI_PATTERNS,
    PCI_PATTERNS,
    FINANCIAL_PATTERNS,
    CONTEXT_KEYWORDS,
    get_all_patterns,
)
from .predictive.validators import get_validator, has_validator
from .predictive.heuristics import (
    PredictiveDetector,
    ContextAnalyzer,
    calculate_entity_confidence,
)
from .anonymizers import (
    AnonymizationStrategy,
    AnonymizationConfig,
    TextAnonymizer,
    AnonymizedEntity,
    anonymize_text,
)

# Optional vault import
try:
    from .vault import SecureVault, VaultConfig, VaultSession, AccessLevel
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False


class FilterAction(Enum):
    """Actions when sensitive data is detected."""
    ALLOW = "allow"           # Allow request to proceed
    ANONYMIZE = "anonymize"   # Anonymize sensitive data and proceed
    BLOCK = "block"           # Block the request entirely
    WARN = "warn"             # Add warning but allow


@dataclass
class DetectedEntity:
    """Represents a detected sensitive entity."""
    value: str
    entity_type: str
    category: EntityCategory
    start: int
    end: int
    confidence: float
    is_validated: bool
    line_number: int
    context_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "entity_type": self.entity_type,
            "category": self.category.value,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "is_validated": self.is_validated,
            "line_number": self.line_number,
            "context_keywords": self.context_keywords,
            "metadata": self.metadata,
        }


@dataclass
class NERResult:
    """Result of NER analysis on text."""
    original_text: str
    anonymized_text: Optional[str]
    entities: List[DetectedEntity]
    filter_action: FilterAction
    risk_score: float
    categories_found: Set[EntityCategory]
    statistics: Dict[str, int]
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text_length": len(self.original_text),
            "anonymized_text": self.anonymized_text,
            "entities": [e.to_dict() for e in self.entities],
            "entity_count": len(self.entities),
            "filter_action": self.filter_action.value,
            "risk_score": round(self.risk_score, 3),
            "categories_found": [c.value for c in self.categories_found],
            "statistics": self.statistics,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "warnings": self.warnings,
            "blocked_reason": self.blocked_reason,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def is_safe(self) -> bool:
        """Check if text is safe to send to LLM."""
        return self.filter_action in (FilterAction.ALLOW, FilterAction.WARN)

    @property
    def has_sensitive_data(self) -> bool:
        """Check if any sensitive data was detected."""
        return len(self.entities) > 0


@dataclass
class FilterPolicy:
    """Policy configuration for LLM request filtering."""
    # Category-specific actions
    pii_action: FilterAction = FilterAction.ANONYMIZE
    phi_action: FilterAction = FilterAction.BLOCK
    pci_action: FilterAction = FilterAction.BLOCK
    financial_action: FilterAction = FilterAction.ANONYMIZE
    business_action: FilterAction = FilterAction.BLOCK
    credentials_action: FilterAction = FilterAction.BLOCK  # API keys, tokens, secrets

    # Thresholds
    min_confidence: float = 0.5
    max_entities_before_block: int = 10
    risk_score_block_threshold: float = 0.8

    # Anonymization config
    anonymization_strategy: AnonymizationStrategy = AnonymizationStrategy.REDACT

    def get_action_for_category(self, category: EntityCategory) -> FilterAction:
        """Get the configured action for a category."""
        mapping = {
            EntityCategory.PII: self.pii_action,
            EntityCategory.PHI: self.phi_action,
            EntityCategory.PCI: self.pci_action,
            EntityCategory.FINANCIAL: self.financial_action,
            EntityCategory.BUSINESS: self.business_action,
            EntityCategory.CREDENTIALS: self.credentials_action,
        }
        return mapping.get(category, FilterAction.WARN)


class SensitiveDataNERAgent:
    """
    Named Entity Recognition agent for sensitive data detection.

    This agent can be used to:
    1. Detect sensitive entities in text
    2. Filter LLM requests to prevent data leakage
    3. Anonymize sensitive data before processing
    4. Integrate with Classification Agent for business terms
    """

    def __init__(
        self,
        business_terms: Optional[List[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        filter_policy: Optional[FilterPolicy] = None,
        strict_mode: bool = False,
        locales: Optional[List[str]] = None,
        vault_config: Optional[Any] = None,
        enable_vault: bool = False,
    ):
        """
        Initialize the NER agent.

        Args:
            business_terms: List of business-sensitive terms to detect
            custom_patterns: Custom regex patterns {name: pattern}
            filter_policy: Policy for filtering LLM requests
            strict_mode: If True, require validation when available
            locales: List of locales to enable (e.g., ["br", "us"])
            vault_config: Configuration for secure vault (VaultConfig)
            enable_vault: Enable vault for storing original/anonymized mappings
        """
        self.filter_policy = filter_policy or FilterPolicy()
        self.strict_mode = strict_mode
        self.locales = set(locales) if locales else None

        # Initialize patterns
        self._patterns: Dict[str, EntityPatternConfig] = {}
        self._load_default_patterns()

        # Business terms
        self._business_terms: Set[str] = set()
        self._compiled_business_terms: List[Tuple[re.Pattern, str]] = []
        if business_terms:
            self.add_business_terms(business_terms)

        # Custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.add_custom_pattern(name, pattern)

        # Initialize predictive detector
        self._detector = PredictiveDetector(
            strict_mode=strict_mode,
            min_confidence_override=self.filter_policy.min_confidence
        )
        self._context_analyzer = ContextAnalyzer()

        # Vault for secure storage
        self._vault: Optional[Any] = None
        self._vault_token: Optional[str] = None
        self._enable_vault = enable_vault

        if enable_vault:
            if not HAS_VAULT:
                raise ImportError(
                    "Vault dependencies not available. "
                    "Install cryptography: pip install cryptography"
                )
            self._vault = SecureVault(vault_config)
            self._vault_initialized = False
        else:
            self._vault_initialized = False

    def _load_default_patterns(self) -> None:
        """Load default entity patterns filtered by locale."""
        all_patterns = get_all_patterns()

        for name, config in all_patterns.items():
            # Filter by locale if specified
            if self.locales is not None:
                if config.locale != "universal" and config.locale not in self.locales:
                    continue
            self._patterns[name] = config

    def add_business_terms(self, terms: List[str]) -> None:
        """
        Add business-sensitive terms to detect.

        These can be strategic project names, acquisition targets,
        confidential initiatives, etc.

        Args:
            terms: List of terms to add
        """
        for term in terms:
            cleaned = term.strip()
            if cleaned and cleaned.lower() not in self._business_terms:
                self._business_terms.add(cleaned.lower())
                # Compile pattern with word boundary
                pattern = re.compile(
                    rf"\b{re.escape(cleaned)}\b",
                    re.IGNORECASE
                )
                self._compiled_business_terms.append((pattern, cleaned))

    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        category: EntityCategory = EntityCategory.BUSINESS,
        description: str = ""
    ) -> None:
        """
        Add a custom detection pattern.

        Args:
            name: Pattern name (identifier)
            pattern: Regex pattern
            category: Entity category
            description: Human-readable description
        """
        self._patterns[f"custom_{name}"] = EntityPatternConfig(
            name=name,
            category=category,
            pattern=pattern,
            description=description or f"Custom pattern: {name}",
            priority=1
        )

    def import_from_classification_agent(
        self,
        classification_report: Dict[str, Any]
    ) -> None:
        """
        Import business terms from a Classification Agent report.

        Args:
            classification_report: Report dict from DataClassificationAgent
        """
        # Extract proprietary/business terms from report
        if "proprietary_columns" in classification_report:
            for col in classification_report["proprietary_columns"]:
                if isinstance(col, str):
                    self.add_business_terms([col])

        # Check for detected business patterns in columns
        if "columns" in classification_report:
            for col in classification_report["columns"]:
                patterns = col.get("detected_patterns", [])
                for pattern in patterns:
                    if pattern.startswith("business:"):
                        term = pattern.replace("business:", "")
                        self.add_business_terms([term])

    def detect_entities(
        self,
        text: str,
        include_low_confidence: bool = False
    ) -> List[DetectedEntity]:
        """
        Detect all sensitive entities in text.

        Args:
            text: Text to analyze
            include_low_confidence: Include entities below threshold

        Returns:
            List of detected entities
        """
        entities: List[DetectedEntity] = []
        seen_positions: Set[Tuple[int, int]] = set()

        # Sort patterns by priority (higher first)
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )

        # Detect using regex patterns
        for pattern_name, config in sorted_patterns:
            for match in config.compiled.finditer(text):
                start, end = match.start(), match.end()

                # Skip overlapping matches
                if any(
                    s <= start < e or s < end <= e
                    for s, e in seen_positions
                ):
                    continue

                value = match.group()

                # Calculate confidence with predictive analysis
                confidence, metadata = calculate_entity_confidence(
                    value, config, text, start, end, self.strict_mode
                )

                # Skip low confidence unless requested
                min_conf = self.filter_policy.min_confidence
                if not include_low_confidence and confidence < min_conf:
                    continue

                entity = DetectedEntity(
                    value=value,
                    entity_type=pattern_name,
                    category=config.category,
                    start=start,
                    end=end,
                    confidence=confidence,
                    is_validated=metadata.get("is_validated", False),
                    line_number=metadata.get("line_number", 1),
                    context_keywords=metadata.get("matched_keywords", []),
                    metadata=metadata
                )
                entities.append(entity)
                seen_positions.add((start, end))

        # Detect business terms
        for pattern, term in self._compiled_business_terms:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                # Skip overlapping
                if any(
                    s <= start < e or s < end <= e
                    for s, e in seen_positions
                ):
                    continue

                # Business terms get context-based confidence
                context = self._context_analyzer.extract_context(text, start, end)
                boost, keywords = self._context_analyzer.find_context_keywords(
                    context, EntityCategory.BUSINESS
                )

                confidence = 0.6 + boost  # Base confidence for exact term match

                entity = DetectedEntity(
                    value=match.group(),
                    entity_type=f"business_term:{term}",
                    category=EntityCategory.BUSINESS,
                    start=start,
                    end=end,
                    confidence=min(confidence, 1.0),
                    is_validated=True,  # Exact term match
                    line_number=text[:start].count('\n') + 1,
                    context_keywords=keywords,
                    metadata={"term": term}
                )
                entities.append(entity)
                seen_positions.add((start, end))

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def analyze(
        self,
        text: str,
        anonymize: bool = True,
        include_low_confidence: bool = False
    ) -> NERResult:
        """
        Perform full NER analysis on text.

        Args:
            text: Text to analyze
            anonymize: Whether to generate anonymized version
            include_low_confidence: Include low-confidence entities

        Returns:
            NERResult with detected entities and analysis
        """
        import time
        start_time = time.time()

        # Detect entities
        entities = self.detect_entities(text, include_low_confidence)

        # Collect statistics
        categories_found: Set[EntityCategory] = set()
        stats = {
            "total": len(entities),
            "pii": 0,
            "phi": 0,
            "pci": 0,
            "financial": 0,
            "business": 0,
            "credentials": 0,
            "validated": 0,
            "high_confidence": 0,
        }

        for entity in entities:
            categories_found.add(entity.category)
            stats[entity.category.value] += 1
            if entity.is_validated:
                stats["validated"] += 1
            if entity.confidence >= 0.8:
                stats["high_confidence"] += 1

        # Calculate risk score
        risk_score = self._calculate_risk_score(entities)

        # Determine filter action
        filter_action, blocked_reason, warnings = self._determine_filter_action(
            entities, categories_found, risk_score
        )

        # Generate anonymized text if requested
        anonymized_text = None
        if anonymize and entities:
            entity_dicts = [
                {
                    "value": e.value,
                    "entity_type": e.entity_type,
                    "category": e.category.value,
                    "start": e.start,
                    "end": e.end,
                }
                for e in entities
            ]
            anonymized_text, _ = anonymize_text(
                text,
                entity_dicts,
                strategy=self.filter_policy.anonymization_strategy
            )

        processing_time = (time.time() - start_time) * 1000

        return NERResult(
            original_text=text,
            anonymized_text=anonymized_text,
            entities=entities,
            filter_action=filter_action,
            risk_score=risk_score,
            categories_found=categories_found,
            statistics=stats,
            processing_time_ms=processing_time,
            warnings=warnings,
            blocked_reason=blocked_reason,
        )

    def _calculate_risk_score(self, entities: List[DetectedEntity]) -> float:
        """
        Calculate overall risk score based on detected entities.

        Risk factors:
        - Number of entities
        - Category severity
        - Confidence levels
        - Validation status
        """
        if not entities:
            return 0.0

        # Category weights
        category_weights = {
            EntityCategory.PII: 0.6,
            EntityCategory.PHI: 0.9,
            EntityCategory.PCI: 0.95,
            EntityCategory.FINANCIAL: 0.7,
            EntityCategory.BUSINESS: 0.8,
            EntityCategory.CREDENTIALS: 0.99,  # Highest risk - API keys, tokens, secrets
        }

        # Calculate weighted sum
        total_weight = 0.0
        for entity in entities:
            weight = category_weights.get(entity.category, 0.5)
            # Boost for validated entities
            if entity.is_validated:
                weight *= 1.2
            # Adjust by confidence
            weight *= entity.confidence
            total_weight += weight

        # Normalize (sigmoid-like scaling)
        # More entities = higher risk, but with diminishing returns
        normalized = total_weight / (total_weight + 5)

        return min(normalized, 1.0)

    def _determine_filter_action(
        self,
        entities: List[DetectedEntity],
        categories: Set[EntityCategory],
        risk_score: float
    ) -> Tuple[FilterAction, Optional[str], List[str]]:
        """
        Determine the appropriate filter action based on policy.

        Returns:
            Tuple of (action, blocked_reason, warnings)
        """
        warnings = []
        blocked_reason = None

        if not entities:
            return FilterAction.ALLOW, None, []

        # Check risk score threshold
        if risk_score >= self.filter_policy.risk_score_block_threshold:
            return (
                FilterAction.BLOCK,
                f"Risk score ({risk_score:.2f}) exceeds threshold",
                []
            )

        # Check entity count
        if len(entities) >= self.filter_policy.max_entities_before_block:
            return (
                FilterAction.BLOCK,
                f"Too many sensitive entities detected ({len(entities)})",
                []
            )

        # Check category-specific actions
        strictest_action = FilterAction.ALLOW

        for category in categories:
            action = self.filter_policy.get_action_for_category(category)

            if action == FilterAction.BLOCK:
                return (
                    FilterAction.BLOCK,
                    f"Policy blocks {category.value} data",
                    []
                )

            if action == FilterAction.ANONYMIZE:
                strictest_action = FilterAction.ANONYMIZE
            elif action == FilterAction.WARN and strictest_action == FilterAction.ALLOW:
                strictest_action = FilterAction.WARN
                warnings.append(
                    f"Warning: {category.value} data detected"
                )

        return strictest_action, blocked_reason, warnings

    def filter_llm_request(
        self,
        prompt: str,
        auto_anonymize: bool = True
    ) -> Tuple[str, NERResult]:
        """
        Filter an LLM request for sensitive data.

        This is the main entry point for using this agent as an
        LLM gateway filter.

        Args:
            prompt: The prompt to send to LLM
            auto_anonymize: Automatically anonymize if action is ANONYMIZE

        Returns:
            Tuple of (safe_prompt, analysis_result)

        Raises:
            ValueError: If the request is blocked by policy
        """
        result = self.analyze(prompt, anonymize=auto_anonymize)

        if result.filter_action == FilterAction.BLOCK:
            raise ValueError(
                f"Request blocked by sensitive data policy: {result.blocked_reason}"
            )

        if result.filter_action == FilterAction.ANONYMIZE and result.anonymized_text:
            return result.anonymized_text, result

        return prompt, result

    def create_safe_prompt(self, prompt: str) -> str:
        """
        Create a safe version of a prompt for LLM.

        Convenience method that returns only the safe prompt.

        Args:
            prompt: Original prompt

        Returns:
            Safe prompt with sensitive data handled

        Raises:
            ValueError: If request is blocked
        """
        safe_prompt, _ = self.filter_llm_request(prompt)
        return safe_prompt

    def batch_analyze(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> Iterator[NERResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze
            show_progress: Print progress (for CLI usage)

        Yields:
            NERResult for each text
        """
        total = len(texts)
        for i, text in enumerate(texts):
            if show_progress:
                print(f"Processing {i+1}/{total}...", end="\r")
            yield self.analyze(text)

    def get_entity_summary(
        self,
        result: NERResult
    ) -> Dict[str, List[str]]:
        """
        Get a summary of detected entities grouped by category.

        Args:
            result: NER analysis result

        Returns:
            Dict mapping category names to list of entity values
        """
        summary: Dict[str, List[str]] = {}

        for entity in result.entities:
            category = entity.category.value
            if category not in summary:
                summary[category] = []
            summary[category].append(entity.value)

        return summary

    def export_patterns(self) -> Dict[str, Any]:
        """Export current pattern configuration."""
        patterns = {}
        for name, config in self._patterns.items():
            patterns[name] = {
                "pattern": config.pattern,
                "category": config.category.value,
                "description": config.description,
                "locale": config.locale,
                "has_validation": config.has_validation,
            }

        return {
            "patterns": patterns,
            "business_terms": list(self._business_terms),
            "policy": {
                "min_confidence": self.filter_policy.min_confidence,
                "anonymization_strategy": self.filter_policy.anonymization_strategy.value,
            }
        }

    # =========================================================================
    # Vault Integration Methods
    # =========================================================================

    def initialize_vault(
        self,
        master_password: Optional[str] = None,
        admin_username: str = "admin",
        admin_password: str = "",
    ) -> str:
        """
        Initialize the secure vault for storing original/anonymized mappings.

        Args:
            master_password: Master password for encryption key derivation.
                           If None, uses VAULT_MASTER_KEY env var.
            admin_username: Username for vault admin user
            admin_password: Password for vault admin user

        Returns:
            Active encryption key ID
        """
        if not self._enable_vault or not self._vault:
            raise RuntimeError(
                "Vault not enabled. Initialize agent with enable_vault=True"
            )

        key_id = self._vault.initialize(master_password)

        # Create admin user if password provided
        if admin_password:
            try:
                self._vault.create_user(
                    admin_username,
                    admin_password,
                    AccessLevel.SUPER_ADMIN
                )
            except ValueError:
                # User already exists
                pass

            # Authenticate
            self._vault_token = self._vault.authenticate(
                admin_username, admin_password
            )

        self._vault_initialized = True
        return key_id

    def vault_authenticate(self, username: str, password: str) -> str:
        """
        Authenticate with the vault.

        Args:
            username: Vault username
            password: Vault password

        Returns:
            Access token
        """
        if not self._vault:
            raise RuntimeError("Vault not initialized")

        self._vault_token = self._vault.authenticate(username, password)
        return self._vault_token

    def vault_logout(self) -> None:
        """Logout from the vault."""
        if self._vault and self._vault_token:
            self._vault.logout(self._vault_token)
            self._vault_token = None

    def filter_and_store(
        self,
        prompt: str,
        auto_anonymize: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, NERResult, Optional[str]]:
        """
        Filter an LLM request and store the mapping in the vault.

        This is the main entry point for secure LLM filtering with
        the ability to later decrypt and recover the original message.

        Args:
            prompt: The prompt to send to LLM
            auto_anonymize: Automatically anonymize if action is ANONYMIZE
            metadata: Additional metadata to store with the session

        Returns:
            Tuple of (safe_prompt, analysis_result, session_id)
            session_id is None if vault is not enabled or no entities found

        Raises:
            ValueError: If the request is blocked by policy
        """
        # First, analyze and filter
        safe_prompt, result = self.filter_llm_request(prompt, auto_anonymize)

        session_id = None

        # Store in vault if enabled and entities were found
        if (
            self._enable_vault
            and self._vault
            and self._vault_token
            and result.entities
        ):
            # Prepare entities for vault storage
            entity_dicts = [
                {
                    "value": e.value,
                    "anonymized": self._get_anonymized_value(e, result),
                    "entity_type": e.entity_type,
                    "category": e.category.value,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "pattern": e.entity_type,
                }
                for e in result.entities
            ]

            # Store session
            vault_session = self._vault.store_session(
                token=self._vault_token,
                original_text=prompt,
                anonymized_text=result.anonymized_text or prompt,
                entities=entity_dicts,
                risk_score=result.risk_score,
                metadata=metadata or {},
            )
            session_id = vault_session.session_id

        return safe_prompt, result, session_id

    def _get_anonymized_value(
        self,
        entity: DetectedEntity,
        result: NERResult
    ) -> str:
        """Get the anonymized value for an entity from the result."""
        if not result.anonymized_text:
            return entity.value

        # The anonymized text has replaced values, we need to extract the label
        # For now, return a generic label based on entity type
        from .anonymizers import AnonymizationConfig
        config = AnonymizationConfig()

        if entity.entity_type in config.entity_labels:
            return config.entity_labels[entity.entity_type]
        elif entity.category.value in config.category_labels:
            return config.category_labels[entity.category.value]
        else:
            return f"[{entity.category.value.upper()}]"

    def decrypt_session(
        self,
        session_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Decrypt a stored session and recover original data.

        Args:
            session_id: Session ID from filter_and_store

        Returns:
            Tuple of (original_text, list of original entities)

        Raises:
            RuntimeError: If vault not initialized
            PermissionError: If access denied
            ValueError: If session not found
        """
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.decrypt_session(self._vault_token, session_id)

    def get_session_anonymized(
        self,
        session_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get anonymized data for a session (without decryption).

        This can be used by users with lower access levels who
        only need to see the anonymized version.

        Args:
            session_id: Session ID

        Returns:
            Tuple of (anonymized_text, anonymized entities)
        """
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.get_anonymized_session(self._vault_token, session_id)

    def list_vault_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List stored sessions from the vault.

        Args:
            limit: Maximum sessions to return
            offset: Pagination offset
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of session metadata
        """
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.list_sessions(
            self._vault_token, limit, offset, start_date, end_date
        )

    def delete_vault_session(self, session_id: str) -> bool:
        """
        Delete a session from the vault.

        Requires ADMIN access level.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
        """
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.delete_session(self._vault_token, session_id)

    def get_vault_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.get_stats(self._vault_token)

    def get_vault_audit_logs(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get vault audit logs."""
        if not self._vault or not self._vault_token:
            raise RuntimeError(
                "Vault not initialized or not authenticated"
            )

        return self._vault.get_audit_logs(
            self._vault_token,
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )

    @property
    def vault_enabled(self) -> bool:
        """Check if vault is enabled."""
        return self._enable_vault

    @property
    def vault_initialized(self) -> bool:
        """Check if vault is initialized and authenticated."""
        return self._vault_initialized and self._vault_token is not None

    def __repr__(self) -> str:
        vault_status = "enabled" if self._enable_vault else "disabled"
        return (
            f"SensitiveDataNERAgent("
            f"patterns={len(self._patterns)}, "
            f"business_terms={len(self._business_terms)}, "
            f"strict_mode={self.strict_mode}, "
            f"vault={vault_status})"
        )


# Convenience function for quick analysis
def analyze_text(
    text: str,
    business_terms: Optional[List[str]] = None,
    strict_mode: bool = False
) -> NERResult:
    """
    Quick text analysis for sensitive data.

    Args:
        text: Text to analyze
        business_terms: Optional business-sensitive terms
        strict_mode: Require validation when available

    Returns:
        NERResult with analysis
    """
    agent = SensitiveDataNERAgent(
        business_terms=business_terms,
        strict_mode=strict_mode
    )
    return agent.analyze(text)


# Export for convenience
__all__ = [
    "SensitiveDataNERAgent",
    "NERResult",
    "DetectedEntity",
    "FilterAction",
    "FilterPolicy",
    "AnonymizationConfig",
    "AnonymizationStrategy",
    "EntityCategory",
    "analyze_text",
]
