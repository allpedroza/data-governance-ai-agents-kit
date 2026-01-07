"""
Predictive Detection Heuristics

This module provides context-aware detection and confidence
scoring for entity recognition beyond simple regex matching.

Key features:
- Context window analysis (surrounding text)
- Keyword proximity scoring
- Statistical pattern analysis
- Multi-entity correlation
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..patterns.entity_patterns import (
    EntityCategory,
    CONTEXT_KEYWORDS,
    EntityPatternConfig,
)
from .validators import get_validator


@dataclass
class ContextWindow:
    """Represents a window of text around a potential entity."""
    text: str
    start: int
    end: int
    entity_value: str
    before_context: str
    after_context: str
    line_number: int = 0


@dataclass
class ConfidenceFactors:
    """Breakdown of confidence score factors."""
    pattern_match: float = 0.0
    validation_pass: float = 0.0
    context_keywords: float = 0.0
    format_quality: float = 0.0
    entity_density: float = 0.0
    negation_penalty: float = 0.0

    @property
    def total(self) -> float:
        """Calculate total confidence score (0.0 to 1.0)."""
        raw = (
            self.pattern_match +
            self.validation_pass +
            self.context_keywords +
            self.format_quality +
            self.entity_density -
            self.negation_penalty
        )
        return max(0.0, min(1.0, raw))


class ContextAnalyzer:
    """
    Analyzes text context around detected entities to improve
    detection confidence and reduce false positives.
    """

    # Negation patterns that reduce confidence
    # Note: "no" removed - in Portuguese it's a preposition ("em + o" = "in the"), not negation
    NEGATION_PATTERNS = [
        r"\b(?:não|not|never|nunca|sem|without)\s+(?:é|is|are|was|were|tem|has|have)?\s*",
        r"\b(?:exemplo|example|sample|teste|test|fake|mock|dummy|placeholder)\b",
        r"\b(?:inválido|invalid|incorreto|incorrect|erro|error)\b",
        r"(?:formato|format):\s*",
        r"(?:x{4,}|0{5,}|1{5,}|9{5,})",  # Require 4+ x's or 5+ repeated digits
    ]

    # Context window size (characters before/after entity)
    CONTEXT_SIZE = 100

    def __init__(self):
        self._compiled_negations = [
            re.compile(p, re.IGNORECASE) for p in self.NEGATION_PATTERNS
        ]

    def extract_context(
        self,
        text: str,
        match_start: int,
        match_end: int
    ) -> ContextWindow:
        """
        Extract context window around a match.

        Args:
            text: Full text
            match_start: Start position of match
            match_end: End position of match

        Returns:
            ContextWindow with before/after context
        """
        before_start = max(0, match_start - self.CONTEXT_SIZE)
        after_end = min(len(text), match_end + self.CONTEXT_SIZE)

        # Count line number
        line_number = text[:match_start].count('\n') + 1

        return ContextWindow(
            text=text[before_start:after_end],
            start=match_start,
            end=match_end,
            entity_value=text[match_start:match_end],
            before_context=text[before_start:match_start].lower(),
            after_context=text[match_end:after_end].lower(),
            line_number=line_number
        )

    def check_negation(self, context: ContextWindow) -> float:
        """
        Check for negation patterns in context.

        Returns:
            Penalty score (0.0 to 0.5) if negation found
        """
        full_context = context.before_context + " " + context.after_context

        for pattern in self._compiled_negations:
            if pattern.search(full_context):
                return 0.3  # Significant penalty for negation context

        # Check for obviously fake values
        entity = context.entity_value
        if re.match(r'^(0+|1+|x+|X+|\*+)$', re.sub(r'\D', '', entity)):
            return 0.5  # High penalty for placeholder values

        return 0.0

    def find_context_keywords(
        self,
        context: ContextWindow,
        category: EntityCategory
    ) -> Tuple[float, List[str]]:
        """
        Find category-related keywords in context.

        Args:
            context: Context window around entity
            category: Entity category to check for

        Returns:
            Tuple of (confidence boost, list of matched keywords)
        """
        if category not in CONTEXT_KEYWORDS:
            return 0.0, []

        keywords = CONTEXT_KEYWORDS[category]
        full_context = context.before_context + " " + context.after_context
        matched = []

        boost = 0.0

        # Check high confidence keywords
        for kw in keywords.get("high_confidence", set()):
            if kw in full_context:
                boost += 0.25
                matched.append(f"+{kw}")

        # Check medium confidence keywords
        for kw in keywords.get("medium_confidence", set()):
            if kw in full_context:
                boost += 0.15
                matched.append(f"~{kw}")

        # Check low confidence keywords
        for kw in keywords.get("low_confidence", set()):
            if kw in full_context:
                boost += 0.05
                matched.append(kw)

        # Cap the boost
        return min(boost, 0.4), matched

    def analyze_format_quality(self, entity: str, pattern_name: str) -> float:
        """
        Analyze the format quality of detected entity.

        Some patterns match loosely; this checks for well-formatted values.

        Returns:
            Quality score (0.0 to 0.15)
        """
        # Remove separators for analysis
        digits = re.sub(r'\D', '', entity)

        # Check for proper formatting (with separators)
        has_formatting = len(entity) > len(digits)

        # Credit cards with spaces/dashes are more likely real
        if pattern_name in ("credit_card", "card_visa", "card_mastercard"):
            if has_formatting:
                return 0.15
            return 0.05

        # CPF/CNPJ with dots and dashes are more likely real
        if pattern_name in ("cpf", "cnpj"):
            if '.' in entity or '-' in entity:
                return 0.15
            return 0.05

        # Email with proper domain is more likely real
        if pattern_name == "email" and '.' in entity.split('@')[-1]:
            return 0.1

        return 0.05


class PredictiveDetector:
    """
    Predictive entity detection using multiple heuristics.

    Combines regex matching with:
    - Checksum validation (when available)
    - Context analysis
    - Keyword proximity
    - Format quality assessment
    - Multi-entity correlation
    """

    # Minimum confidence thresholds by category
    MIN_CONFIDENCE = {
        EntityCategory.PII: 0.4,
        EntityCategory.PHI: 0.5,
        EntityCategory.PCI: 0.6,  # Higher bar for payment data
        EntityCategory.FINANCIAL: 0.5,
        EntityCategory.BUSINESS: 0.3,  # Lower bar (term matching)
    }

    def __init__(
        self,
        strict_mode: bool = False,
        min_confidence_override: Optional[float] = None
    ):
        """
        Initialize predictive detector.

        Args:
            strict_mode: If True, requires validation when available
            min_confidence_override: Override default minimum confidence
        """
        self.strict_mode = strict_mode
        self.min_confidence_override = min_confidence_override
        self.context_analyzer = ContextAnalyzer()
        self._entity_cache: Dict[str, float] = {}

    def calculate_confidence(
        self,
        entity_value: str,
        pattern_config: EntityPatternConfig,
        context: Optional[ContextWindow] = None,
        nearby_entities: Optional[List[str]] = None
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Calculate confidence score for a detected entity.

        Args:
            entity_value: The matched entity value
            pattern_config: Pattern configuration used for match
            context: Optional context window around entity
            nearby_entities: Other entities detected nearby

        Returns:
            Tuple of (final confidence, confidence factors breakdown)
        """
        factors = ConfidenceFactors()

        # Base score for pattern match
        factors.pattern_match = 0.35

        # Validation check
        validator = get_validator(pattern_config.name)
        if validator:
            if validator(entity_value):
                factors.validation_pass = 0.35  # Strong boost for valid checksum
            else:
                if self.strict_mode:
                    factors.validation_pass = -0.5  # Penalty in strict mode
                else:
                    factors.validation_pass = -0.15  # Lighter penalty

        # Context analysis
        if context:
            # Check for negation
            factors.negation_penalty = self.context_analyzer.check_negation(context)

            # Find context keywords
            boost, _ = self.context_analyzer.find_context_keywords(
                context, pattern_config.category
            )
            factors.context_keywords = boost

            # Format quality
            factors.format_quality = self.context_analyzer.analyze_format_quality(
                entity_value, pattern_config.name
            )

        # Entity density (more entities nearby = higher confidence)
        if nearby_entities and len(nearby_entities) > 1:
            factors.entity_density = min(0.1, len(nearby_entities) * 0.02)

        # Apply pattern-specific context boost
        if context:
            factors.context_keywords += pattern_config.context_boost * 0.5

        return factors.total, factors

    def get_min_confidence(self, category: EntityCategory) -> float:
        """Get minimum confidence threshold for a category."""
        if self.min_confidence_override is not None:
            return self.min_confidence_override
        return self.MIN_CONFIDENCE.get(category, 0.4)

    def should_accept(
        self,
        confidence: float,
        category: EntityCategory
    ) -> bool:
        """
        Determine if entity should be accepted based on confidence.

        Args:
            confidence: Calculated confidence score
            category: Entity category

        Returns:
            True if entity should be accepted
        """
        return confidence >= self.get_min_confidence(category)


def calculate_entity_confidence(
    entity_value: str,
    pattern_config: EntityPatternConfig,
    full_text: str,
    match_start: int,
    match_end: int,
    strict_mode: bool = False
) -> Tuple[float, Dict[str, any]]:
    """
    Convenience function to calculate entity confidence.

    Args:
        entity_value: Matched entity value
        pattern_config: Pattern configuration
        full_text: Full text being analyzed
        match_start: Start position of match
        match_end: End position of match
        strict_mode: Whether to use strict validation

    Returns:
        Tuple of (confidence score, metadata dict)
    """
    detector = PredictiveDetector(strict_mode=strict_mode)
    context_analyzer = ContextAnalyzer()

    context = context_analyzer.extract_context(full_text, match_start, match_end)
    confidence, factors = detector.calculate_confidence(
        entity_value, pattern_config, context
    )

    # Get matched keywords for metadata
    _, matched_keywords = context_analyzer.find_context_keywords(
        context, pattern_config.category
    )

    metadata = {
        "confidence": confidence,
        "factors": {
            "pattern_match": factors.pattern_match,
            "validation_pass": factors.validation_pass,
            "context_keywords": factors.context_keywords,
            "format_quality": factors.format_quality,
            "entity_density": factors.entity_density,
            "negation_penalty": factors.negation_penalty,
        },
        "matched_keywords": matched_keywords,
        "line_number": context.line_number,
        "is_validated": factors.validation_pass > 0,
    }

    return confidence, metadata
