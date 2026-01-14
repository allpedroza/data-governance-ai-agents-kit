"""
SpaCy Helper Module for Enhanced NER Detection

This module provides SpaCy-based entity detection and POS tagging
to improve accuracy in distinguishing person names from verbs,
common nouns, and other false positives.

Features:
- Person name validation using SpaCy NER (PER label)
- Verb detection using POS tagging
- Caching for performance optimization
- Graceful fallback when SpaCy is not available
"""

import logging
from typing import Optional, List, Dict, Set, Tuple
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# SpaCy availability flag
_SPACY_AVAILABLE = False
_nlp = None
_MODEL_NAME = "pt_core_news_md"  # Medium model, good balance of speed/accuracy

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    logger.info("SpaCy not available. Using regex-only mode for NER.")


@dataclass
class SpaCyEntity:
    """Represents an entity detected by SpaCy."""
    text: str
    label: str  # PER, ORG, LOC, MISC, etc.
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class SpaCyToken:
    """Represents a token with POS information."""
    text: str
    pos: str  # VERB, NOUN, PROPN, etc.
    lemma: str
    is_stop: bool


def is_spacy_available() -> bool:
    """Check if SpaCy is available."""
    return _SPACY_AVAILABLE


def get_model_name() -> str:
    """Get the SpaCy model name being used."""
    return _MODEL_NAME


def load_spacy_model(model_name: Optional[str] = None) -> bool:
    """
    Load SpaCy model for Portuguese NER.

    Args:
        model_name: Optional model name override

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _nlp, _MODEL_NAME

    if not _SPACY_AVAILABLE:
        return False

    if _nlp is not None:
        return True

    model = model_name or _MODEL_NAME

    # Try models in order of preference
    models_to_try = [model, "pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"]

    for m in models_to_try:
        try:
            _nlp = spacy.load(m)
            _MODEL_NAME = m
            logger.info(f"Loaded SpaCy model: {m}")
            return True
        except OSError:
            continue

    logger.warning(
        f"Could not load any Portuguese SpaCy model. "
        f"Install with: python -m spacy download pt_core_news_md"
    )
    return False


def get_nlp():
    """Get the loaded SpaCy NLP pipeline."""
    if _nlp is None:
        load_spacy_model()
    return _nlp


@lru_cache(maxsize=1000)
def analyze_text(text: str) -> Optional[Dict]:
    """
    Analyze text with SpaCy and cache results.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with entities, tokens, and analysis results
    """
    nlp = get_nlp()
    if nlp is None:
        return None

    try:
        doc = nlp(text)

        entities = [
            SpaCyEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            )
            for ent in doc.ents
        ]

        tokens = [
            SpaCyToken(
                text=token.text,
                pos=token.pos_,
                lemma=token.lemma_,
                is_stop=token.is_stop
            )
            for token in doc
        ]

        return {
            "entities": entities,
            "tokens": tokens,
            "text": text
        }
    except Exception as e:
        logger.warning(f"SpaCy analysis failed: {str(e)}")
        return None


def is_person_name_spacy(text: str) -> Tuple[bool, float]:
    """
    Check if text is a person name using SpaCy NER.

    Args:
        text: Text to check

    Returns:
        Tuple of (is_person_name, confidence)
        - is_person_name: True if SpaCy classifies as PER
        - confidence: Confidence score (0.0-1.0)
    """
    if not _SPACY_AVAILABLE:
        return (True, 0.5)  # Uncertain, let regex decide

    analysis = analyze_text(text)
    if analysis is None:
        return (True, 0.5)

    entities = analysis["entities"]

    # Check if any entity is classified as PER (Person)
    for ent in entities:
        if ent.label_ == "PER" and ent.text.lower() == text.lower():
            return (True, 0.95)

    # Check if the text spans a PER entity
    for ent in entities:
        if ent.label_ == "PER" and (
            ent.text.lower() in text.lower() or
            text.lower() in ent.text.lower()
        ):
            return (True, 0.85)

    # Check if any entity is ORG, LOC, or MISC - not a person
    for ent in entities:
        if ent.label_ in ("ORG", "LOC", "MISC") and ent.text.lower() == text.lower():
            return (False, 0.9)

    # No entity found - could be unknown
    return (True, 0.5)


def contains_verb(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains verbs using SpaCy POS tagging.

    Args:
        text: Text to check

    Returns:
        Tuple of (contains_verb, list_of_verbs)
    """
    if not _SPACY_AVAILABLE:
        return (False, [])

    analysis = analyze_text(text)
    if analysis is None:
        return (False, [])

    verbs = [
        token.text for token in analysis["tokens"]
        if token.pos == "VERB"
    ]

    return (len(verbs) > 0, verbs)


def get_pos_tags(text: str) -> List[Tuple[str, str]]:
    """
    Get POS tags for all tokens in text.

    Args:
        text: Text to analyze

    Returns:
        List of (token, pos_tag) tuples
    """
    if not _SPACY_AVAILABLE:
        return []

    analysis = analyze_text(text)
    if analysis is None:
        return []

    return [(token.text, token.pos) for token in analysis["tokens"]]


def is_proper_noun(text: str) -> Tuple[bool, float]:
    """
    Check if text consists of proper nouns (PROPN).

    Args:
        text: Text to check

    Returns:
        Tuple of (is_proper_noun, confidence)
    """
    if not _SPACY_AVAILABLE:
        return (True, 0.5)

    analysis = analyze_text(text)
    if analysis is None:
        return (True, 0.5)

    tokens = analysis["tokens"]

    if not tokens:
        return (False, 0.5)

    # Count proper nouns vs other types
    propn_count = sum(1 for t in tokens if t.pos == "PROPN")
    noun_count = sum(1 for t in tokens if t.pos == "NOUN")
    verb_count = sum(1 for t in tokens if t.pos == "VERB")
    adj_count = sum(1 for t in tokens if t.pos == "ADJ")

    total_content_words = propn_count + noun_count + verb_count + adj_count

    if total_content_words == 0:
        return (True, 0.5)  # Uncertain

    # If mostly proper nouns, likely a name
    propn_ratio = propn_count / total_content_words

    if propn_ratio >= 0.8:
        return (True, 0.9)
    elif propn_ratio >= 0.5:
        return (True, 0.7)
    elif verb_count > 0:
        return (False, 0.85)  # Contains verb - likely not a name
    elif noun_count > propn_count:
        return (False, 0.7)  # More common nouns than proper nouns
    else:
        return (True, 0.5)


def validate_person_name_with_spacy(name: str) -> Tuple[bool, float, str]:
    """
    Comprehensive person name validation using SpaCy.

    Combines multiple SpaCy checks:
    1. NER entity classification (PER label)
    2. POS tagging (PROPN vs VERB/NOUN)
    3. Context analysis

    Args:
        name: Candidate person name to validate

    Returns:
        Tuple of (is_valid, confidence, reason)
    """
    if not _SPACY_AVAILABLE:
        return (True, 0.5, "spacy_not_available")

    # Check if it's a verb
    has_verb, verbs = contains_verb(name)
    if has_verb:
        # Check if it's ONLY a verb (like "Critical Error")
        analysis = analyze_text(name)
        if analysis:
            tokens = [t for t in analysis["tokens"] if not t.is_stop]
            verb_tokens = [t for t in tokens if t.pos == "VERB"]

            # If all content words are verbs, reject
            if len(verb_tokens) == len(tokens) and len(tokens) > 0:
                return (False, 0.9, f"all_verbs:{','.join(verbs)}")

            # If first word is a verb, likely not a name
            if tokens and tokens[0].pos == "VERB":
                return (False, 0.85, f"starts_with_verb:{tokens[0].text}")

    # Check NER classification
    is_per, per_conf = is_person_name_spacy(name)
    if not is_per:
        return (False, per_conf, "ner_not_person")

    # Check proper noun ratio
    is_propn, propn_conf = is_proper_noun(name)
    if not is_propn and propn_conf > 0.7:
        return (False, propn_conf, "not_proper_noun")

    # If SpaCy confirms as PER with high confidence
    if is_per and per_conf >= 0.85:
        return (True, per_conf, "ner_confirmed_person")

    # Default: uncertain, let regex rules decide
    return (True, 0.5, "uncertain")


def extract_entities(text: str, entity_types: Optional[Set[str]] = None) -> List[SpaCyEntity]:
    """
    Extract all entities from text using SpaCy NER.

    Args:
        text: Text to analyze
        entity_types: Optional set of entity types to filter (PER, ORG, LOC, MISC)

    Returns:
        List of detected entities
    """
    if not _SPACY_AVAILABLE:
        return []

    analysis = analyze_text(text)
    if analysis is None:
        return []

    entities = analysis["entities"]

    if entity_types:
        entities = [e for e in entities if e.label in entity_types]

    return entities


def detect_portuguese_verb_patterns(text: str) -> List[Dict]:
    """
    Detect Portuguese verb patterns using both SpaCy and regex.

    This helps identify conjugated verbs that might be false positives
    for person names.

    Args:
        text: Text to analyze

    Returns:
        List of detected verb patterns with details
    """
    import re

    verbs_found = []

    # SpaCy-based detection
    if _SPACY_AVAILABLE:
        analysis = analyze_text(text)
        if analysis:
            for token in analysis["tokens"]:
                if token.pos == "VERB":
                    verbs_found.append({
                        "text": token.text,
                        "lemma": token.lemma,
                        "source": "spacy",
                        "confidence": 0.95
                    })

    # Regex-based verb patterns (Portuguese morphology)
    verb_patterns = [
        # Infinitives
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:ar|er|ir))\b', 'infinitive'),
        # Gerund (-ando, -endo, -indo)
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:ando|endo|indo))\b', 'gerund'),
        # Past participle (-ado, -ido)
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:ado|ido))\b', 'participle'),
        # Common conjugations
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:ou|eu|iu))\b', 'past_3s'),
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:amos|emos|imos))\b', 'past_1p'),
        (r'\b([a-záàâãéèêíïóôõöúç]+(?:aram|eram|iram))\b', 'past_3p'),
    ]

    # Words that look like verb patterns but are not verbs
    not_verbs = {
        "estado", "estudo", "cargo", "caso", "dado", "lado",
        "passado", "mercado", "resultado", "sentido", "pedido",
        "partido", "conteúdo", "conteudo", "período", "periodo"
    }

    for pattern, verb_type in verb_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            word = match.group(1).lower()

            # Skip known non-verbs
            if word in not_verbs:
                continue

            # Check if already found by SpaCy
            if any(v["text"].lower() == word for v in verbs_found):
                continue

            verbs_found.append({
                "text": match.group(1),
                "lemma": None,
                "source": "regex",
                "verb_type": verb_type,
                "confidence": 0.7
            })

    return verbs_found


def clear_cache():
    """Clear the SpaCy analysis cache."""
    analyze_text.cache_clear()


# Pre-load model on import if available
if _SPACY_AVAILABLE:
    try:
        load_spacy_model()
    except Exception as e:
        logger.warning(f"Could not pre-load SpaCy model: {str(e)}")
