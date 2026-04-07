"""
Shared utilities for data-governance-ai-agents-kit.

Modules:
    serialization: SerializableMixin — provides to_json() for dataclasses.
    persistence:   JsonStorageMixin  — provides JSON file I/O helpers.
"""

from shared.serialization import SerializableMixin
from shared.persistence import JsonStorageMixin

__all__ = ["SerializableMixin", "JsonStorageMixin"]
