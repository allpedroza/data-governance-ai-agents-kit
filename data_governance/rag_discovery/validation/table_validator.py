"""
Table Validator - Validates discovered tables against a catalog
Eliminates LLM hallucinations by ensuring tables actually exist
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of table validation"""
    table_name: str
    is_valid: bool
    matched_name: Optional[str] = None  # Correct name if partial match
    original_name: Optional[str] = None  # Original name before correction
    reason: Optional[str] = None
    confidence: str = "high"  # high, medium, low


class TableValidator:
    """
    Validates discovered tables against a catalog of available tables

    Features:
    - Exact match validation
    - Partial/fuzzy matching for typos
    - Multiple catalog source formats (TXT, JSON, API)
    - Statistics and reporting

    This is crucial for eliminating LLM hallucinations - only
    tables that actually exist in the catalog are returned.
    """

    def __init__(
        self,
        catalog_source: Optional[str] = None,
        case_sensitive: bool = False,
        allow_partial_match: bool = True
    ):
        """
        Initialize table validator

        Args:
            catalog_source: Path to catalog file or connection string
            case_sensitive: Whether table name matching is case-sensitive
            allow_partial_match: Allow partial name matches (e.g., 'customers' matches 'prod.public.customers')
        """
        self._available_tables: Set[str] = set()
        self._table_metadata: Dict[str, Dict[str, Any]] = {}
        self._case_sensitive = case_sensitive
        self._allow_partial_match = allow_partial_match

        if catalog_source:
            self.load_catalog(catalog_source)

    def load_catalog(self, source: str) -> int:
        """
        Load catalog from various sources

        Supported formats:
        - .txt: One table per line or BigQuery format in text
        - .json: Array of table objects or dict with table names as keys
        - .csv: CSV with table_name column

        Args:
            source: Path to catalog file

        Returns:
            Number of tables loaded
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Catalog file not found: {source}")

        source_lower = source.lower()

        if source_lower.endswith('.txt'):
            return self._load_from_txt(source)
        elif source_lower.endswith('.json'):
            return self._load_from_json(source)
        elif source_lower.endswith('.csv'):
            return self._load_from_csv(source)
        else:
            # Try to auto-detect format
            return self._load_auto_detect(source)

    def _load_from_txt(self, filepath: str) -> int:
        """Load tables from TXT file"""
        # Pattern for BigQuery-style table names: project.dataset.table
        table_pattern = r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)'

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tables = re.findall(table_pattern, content)
        self._available_tables = set(tables)

        return len(self._available_tables)

    def _load_from_json(self, filepath: str) -> int:
        """Load tables from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Array of objects
            for item in data:
                if isinstance(item, str):
                    self._available_tables.add(item)
                elif isinstance(item, dict):
                    name = (
                        item.get('name') or
                        item.get('table_name') or
                        item.get('full_name') or
                        item.get('id')
                    )
                    if name:
                        self._available_tables.add(name)
                        self._table_metadata[name] = item

        elif isinstance(data, dict):
            # Dict with table names as keys
            for name, metadata in data.items():
                self._available_tables.add(name)
                if isinstance(metadata, dict):
                    self._table_metadata[name] = metadata

        return len(self._available_tables)

    def _load_from_csv(self, filepath: str) -> int:
        """Load tables from CSV file"""
        import csv

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Try common column names
                name = (
                    row.get('table_name') or
                    row.get('name') or
                    row.get('full_name') or
                    row.get('table_id')
                )

                if name:
                    self._available_tables.add(name)
                    self._table_metadata[name] = dict(row)

        return len(self._available_tables)

    def _load_auto_detect(self, filepath: str) -> int:
        """Try to auto-detect file format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars

        # Try JSON
        if content.strip().startswith('{') or content.strip().startswith('['):
            return self._load_from_json(filepath)

        # Default to TXT
        return self._load_from_txt(filepath)

    def add_tables(self, tables: List[str]) -> None:
        """Add tables programmatically"""
        self._available_tables.update(tables)

    def _normalize_name(self, name: str) -> str:
        """Normalize table name for comparison"""
        if self._case_sensitive:
            return name.strip()
        return name.strip().lower()

    def _find_match(self, table_name: str) -> Tuple[bool, Optional[str], str]:
        """
        Find matching table in catalog

        Returns:
            Tuple of (is_valid, matched_name, confidence)
        """
        normalized_query = self._normalize_name(table_name)

        # Exact match
        for available in self._available_tables:
            if self._normalize_name(available) == normalized_query:
                return True, available, "high"

        if not self._allow_partial_match:
            return False, None, "none"

        # Partial match: query is suffix of available
        for available in self._available_tables:
            norm_available = self._normalize_name(available)
            if norm_available.endswith(normalized_query):
                return True, available, "medium"

        # Partial match: query contains available's table part
        for available in self._available_tables:
            parts = available.split('.')
            if parts:
                table_part = self._normalize_name(parts[-1])
                if table_part == normalized_query:
                    return True, available, "medium"

        # Partial match: available contains query
        for available in self._available_tables:
            if normalized_query in self._normalize_name(available):
                return True, available, "low"

        return False, None, "none"

    def validate_single(self, table_name: str) -> ValidationResult:
        """
        Validate a single table name

        Args:
            table_name: Table name to validate

        Returns:
            ValidationResult with match information
        """
        if not table_name:
            return ValidationResult(
                table_name="",
                is_valid=False,
                reason="Empty table name"
            )

        is_valid, matched_name, confidence = self._find_match(table_name)

        if is_valid:
            return ValidationResult(
                table_name=matched_name or table_name,
                is_valid=True,
                matched_name=matched_name,
                original_name=table_name if matched_name != table_name else None,
                confidence=confidence
            )
        else:
            return ValidationResult(
                table_name=table_name,
                is_valid=False,
                reason="Table not found in catalog"
            )

    def validate(
        self,
        discovered_tables: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a list of discovered tables

        Args:
            discovered_tables: List of table dicts with 'table_name' key

        Returns:
            Tuple of (valid_tables, invalid_tables)
        """
        valid_tables = []
        invalid_tables = []

        for table in discovered_tables:
            table_name = table.get("table_name", "")

            if not table_name:
                invalid_tables.append({
                    **table,
                    "reason": "Missing or empty table name"
                })
                continue

            result = self.validate_single(table_name)

            if result.is_valid:
                validated_table = {**table}

                # Update with corrected name if needed
                if result.matched_name and result.matched_name != table_name:
                    validated_table["table_name"] = result.matched_name
                    validated_table["original_name"] = table_name
                    validated_table["match_confidence"] = result.confidence

                # Add metadata if available
                if result.matched_name and result.matched_name in self._table_metadata:
                    validated_table["catalog_metadata"] = self._table_metadata[result.matched_name]

                valid_tables.append(validated_table)
            else:
                invalid_tables.append({
                    **table,
                    "reason": result.reason
                })

        return valid_tables, invalid_tables

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics"""
        # Count by project/database
        projects = {}
        for table in self._available_tables:
            parts = table.split('.')
            if len(parts) >= 1:
                project = parts[0]
                projects[project] = projects.get(project, 0) + 1

        return {
            "total_tables": len(self._available_tables),
            "tables_with_metadata": len(self._table_metadata),
            "projects": projects
        }

    def search_catalog(
        self,
        query: str,
        limit: int = 10
    ) -> List[str]:
        """
        Search catalog for tables matching query

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching table names
        """
        normalized_query = self._normalize_name(query)
        matches = []

        for table in self._available_tables:
            if normalized_query in self._normalize_name(table):
                matches.append(table)
                if len(matches) >= limit:
                    break

        return matches

    @property
    def catalog_size(self) -> int:
        """Number of tables in catalog"""
        return len(self._available_tables)

    @property
    def tables(self) -> Set[str]:
        """Get all table names"""
        return self._available_tables.copy()
