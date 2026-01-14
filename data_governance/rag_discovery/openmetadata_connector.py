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
"""Connector util to fetch table metadata from OpenMetadata for RAG indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

try:
    from rag_discovery.models import TableMetadata
except ImportError:
    from .models import TableMetadata


class OpenMetadataConnectorError(Exception):
    """Raised when communication with OpenMetadata fails."""


@dataclass
class OpenMetadataConnector:
    """Lightweight helper to fetch table metadata from an OpenMetadata server."""

    server_url: str
    api_token: str | None = None
    timeout: int = 15
    page_size: int = 50

    def __post_init__(self) -> None:
        self.server_url = self.server_url.rstrip("/")
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})

    def fetch_tables(
        self, max_tables: int = 200, service_filter: Optional[str] = None
    ) -> List[TableMetadata]:
        """
        Retrieve table entities and convert them to ``TableMetadata`` objects.

        Args:
            max_tables: Maximum number of tables to return.
            service_filter: Optional service name filter (matches OpenMetadata service).

        Returns:
            List of ``TableMetadata`` entries ready for indexing.
        """

        tables: List[TableMetadata] = []
        after = None

        while len(tables) < max_tables:
            params: Dict[str, str | int] = {"limit": min(self.page_size, max_tables - len(tables))}
            if after:
                params["after"] = after

            response = self.session.get(
                f"{self.server_url}/api/v1/tables",
                params=params,
                timeout=self.timeout,
            )

            if not response.ok:
                raise OpenMetadataConnectorError(
                    f"Falha ao consultar o OpenMetadata ({response.status_code}): {response.text}"
                )

            payload = response.json()
            entities = payload.get("data", [])

            for entity in entities:
                service_name = self._extract_service_name(entity)
                if service_filter and service_name:
                    if service_filter.lower() not in service_name.lower():
                        continue

                tables.append(self._convert_table(entity))
                if len(tables) >= max_tables:
                    break

            paging = payload.get("paging", {})
            after = paging.get("after")
            if not after:
                break

        return tables

    def _extract_service_name(self, entity: Dict[str, object]) -> str:
        service = entity.get("service")
        if isinstance(service, dict):
            return str(service.get("name", ""))
        if isinstance(service, str):
            return service
        return ""

    def _convert_table(self, entity: Dict[str, object]) -> TableMetadata:
        columns_payload = entity.get("columns", []) if isinstance(entity, dict) else []
        columns: List[Dict[str, str]] = []
        for col in columns_payload or []:
            if not isinstance(col, dict):
                continue
            columns.append(
                {
                    "name": str(col.get("name", "")),
                    "type": str(col.get("dataTypeDisplay") or col.get("dataType") or ""),
                    "description": str(col.get("description", "")),
                }
            )

        tags_payload = entity.get("tags") if isinstance(entity, dict) else []
        tags: List[str] = []
        if isinstance(tags_payload, list):
            for tag in tags_payload:
                if isinstance(tag, dict):
                    value = tag.get("tagFQN") or tag.get("name")
                    if value:
                        tags.append(str(value))

        fqn = ""
        if isinstance(entity, dict):
            fqn = str(entity.get("fullyQualifiedName", ""))

        database, schema = self._parse_fqn(fqn)

        return TableMetadata(
            name=str(entity.get("name", "")) if isinstance(entity, dict) else "",
            database=database,
            schema=schema,
            description=str(entity.get("description", "")) if isinstance(entity, dict) else "",
            columns=columns,
            owner=self._parse_owner(entity),
            tags=tags,
        )

    def _parse_owner(self, entity: Dict[str, object]) -> str:
        owner = entity.get("owner") if isinstance(entity, dict) else None
        if isinstance(owner, dict):
            return str(owner.get("name", ""))
        if isinstance(owner, str):
            return owner
        return ""

    def _parse_fqn(self, fqn: str) -> tuple[str, str]:
        if not fqn:
            return "", ""
        parts = fqn.split(".")
        if len(parts) >= 4:
            return parts[-3], parts[-2]
        if len(parts) == 3:
            return parts[0], parts[1]
        return "", ""
