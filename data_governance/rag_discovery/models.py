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
"""Shared data models for RAG discovery module.

This module contains dataclasses used across the RAG discovery components,
extracted to avoid circular imports and heavy dependency chains.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TableMetadata:
    """Representa metadados de uma tabela"""
    name: str
    database: str = ""
    schema: str = ""
    description: str = ""
    columns: List[Dict[str, Any]] = field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    location: str = ""
    format: str = ""  # parquet, delta, csv, etc
    partition_keys: List[str] = field(default_factory=list)
    sample_data: Optional[Dict[str, List]] = None

    def to_text_representation(self) -> str:
        """Converte metadados para representacao textual para embedding"""
        parts = []

        # Nome completo da tabela
        full_name = f"{self.database}.{self.schema}.{self.name}" if self.database and self.schema else self.name
        parts.append(f"Tabela: {full_name}")

        # Descricao
        if self.description:
            parts.append(f"Descricao: {self.description}")

        # Colunas
        if self.columns:
            parts.append("Colunas:")
            for col in self.columns:
                col_info = f"  - {col.get('name', 'unknown')} ({col.get('type', 'unknown')})"
                if col.get('description'):
                    col_info += f": {col['description']}"
                parts.append(col_info)

        # Informacoes adicionais
        if self.owner:
            parts.append(f"Proprietario: {self.owner}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.format:
            parts.append(f"Formato: {self.format}")

        if self.partition_keys:
            parts.append(f"Particionado por: {', '.join(self.partition_keys)}")

        if self.location:
            parts.append(f"Localizacao: {self.location}")

        if self.row_count:
            parts.append(f"Numero de linhas: {self.row_count:,}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario"""
        return {
            'name': self.name,
            'database': self.database,
            'schema': self.schema,
            'description': self.description,
            'columns': self.columns,
            'row_count': self.row_count,
            'size_bytes': self.size_bytes,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'owner': self.owner,
            'tags': self.tags,
            'location': self.location,
            'format': self.format,
            'partition_keys': self.partition_keys,
            'sample_data': self.sample_data
        }


@dataclass
class SearchResult:
    """Representa um resultado de busca"""
    table: TableMetadata
    relevance_score: float
    matching_reason: str
    snippet: str
