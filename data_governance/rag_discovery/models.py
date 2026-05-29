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

Extended to support the 6-step Data Discovery framework:
1. Business Context
2. Needs Mapping
3. Asset Inventory
4. Viability Analysis
5. Technical Assessment
6. Delivery Plan
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Maturity level constants (L0-L4)
# ---------------------------------------------------------------------------
MATURITY_INEXISTENT = 0   # Data does not exist yet, needs collection
MATURITY_RAW = 1          # Ingested but untreated (bronze)
MATURITY_STAGING = 2      # Cleaned, deduplicated (silver)
MATURITY_CURATED = 3      # Modeled with business rules (gold)
MATURITY_DATA_PRODUCT = 4 # Documented, contracted, monitored, with SLA

MATURITY_LABELS = {
    MATURITY_INEXISTENT: "L0 - Inexistente",
    MATURITY_RAW: "L1 - Raw / Bronze",
    MATURITY_STAGING: "L2 - Staging / Silver",
    MATURITY_CURATED: "L3 - Curated / Gold",
    MATURITY_DATA_PRODUCT: "L4 - Data Product",
}

# ---------------------------------------------------------------------------
# Asset type constants
# ---------------------------------------------------------------------------
ASSET_TYPE_FACT = "tabela_fato"
ASSET_TYPE_DIMENSION = "tabela_dimensão"
ASSET_TYPE_VIEW = "view"
ASSET_TYPE_SEMANTIC_MODEL = "modelo_semântico"
ASSET_TYPE_UNKNOWN = ""

# ---------------------------------------------------------------------------
# Data classification constants
# ---------------------------------------------------------------------------
CLASSIFICATION_PUBLIC = "público"
CLASSIFICATION_INTERNAL = "interno"
CLASSIFICATION_CONFIDENTIAL = "confidencial"
CLASSIFICATION_RESTRICTED = "restrito"


@dataclass
class TableMetadata:
    """Representa metadados de uma tabela — alinhado ao framework de Data Discovery.

    Campos originais mantidos para retrocompatibilidade.
    Novos campos adicionados com defaults para zero breaking changes.
    """
    # --- Identificação (existente) ---
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

    # --- Etapa 1: Contexto de Negócio (input humano) ---
    business_domain: str = ""       # "Vendas", "Marketing", "Financeiro"
    business_process: str = ""      # "Ciclo de Vendas", "Onboarding"
    stakeholders: List[str] = field(default_factory=list)

    # --- Etapa 3: Classificação do Asset ---
    asset_type: str = ""            # ASSET_TYPE_* constants
    source_system: str = ""         # "SAP ERP", "CRM", "Google Analytics"
    maturity_level: int = 0         # MATURITY_* constants (0-4)

    # --- Etapa 5: Mapeamento Técnico ---
    granularity: str = ""           # "sessão", "pedido", "cliente", "dia"
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    freshness_hours: Optional[float] = None  # SLA de freshness
    data_quality_score: Optional[float] = None  # Score 0-1 do DataQualityAgent
    classification: str = ""        # CLASSIFICATION_* constants

    def to_text_representation(self) -> str:
        """Converte metadados para representacao textual para embedding.

        Inclui campos de negócio, classificação e técnicos para que a busca
        semântica considere todos os aspectos do asset.
        """
        parts = []

        # Nome completo da tabela
        full_name = f"{self.database}.{self.schema}.{self.name}" if self.database and self.schema else self.name
        parts.append(f"Tabela: {full_name}")

        # Descricao
        if self.description:
            parts.append(f"Descricao: {self.description}")

        # --- Contexto de negócio ---
        if self.business_domain:
            parts.append(f"Domínio de negócio: {self.business_domain}")

        if self.business_process:
            parts.append(f"Processo: {self.business_process}")

        # --- Classificação do asset ---
        if self.asset_type:
            parts.append(f"Tipo de asset: {self.asset_type}")

        if self.source_system:
            parts.append(f"Sistema de origem: {self.source_system}")

        maturity_label = MATURITY_LABELS.get(self.maturity_level, "")
        if maturity_label:
            parts.append(f"Maturidade: {maturity_label}")

        if self.classification:
            parts.append(f"Classificação: {self.classification}")

        # --- Colunas ---
        if self.columns:
            parts.append("Colunas:")
            for col in self.columns:
                col_info = f"  - {col.get('name', 'unknown')} ({col.get('type', 'unknown')})"
                if col.get('description'):
                    col_info += f": {col['description']}"
                parts.append(col_info)

        # --- Informacoes técnicas ---
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

        if self.granularity:
            parts.append(f"Granularidade: {self.granularity}")

        if self.primary_keys:
            parts.append(f"Chaves primárias: {', '.join(self.primary_keys)}")

        if self.foreign_keys:
            fk_parts = [f"{fk.get('column', '?')} → {fk.get('references', '?')}" for fk in self.foreign_keys]
            parts.append(f"Chaves estrangeiras: {'; '.join(fk_parts)}")

        if self.freshness_hours is not None:
            parts.append(f"SLA de freshness: {self.freshness_hours}h")

        if self.data_quality_score is not None:
            parts.append(f"Quality score: {self.data_quality_score:.0%}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario completo (inclui campos novos)"""
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
            'sample_data': self.sample_data,
            # Novos campos
            'business_domain': self.business_domain,
            'business_process': self.business_process,
            'stakeholders': self.stakeholders,
            'asset_type': self.asset_type,
            'source_system': self.source_system,
            'maturity_level': self.maturity_level,
            'granularity': self.granularity,
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys,
            'freshness_hours': self.freshness_hours,
            'data_quality_score': self.data_quality_score,
            'classification': self.classification,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableMetadata":
        """Cria instância a partir de dicionário.

        Retrocompatível: ignora campos desconhecidos e aplica defaults
        para campos novos ausentes.
        """
        # Campos que são listas precisam de tratamento especial
        stakeholders = data.get('stakeholders', [])
        if isinstance(stakeholders, str):
            stakeholders = [s.strip() for s in stakeholders.split(',') if s.strip()]

        tags = data.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]

        primary_keys = data.get('primary_keys', [])
        if isinstance(primary_keys, str):
            primary_keys = [k.strip() for k in primary_keys.split(',') if k.strip()]

        partition_keys = data.get('partition_keys', [])
        if isinstance(partition_keys, str):
            partition_keys = [k.strip() for k in partition_keys.split(',') if k.strip()]

        foreign_keys = data.get('foreign_keys', [])
        if isinstance(foreign_keys, str):
            try:
                foreign_keys = json.loads(foreign_keys)
            except (json.JSONDecodeError, TypeError):
                foreign_keys = []

        return cls(
            name=data.get('name', ''),
            database=data.get('database', ''),
            schema=data.get('schema', ''),
            description=data.get('description', ''),
            columns=data.get('columns', []),
            row_count=data.get('row_count'),
            size_bytes=data.get('size_bytes'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            owner=data.get('owner', ''),
            tags=tags,
            location=data.get('location', ''),
            format=data.get('format', ''),
            partition_keys=partition_keys,
            sample_data=data.get('sample_data'),
            # Novos campos
            business_domain=data.get('business_domain', ''),
            business_process=data.get('business_process', ''),
            stakeholders=stakeholders,
            asset_type=data.get('asset_type', ''),
            source_system=data.get('source_system', ''),
            maturity_level=data.get('maturity_level', 0),
            granularity=data.get('granularity', ''),
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            freshness_hours=data.get('freshness_hours'),
            data_quality_score=data.get('data_quality_score'),
            classification=data.get('classification', ''),
        )

    def flatten_for_chroma(self) -> Dict[str, Any]:
        """Converte para metadados compatíveis com ChromaDB.

        ChromaDB aceita apenas str, int, float, bool como valores de metadados.
        Listas e dicts são serializados como JSON strings.
        """
        return {
            'name': self.name,
            'database': self.database,
            'schema': self.schema,
            'description': (self.description[:500] if self.description else ""),
            'owner': self.owner,
            'format': self.format,
            'location': self.location,
            'num_columns': len(self.columns),
            'row_count': self.row_count or 0,
            'tags': ','.join(self.tags) if self.tags else "",
            # Novos campos (primitivos)
            'business_domain': self.business_domain,
            'business_process': self.business_process,
            'asset_type': self.asset_type,
            'source_system': self.source_system,
            'maturity_level': self.maturity_level,
            'granularity': self.granularity,
            'freshness_hours': self.freshness_hours or 0.0,
            'data_quality_score': self.data_quality_score or 0.0,
            'classification': self.classification,
            # Listas serializadas como CSV
            'stakeholders': ','.join(self.stakeholders) if self.stakeholders else "",
            'primary_keys': ','.join(self.primary_keys) if self.primary_keys else "",
            'partition_keys': ','.join(self.partition_keys) if self.partition_keys else "",
            # Dicts serializados como JSON
            'foreign_keys': json.dumps(self.foreign_keys) if self.foreign_keys else "",
        }


@dataclass
class SearchResult:
    """Representa um resultado de busca"""
    table: TableMetadata
    relevance_score: float
    matching_reason: str
    snippet: str
