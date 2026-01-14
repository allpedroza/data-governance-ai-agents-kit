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
Data Governance Agents

This package contains agents focused on data governance, including:
- Data Lineage: Mapping and analyzing data pipeline dependencies
- RAG Discovery: Semantic search for data discovery
- Metadata Enrichment: Auto-generating metadata for data assets
- Data Classification: Classifying data by sensitivity level
- Data Quality: Monitoring and analyzing data quality
- Data Asset Value: Analyzing business value of data assets
- Data Product Scoring: Scoring layer for data products

These agents help organizations implement comprehensive data governance
practices across their data ecosystem.
"""

# Import agents when available
try:
    from .lineage import DataLineageAgent
    _lineage_available = True
except ImportError:
    _lineage_available = False

try:
    from .rag_discovery import RAGDiscoveryAgent
    _rag_available = True
except ImportError:
    _rag_available = False

try:
    from .metadata_enrichment import MetadataEnrichmentAgent
    _metadata_available = True
except ImportError:
    _metadata_available = False

try:
    from .data_classification import DataClassificationAgent
    _classification_available = True
except ImportError:
    _classification_available = False

try:
    from .data_quality import DataQualityAgent
    _quality_available = True
except ImportError:
    _quality_available = False

try:
    from .data_contracts import DataContractAgent
    _contracts_available = True
except ImportError:
    _contracts_available = False

try:
    from .data_asset_value import (
        DataAssetValueAgent,
        AssetValueReport,
        AssetValueScore,
    )
    _asset_value_available = True
except ImportError:
    _asset_value_available = False

from .data_product_scoring import (
    DataProductScoringAgent,
    DataProductScoringReport,
    DataProductScore,
    DataProductDefinition,
    DataProductContract,
    DataProductGovernance,
    DataProductScoringWeights,
)

__all__ = []

if _lineage_available:
    __all__.append('DataLineageAgent')

if _rag_available:
    __all__.append('RAGDiscoveryAgent')

if _metadata_available:
    __all__.append('MetadataEnrichmentAgent')

if _classification_available:
    __all__.append('DataClassificationAgent')

if _quality_available:
    __all__.append('DataQualityAgent')

if _contracts_available:
    __all__.append('DataContractAgent')

if _asset_value_available:
    __all__.extend([
        'DataAssetValueAgent',
        'AssetValueReport',
        'AssetValueScore',
    ])

__all__.extend([
    'DataProductScoringAgent',
    'DataProductScoringReport',
    'DataProductScore',
    'DataProductDefinition',
    'DataProductContract',
    'DataProductGovernance',
    'DataProductScoringWeights',
])

__version__ = '1.0.0'
