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
