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
"""Example: Sync OpenMetadata tables and index them for semantic discovery."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_discovery_rag_agent import DataDiscoveryRAGAgent
from openmetadata_connector import OpenMetadataConnector


def main() -> None:
    base_url = os.getenv("OPENMETADATA_HOST", "http://localhost:8585")
    token = os.getenv("OPENMETADATA_API_TOKEN", "")
    service = os.getenv("OPENMETADATA_SERVICE", "")

    if not token:
        raise SystemExit("Defina OPENMETADATA_API_TOKEN para consultar o catálogo.")

    connector = OpenMetadataConnector(server_url=base_url, api_token=token)
    agent = DataDiscoveryRAGAgent(
        collection_name="openmetadata_catalog",
        persist_directory=str(Path(__file__).parent / ".chroma_openmetadata"),
    )

    print("📥 Buscando tabelas no OpenMetadata...")
    tables = connector.fetch_tables(max_tables=200, service_filter=service or None)
    print(f"Encontradas {len(tables)} tabelas. Indexando...")

    agent.index_tables_batch(tables, force_update=True)
    print("✅ Catálogo disponível para busca semântica")

    sample_query = "Quais tabelas contêm dados de clientes?"
    results = agent.search(sample_query, n_results=3)
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result.table.database}.{result.table.name} → {result.relevance_score:.0%}")
        if result.table.description:
            print(f"   {result.table.description[:100]}")


if __name__ == "__main__":
    main()
