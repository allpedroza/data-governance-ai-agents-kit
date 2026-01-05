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
