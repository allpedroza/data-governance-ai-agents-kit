"""
Example: Using OpenMetadata as metadata source for Data Discovery

OpenMetadata is an open-source metadata platform for data discovery,
data observability, and data governance.

This example shows how to:
1. Connect to OpenMetadata
2. Fetch table metadata
3. Index in Data Discovery Agent
4. Run queries

Prerequisites:
- OpenMetadata server running (default: http://localhost:8585)
- JWT token for authentication (or configure in settings)
- pip install openmetadata-ingestion (optional, for SDK mode)

References:
- https://docs.open-metadata.org
- https://github.com/open-metadata/OpenMetadata
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def example_basic_usage():
    """Basic usage: Connect and fetch tables"""
    print("=" * 60)
    print("Example 1: Basic OpenMetadata Connection")
    print("=" * 60)

    from rag_discovery.connectors import OpenMetadataConnector, OpenMetadataConfig

    # Configuration from environment or explicit
    config = OpenMetadataConfig(
        server_host=os.getenv("OPENMETADATA_HOST", "http://localhost:8585"),
        jwt_token=os.getenv("OPENMETADATA_JWT_TOKEN"),
        auth_type="jwt"
    )

    print(f"Connecting to: {config.server_host}")

    try:
        connector = OpenMetadataConnector(config)

        # Check health
        if connector.health_check():
            print("✓ Connected to OpenMetadata")
        else:
            print("✗ Failed to connect")
            return

        # List services
        print("\nDatabase Services:")
        services = connector.get_services()
        for svc in services[:5]:
            print(f"  - {svc['name']} ({svc['service_type']})")

        # Fetch tables
        print("\nFetching tables...")
        tables = connector.get_tables(limit=10)

        print(f"\nFound {len(tables)} tables:")
        for table in tables[:5]:
            print(f"\n  {table.fully_qualified_name}")
            print(f"    Description: {table.description[:100]}..." if table.description else "    No description")
            print(f"    Columns: {len(table.columns)}")
            print(f"    Tags: {', '.join(table.tags) if table.tags else 'None'}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure OpenMetadata is running and accessible.")
        print("You can start it with: docker-compose up -d")


def example_export_metadata():
    """Export metadata to files for offline use"""
    print("\n" + "=" * 60)
    print("Example 2: Export Metadata to Files")
    print("=" * 60)

    from rag_discovery.connectors import OpenMetadataConnector, OpenMetadataConfig

    config = OpenMetadataConfig.from_env()

    try:
        connector = OpenMetadataConnector(config)

        # Export to JSON (for indexing)
        output_json = "./exports/openmetadata_tables.json"
        count = connector.export_to_json(
            output_path=output_json,
            limit=100
        )
        print(f"✓ Exported {count} tables to {output_json}")

        # Export to TXT (for catalog validation)
        output_txt = "./exports/openmetadata_catalog.txt"
        count = connector.export_to_catalog_txt(
            output_path=output_txt,
            limit=100
        )
        print(f"✓ Exported {count} table names to {output_txt}")

    except Exception as e:
        print(f"Error: {e}")


def example_with_data_discovery_agent():
    """Full integration with Data Discovery Agent"""
    print("\n" + "=" * 60)
    print("Example 3: Data Discovery with OpenMetadata")
    print("=" * 60)

    from rag_discovery import DataDiscoveryAgent
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore

    # Initialize providers
    print("\nInitializing providers...")

    embedding_provider = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        llm_provider = OpenAILLM(model="gpt-4o-mini")
    except ValueError:
        print("Warning: OPENAI_API_KEY not set. Using search-only mode.")
        llm_provider = None

    vector_store = ChromaStore(
        collection_name="openmetadata_demo",
        persist_directory="./chroma_openmetadata"
    )

    if llm_provider:
        # Create agent
        agent = DataDiscoveryAgent(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_store=vector_store
        )

        # Index from OpenMetadata
        try:
            count = agent.index_from_openmetadata(
                host=os.getenv("OPENMETADATA_HOST", "http://localhost:8585"),
                jwt_token=os.getenv("OPENMETADATA_JWT_TOKEN"),
                limit=50  # Limit for demo
            )

            if count > 0:
                print(f"\n✓ Indexed {count} tables from OpenMetadata")

                # Run queries
                queries = [
                    "Where are customer data stored?",
                    "Tables with financial transactions",
                    "Show me all tables with PII data"
                ]

                for query in queries:
                    print(f"\n{'─' * 50}")
                    print(f"Query: {query}")
                    print('─' * 50)

                    result = agent.discover(query, top_k=3)

                    print(f"\nAnswer: {result.answer[:300]}...")
                    print(f"\nTables: {len(result.validated_tables)}")
                    print(f"Confidence: {result.confidence:.1%}")
                    print(f"Latency: {result.latency_ms}ms")

        except Exception as e:
            print(f"Error connecting to OpenMetadata: {e}")
            print("Continuing with demo data...")

    else:
        print("Skipping agent demo (no LLM provider)")


def example_factory_method():
    """Using the factory method for quick setup"""
    print("\n" + "=" * 60)
    print("Example 4: Factory Method (Quick Setup)")
    print("=" * 60)

    from rag_discovery import DataDiscoveryAgent
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore

    try:
        # One-liner to create agent with OpenMetadata data
        agent = DataDiscoveryAgent.from_openmetadata(
            embedding_provider=SentenceTransformerEmbeddings(),
            llm_provider=OpenAILLM(model="gpt-4o-mini"),
            vector_store=ChromaStore(collection_name="quick_setup"),
            openmetadata_host=os.getenv("OPENMETADATA_HOST", "http://localhost:8585"),
            openmetadata_token=os.getenv("OPENMETADATA_JWT_TOKEN"),
            limit=20
        )

        print("✓ Agent created with OpenMetadata data")

        # Use it
        result = agent.ask("What tables contain user information?")
        print(f"\nAnswer: {result['answer']}")

    except ValueError as e:
        print(f"Skipping: {e}")
    except Exception as e:
        print(f"Error: {e}")


def example_filtered_fetch():
    """Fetch tables with filters"""
    print("\n" + "=" * 60)
    print("Example 5: Filtered Fetch")
    print("=" * 60)

    from rag_discovery.connectors import OpenMetadataConnector, OpenMetadataConfig

    config = OpenMetadataConfig.from_env()

    try:
        connector = OpenMetadataConnector(config)

        # Get services first
        services = connector.get_services()

        if services:
            print(f"Available services: {[s['name'] for s in services]}")

            # Fetch from specific service
            service_name = services[0]['name']
            print(f"\nFetching tables from service: {service_name}")

            tables = connector.get_tables(
                service=service_name,
                limit=10
            )

            print(f"Found {len(tables)} tables")
            for t in tables[:5]:
                print(f"  - {t.name}: {len(t.columns)} columns")

    except Exception as e:
        print(f"Error: {e}")


def example_get_specific_table():
    """Fetch a specific table by FQN"""
    print("\n" + "=" * 60)
    print("Example 6: Get Specific Table")
    print("=" * 60)

    from rag_discovery.connectors import OpenMetadataConnector, OpenMetadataConfig

    config = OpenMetadataConfig.from_env()

    try:
        connector = OpenMetadataConnector(config)

        # Get some tables first to find a FQN
        tables = connector.get_tables(limit=1)

        if tables:
            fqn = tables[0].fully_qualified_name
            print(f"Fetching table: {fqn}")

            table = connector.get_table_by_fqn(fqn)

            if table:
                print(f"\nTable: {table.name}")
                print(f"FQN: {table.fully_qualified_name}")
                print(f"Description: {table.description or 'N/A'}")
                print(f"Service: {table.service} ({table.service_type})")
                print(f"Table Type: {table.table_type}")

                print(f"\nColumns ({len(table.columns)}):")
                for col in table.columns[:5]:
                    print(f"  - {col['name']} ({col['type']})")
                    if col.get('description'):
                        print(f"    {col['description'][:50]}...")

                if table.tags:
                    print(f"\nTags: {', '.join(table.tags)}")

                if table.owners:
                    print(f"Owners: {', '.join(table.owners)}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("OpenMetadata Integration Examples")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  OPENMETADATA_HOST: {os.getenv('OPENMETADATA_HOST', 'http://localhost:8585')}")
    print(f"  OPENMETADATA_JWT_TOKEN: {'Set' if os.getenv('OPENMETADATA_JWT_TOKEN') else 'Not set'}")
    print(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print()

    # Run examples
    example_basic_usage()
    # example_export_metadata()
    # example_with_data_discovery_agent()
    # example_factory_method()
    # example_filtered_fetch()
    # example_get_specific_table()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print()
    print("To run more examples, uncomment them in main()")
    print("=" * 60)


if __name__ == "__main__":
    main()
