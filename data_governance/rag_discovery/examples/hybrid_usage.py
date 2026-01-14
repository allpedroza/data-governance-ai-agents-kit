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
Example: Hybrid Data Discovery Agent v2
Demonstrates the new Dartboard Ranking with pluggable providers
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_discovery import DataDiscoveryAgent, TableMetadata
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore


def create_sample_tables():
    """Create sample table metadata for testing"""
    return [
        TableMetadata(
            name="customers",
            database="production",
            schema="public",
            description="Tabela principal de clientes contendo informações cadastrais e de contato",
            columns=[
                {"name": "customer_id", "type": "bigint", "description": "Identificador único do cliente"},
                {"name": "name", "type": "varchar(200)", "description": "Nome completo do cliente"},
                {"name": "email", "type": "varchar(100)", "description": "Email de contato"},
                {"name": "phone", "type": "varchar(20)", "description": "Telefone"},
                {"name": "created_at", "type": "timestamp", "description": "Data de cadastro"},
                {"name": "country", "type": "varchar(50)", "description": "País do cliente"}
            ],
            row_count=1500000,
            owner="data-team",
            tags=["pii", "critical", "customer-data"],
            location="s3://data-lake/production/customers/",
            format="delta",
            partition_keys=["country", "created_at"]
        ),
        TableMetadata(
            name="orders",
            database="production",
            schema="public",
            description="Histórico de pedidos realizados pelos clientes",
            columns=[
                {"name": "order_id", "type": "bigint", "description": "ID único do pedido"},
                {"name": "customer_id", "type": "bigint", "description": "FK para customers"},
                {"name": "order_date", "type": "timestamp", "description": "Data do pedido"},
                {"name": "total_amount", "type": "decimal(10,2)", "description": "Valor total"},
                {"name": "status", "type": "varchar(50)", "description": "Status do pedido"},
                {"name": "payment_method", "type": "varchar(50)", "description": "Forma de pagamento"}
            ],
            row_count=5000000,
            owner="data-team",
            tags=["transactional", "critical"],
            location="s3://data-lake/production/orders/",
            format="parquet",
            partition_keys=["order_date"]
        ),
        TableMetadata(
            name="product_catalog",
            database="production",
            schema="public",
            description="Catálogo de produtos disponíveis para venda",
            columns=[
                {"name": "product_id", "type": "bigint", "description": "ID do produto"},
                {"name": "name", "type": "varchar(200)", "description": "Nome do produto"},
                {"name": "category", "type": "varchar(100)", "description": "Categoria"},
                {"name": "price", "type": "decimal(10,2)", "description": "Preço unitário"},
                {"name": "stock_quantity", "type": "int", "description": "Quantidade em estoque"}
            ],
            row_count=50000,
            owner="product-team",
            tags=["product", "catalog"],
            location="s3://data-lake/production/products/",
            format="delta"
        ),
        TableMetadata(
            name="user_activity_logs",
            database="analytics",
            schema="logs",
            description="Logs de atividade dos usuários no aplicativo e website",
            columns=[
                {"name": "event_id", "type": "uuid", "description": "ID do evento"},
                {"name": "user_id", "type": "bigint", "description": "ID do usuário"},
                {"name": "event_type", "type": "varchar(100)", "description": "Tipo de evento"},
                {"name": "timestamp", "type": "timestamp", "description": "Momento do evento"},
                {"name": "page_url", "type": "varchar(500)", "description": "URL acessada"},
                {"name": "user_agent", "type": "varchar(500)", "description": "Browser/app do usuário"}
            ],
            row_count=100000000,
            owner="analytics-team",
            tags=["logs", "analytics", "behavioral"],
            location="s3://data-lake/analytics/user_activity/",
            format="parquet",
            partition_keys=["timestamp"]
        ),
        TableMetadata(
            name="financial_transactions",
            database="finance",
            schema="transactions",
            description="Transações financeiras detalhadas incluindo pagamentos e reembolsos",
            columns=[
                {"name": "transaction_id", "type": "uuid", "description": "ID da transação"},
                {"name": "order_id", "type": "bigint", "description": "ID do pedido relacionado"},
                {"name": "amount", "type": "decimal(15,2)", "description": "Valor da transação"},
                {"name": "currency", "type": "char(3)", "description": "Moeda (USD, BRL, etc)"},
                {"name": "transaction_date", "type": "timestamp", "description": "Data da transação"},
                {"name": "status", "type": "varchar(50)", "description": "Status (completed, pending, failed)"}
            ],
            row_count=8000000,
            owner="finance-team",
            tags=["pii", "financial", "critical", "audit"],
            location="s3://data-lake/finance/transactions/",
            format="delta",
            partition_keys=["transaction_date", "currency"]
        ),
        TableMetadata(
            name="data_consumption_hourly",
            database="telecom",
            schema="usage",
            description="Consumo de dados por hora por cliente em redes 3G, 4G e 5G",
            columns=[
                {"name": "record_id", "type": "bigint", "description": "ID do registro"},
                {"name": "customer_id", "type": "bigint", "description": "ID do cliente"},
                {"name": "hour_timestamp", "type": "timestamp", "description": "Hora do consumo"},
                {"name": "download_mb", "type": "decimal(10,2)", "description": "Download em MB"},
                {"name": "upload_mb", "type": "decimal(10,2)", "description": "Upload em MB"},
                {"name": "network_type", "type": "varchar(10)", "description": "Tipo de rede (3G/4G/5G)"},
                {"name": "cell_id", "type": "varchar(50)", "description": "ID da célula"}
            ],
            row_count=500000000,
            owner="network-team",
            tags=["usage", "telecom", "high-volume"],
            location="s3://data-lake/telecom/consumption/",
            format="parquet",
            partition_keys=["hour_timestamp", "network_type"]
        )
    ]


def main():
    print("=" * 70)
    print("Data Discovery Agent v2 - Hybrid Example")
    print("=" * 70)
    print()

    # =========================================
    # 1. Initialize providers
    # =========================================
    print("Step 1: Initializing providers...")

    # Local embeddings (no API cost!)
    embedding_provider = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # OpenAI LLM for response generation
    # Note: Requires OPENAI_API_KEY environment variable
    try:
        llm_provider = OpenAILLM(
            model="gpt-4o-mini",
            default_temperature=0.0
        )
    except ValueError as e:
        print(f"Warning: {e}")
        print("Running without LLM (search only mode)")
        llm_provider = None

    # ChromaDB for vector storage
    vector_store = ChromaStore(
        collection_name="hybrid_demo",
        persist_directory="./chroma_hybrid_demo"
    )

    print()

    # =========================================
    # 2. Initialize agent
    # =========================================
    print("Step 2: Initializing Data Discovery Agent...")

    if llm_provider:
        agent = DataDiscoveryAgent(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_store=vector_store,
            catalog_source=None,  # No catalog validation for demo
            alpha=0.7,   # Semantic weight
            beta=0.2,    # Lexical weight
            gamma=0.1    # Importance weight
        )
    else:
        # For demo without LLM, we can still use the retriever directly
        from rag_discovery.retrieval import HybridRetriever, HybridRetrieverConfig

        config = HybridRetrieverConfig(alpha=0.7, beta=0.2, gamma=0.1)
        retriever = HybridRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            config=config
        )
        agent = None

    print()

    # =========================================
    # 3. Index sample tables
    # =========================================
    print("Step 3: Indexing sample tables...")

    tables = create_sample_tables()

    if agent:
        agent.index_metadata(tables)
    else:
        # Direct indexing with retriever
        documents = []
        for t in tables:
            full_name = f"{t.database}.{t.schema}.{t.name}"
            documents.append({
                "id": full_name,
                "text": t.to_text(),
                "metadata": t.to_dict()
            })
        retriever.index_documents(documents)

    print()

    # =========================================
    # 4. Run queries
    # =========================================
    print("=" * 70)
    print("HYBRID SEARCH EXAMPLES (Dartboard Ranking)")
    print("=" * 70)

    queries = [
        "Onde estão os dados de clientes?",
        "Tabelas com informações financeiras e transações",
        "Consumo de dados em redes 4G e 5G",
        "Logs de atividade do usuário",
        "Dados para análise de vendas por produto"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print('─' * 70)

        if agent:
            # Full discovery with LLM
            result = agent.discover(query, top_k=3, validate=False)

            print(f"\nAnswer:\n{result.answer}")
            print(f"\nTables found: {len(result.validated_tables)}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Latency: {result.latency_ms}ms")

            # Show score breakdown
            print("\nScore breakdown:")
            for r in result.retrieval_results[:3]:
                print(f"  {r.chunk_id}:")
                print(f"    Combined: {r.combined_score:.3f}")
                print(f"    Semantic: {r.semantic_score:.3f}")
                print(f"    Lexical:  {r.lexical_score:.3f}")
                print(f"    Import.:  {r.importance_score:.3f}")

        else:
            # Search only (no LLM)
            results = retriever.retrieve(query, top_k=3)

            print("\nResults (search only mode):")
            for j, r in enumerate(results, 1):
                print(f"\n  {j}. {r.chunk_id}")
                print(f"     Combined: {r.combined_score:.3f}")
                print(f"     Semantic: {r.semantic_score:.3f}")
                print(f"     Lexical:  {r.lexical_score:.3f}")

    # =========================================
    # 5. Show statistics
    # =========================================
    print("\n\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    if agent:
        stats = agent.get_statistics()
        print(f"\nRetriever stats:")
        print(f"  Total documents: {stats['retriever']['total_documents']}")
        print(f"  Vocabulary size: {stats['retriever']['lexical_vocabulary_size']}")
        print(f"  Weights: α={stats['retriever']['weights']['alpha']}, "
              f"β={stats['retriever']['weights']['beta']}, "
              f"γ={stats['retriever']['weights']['gamma']}")

        print(f"\nSession stats:")
        session = stats['logs']
        print(f"  Total queries: {session.get('total_queries', 0)}")
        if session.get('total_queries', 0) > 0:
            print(f"  Avg latency: {session.get('avg_latency_ms', 0):.0f}ms")
    else:
        stats = retriever.get_statistics()
        print(f"Total documents: {stats['total_documents']}")
        print(f"Vocabulary size: {stats['lexical_vocabulary_size']}")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
