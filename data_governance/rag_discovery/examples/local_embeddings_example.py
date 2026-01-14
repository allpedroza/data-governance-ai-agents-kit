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
Example: Using local embeddings with SentenceTransformers
No API costs, fast inference, works offline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.vectorstore import FAISSStore, ChromaStore
from rag_discovery.retrieval import HybridRetriever, HybridRetrieverConfig


def main():
    print("=" * 60)
    print("Local Embeddings Example (No API Cost)")
    print("=" * 60)
    print()

    # =========================================
    # 1. Initialize local embeddings
    # =========================================
    print("Loading SentenceTransformer model...")
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # 384 dimensions, fast
    )
    print(f"Model loaded: {embeddings.model_name}")
    print(f"Dimension: {embeddings.dimension}")

    # =========================================
    # 2. Test embedding generation
    # =========================================
    print("\nGenerating embeddings...")

    texts = [
        "Tabela de clientes com informações cadastrais",
        "Histórico de pedidos e transações de vendas",
        "Logs de atividade do usuário no sistema",
        "Métricas de consumo de dados em tempo real"
    ]

    # Single embedding
    single = embeddings.embed(texts[0])
    print(f"Single embedding: {len(single.vector)} dimensions")

    # Batch embedding (more efficient)
    batch = embeddings.embed_batch(texts)
    print(f"Batch embeddings: {len(batch)} texts processed")

    # =========================================
    # 3. Use with FAISS (local vector store)
    # =========================================
    print("\n" + "-" * 40)
    print("Using with FAISS (local vector store)")
    print("-" * 40)

    faiss_store = FAISSStore(
        collection_name="local_demo",
        persist_directory="./faiss_local_demo",
        dimension=embeddings.dimension
    )

    # Index documents
    ids = [f"doc_{i}" for i in range(len(texts))]
    vectors = [e.vector for e in batch]
    metadatas = [{"text": t, "index": i} for i, t in enumerate(texts)]

    faiss_store.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Indexed {faiss_store.count()} documents in FAISS")

    # Search
    query = "informações de clientes"
    query_embedding = embeddings.embed(query)

    results = faiss_store.search(
        embedding=query_embedding.vector,
        n_results=2
    )

    print(f"\nQuery: '{query}'")
    print("Results:")
    for i, (doc_id, doc, dist) in enumerate(zip(
        results.ids[0], results.documents[0], results.distances[0]
    )):
        print(f"  {i+1}. [{doc_id}] Score: {dist:.3f}")
        print(f"     {doc[:50]}...")

    # =========================================
    # 4. Use with Hybrid Retriever
    # =========================================
    print("\n" + "-" * 40)
    print("Using with Hybrid Retriever (Dartboard)")
    print("-" * 40)

    # Use ChromaDB for hybrid retriever
    chroma_store = ChromaStore(
        collection_name="hybrid_local_demo",
        persist_directory="./chroma_local_demo"
    )

    config = HybridRetrieverConfig(
        alpha=0.7,   # Semantic
        beta=0.2,    # Lexical
        gamma=0.1    # Importance
    )

    retriever = HybridRetriever(
        embedding_provider=embeddings,
        vector_store=chroma_store,
        config=config
    )

    # Index with richer documents
    documents = [
        {
            "id": "customers",
            "text": "Tabela customers: dados cadastrais de clientes incluindo nome, email, telefone, endereço. Contém informações PII sensíveis.",
            "metadata": {"type": "customer", "pii": True}
        },
        {
            "id": "orders",
            "text": "Tabela orders: histórico de pedidos com data, valor, status, forma de pagamento. Relacionada com customers.",
            "metadata": {"type": "transaction", "pii": False}
        },
        {
            "id": "activity_logs",
            "text": "Tabela activity_logs: logs de atividade do usuário no app e website. Eventos, timestamps, URLs visitadas.",
            "metadata": {"type": "logs", "pii": False}
        },
        {
            "id": "data_usage",
            "text": "Tabela data_usage: consumo de dados móveis por cliente. Download, upload, tipo de rede 3G/4G/5G.",
            "metadata": {"type": "usage", "pii": False}
        }
    ]

    retriever.index_documents(documents, show_progress=True)

    # Hybrid search
    queries = [
        "dados pessoais de clientes",
        "consumo de dados 4G",
        "histórico de compras"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query, top_k=2)

        for r in results:
            print(f"  - {r.chunk_id}: {r.combined_score:.3f}")
            print(f"    (sem={r.semantic_score:.2f}, lex={r.lexical_score:.2f}, imp={r.importance_score:.2f})")

    # =========================================
    # 5. Performance comparison
    # =========================================
    print("\n" + "-" * 40)
    print("Performance (Local vs API)")
    print("-" * 40)

    import time

    # Benchmark local embeddings
    test_texts = ["Test sentence " + str(i) for i in range(100)]

    start = time.time()
    _ = embeddings.embed_batch(test_texts)
    local_time = time.time() - start

    print(f"Local embeddings (100 texts): {local_time*1000:.0f}ms")
    print(f"Average per text: {local_time*10:.1f}ms")
    print("\nNote: API embeddings would take ~200-500ms per request")
    print("Local embeddings are 10-50x faster!")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
