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
Basic usage example for Metadata Enrichment Agent

This example demonstrates how to:
1. Initialize the agent with providers
2. Index architecture standards
3. Enrich metadata from a CSV file
4. Export results to different formats
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from metadata_enrichment.agent import MetadataEnrichmentAgent
from metadata_enrichment.standards import StandardDocument
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore


def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    # Initialize providers
    print("Initializing providers...")
    embedding_provider = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Fast, local embeddings
    )

    llm_provider = OpenAILLM(
        model="gpt-4o-mini"  # Cost-effective for metadata generation
    )

    vector_store = ChromaStore(
        collection_name="metadata_standards",
        persist_directory="./chroma_standards"
    )

    # Create agent
    agent = MetadataEnrichmentAgent(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        vector_store=vector_store,
        language="pt-br"
    )

    # Index some sample standards
    print("\nIndexing sample standards...")
    sample_standards = [
        StandardDocument(
            id="naming-001",
            title="Convenção de Nomenclatura de Tabelas",
            content="""
            Tabelas devem seguir o padrão: dominio_entidade_tipo
            Exemplos:
            - customer_profile_dim (dimensão de perfil de cliente)
            - sales_transaction_fact (fato de transação de vendas)
            - product_inventory_snapshot (snapshot de inventário)

            Prefixos por tipo:
            - dim_ ou _dim: Dimensões
            - fact_ ou _fact: Fatos
            - stg_: Staging
            - raw_: Dados brutos
            """,
            category="naming_convention",
            tags=["nomenclatura", "tabelas", "padrão"]
        ),
        StandardDocument(
            id="pii-001",
            title="Classificação de Dados Pessoais (PII)",
            content="""
            Dados pessoais que exigem proteção especial:

            Alta sensibilidade (restricted):
            - CPF, CNPJ, RG
            - Dados de saúde
            - Dados financeiros pessoais
            - Senhas e credenciais

            Média sensibilidade (confidential):
            - Nome completo
            - Email pessoal
            - Telefone
            - Endereço

            Identificadores (internal):
            - IDs de cliente
            - IDs de conta
            """,
            category="data_classification",
            tags=["pii", "lgpd", "classificação", "dados pessoais"]
        ),
        StandardDocument(
            id="glossary-001",
            title="Glossário de Negócios - Clientes",
            content="""
            Termos relacionados a clientes:

            - Cliente (Customer): Pessoa física ou jurídica que adquire produtos/serviços
            - Prospect: Potencial cliente em processo de qualificação
            - Lead: Contato inicial que demonstrou interesse
            - Churn: Cliente que cancelou ou deixou de comprar
            - LTV (Lifetime Value): Valor total que um cliente gera
            - CAC (Customer Acquisition Cost): Custo de aquisição de cliente
            """,
            category="glossary",
            tags=["cliente", "customer", "glossário", "negócio"]
        )
    ]

    count = agent.index_standards(sample_standards)
    print(f"Indexed {count} standards")

    # Create sample CSV for testing
    sample_csv = """customer_id,nome_completo,email,cpf,data_nascimento,telefone,endereco,cidade,estado,valor_total_compras
1001,João Silva,joao.silva@email.com,123.456.789-00,1985-03-15,11987654321,Rua das Flores 123,São Paulo,SP,15750.50
1002,Maria Santos,maria@empresa.com,987.654.321-00,1990-07-22,21999887766,Av Brasil 456,Rio de Janeiro,RJ,8320.00
1003,Pedro Oliveira,pedro.oliveira@gmail.com,456.789.123-00,1978-11-30,31988776655,Rua Central 789,Belo Horizonte,MG,22100.75
"""

    # Save sample CSV
    sample_file = Path("./sample_customers.csv")
    sample_file.write_text(sample_csv)

    # Enrich metadata
    print("\nEnriching metadata from CSV...")
    result = agent.enrich_from_csv(
        file_path=str(sample_file),
        sample_size=100,
        additional_context="Tabela de clientes do sistema de CRM"
    )

    # Print results
    print("\n" + "=" * 60)
    print("ENRICHMENT RESULTS")
    print("=" * 60)

    print(f"\nTable: {result.table_name}")
    print(f"Business Name: {result.business_name}")
    print(f"Domain: {result.domain}")
    print(f"Classification: {result.classification}")
    print(f"Has PII: {result.has_pii}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Processing Time: {result.processing_time_ms}ms")

    print(f"\nDescription:\n{result.description}")

    print(f"\nTags: {', '.join(result.tags)}")

    if result.pii_columns:
        print(f"\n⚠️ PII Columns: {', '.join(result.pii_columns)}")

    print("\nColumns:")
    for col in result.columns:
        pii_marker = " [PII]" if col.is_pii else ""
        print(f"  - {col.name}{pii_marker}: {col.description[:60]}...")

    # Export to different formats
    print("\nExporting results...")

    # JSON
    json_path = Path("./sample_customers_metadata.json")
    json_path.write_text(result.to_json())
    print(f"  - JSON: {json_path}")

    # Markdown
    md_path = Path("./sample_customers_metadata.md")
    md_path.write_text(result.to_markdown())
    print(f"  - Markdown: {md_path}")

    print("\n✓ Done!")

    # Cleanup
    sample_file.unlink()


if __name__ == "__main__":
    main()
