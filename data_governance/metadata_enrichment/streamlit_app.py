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
Streamlit UI for Metadata Enrichment Agent
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import streamlit as st

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def init_agent():
    """Initialize the Metadata Enrichment Agent"""
    from metadata_enrichment.agent import MetadataEnrichmentAgent
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY não configurada. Configure a variável de ambiente.")
        st.stop()

    # Initialize providers
    embedding_provider = SentenceTransformerEmbeddings()
    llm_provider = OpenAILLM(model="gpt-4o-mini")
    vector_store = ChromaStore(
        collection_name="metadata_standards",
        persist_directory="./chroma_standards"
    )

    # Create agent
    agent = MetadataEnrichmentAgent(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        vector_store=vector_store,
        standards_persist_dir="./standards_index"
    )

    return agent


def render_enrichment_result(result):
    """Render enrichment result in Streamlit"""
    from metadata_enrichment.agent import EnrichmentResult

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(f"📊 {result.business_name}")
        st.caption(f"`{result.table_name}`")
    with col2:
        if result.has_pii:
            st.error("⚠️ Contém PII")
        else:
            st.success("✓ Sem PII")
    with col3:
        st.metric("Confiança", f"{result.confidence:.0%}")

    # Description
    st.markdown("### Descrição")
    st.write(result.description)

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Domínio:** {result.domain}")
    with col2:
        st.markdown(f"**Classificação:** {result.classification}")
    with col3:
        st.markdown(f"**Linhas:** {result.row_count:,}")
    with col4:
        st.markdown(f"**Tempo:** {result.processing_time_ms}ms")

    # Tags
    if result.tags:
        st.markdown("**Tags:** " + " ".join([f"`{tag}`" for tag in result.tags]))

    # Columns table
    st.markdown("### Colunas")

    columns_data = []
    for col in result.columns:
        pii_marker = "⚠️" if col.is_pii else ""
        columns_data.append({
            "Coluna": f"{col.name} {pii_marker}",
            "Tipo": col.original_type,
            "Descrição": col.description[:80] + "..." if len(col.description) > 80 else col.description,
            "Classificação": col.classification,
            "Tags": ", ".join(col.tags[:3]) if col.tags else "-"
        })

    st.dataframe(columns_data, use_container_width=True)

    # PII Warning
    if result.pii_columns:
        st.warning(f"⚠️ Colunas com dados pessoais (PII): {', '.join(result.pii_columns)}")

    # Export options
    st.markdown("### Exportar")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "📥 JSON",
            result.to_json(),
            file_name=f"{result.table_name}_metadata.json",
            mime="application/json"
        )

    with col2:
        st.download_button(
            "📥 Markdown",
            result.to_markdown(),
            file_name=f"{result.table_name}_metadata.md",
            mime="text/markdown"
        )


def main():
    st.set_page_config(
        page_title="Metadata Enrichment Agent",
        page_icon="🏷️",
        layout="wide"
    )

    st.title("🏷️ Metadata Enrichment Agent")
    st.markdown("Gere descrições, tags e classificações automaticamente para suas tabelas de dados.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuração")

        # Standards management
        st.subheader("📚 Normativos")

        standards_file = st.file_uploader(
            "Carregar normativos (JSON)",
            type=["json"],
            help="Arquivo JSON com padrões de nomenclatura, glossário, etc."
        )

        if standards_file:
            if st.button("Indexar Normativos"):
                with st.spinner("Indexando..."):
                    agent = init_agent()
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
                        f.write(standards_file.read())
                        temp_path = f.name

                    count = agent.index_standards_from_json(temp_path)
                    st.success(f"✓ {count} documentos indexados")
                    os.unlink(temp_path)

        st.divider()

        # Statistics
        st.subheader("📊 Estatísticas")
        if st.button("Atualizar"):
            agent = init_agent()
            stats = agent.get_statistics()
            st.json(stats)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Arquivo", "🗄️ SQL", "📋 Batch"])

    with tab1:
        st.subheader("Enriquecer arquivo")

        file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"])

        uploaded_file = st.file_uploader(
            f"Selecione o arquivo {file_type}",
            type=["csv"] if file_type == "CSV" else ["parquet"]
        )

        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.number_input("Tamanho da amostra", min_value=10, max_value=1000, value=100)
        with col2:
            if file_type == "CSV":
                encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"])
                separator = st.selectbox("Separador", [",", ";", "|", "\t"])

        additional_context = st.text_area(
            "Contexto adicional (opcional)",
            placeholder="Informações adicionais sobre a tabela, domínio de negócio, etc."
        )

        if uploaded_file and st.button("🚀 Enriquecer Metadados", type="primary"):
            with st.spinner("Analisando dados e gerando metadados..."):
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type.lower()}") as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name

                try:
                    agent = init_agent()

                    if file_type == "CSV":
                        result = agent.enrich_from_csv(
                            temp_path,
                            sample_size=sample_size,
                            encoding=encoding,
                            separator=separator,
                            additional_context=additional_context if additional_context else None
                        )
                    else:
                        result = agent.enrich_from_parquet(
                            temp_path,
                            sample_size=sample_size,
                            additional_context=additional_context if additional_context else None
                        )

                    st.success("✓ Metadados gerados com sucesso!")
                    st.divider()
                    render_enrichment_result(result)

                except Exception as e:
                    st.error(f"Erro: {str(e)}")
                finally:
                    os.unlink(temp_path)

    with tab2:
        st.subheader("Enriquecer tabela SQL")

        connection_string = st.text_input(
            "Connection String",
            placeholder="postgresql://user:pass@host:5432/database",
            type="password"
        )

        col1, col2 = st.columns(2)
        with col1:
            table_name = st.text_input("Nome da tabela", placeholder="schema.table_name")
        with col2:
            sql_sample_size = st.number_input("Amostra", min_value=10, max_value=1000, value=100, key="sql_sample")

        sql_context = st.text_area(
            "Contexto adicional (opcional)",
            placeholder="Informações sobre a tabela...",
            key="sql_context"
        )

        if connection_string and table_name and st.button("🚀 Enriquecer Tabela SQL", type="primary"):
            with st.spinner("Conectando e analisando..."):
                try:
                    agent = init_agent()

                    # Parse schema and table
                    if "." in table_name:
                        schema, tbl = table_name.rsplit(".", 1)
                    else:
                        schema, tbl = None, table_name

                    result = agent.enrich_from_sql(
                        table_name=tbl,
                        connection_string=connection_string,
                        schema=schema,
                        sample_size=sql_sample_size,
                        additional_context=sql_context if sql_context else None
                    )

                    st.success("✓ Metadados gerados com sucesso!")
                    st.divider()
                    render_enrichment_result(result)

                except Exception as e:
                    st.error(f"Erro: {str(e)}")

    with tab3:
        st.subheader("Processamento em lote")
        st.info("Faça upload de múltiplos arquivos para enriquecer em batch.")

        batch_files = st.file_uploader(
            "Selecione os arquivos",
            type=["csv", "parquet"],
            accept_multiple_files=True
        )

        if batch_files and st.button("🚀 Processar Batch", type="primary"):
            progress = st.progress(0)
            results = []

            agent = init_agent()

            for i, file in enumerate(batch_files):
                st.write(f"Processando: {file.name}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as f:
                    f.write(file.read())
                    temp_path = f.name

                try:
                    if file.name.endswith(".csv"):
                        result = agent.enrich_from_csv(temp_path)
                    else:
                        result = agent.enrich_from_parquet(temp_path)

                    results.append(result)
                    st.success(f"✓ {file.name}")

                except Exception as e:
                    st.error(f"✗ {file.name}: {str(e)}")
                finally:
                    os.unlink(temp_path)

                progress.progress((i + 1) / len(batch_files))

            if results:
                st.divider()
                st.subheader("Resultados")

                # Summary
                total_tables = len(results)
                pii_tables = sum(1 for r in results if r.has_pii)
                total_columns = sum(len(r.columns) for r in results)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tabelas", total_tables)
                with col2:
                    st.metric("Com PII", pii_tables)
                with col3:
                    st.metric("Colunas", total_columns)

                # Export catalog
                catalog_json = json.dumps({
                    "tables": [r.to_dict() for r in results]
                }, ensure_ascii=False, indent=2)

                st.download_button(
                    "📥 Baixar Catálogo Completo (JSON)",
                    catalog_json,
                    file_name="data_catalog.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
