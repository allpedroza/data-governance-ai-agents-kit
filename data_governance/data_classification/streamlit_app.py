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
Streamlit UI for Data Classification Agent

Standalone interface for data classification and sensitivity detection.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_classification.agent import DataClassificationAgent, ClassificationReport

st.set_page_config(
    page_title="Data Classification Agent",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Data Classification Agent")
st.markdown(
    "Classifique dados automaticamente por níveis de sensibilidade: "
    "PII (dados pessoais), PHI (dados de saúde), PCI (dados de pagamento) e dados financeiros."
)


@st.cache_resource
def get_agent():
    """Initialize the classification agent."""
    return DataClassificationAgent()


agent = get_agent()

# Sidebar configuration
with st.sidebar:
    st.header("Configurações")

    sample_size = st.number_input(
        "Tamanho da amostra",
        min_value=100,
        max_value=100000,
        value=1000,
        help="Número de linhas para análise"
    )

    st.divider()

    st.markdown("### Padrões customizados")
    custom_pattern_name = st.text_input("Nome do padrão", placeholder="ex: protocolo_interno")
    custom_pattern_regex = st.text_input("Regex", placeholder=r"ex: PROT-\d{8}")

    if st.button("Adicionar padrão") and custom_pattern_name and custom_pattern_regex:
        agent.add_custom_pattern(custom_pattern_name, custom_pattern_regex)
        st.success(f"Padrão '{custom_pattern_name}' adicionado!")

    st.divider()

    stats = agent.get_statistics()
    st.caption(f"📊 {stats['pii_patterns']} padrões PII")
    st.caption(f"📊 {stats['phi_patterns']} padrões PHI")
    st.caption(f"📊 {stats['financial_patterns']} padrões financeiros")
    st.caption(f"📊 {stats['custom_patterns']} padrões customizados")

# Main content
st.markdown("### Classificar Dados")

file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"])

uploaded_file = st.file_uploader(
    f"Selecione o arquivo {file_type}",
    type=["csv"] if file_type == "CSV" else ["parquet"]
)

if file_type == "CSV":
    col1, col2 = st.columns(2)
    with col1:
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"])
    with col2:
        separator = st.selectbox("Separador", [",", ";", "|", "\t"])

if uploaded_file and st.button("🔍 Classificar Dados", type="primary"):
    with st.spinner("Analisando dados e detectando padrões sensíveis..."):
        suffix = ".csv" if file_type == "CSV" else ".parquet"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.read())
            temp_path = f.name

        try:
            if file_type == "CSV":
                report = agent.classify_from_csv(
                    temp_path,
                    encoding=encoding,
                    separator=separator,
                    sample_size=sample_size
                )
            else:
                report = agent.classify_from_parquet(temp_path, sample_size=sample_size)

            st.session_state.classification_report = report
            st.success("✅ Classificação concluída!")

        except Exception as e:
            st.error(f"Erro ao classificar: {e}")
        finally:
            os.unlink(temp_path)

# Display results
if "classification_report" in st.session_state:
    report = st.session_state.classification_report
    st.divider()

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sensitivity_colors = {
            "public": "🟢",
            "internal": "🔵",
            "confidential": "🟡",
            "restricted": "🔴"
        }
        icon = sensitivity_colors.get(report.overall_sensitivity, "⚪")
        st.metric("Sensibilidade", f"{icon} {report.overall_sensitivity.upper()}")

    with col2:
        st.metric("Colunas Analisadas", report.columns_analyzed)

    with col3:
        st.metric("Alto Risco", report.high_risk_count)

    with col4:
        st.metric("Linhas", f"{report.row_count:,}")

    # Categories found
    st.markdown("### Categorias Detectadas")

    cat_cols = st.columns(4)
    with cat_cols[0]:
        if report.pii_columns:
            st.error(f"**PII**: {len(report.pii_columns)} colunas")
            for col in report.pii_columns[:5]:
                st.caption(f"  • {col}")
        else:
            st.success("**PII**: Não detectado")

    with cat_cols[1]:
        if report.phi_columns:
            st.error(f"**PHI**: {len(report.phi_columns)} colunas")
            for col in report.phi_columns[:5]:
                st.caption(f"  • {col}")
        else:
            st.success("**PHI**: Não detectado")

    with cat_cols[2]:
        if report.pci_columns:
            st.error(f"**PCI**: {len(report.pci_columns)} colunas")
            for col in report.pci_columns[:5]:
                st.caption(f"  • {col}")
        else:
            st.success("**PCI**: Não detectado")

    with cat_cols[3]:
        if report.financial_columns:
            st.warning(f"**Financeiro**: {len(report.financial_columns)} colunas")
            for col in report.financial_columns[:5]:
                st.caption(f"  • {col}")
        else:
            st.success("**Financeiro**: Não detectado")

    # Column details
    st.markdown("### Detalhes por Coluna")

    columns_data = []
    for col in report.columns:
        sensitivity_icon = {
            "public": "🟢",
            "internal": "🔵",
            "confidential": "🟡",
            "restricted": "🔴"
        }.get(col.sensitivity_level, "⚪")

        columns_data.append({
            "Coluna": col.name,
            "Sensibilidade": f"{sensitivity_icon} {col.sensitivity_level}",
            "Categorias": ", ".join(col.categories) if col.categories else "-",
            "Tipo PII": col.pii_type or "-",
            "Confiança": f"{col.confidence:.0%}",
            "Padrões": ", ".join(col.detected_patterns[:2]) if col.detected_patterns else "-"
        })

    st.dataframe(columns_data, use_container_width=True, hide_index=True)

    # Compliance flags
    if report.compliance_flags:
        st.markdown("### Conformidade Regulatória")
        for flag in report.compliance_flags:
            st.info(f"📋 {flag}")

    # Recommendations
    if report.recommendations:
        st.markdown("### Recomendações")
        for rec in report.recommendations:
            st.warning(f"💡 {rec}")

    # Export
    st.markdown("### Exportar Relatório")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📥 JSON",
            report.to_json(),
            file_name=f"{report.source_name}_classification.json",
            mime="application/json"
        )

    with col2:
        st.download_button(
            "📥 Markdown",
            report.to_markdown(),
            file_name=f"{report.source_name}_classification.md",
            mime="text/markdown"
        )
