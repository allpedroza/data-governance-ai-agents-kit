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
Streamlit App for Sensitive Data NER Agent

Interactive interface for detecting and anonymizing sensitive data in text.
"""

import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any

from .agent import (
    SensitiveDataNERAgent,
    NERResult,
    FilterPolicy,
    FilterAction,
    EntityCategory,
)
from .anonymizers import AnonymizationStrategy


def render_entity_badge(category: EntityCategory) -> str:
    """Render category as colored badge."""
    colors = {
        EntityCategory.PII: "#FF6B6B",
        EntityCategory.PHI: "#4ECDC4",
        EntityCategory.PCI: "#FFE66D",
        EntityCategory.FINANCIAL: "#95E1D3",
        EntityCategory.BUSINESS: "#DDA0DD",
    }
    color = colors.get(category, "#CCCCCC")
    return f'<span style="background-color:{color};padding:2px 8px;border-radius:4px;font-size:12px;">{category.value.upper()}</span>'


def highlight_entities(text: str, entities: List[Dict]) -> str:
    """Highlight detected entities in text."""
    if not entities:
        return text

    # Sort by position (reverse)
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    result = text
    for entity in sorted_entities:
        start, end = entity["start"], entity["end"]
        category = entity["category"]

        colors = {
            "pii": "#FFCCCC",
            "phi": "#CCFFFF",
            "pci": "#FFFFCC",
            "financial": "#CCFFCC",
            "business": "#E6CCFF",
        }
        color = colors.get(category, "#EEEEEE")

        highlighted = f'<mark style="background-color:{color};padding:2px;">{text[start:end]}</mark>'
        result = result[:start] + highlighted + result[end:]

    return result


def main():
    st.set_page_config(
        page_title="Sensitive Data NER",
        page_icon="🔒",
        layout="wide"
    )

    st.title("🔒 Sensitive Data NER Agent")
    st.markdown("""
    Detecção e anonimização de dados sensíveis para proteção de requisições a LLMs.

    **Categorias detectadas:** PII, PHI, PCI, Dados Financeiros, Dados Estratégicos
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configurações")

    # Filter Policy
    st.sidebar.subheader("Política de Filtro")

    pii_action = st.sidebar.selectbox(
        "Ação para PII",
        options=["anonymize", "block", "warn", "allow"],
        index=0
    )
    phi_action = st.sidebar.selectbox(
        "Ação para PHI",
        options=["block", "anonymize", "warn", "allow"],
        index=0
    )
    pci_action = st.sidebar.selectbox(
        "Ação para PCI",
        options=["block", "anonymize", "warn", "allow"],
        index=0
    )
    financial_action = st.sidebar.selectbox(
        "Ação para Financeiro",
        options=["anonymize", "block", "warn", "allow"],
        index=0
    )
    business_action = st.sidebar.selectbox(
        "Ação para Negócios",
        options=["block", "anonymize", "warn", "allow"],
        index=0
    )

    # Anonymization Strategy
    st.sidebar.subheader("Estratégia de Anonimização")
    anon_strategy = st.sidebar.selectbox(
        "Estratégia",
        options=["redact", "mask", "hash", "partial", "pseudonymize"],
        index=0,
        help="""
        - **redact**: Substitui por rótulo [PII], [CPF], etc.
        - **mask**: Substitui por asteriscos ***
        - **hash**: Substitui por hash determinístico
        - **partial**: Mascara parcialmente, mantém início/fim
        - **pseudonymize**: Substitui por dados falsos consistentes
        """
    )

    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Confiança Mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Entidades com confiança menor que este valor serão ignoradas"
    )

    strict_mode = st.sidebar.checkbox(
        "Modo Estrito",
        value=False,
        help="Requer validação de checksum quando disponível"
    )

    # Business terms
    st.sidebar.subheader("Termos de Negócio")
    business_terms_input = st.sidebar.text_area(
        "Termos sensíveis (um por linha)",
        placeholder="Projeto Confidencial\nAquisição XYZ\nParceria ABC",
        height=100
    )
    business_terms = [
        t.strip() for t in business_terms_input.split("\n")
        if t.strip()
    ]

    # Locales
    st.sidebar.subheader("Locais")
    locale_br = st.sidebar.checkbox("Brasil (BR)", value=True)
    locale_us = st.sidebar.checkbox("Estados Unidos (US)", value=True)
    locale_eu = st.sidebar.checkbox("Europa (EU)", value=True)

    locales = []
    if locale_br:
        locales.append("br")
    if locale_us:
        locales.append("us")
    if locale_eu:
        locales.extend(["es", "pt"])

    # Create policy
    policy = FilterPolicy(
        pii_action=FilterAction(pii_action),
        phi_action=FilterAction(phi_action),
        pci_action=FilterAction(pci_action),
        financial_action=FilterAction(financial_action),
        business_action=FilterAction(business_action),
        min_confidence=min_confidence,
        anonymization_strategy=AnonymizationStrategy(anon_strategy),
    )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Texto de Entrada")
        input_text = st.text_area(
            "Cole o texto para análise",
            height=300,
            placeholder="""Exemplo:
O cliente João da Silva, CPF 123.456.789-09, solicitou atualização
de seus dados. Email: joao.silva@email.com, telefone (11) 98765-4321.

Diagnóstico: CID-10 J45.0 - Asma predominantemente alérgica.
Cartão SUS: 123456789012345.

Pagamento via cartão 4532 1234 5678 9010, validade 12/25.
Conta bancária: AG 1234 CC 12345678-9.

Referente ao Projeto Confidencial para aquisição da empresa XYZ."""
        )

    # Create agent and analyze
    if input_text:
        try:
            agent = SensitiveDataNERAgent(
                business_terms=business_terms if business_terms else None,
                filter_policy=policy,
                strict_mode=strict_mode,
                locales=locales if locales else None,
            )

            result = agent.analyze(input_text, anonymize=True)

            with col2:
                st.subheader("🔍 Resultado da Análise")

                # Filter action badge
                action_colors = {
                    FilterAction.ALLOW: "🟢",
                    FilterAction.WARN: "🟡",
                    FilterAction.ANONYMIZE: "🟠",
                    FilterAction.BLOCK: "🔴",
                }
                st.markdown(f"### {action_colors[result.filter_action]} Ação: **{result.filter_action.value.upper()}**")

                if result.blocked_reason:
                    st.error(f"Motivo do bloqueio: {result.blocked_reason}")

                for warning in result.warnings:
                    st.warning(warning)

                # Metrics
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("Entidades Detectadas", result.statistics["total"])
                with met_col2:
                    st.metric("Score de Risco", f"{result.risk_score:.1%}")
                with met_col3:
                    st.metric("Tempo (ms)", f"{result.processing_time_ms:.1f}")

            # Tabs for detailed results
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Entidades",
                "🔐 Texto Anonimizado",
                "📈 Estatísticas",
                "📋 JSON"
            ])

            with tab1:
                if result.entities:
                    st.markdown("### Entidades Detectadas")

                    # Entity table
                    entity_data = []
                    for e in result.entities:
                        entity_data.append({
                            "Valor": e.value[:30] + "..." if len(e.value) > 30 else e.value,
                            "Tipo": e.entity_type,
                            "Categoria": e.category.value.upper(),
                            "Confiança": f"{e.confidence:.1%}",
                            "Validado": "✓" if e.is_validated else "✗",
                            "Linha": e.line_number,
                        })

                    df = pd.DataFrame(entity_data)
                    st.dataframe(df, use_container_width=True)

                    # Highlighted text
                    st.markdown("### Texto com Destaque")
                    entities_dict = [e.to_dict() for e in result.entities]
                    highlighted = highlight_entities(input_text, entities_dict)
                    st.markdown(f'<div style="white-space:pre-wrap;font-family:monospace;background:#f5f5f5;padding:15px;border-radius:5px;">{highlighted}</div>', unsafe_allow_html=True)
                else:
                    st.success("Nenhuma entidade sensível detectada!")

            with tab2:
                if result.anonymized_text:
                    st.markdown("### Texto Anonimizado")
                    st.code(result.anonymized_text, language=None)

                    # Copy button
                    st.download_button(
                        "📋 Baixar texto anonimizado",
                        result.anonymized_text,
                        file_name="texto_anonimizado.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("Nenhuma anonimização necessária")

            with tab3:
                st.markdown("### Estatísticas por Categoria")

                stats_data = {
                    "Categoria": ["PII", "PHI", "PCI", "Financeiro", "Negócios"],
                    "Quantidade": [
                        result.statistics.get("pii", 0),
                        result.statistics.get("phi", 0),
                        result.statistics.get("pci", 0),
                        result.statistics.get("financial", 0),
                        result.statistics.get("business", 0),
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                st.bar_chart(df_stats.set_index("Categoria"))

                st.markdown("### Métricas de Qualidade")
                qual_col1, qual_col2 = st.columns(2)
                with qual_col1:
                    st.metric(
                        "Entidades Validadas",
                        f"{result.statistics.get('validated', 0)}/{result.statistics['total']}"
                    )
                with qual_col2:
                    st.metric(
                        "Alta Confiança (≥80%)",
                        result.statistics.get("high_confidence", 0)
                    )

            with tab4:
                st.markdown("### Resultado em JSON")
                st.json(result.to_dict())

        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Sensitive Data NER Agent** - Parte do Data Governance AI Agents Kit

    Use este agente como filtro para proteger requisições a LLMs contra vazamento de dados sensíveis.
    """)


if __name__ == "__main__":
    main()
