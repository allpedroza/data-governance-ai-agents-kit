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
Streamlit UI for Data Quality Agent
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def init_agent():
    """Initialize the Data Quality Agent"""
    from data_quality.agent import DataQualityAgent

    return DataQualityAgent(
        persist_dir="./quality_data",
        enable_schema_tracking=True
    )


def render_dimension_card(dimension: str, data: dict):
    """Render a dimension score card"""
    score = data.get("score", 0)
    status = data.get("status", "unknown")

    if status == "passed":
        color = "#22c55e"
        icon = "✅"
    elif status == "warning":
        color = "#f59e0b"
        icon = "⚠️"
    else:
        color = "#ef4444"
        icon = "❌"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 24px;">{icon}</div>
        <div style="font-size: 14px; color: #888; margin-top: 4px;">{dimension.upper()}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{score:.0%}</div>
    </div>
    """, unsafe_allow_html=True)


def render_alerts(alerts: list):
    """Render alerts section"""
    if not alerts:
        st.success("✅ Nenhum alerta ativo")
        return

    for alert in alerts:
        level = alert.get("level", "info")
        if level == "critical":
            st.error(f"🔴 **{alert.get('rule_name')}**: {alert.get('message')}")
        elif level == "warning":
            st.warning(f"🟡 **{alert.get('rule_name')}**: {alert.get('message')}")
        else:
            st.info(f"🔵 **{alert.get('rule_name')}**: {alert.get('message')}")


def render_schema_drift(drift_data: dict):
    """Render schema drift section"""
    if not drift_data or not drift_data.get("has_drift"):
        st.success("✅ Nenhuma mudança de schema detectada")
        return

    st.warning(f"⚠️ {drift_data.get('summary')}")

    changes = drift_data.get("changes", [])
    if changes:
        for change in changes[:10]:
            severity = change.get("severity", "info")
            icon = "🔴" if severity == "critical" else "🟡" if severity == "warning" else "🔵"
            st.markdown(f"- {icon} {change.get('message')}")


def main():
    st.set_page_config(
        page_title="Data Quality Agent",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Data Quality Agent")
    st.markdown("Monitore a qualidade dos seus dados com métricas automáticas e alertas de SLA.")

    # Initialize agent
    if "quality_agent" not in st.session_state:
        st.session_state.quality_agent = init_agent()

    agent = st.session_state.quality_agent

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuração")

        # Rules management
        st.subheader("📋 Regras de Qualidade")

        rules_file = st.file_uploader(
            "Carregar regras (JSON)",
            type=["json"],
            help="Arquivo JSON com regras de qualidade"
        )

        if rules_file and st.button("Carregar Regras"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
                f.write(rules_file.read())
                temp_path = f.name

            try:
                count = agent.load_rules_from_file(temp_path)
                st.success(f"✓ {count} regras carregadas")
            except Exception as e:
                st.error(f"Erro: {e}")
            finally:
                os.unlink(temp_path)

        stats = agent.get_statistics()
        st.caption(f"📊 {stats.get('rules', {}).get('total_rules', 0)} regras configuradas")

        st.divider()

        # Freshness SLA config
        st.subheader("⏱️ SLA de Freshness")
        default_sla = st.number_input("SLA padrão (horas)", min_value=1, max_value=168, value=24)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Avaliar Arquivo", "📊 Alertas Ativos", "📈 Histórico"])

    with tab1:
        st.subheader("Avaliar Qualidade de Dados")

        file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"])

        uploaded_file = st.file_uploader(
            f"Selecione o arquivo {file_type}",
            type=["csv"] if file_type == "CSV" else ["parquet"]
        )

        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.number_input(
                "Tamanho da amostra",
                min_value=100,
                max_value=100000,
                value=10000
            )
        with col2:
            if file_type == "CSV":
                encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"])
                separator = st.selectbox("Separador", [",", ";", "|"])

        # Freshness config
        with st.expander("⏱️ Configuração de Freshness"):
            check_freshness = st.checkbox("Verificar freshness", value=False)
            if check_freshness:
                timestamp_col = st.text_input(
                    "Coluna de timestamp",
                    placeholder="updated_at, created_at, etc."
                )
                sla_hours = st.number_input(
                    "SLA (horas)",
                    min_value=1,
                    max_value=168,
                    value=default_sla
                )

        # Validity configs
        with st.expander("✓ Configuração de Validação"):
            add_validity = st.checkbox("Adicionar validação de formato", value=False)
            if add_validity:
                val_column = st.text_input("Coluna para validar")
                val_pattern = st.selectbox(
                    "Padrão",
                    ["email", "cpf", "cnpj", "phone_br", "date_iso", "uuid"]
                )

        if uploaded_file and st.button("🚀 Avaliar Qualidade", type="primary"):
            with st.spinner("Analisando dados..."):
                # Save uploaded file
                suffix = ".csv" if file_type == "CSV" else ".parquet"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name

                try:
                    # Build config
                    kwargs = {"sample_size": sample_size}

                    if check_freshness and timestamp_col:
                        kwargs["freshness_config"] = {
                            "timestamp_column": timestamp_col,
                            "sla_hours": sla_hours,
                            "max_age_hours": sla_hours * 2
                        }

                    if add_validity and val_column:
                        kwargs["validity_configs"] = [{
                            "column": val_column,
                            "pattern_name": val_pattern,
                            "threshold": 0.95
                        }]

                    # Evaluate
                    report = agent.evaluate_file(temp_path, **kwargs)
                    st.session_state.last_report = report

                    st.success(f"✓ Análise concluída em {report.processing_time_ms}ms")

                except Exception as e:
                    st.error(f"Erro: {e}")
                finally:
                    os.unlink(temp_path)

        # Display results
        if "last_report" in st.session_state:
            report = st.session_state.last_report

            st.divider()

            # Overall score
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score_color = "#22c55e" if report.overall_status == "passed" else "#f59e0b" if report.overall_status == "warning" else "#ef4444"
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 48px; font-weight: bold; color: {score_color};">
                        {report.overall_score:.0%}
                    </div>
                    <div style="color: #888;">Score Geral</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.metric("Status", report.overall_status.upper())
            with col3:
                st.metric("Linhas", f"{report.row_count:,}")
            with col4:
                st.metric("Colunas", report.columns_checked)

            # Dimensions
            st.markdown("### Dimensões de Qualidade")
            dim_cols = st.columns(len(report.dimensions))
            for i, (dim_name, dim_data) in enumerate(report.dimensions.items()):
                with dim_cols[i]:
                    render_dimension_card(dim_name, dim_data)

            # Alerts
            st.markdown("### Alertas")
            render_alerts(report.alerts)

            # Schema drift
            if report.schema_drift:
                st.markdown("### Schema Drift")
                render_schema_drift(report.schema_drift)

            # Detailed metrics
            with st.expander("📊 Métricas Detalhadas"):
                for metric in report.metrics:
                    status_icon = "✅" if metric.get("passed") else "❌"
                    st.markdown(f"""
                    **{status_icon} {metric.get('metric_name')}** ({metric.get('dimension')})
                    - Valor: {metric.get('value_percent')}
                    - Threshold: {metric.get('threshold'):.0%}
                    - {metric.get('message')}
                    """)

            # Export
            st.markdown("### Exportar")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 JSON",
                    report.to_json(),
                    file_name=f"{report.source_name}_quality_report.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    "📥 Markdown",
                    report.to_markdown(),
                    file_name=f"{report.source_name}_quality_report.md",
                    mime="text/markdown"
                )

    with tab2:
        st.subheader("Alertas Ativos")

        alerts = agent.get_active_alerts()

        if not alerts:
            st.success("✅ Nenhum alerta ativo no momento")
        else:
            st.warning(f"⚠️ {len(alerts)} alertas ativos")

            for alert in alerts:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        level = alert.level.value
                        icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🔵"
                        st.markdown(f"""
                        {icon} **{alert.rule_name}**
                        - Tabela: `{alert.table_name}`
                        - Dimensão: {alert.dimension}
                        - Valor: {alert.value:.2%} (threshold: {alert.threshold:.2%})
                        - {alert.message}
                        """)
                    with col2:
                        if st.button("Reconhecer", key=alert.alert_id):
                            agent.acknowledge_alert(alert.alert_id)
                            st.rerun()

    with tab3:
        st.subheader("Histórico de Avaliações")

        reports = agent.get_report_history(limit=20)

        if not reports:
            st.info("Nenhuma avaliação realizada ainda")
        else:
            history_data = []
            for r in reports:
                history_data.append({
                    "Timestamp": r.timestamp[:19],
                    "Source": r.source_name,
                    "Score": f"{r.overall_score:.0%}",
                    "Status": r.overall_status,
                    "Alertas": len(r.alerts),
                    "Tempo (ms)": r.processing_time_ms
                })

            st.dataframe(history_data, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
