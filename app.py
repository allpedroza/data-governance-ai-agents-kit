"""Unified Streamlit UI to orchestrate the five available agents."""
import csv
import json
import os
import sys
import tempfile
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Ensure repository modules are importable when running the app from repo root
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "lineage"))
sys.path.append(str(BASE_DIR / "rag_discovery"))

# External catalog connectors
from rag_discovery.openmetadata_connector import (  # noqa: E402
    OpenMetadataConnector,
    OpenMetadataConnectorError,
)

missing_core_modules = [module for module in ["openai"] if find_spec(module) is None]
if missing_core_modules:
    missing_list = ", ".join(sorted(missing_core_modules))
    raise ModuleNotFoundError(
        "Required dependency not found: {}. "
        "Install with 'python -m pip install -r requirements.txt' "
        "using the same interpreter that runs Streamlit.".format(missing_list)
    )

from lineage.data_lineage_agent import DataLineageAgent  # noqa: E402
from rag_discovery.data_discovery_rag_agent import (  # noqa: E402
    DataDiscoveryRAGAgent,
    TableMetadata,
)

# Metadata Enrichment Agent imports
sys.path.append(str(BASE_DIR / "metadata_enrichment"))
ENRICHMENT_AVAILABLE = True
try:
    from metadata_enrichment.agent import MetadataEnrichmentAgent, EnrichmentResult
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore
except ImportError:
    ENRICHMENT_AVAILABLE = False

# Data Quality Agent imports
sys.path.append(str(BASE_DIR / "data_quality"))
QUALITY_AVAILABLE = True
try:
    from data_quality.agent import DataQualityAgent, QualityReport
    from data_quality.rules import QualityRule, AlertLevel
except ImportError:
    QUALITY_AVAILABLE = False

# Data Classification Agent imports
sys.path.append(str(BASE_DIR / "data_classification"))
CLASSIFICATION_AVAILABLE = True
try:
    from data_classification.agent import DataClassificationAgent, ClassificationReport
except ImportError:
    CLASSIFICATION_AVAILABLE = False

# Data Asset Value Agent imports
sys.path.append(str(BASE_DIR / "data_asset_value"))
VALUE_AVAILABLE = True
try:
    from data_asset_value.agent import DataAssetValueAgent, AssetValueReport
except ImportError:
    VALUE_AVAILABLE = False

# Sensitive Data NER Agent imports
sys.path.append(str(BASE_DIR / "sensitive_data_ner"))
NER_AVAILABLE = True
VAULT_AVAILABLE = False
try:
    from sensitive_data_ner.agent import (
        SensitiveDataNERAgent,
        NERResult,
        FilterPolicy,
        FilterAction,
        EntityCategory,
    )
    from sensitive_data_ner.anonymizers import AnonymizationStrategy
    # Vault imports (optional - requires cryptography)
    try:
        from sensitive_data_ner.vault import (
            SecureVault,
            VaultConfig,
            AccessLevel,
            AuditEventType,
        )
        VAULT_AVAILABLE = True
    except ImportError:
        VAULT_AVAILABLE = False
except ImportError:
    NER_AVAILABLE = False


st.set_page_config(
    page_title="Data Governance AI Agents",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
            --text: #0f172a;
            --muted: #475569;
            --accent: #2563eb;
        }

        .main {
            background: var(--bg);
        }

        .section-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            box-shadow: 0 10px 35px rgba(15, 23, 42, 0.06);
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: #eef2ff;
            color: var(--text);
            margin-right: 0.4rem;
            border: 1px solid var(--border);
        }

        .callout {
            border-left: 4px solid var(--accent);
            background: #eef2ff;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            color: var(--text);
        }

        .compact-header h1, .compact-header p {
            margin-bottom: 0.35rem;
            color: var(--text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title: str, description: str | None = None) -> None:
    """Consistent header used across tabs to keep the UI minimal."""

    st.markdown("<div class='compact-header'>", unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.markdown(f"<p style='color:#475569;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def card_container():
    """Provide a lightly styled container similar to OpenMetadata panels."""

    return st.container(border=True)


def _initialize_rag_agent() -> None:
    """Initialize the RAG agent using the current environment variables."""
    try:
        st.session_state.rag_agent = DataDiscoveryRAGAgent(
            persist_directory=str(BASE_DIR / "rag_discovery" / ".chroma_ui"),
            collection_name="ui_catalog",
        )
        st.session_state.rag_agent_error = None
    except Exception as exc:  # noqa: BLE001 - surface initialization issues
        st.session_state.rag_agent = None
        st.session_state.rag_agent_error = str(exc)


def init_session_state() -> None:
    """Prepare shared agents and caches."""
    if "lineage_agent" not in st.session_state:
        st.session_state.lineage_agent = DataLineageAgent()
        st.session_state.lineage_results = None

    if "rag_catalog" not in st.session_state:
        st.session_state.rag_catalog: List[TableMetadata] = []

    if "connected_catalogs" not in st.session_state:
        st.session_state.connected_catalogs: List[Dict[str, Any]] = []

    if "discovery_messages" not in st.session_state:
        st.session_state.discovery_messages = []

    if "connection_settings" not in st.session_state:
        st.session_state.connection_settings = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_api_url": os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
            "llm_provider": os.environ.get("LLM_PROVIDER", "openai"),
            "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "gemini_api_key": os.environ.get("GOOGLE_API_KEY", ""),
            "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "deepseek_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "deepseek_api_url": os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com"),
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic_model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            "atlas_host": os.environ.get("ATLAS_HOST", ""),
            "glue_region": os.environ.get("AWS_REGION", ""),
            "chroma_persist": str(BASE_DIR / "rag_discovery" / ".chroma_ui"),
            "openmetadata_host": os.environ.get("OPENMETADATA_HOST", ""),
            "openmetadata_api_token": os.environ.get("OPENMETADATA_API_TOKEN", ""),
        }

    if "rag_agent" not in st.session_state:
        _initialize_rag_agent()


def hero_section() -> None:
    """Top banner with quick context."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.title("Data Governance AI Agents")
            st.markdown(
                "Uma interface simples para orquestrar o framework de IA Generativa que acelerará seu programa de Governança de dados."
            )
            st.markdown(
                "<div class='callout'>Defina sua `OPENAI_API_KEY` para ativar buscas vetoriais, geração de metadados e relatórios ricos.</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Agentes disponíveis", "5", help="Lineage, Discovery, Enrichment, Classification e Quality")
            st.metric("Status da sessão", "Pronto", help="Sessão inicializada com os caches padrões")


def render_lineage_tab() -> None:
    """UI for running the Data Lineage Agent."""
    section_header(
        "🔗 Data Lineage Agent",
        "Envie pipelines e acompanhe a análise de forma direta, sem painéis complexos.",
    )

    # Check for configured warehouses
    configured_warehouses = list(st.session_state.get("warehouse_connections", {}).keys())

    with card_container():
        # Source selection
        source_options = ["Arquivos enviados"]
        if configured_warehouses:
            source_options.append("DDL do Data Warehouse")

        lineage_source = st.radio(
            "Fonte dos dados",
            source_options,
            horizontal=True,
            key="lineage_source_mode"
        )

        if lineage_source == "Arquivos enviados":
            with st.form("lineage_form"):
                uploads = st.file_uploader(
                    "Envie arquivos do pipeline (SQL, Python, Terraform, etc.)",
                    type=["py", "sql", "tf", "json", "scala", "dag"],
                    accept_multiple_files=True,
                )
                st.checkbox("Detectar padrões de streaming", value=True, key="detect_streaming")
                analyze = st.form_submit_button("Gerar linhagem")

            if analyze:
                if not uploads:
                    st.warning("Envie ao menos um arquivo para análise.")
                else:
                    with st.status("Processando arquivos", expanded=True) as status:
                        tmp_paths: List[str] = []
                        with tempfile.TemporaryDirectory() as temp_dir:
                            for upload in uploads:
                                dest = Path(temp_dir) / upload.name
                                dest.write_bytes(upload.read())
                                tmp_paths.append(str(dest))
                            status.write(f"{len(tmp_paths)} arquivos enviados")
                            status.write("Executando análise de linhagem...")
                            results = st.session_state.lineage_agent.analyze_pipeline(tmp_paths)
                            st.session_state.lineage_results = results
                            status.update(label="Análise concluída", state="complete")

        elif lineage_source == "DDL do Data Warehouse":
            st.markdown("**Extrair DDL diretamente do Data Warehouse conectado**")

            selected_wh = st.selectbox(
                "Data Warehouse",
                configured_warehouses,
                key="lineage_dw_select"
            )

            col1, col2 = st.columns(2)
            with col1:
                wh_database = st.text_input("Database (opcional)", key="lineage_dw_database")
            with col2:
                wh_schema = st.text_input("Schema (opcional)", key="lineage_dw_schema")

            # Table selection mode
            table_mode = st.radio(
                "Modo de seleção",
                ["Todas as tabelas", "Tabelas específicas"],
                horizontal=True,
                key="lineage_table_mode"
            )

            specific_tables = []
            if table_mode == "Tabelas específicas":
                tables_input = st.text_area(
                    "Nomes das tabelas (uma por linha)",
                    placeholder="tabela1\ntabela2\ntabela3",
                    key="lineage_specific_tables"
                )
                if tables_input:
                    specific_tables = [t.strip() for t in tables_input.strip().split("\n") if t.strip()]

            st.checkbox("Detectar padrões de streaming", value=True, key="detect_streaming_dw")

            if st.button("Extrair DDL e Gerar Linhagem", key="lineage_dw_analyze"):
                if selected_wh:
                    wh_config = st.session_state.warehouse_connections.get(selected_wh, {})

                    with st.status("Conectando ao Data Warehouse...", expanded=True) as status:
                        try:
                            dw_connector = get_warehouse_connector(selected_wh, wh_config)
                            dw_connector.connect()
                            status.write(f"Conectado a {selected_wh}")

                            # Get tables to process
                            if specific_tables:
                                tables_to_process = specific_tables
                            else:
                                status.write("Listando tabelas...")
                                table_list = dw_connector.list_tables(
                                    schema=wh_schema or None,
                                    database=wh_database or None
                                )
                                tables_to_process = [t.name for t in table_list]

                            status.write(f"Extraindo DDL de {len(tables_to_process)} tabelas...")

                            # Extract DDL for each table
                            ddl_contents = []
                            for tbl_name in tables_to_process:
                                try:
                                    ddl = dw_connector.get_ddl(
                                        tbl_name,
                                        schema=wh_schema or None,
                                        database=wh_database or None
                                    )
                                    if ddl:
                                        ddl_contents.append(f"-- Table: {tbl_name}\n{ddl}\n")
                                except Exception as e:
                                    status.write(f"Erro ao extrair DDL de {tbl_name}: {e}")

                            if not ddl_contents:
                                st.warning("Nenhum DDL foi extraído.")
                            else:
                                # Write DDL to temp file and analyze
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    ddl_file = Path(temp_dir) / f"{selected_wh}_ddl.sql"
                                    ddl_file.write_text("\n\n".join(ddl_contents))

                                    status.write(f"DDL extraído: {len(ddl_contents)} tabelas")
                                    status.write("Executando análise de linhagem...")

                                    results = st.session_state.lineage_agent.analyze_pipeline([str(ddl_file)])
                                    st.session_state.lineage_results = results

                                status.update(label="Análise concluída", state="complete")

                            dw_connector.disconnect()

                        except Exception as e:
                            st.error(f"Erro ao conectar ao Data Warehouse: {e}")

    results = st.session_state.get("lineage_results")
    if results:
        metrics = results.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets", metrics.get("total_assets", 0))
        col2.metric("Transformações", metrics.get("total_transformations", 0))
        complexity: Dict[str, int] = metrics.get("complexity_metrics", {})
        col3.metric("Nós", complexity.get("nodes", 0))
        col4.metric("Arestas", complexity.get("edges", 0))

        with st.expander("Insights automáticos", expanded=True):
            insights = results.get("insights", {})
            st.markdown(insights.get("summary", "Sem resumo disponível."))
            for rec in insights.get("recommendations", []):
                st.markdown(f"- {rec}")

        critical = results.get("critical_components", {})
        with st.expander("Componentes críticos"):
            st.write(critical)

        st.info(
            "Para visualização completa do grafo, execute `streamlit run lineage/app.py`."
        )


def _apply_connection_settings(settings: Dict[str, str]) -> None:
    """Persist connection settings in memory and environment, then refresh the agent."""
    st.session_state.connection_settings.update(settings)
    if settings.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = settings["openai_api_key"]
    if settings.get("openai_api_url"):
        os.environ["OPENAI_API_URL"] = settings["openai_api_url"]
    if settings.get("atlas_host"):
        os.environ["ATLAS_HOST"] = settings["atlas_host"]
    if settings.get("glue_region"):
        os.environ["AWS_REGION"] = settings["glue_region"]
    if settings.get("llm_provider"):
        os.environ["LLM_PROVIDER"] = settings["llm_provider"]
    if settings.get("llm_model"):
        os.environ["LLM_MODEL"] = settings["llm_model"]
    if settings.get("gemini_api_key"):
        os.environ["GOOGLE_API_KEY"] = settings["gemini_api_key"]
    if settings.get("gemini_model"):
        os.environ["GEMINI_MODEL"] = settings["gemini_model"]
    if settings.get("deepseek_api_key"):
        os.environ["DEEPSEEK_API_KEY"] = settings["deepseek_api_key"]
    if settings.get("deepseek_model"):
        os.environ["DEEPSEEK_MODEL"] = settings["deepseek_model"]
    if settings.get("deepseek_api_url"):
        os.environ["DEEPSEEK_API_URL"] = settings["deepseek_api_url"]
    if settings.get("anthropic_api_key"):
        os.environ["ANTHROPIC_API_KEY"] = settings["anthropic_api_key"]
    if settings.get("anthropic_model"):
        os.environ["ANTHROPIC_MODEL"] = settings["anthropic_model"]
    if settings.get("openmetadata_host"):
        os.environ["OPENMETADATA_HOST"] = settings["openmetadata_host"]
    if settings.get("openmetadata_api_token"):
        os.environ["OPENMETADATA_API_TOKEN"] = settings["openmetadata_api_token"]

    _initialize_rag_agent()


def _render_table_catalog(catalog: List[TableMetadata]) -> None:
    """Display a simple catalog grid."""
    if not catalog:
        st.markdown("Catálogo vazio. Adicione uma tabela para começar.")
        return

    rows = []
    for table in catalog:
        rows.append(
            {
                "Tabela": table.name,
                "Base": table.database,
                "Schema": table.schema,
                "Tags": ", ".join(table.tags),
                "Descrição": table.description,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _index_table_if_possible(table: TableMetadata) -> None:
    """Try to index the table in ChromaDB when the agent is available."""
    agent = st.session_state.get("rag_agent")
    if not agent:
        return
    try:
        agent.index_table(table, force_update=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Não foi possível indexar esta tabela no ChromaDB: {exc}")


def _sync_openmetadata_tables(
    server_url: str,
    api_token: str,
    service_filter: str,
    limit: int,
) -> List[TableMetadata]:
    """Fetch tables from OpenMetadata and index them for discovery."""

    connector = OpenMetadataConnector(server_url=server_url, api_token=api_token)
    with st.status("Sincronizando metadados do OpenMetadata", expanded=True) as status:
        status.write("Chamando API do OpenMetadata...")
        tables = connector.fetch_tables(
            max_tables=limit,
            service_filter=service_filter or None,
        )

        status.write(f"{len(tables)} tabelas retornadas; indexando no catálogo local...")
        for table in tables:
            st.session_state.rag_catalog.append(table)
            _index_table_if_possible(table)

        status.update(
            label="Metadados sincronizados do OpenMetadata",
            state="complete",
        )

    return tables


def _search_with_fallback(query: str) -> List[Dict[str, str]]:
    """Execute vector search or fallback to local keyword matching."""
    agent = st.session_state.get("rag_agent")
    catalog: List[TableMetadata] = st.session_state.get("rag_catalog", [])

    if agent and catalog:
        try:
            results = agent.search(query, n_results=5)
            return [
                {
                    "Tabela": res.table.name,
                    "Confiança": f"{res.relevance_score:.0%}",
                    "Motivo": res.matching_reason,
                    "Trecho": res.snippet,
                }
                for res in results
            ]
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Busca vetorial falhou, usando fallback simples: {exc}")

    lowered = query.lower()
    fallback = []
    for table in catalog:
        blob = " ".join(
            [
                table.name,
                table.database,
                table.schema,
                table.description,
                " ".join(table.tags),
            ]
        ).lower()
        if lowered in blob:
            fallback.append(
                {
                    "Tabela": table.name,
                    "Confiança": "—",
                    "Motivo": "Encontrado via busca simples",
                    "Trecho": table.description[:160],
                }
            )
    return fallback


def _render_connected_catalogs() -> None:
    """Show connected catalog badges for quick context."""
    if not st.session_state.get("connected_catalogs"):
        st.info("Nenhum catálogo conectado ainda. Conecte um catálogo para conversar com o agente.")
        return

    st.markdown("**Catálogos conectados**")
    badges = []
    for catalog in st.session_state.connected_catalogs:
        label = f"{catalog['name']} • {catalog['source']}"
        badges.append(f"<span class='pill'>{label}</span>")
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def _render_connection_guide() -> None:
    """Display a guided experience for configuring external connections."""
    with card_container():
        st.markdown("### Guia de conexões essenciais")
        settings: Dict[str, str] = st.session_state.connection_settings

    status_labels = []
    status_labels.append(
        "✅ OPENAI_API_KEY configurada" if settings.get("openai_api_key") else "⚠️ OPENAI_API_KEY ausente"
    )
    status_labels.append(
        "✅ GOOGLE_API_KEY configurada" if settings.get("gemini_api_key") else "⚠️ GOOGLE_API_KEY ausente"
    )
    status_labels.append(
        "✅ DEEPSEEK_API_KEY configurada" if settings.get("deepseek_api_key") else "⚠️ DEEPSEEK_API_KEY ausente"
    )
    status_labels.append(
        "✅ ANTHROPIC_API_KEY configurada" if settings.get("anthropic_api_key") else "⚠️ ANTHROPIC_API_KEY ausente"
    )
    status_labels.append(
        "✅ Persistência local do Chroma pronta" if settings.get("chroma_persist") else "⚠️ Revise o diretório do Chroma"
    )
    if settings.get("openmetadata_host"):
        status_labels.append("✅ Endpoint do OpenMetadata definido")
    else:
        status_labels.append("⚠️ Defina a URL do OpenMetadata para sincronizar catálogos")
    st.markdown("; ".join(status_labels))

    with st.expander("Configurar provedores de LLM e vetorização", expanded=not settings.get("openai_api_key")):
        st.markdown(
            "- Defina a chave de cada provedor para habilitar fluxos de IA no framework.\n"
            "- Selecione o modelo padrão por provedor e, opcionalmente, defina qual será o padrão global (`LLM_PROVIDER`).\n"
            "- O catálogo vetorial usa ChromaDB local em `{}`; nenhum serviço externo é necessário.".format(
                settings.get("chroma_persist")
            )
        )

        openai_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-3.5-turbo",
        ]
        gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]
        deepseek_models = ["deepseek-chat", "deepseek-reasoner"]
        claude_models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

        openai_tab, gemini_tab, deepseek_tab, claude_tab = st.tabs(
            ["OpenAI", "Gemini", "DeepSeek", "Claude"]
        )

        with openai_tab:
            with st.form("openai_settings_form"):
                openai_key = st.text_input(
                    "OPENAI_API_KEY",
                    value=settings.get("openai_api_key", ""),
                    type="password",
                    help="Copie a chave exata do painel da OpenAI para evitar erros de autenticação.",
                )
                openai_url = st.text_input(
                    "OPENAI_API_URL",
                    value=settings.get("openai_api_url", "https://api.openai.com/v1"),
                    help="Use o endpoint padrão ou um compatível (Azure/OpenAI compatível).",
                )
                openai_model = st.selectbox(
                    "Modelo OpenAI",
                    options=openai_models,
                    index=openai_models.index(settings.get("llm_model", "gpt-4o-mini"))
                    if settings.get("llm_model") in openai_models
                    else 0,
                )
                set_default = st.checkbox(
                    "Usar OpenAI como provedor padrão", value=settings.get("llm_provider") == "openai"
                )
                apply_credentials = st.form_submit_button("Salvar configuração OpenAI")

            if apply_credentials:
                _apply_connection_settings(
                    {
                        "openai_api_key": openai_key.strip(),
                        "openai_api_url": openai_url.strip(),
                        "llm_model": openai_model,
                        "llm_provider": "openai" if set_default else settings.get("llm_provider", "openai"),
                    }
                )
                st.success("Configuração OpenAI salva.")

        with gemini_tab:
            with st.form("gemini_settings_form"):
                gemini_key = st.text_input(
                    "GOOGLE_API_KEY",
                    value=settings.get("gemini_api_key", ""),
                    type="password",
                    help="Chave da API Gemini (Google AI/Vertex).",
                )
                gemini_model = st.selectbox(
                    "Modelo Gemini",
                    options=gemini_models,
                    index=gemini_models.index(settings.get("gemini_model", "gemini-1.5-flash"))
                    if settings.get("gemini_model") in gemini_models
                    else 0,
                )
                set_default_gemini = st.checkbox(
                    "Usar Gemini como provedor padrão", value=settings.get("llm_provider") == "gemini"
                )
                apply_gemini = st.form_submit_button("Salvar configuração Gemini")

            if apply_gemini:
                _apply_connection_settings(
                    {
                        "gemini_api_key": gemini_key.strip(),
                        "gemini_model": gemini_model,
                        "llm_provider": "gemini" if set_default_gemini else settings.get("llm_provider", "openai"),
                        "llm_model": gemini_model if set_default_gemini else settings.get("llm_model", "gpt-4o-mini"),
                    }
                )
                st.success("Configuração Gemini salva.")

        with deepseek_tab:
            with st.form("deepseek_settings_form"):
                deepseek_key = st.text_input(
                    "DEEPSEEK_API_KEY",
                    value=settings.get("deepseek_api_key", ""),
                    type="password",
                    help="Chave da API DeepSeek (compatível com OpenAI).",
                )
                deepseek_url = st.text_input(
                    "DEEPSEEK_API_URL",
                    value=settings.get("deepseek_api_url", "https://api.deepseek.com"),
                    help="Endpoint compatível com OpenAI para DeepSeek.",
                )
                deepseek_model = st.selectbox(
                    "Modelo DeepSeek",
                    options=deepseek_models,
                    index=deepseek_models.index(settings.get("deepseek_model", "deepseek-chat"))
                    if settings.get("deepseek_model") in deepseek_models
                    else 0,
                )
                set_default_deepseek = st.checkbox(
                    "Usar DeepSeek como provedor padrão", value=settings.get("llm_provider") == "deepseek"
                )
                apply_deepseek = st.form_submit_button("Salvar configuração DeepSeek")

            if apply_deepseek:
                _apply_connection_settings(
                    {
                        "deepseek_api_key": deepseek_key.strip(),
                        "deepseek_api_url": deepseek_url.strip(),
                        "deepseek_model": deepseek_model,
                        "llm_provider": "deepseek"
                        if set_default_deepseek
                        else settings.get("llm_provider", "openai"),
                        "llm_model": deepseek_model if set_default_deepseek else settings.get("llm_model", "gpt-4o-mini"),
                    }
                )
                st.success("Configuração DeepSeek salva.")

        with claude_tab:
            with st.form("claude_settings_form"):
                claude_key = st.text_input(
                    "ANTHROPIC_API_KEY",
                    value=settings.get("anthropic_api_key", ""),
                    type="password",
                    help="Chave da API Anthropic Claude.",
                )
                claude_model = st.selectbox(
                    "Modelo Claude",
                    options=claude_models,
                    index=claude_models.index(settings.get("anthropic_model", "claude-3-5-sonnet-20240620"))
                    if settings.get("anthropic_model") in claude_models
                    else 0,
                )
                set_default_claude = st.checkbox(
                    "Usar Claude como provedor padrão", value=settings.get("llm_provider") == "claude"
                )
                apply_claude = st.form_submit_button("Salvar configuração Claude")

            if apply_claude:
                _apply_connection_settings(
                    {
                        "anthropic_api_key": claude_key.strip(),
                        "anthropic_model": claude_model,
                        "llm_provider": "claude"
                        if set_default_claude
                        else settings.get("llm_provider", "openai"),
                        "llm_model": claude_model if set_default_claude else settings.get("llm_model", "gpt-4o-mini"),
                    }
                )
                st.success("Configuração Claude salva.")

    with st.expander("Conexões de catálogos corporativos"):
        st.markdown(
            "- **Apache Atlas**: informe o host em `ATLAS_HOST` (ex.: `http://atlas-host:21000`).\n"
            "- **Glue Data Catalog**: use a região em `AWS_REGION` e credenciais AWS no ambiente.\n"
            "- **BigQuery/Snowflake**: a conexão é guiada na seção de conectores; inclua o projeto/warehouse ao selecionar.\n"
            "- **OpenMetadata**: defina `OPENMETADATA_HOST` e o token (`OPENMETADATA_API_TOKEN`) ou informe-os na conexão guiada.\n"
            "- Após salvar, volte à aba de conexão de catálogo para sincronizar metadados."
        )


def _parse_tables_from_file(uploaded_file) -> List[TableMetadata]:
    """Parse uploaded catalog files (JSON or CSV) into TableMetadata objects."""
    suffix = Path(uploaded_file.name).suffix.lower()
    content = uploaded_file.read()

    if suffix == ".json":
        data = json.loads(content.decode("utf-8"))
        raw_tables = data if isinstance(data, list) else data.get("tables", [])
    elif suffix in {".csv", ".txt"}:
        decoded = content.decode("utf-8")
        reader = csv.DictReader(StringIO(decoded))
        raw_tables = list(reader)
    else:
        raise ValueError("Formato não suportado. Envie JSON ou CSV.")

    tables: List[TableMetadata] = []
    for raw in raw_tables:
        tags = raw.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        tables.append(
            TableMetadata(
                name=raw.get("name", ""),
                database=raw.get("database", ""),
                schema=raw.get("schema", ""),
                description=raw.get("description", ""),
                tags=tags,
            )
        )
    return tables


def _answer_discovery_question(prompt: str) -> str:
    """Generate an answer using the RAG agent or a simple fallback search."""
    agent = st.session_state.get("rag_agent")
    if agent:
        try:
            response = agent.ask(prompt, n_context=4)
            return response.get("answer") or "Não foi possível gerar uma resposta agora."
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Falha ao usar RAG completo. Retornando uma busca simples: {exc}")

    results = _search_with_fallback(prompt)
    if not results:
        return "Nenhum resultado encontrado. Adicione catálogos ou tabelas antes de perguntar."

    lines = ["Resultados encontrados via busca simples:"]
    for idx, res in enumerate(results, start=1):
        lines.append(
            f"{idx}. {res['Tabela']} — {res['Motivo']} (Trecho: {res['Trecho']})"
        )
    return "\n".join(lines)


def render_rag_tab() -> None:
    """UI for the RAG discovery agent."""
    section_header(
        "🔍 Data Discovery RAG Agent",
        "Construa um catálogo rápido e converse em linguagem natural com uma experiência limpa.",
    )

    if st.session_state.get("rag_agent_error"):
        st.warning(
            "Não foi possível inicializar o agente RAG automaticamente. "
            f"Detalhes: {st.session_state['rag_agent_error']}"
        )

    _render_connection_guide()

    connection_tab, chat_tab = st.tabs(["Conectar catálogo", "Conversar com o agente"])

    with connection_tab:
        st.markdown("Escolha como quer conectar seu catálogo antes de iniciar a conversa.")
        settings = st.session_state.connection_settings
        configured_warehouses = get_configured_warehouses()

        with st.form("catalog_connection_form"):
            # Add DW option if warehouses are configured
            mode_options = ["Arquivo de catálogo", "Conectores disponíveis"]
            if configured_warehouses:
                mode_options.append("Data Warehouse conectado")

            mode = st.radio(
                "Origem do catálogo",
                options=mode_options,
                horizontal=True,
            )
            catalog_name = st.text_input("Nome do catálogo", placeholder="Data Lake Principal")
            dataset_hint = ""
            service_filter = ""
            om_host = settings.get("openmetadata_host", "")
            om_token = settings.get("openmetadata_api_token", "")
            table_limit = 200
            uploaded = None
            connector = None
            selected_wh = None
            wh_database = ""
            wh_schema = ""

            if mode == "Arquivo de catálogo":
                uploaded = st.file_uploader(
                    "Envie um arquivo JSON ou CSV com as tabelas",
                    type=["json", "csv", "txt"],
                    accept_multiple_files=False,
                )
            elif mode == "Data Warehouse conectado":
                # Data Warehouse sync options
                warehouse_labels = {
                    "snowflake": "❄️ Snowflake",
                    "redshift": "🔴 Amazon Redshift",
                    "bigquery": "🔵 Google BigQuery",
                    "synapse": "🟣 Azure Synapse"
                }
                selected_wh = st.selectbox(
                    "Data Warehouse",
                    options=configured_warehouses,
                    format_func=lambda x: warehouse_labels.get(x, x),
                )
                wh_config = st.session_state.warehouse_connections.get(selected_wh, {})

                col1, col2 = st.columns(2)
                with col1:
                    if selected_wh == "bigquery":
                        wh_database = st.text_input("Project ID", value=wh_config.get("project_id", ""))
                    else:
                        wh_database = st.text_input("Database", value=wh_config.get("database", ""))
                with col2:
                    if selected_wh == "bigquery":
                        wh_schema = st.text_input("Dataset (vazio = todos)", value=wh_config.get("dataset", ""))
                    else:
                        wh_schema = st.text_input("Schema (vazio = todos)", value=wh_config.get("schema", ""))

                table_limit = int(
                    st.number_input(
                        "Limite de tabelas a sincronizar",
                        min_value=10,
                        max_value=1000,
                        value=200,
                        step=10,
                        key="dw_table_limit"
                    )
                )
                dataset_hint = f"{selected_wh}:{wh_database}"
            else:
                connector = st.selectbox(
                    "Selecione um conector existente",
                    [
                        "OpenMetadata",
                        "Apache Atlas",
                        "Glue Data Catalog",
                        "BigQuery",
                        "Snowflake",
                    ],
                )
                if connector == "OpenMetadata":
                    om_host = st.text_input(
                        "Endpoint do OpenMetadata",
                        value=om_host,
                        placeholder="https://openmetadata:8585",
                        help="Use o endpoint da API (porta padrão 8585).",
                    )
                    om_token = st.text_input(
                        "Token de autenticação",
                        value=om_token,
                        type="password",
                        help="Cole o JWT de serviço do OpenMetadata ou um token pessoal.",
                    )
                    service_filter = st.text_input(
                        "Filtrar por serviço (opcional)",
                        placeholder="ex: lakehouse_service",
                        help="Use para sincronizar apenas tabelas de um serviço específico.",
                    )
                    table_limit = int(
                        st.number_input(
                            "Limite de tabelas a sincronizar",
                            min_value=10,
                            max_value=1000,
                            value=200,
                            step=10,
                        )
                    )
                    dataset_hint = service_filter or om_host
                else:
                    dataset_hint = st.text_input(
                        "Escopo ou dataset", placeholder="ex: projeto-analytics"
                    )
                uploaded = None

            connect = st.form_submit_button("Conectar catálogo")

        if connect:
            if mode == "Arquivo de catálogo" and not uploaded:
                st.warning("Envie um arquivo de metadados para conectar.")
            elif mode == "Conectores disponíveis" and connector == "OpenMetadata" and (
                not om_host or not om_token
            ):
                st.warning("Informe o endpoint e o token do OpenMetadata para sincronizar.")
            elif mode == "Data Warehouse conectado" and not selected_wh:
                st.warning("Selecione um Data Warehouse para sincronizar.")
            else:
                catalog_label = catalog_name or (
                    uploaded.name if uploaded else (selected_wh if mode == "Data Warehouse conectado" else connector)
                )
                tables_added = 0

                if uploaded:
                    try:
                        tables = _parse_tables_from_file(uploaded)
                        for table in tables:
                            st.session_state.rag_catalog.append(table)
                            _index_table_if_possible(table)
                        tables_added = len(tables)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Não foi possível processar o arquivo: {exc}")
                        tables_added = 0
                elif mode == "Data Warehouse conectado" and selected_wh:
                    # Sync tables from Data Warehouse
                    try:
                        wh_config = st.session_state.warehouse_connections.get(selected_wh, {})
                        dw_connector = get_warehouse_connector(selected_wh, wh_config)

                        if dw_connector:
                            with st.status("Sincronizando catálogo do Data Warehouse...", expanded=True) as status:
                                status.write(f"Conectando ao {selected_wh}...")
                                dw_connector.connect()

                                status.write("Listando tabelas...")
                                dw_tables = dw_connector.list_tables(
                                    schema=wh_schema if wh_schema else None,
                                    database=wh_database if wh_database else None,
                                    include_views=True
                                )

                                # Limit tables
                                dw_tables = dw_tables[:table_limit]

                                status.write(f"Processando {len(dw_tables)} tabelas...")
                                for dw_table in dw_tables:
                                    # Get schema info for description
                                    try:
                                        columns = dw_connector.get_table_schema(
                                            dw_table.name,
                                            schema=dw_table.schema,
                                            database=dw_table.database
                                        )
                                        col_names = [c["name"] for c in columns[:10]]
                                        col_desc = f"Colunas: {', '.join(col_names)}"
                                        if len(columns) > 10:
                                            col_desc += f" (+{len(columns) - 10} mais)"
                                    except Exception:
                                        col_desc = ""

                                    table_meta = TableMetadata(
                                        name=dw_table.name,
                                        database=dw_table.database or wh_database,
                                        schema=dw_table.schema or wh_schema,
                                        description=f"Tabela do {selected_wh}. {col_desc}",
                                        tags=[selected_wh, dw_table.table_type.lower()]
                                    )
                                    st.session_state.rag_catalog.append(table_meta)
                                    _index_table_if_possible(table_meta)

                                tables_added = len(dw_tables)
                                dw_connector.disconnect()
                                status.update(label=f"Sincronizadas {tables_added} tabelas!", state="complete")
                        else:
                            st.error("Não foi possível criar conexão com o Data Warehouse.")
                    except Exception as exc:
                        st.error(f"Erro ao sincronizar com Data Warehouse: {exc}")
                elif mode == "Conectores disponíveis" and connector == "OpenMetadata":
                    try:
                        _apply_connection_settings(
                            {
                                "openmetadata_host": om_host.strip(),
                                "openmetadata_api_token": om_token.strip(),
                            }
                        )
                        tables = _sync_openmetadata_tables(
                            om_host,
                            om_token,
                            service_filter,
                            table_limit,
                        )
                        tables_added = len(tables)
                    except OpenMetadataConnectorError as exc:
                        st.error(f"Falha ao sincronizar com o OpenMetadata: {exc}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Erro inesperado ao ler o OpenMetadata: {exc}")

                source_label = "Arquivo" if uploaded else (selected_wh if mode == "Data Warehouse conectado" else connector)
                st.session_state.connected_catalogs.append(
                    {
                        "name": catalog_label,
                        "source": source_label,
                        "tables": tables_added,
                        "scope": dataset_hint if mode != "Arquivo de catálogo" else "",
                    }
                )
                if tables_added:
                    st.success(
                        f"Catálogo '{catalog_label}' conectado e {tables_added} tabelas adicionadas."
                    )
                else:
                    st.info(
                        f"Catálogo '{catalog_label}' conectado. Adicione tabelas manualmente ou sincronize depois."
                    )

        with st.expander("Adicionar tabela manualmente"):
            with st.form("table_form"):
                name = st.text_input("Nome da tabela", placeholder="customers")
                database = st.text_input("Banco de dados", placeholder="production")
                schema = st.text_input("Schema", placeholder="public")
                description = st.text_area(
                    "Descrição",
                    placeholder=(
                        "Dados de clientes, incluindo status de assinatura e país."
                    ),
                )
                tags = st.multiselect(
                    "Tags",
                    options=[
                        "pii",
                        "critical",
                        "finance",
                        "marketing",
                        "raw",
                        "silver",
                        "gold",
                    ],
                    default=["critical"],
                )
                add_table = st.form_submit_button("Adicionar ao catálogo")

            if add_table and name:
                metadata = TableMetadata(
                    name=name,
                    database=database,
                    schema=schema,
                    description=description,
                    tags=tags,
                )
                st.session_state.rag_catalog.append(metadata)
                _index_table_if_possible(metadata)
                st.success(f"Tabela {name} adicionada ao catálogo")

        _render_connected_catalogs()
        _render_table_catalog(st.session_state.get("rag_catalog", []))

    with chat_tab:
        _render_connected_catalogs()
        st.divider()

        for message in st.session_state.discovery_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Pergunte sobre seus dados")
        if prompt:
            st.session_state.discovery_messages.append({"role": "user", "content": prompt})
            answer = _answer_discovery_question(prompt)
            st.session_state.discovery_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.experimental_rerun()


def _initialize_enrichment_agent():
    """Initialize the Metadata Enrichment Agent."""
    if not ENRICHMENT_AVAILABLE:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        embedding_provider = SentenceTransformerEmbeddings()
        llm_provider = OpenAILLM(model="gpt-4o-mini")
        vector_store = ChromaStore(
            collection_name="metadata_standards",
            persist_directory=str(BASE_DIR / "metadata_enrichment" / ".chroma_standards")
        )

        return MetadataEnrichmentAgent(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_store=vector_store,
            standards_persist_dir=str(BASE_DIR / "metadata_enrichment" / ".standards_index")
        )
    except Exception:
        return None


def render_enrichment_tab() -> None:
    """UI for the Metadata Enrichment Agent."""
    st.subheader("🏷️ Metadata Enrichment Agent")
    st.markdown(
        "Gere automaticamente descrições, tags e classificações para suas tabelas de dados. "
        "O agente analisa amostras de dados e usa padrões de arquitetura para inferir metadados."
    )

    if not ENRICHMENT_AVAILABLE:
        st.error(
            "⚠️ Metadata Enrichment Agent não disponível. "
            "Verifique se as dependências estão instaladas: `pip install -r metadata_enrichment/requirements.txt`"
        )
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ Configure a `OPENAI_API_KEY` para usar o agente de enriquecimento.")
        return

    # Initialize agent in session state
    if "enrichment_agent" not in st.session_state:
        with st.spinner("Inicializando agente de enriquecimento..."):
            st.session_state.enrichment_agent = _initialize_enrichment_agent()

    agent = st.session_state.get("enrichment_agent")
    if not agent:
        st.error("Não foi possível inicializar o agente. Verifique as configurações.")
        return

    # Standards management
    with st.expander("📚 Gerenciar Normativos e Padrões"):
        st.markdown(
            "Indexe documentos de padrões de nomenclatura, glossário de negócios e políticas de classificação."
        )

        standards_file = st.file_uploader(
            "Carregar arquivo de normativos (JSON)",
            type=["json"],
            help="Arquivo JSON com padrões de nomenclatura, glossário, etc.",
            key="standards_upload"
        )

        if standards_file and st.button("Indexar Normativos", key="index_standards"):
            with st.spinner("Indexando normativos..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
                    f.write(standards_file.read())
                    temp_path = f.name

                try:
                    count = agent.index_standards_from_json(temp_path)
                    st.success(f"✓ {count} documentos indexados com sucesso!")
                except Exception as exc:
                    st.error(f"Erro ao indexar: {exc}")
                finally:
                    os.unlink(temp_path)

        stats = agent.get_statistics()
        st.caption(f"📊 {stats.get('standards', {}).get('total_documents', 0)} normativos indexados")

    st.divider()

    # Data source selection
    st.markdown("### Enriquecer Metadados")

    # Use the warehouse selector component
    source_config = render_warehouse_selector("enrichment")

    uploaded_file = None
    file_type = None
    encoding = "utf-8"
    separator = ","

    if source_config["source_type"] == "file":
        file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"], key="enrich_file_type")

        uploaded_file = st.file_uploader(
            f"Selecione o arquivo {file_type}",
            type=["csv"] if file_type == "CSV" else ["parquet"],
            key="enrich_file"
        )

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.number_input(
            "Tamanho da amostra",
            min_value=10,
            max_value=1000,
            value=100,
            key="enrich_sample_size"
        )
    with col2:
        if source_config["source_type"] == "file" and file_type == "CSV":
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], key="csv_encoding")
            separator = st.selectbox("Separador", [",", ";", "|"], key="csv_sep")

    additional_context = st.text_area(
        "Contexto adicional (opcional)",
        placeholder="Informações adicionais sobre a tabela, domínio de negócio, etc.",
        key="enrich_context"
    )

    # Determine if we can run
    can_run = False
    if source_config["source_type"] == "file" and uploaded_file:
        can_run = True
    elif source_config["source_type"] == "warehouse" and source_config.get("table"):
        can_run = True

    if can_run and st.button("🚀 Gerar Metadados", type="primary", key="run_enrichment"):
        with st.spinner("Analisando dados e gerando metadados..."):
            try:
                if source_config["source_type"] == "file":
                    # File-based enrichment
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type.lower()}") as f:
                        f.write(uploaded_file.read())
                        temp_path = f.name

                    try:
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
                    finally:
                        os.unlink(temp_path)
                else:
                    # Warehouse-based enrichment
                    connector = get_warehouse_connector(
                        source_config["warehouse"],
                        source_config["warehouse_config"]
                    )

                    if connector:
                        with st.status("Conectando ao Data Warehouse...", expanded=True) as status:
                            status.write(f"Conectando ao {source_config['warehouse']}...")
                            connector.connect()
                            status.write(f"Lendo amostra de {source_config['table']}...")

                            sample_result = connector.read_sample(
                                source_config["table"],
                                schema=source_config.get("schema"),
                                database=source_config.get("database"),
                                n_rows=sample_size
                            )

                            status.write(f"Lidas {sample_result.row_count} linhas. Enriquecendo metadados...")

                            import pandas as pd
                            df = pd.DataFrame(sample_result.rows)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
                                df.to_parquet(f.name)
                                temp_path = f.name

                            try:
                                # Add table context
                                table_context = f"Tabela: {source_config['table']} do {source_config['warehouse']}"
                                if additional_context:
                                    table_context = f"{table_context}. {additional_context}"

                                result = agent.enrich_from_parquet(
                                    temp_path,
                                    sample_size=sample_size,
                                    additional_context=table_context
                                )
                            finally:
                                os.unlink(temp_path)

                            connector.disconnect()
                            status.update(label="Enriquecimento concluído!", state="complete")
                    else:
                        st.error("Não foi possível criar conexão com o Data Warehouse.")
                        return

                st.session_state.enrichment_result = result
                st.success("✓ Metadados gerados com sucesso!")

            except Exception as exc:
                st.error(f"Erro ao processar: {exc}")

    # Display results
    if "enrichment_result" in st.session_state:
        result = st.session_state.enrichment_result
        st.divider()

        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tabela", result.business_name)
        with col2:
            if result.has_pii:
                st.metric("PII", "⚠️ Detectado", delta=f"{len(result.pii_columns)} colunas")
            else:
                st.metric("PII", "✓ Não detectado")
        with col3:
            st.metric("Classificação", result.classification)
        with col4:
            st.metric("Confiança", f"{result.confidence:.0%}")

        # Description
        st.markdown("### Descrição")
        st.write(result.description)

        # Metadata
        st.markdown(f"**Domínio:** {result.domain} | **Tags:** {', '.join(result.tags)}")
        st.markdown(f"**Proprietário sugerido:** {result.owner_suggestion}")

        # Columns table
        st.markdown("### Colunas")
        columns_data = []
        for col in result.columns:
            pii_marker = "⚠️" if col.is_pii else ""
            columns_data.append({
                "Coluna": f"{col.name} {pii_marker}",
                "Tipo": col.original_type,
                "Descrição": col.description[:60] + "..." if len(col.description) > 60 else col.description,
                "Classificação": col.classification,
                "Tags": ", ".join(col.tags[:3]) if col.tags else "-"
            })

        st.dataframe(columns_data, use_container_width=True, hide_index=True)

        # PII Warning
        if result.pii_columns:
            st.warning(f"⚠️ Colunas com dados pessoais (PII): {', '.join(result.pii_columns)}")

        # Export
        st.markdown("### Exportar")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Baixar JSON",
                result.to_json(),
                file_name=f"{result.table_name}_metadata.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                "📥 Baixar Markdown",
                result.to_markdown(),
                file_name=f"{result.table_name}_metadata.md",
                mime="text/markdown"
            )


def _initialize_quality_agent():
    """Initialize the Data Quality Agent."""
    if not QUALITY_AVAILABLE:
        return None

    try:
        return DataQualityAgent(
            persist_dir=str(BASE_DIR / "data_quality" / ".quality_data"),
            enable_schema_tracking=True
        )
    except Exception:
        return None


def render_quality_tab() -> None:
    """UI for the Data Quality Agent."""
    st.subheader("📊 Data Quality Agent")
    st.markdown(
        "Monitore métricas de qualidade de dados: completude, unicidade, validade, "
        "consistência, freshness (SLA) e detecção de schema drift."
    )

    if not QUALITY_AVAILABLE:
        st.error(
            "⚠️ Data Quality Agent não disponível. "
            "Verifique se as dependências estão instaladas: `pip install -r data_quality/requirements.txt`"
        )
        return

    # Initialize agent in session state
    if "quality_agent" not in st.session_state:
        with st.spinner("Inicializando agente de qualidade..."):
            st.session_state.quality_agent = _initialize_quality_agent()

    agent = st.session_state.get("quality_agent")
    if not agent:
        st.error("Não foi possível inicializar o agente.")
        return

    # Data source selection
    st.markdown("### Avaliar Qualidade")

    # Use the warehouse selector component
    source_config = render_warehouse_selector("quality")

    uploaded_file = None
    file_type = None

    if source_config["source_type"] == "file":
        # File upload mode
        file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"], key="quality_file_type")

        uploaded_file = st.file_uploader(
            f"Selecione o arquivo {file_type}",
            type=["csv"] if file_type == "CSV" else ["parquet"],
            key="quality_file"
        )

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.number_input(
            "Tamanho da amostra",
            min_value=100,
            max_value=100000,
            value=10000,
            key="quality_sample_size"
        )

    # Freshness config
    with st.expander("⏱️ Configuração de Freshness/SLA"):
        check_freshness = st.checkbox("Verificar freshness", value=False, key="check_fresh")
        if check_freshness:
            timestamp_col = st.text_input(
                "Coluna de timestamp",
                placeholder="updated_at",
                key="ts_col"
            )
            sla_hours = st.number_input("SLA (horas)", min_value=1, max_value=168, value=24, key="sla_h")

    # Determine if we can run analysis
    can_run = False
    if source_config["source_type"] == "file" and uploaded_file:
        can_run = True
    elif source_config["source_type"] == "warehouse" and source_config.get("table"):
        can_run = True

    if can_run and st.button("🚀 Avaliar Qualidade", type="primary", key="run_quality"):
        with st.spinner("Analisando dados..."):
            try:
                kwargs = {"sample_size": sample_size}

                if check_freshness and timestamp_col:
                    kwargs["freshness_config"] = {
                        "timestamp_column": timestamp_col,
                        "sla_hours": sla_hours,
                        "max_age_hours": sla_hours * 2
                    }

                if source_config["source_type"] == "file":
                    # File-based analysis
                    suffix = ".csv" if file_type == "CSV" else ".parquet"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(uploaded_file.read())
                        temp_path = f.name

                    try:
                        report = agent.evaluate_file(temp_path, **kwargs)
                    finally:
                        os.unlink(temp_path)
                else:
                    # Warehouse-based analysis
                    connector = get_warehouse_connector(
                        source_config["warehouse"],
                        source_config["warehouse_config"]
                    )

                    if connector:
                        with st.status("Conectando ao Data Warehouse...", expanded=True) as status:
                            status.write(f"Conectando ao {source_config['warehouse']}...")
                            connector.connect()
                            status.write(f"Lendo amostra de {source_config['table']}...")

                            # Read sample data to temp file for analysis
                            result = connector.read_sample(
                                source_config["table"],
                                schema=source_config.get("schema"),
                                database=source_config.get("database"),
                                n_rows=sample_size
                            )

                            status.write(f"Lidas {result.row_count} linhas. Analisando qualidade...")

                            # Convert to DataFrame and save as temp parquet
                            import pandas as pd
                            df = pd.DataFrame(result.rows)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
                                df.to_parquet(f.name)
                                temp_path = f.name

                            try:
                                report = agent.evaluate_file(temp_path, **kwargs)
                                # Update source name to reflect warehouse table
                                report.source_name = f"{source_config['warehouse']}.{source_config.get('schema', '')}.{source_config['table']}"
                            finally:
                                os.unlink(temp_path)

                            connector.disconnect()
                            status.update(label="Análise concluída!", state="complete")
                    else:
                        st.error("Não foi possível criar conexão com o Data Warehouse.")
                        return

                st.session_state.quality_report = report
                st.success(f"✓ Análise concluída em {report.processing_time_ms}ms")

            except Exception as exc:
                st.error(f"Erro: {exc}")

    # Display results
    if "quality_report" in st.session_state:
        report = st.session_state.quality_report
        st.divider()

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score_color = "#22c55e" if report.overall_status == "passed" else "#f59e0b" if report.overall_status == "warning" else "#ef4444"
            st.metric("Score Geral", f"{report.overall_score:.0%}")
        with col2:
            status_emoji = "✅" if report.overall_status == "passed" else "⚠️" if report.overall_status == "warning" else "❌"
            st.metric("Status", f"{status_emoji} {report.overall_status.upper()}")
        with col3:
            st.metric("Linhas", f"{report.row_count:,}")
        with col4:
            st.metric("Colunas", report.columns_checked)

        # Dimensions
        st.markdown("### Dimensões de Qualidade")
        dim_cols = st.columns(len(report.dimensions))
        for i, (dim_name, dim_data) in enumerate(report.dimensions.items()):
            with dim_cols[i]:
                score = dim_data.get("score", 0)
                status = dim_data.get("status", "unknown")
                icon = "✅" if status == "passed" else "⚠️" if status == "warning" else "❌"
                st.metric(dim_name.capitalize(), f"{icon} {score:.0%}")

        # Alerts
        if report.alerts:
            st.markdown("### Alertas")
            for alert in report.alerts:
                level = alert.get("level", "info")
                if level == "critical":
                    st.error(f"🔴 **{alert.get('rule_name')}**: {alert.get('message')}")
                elif level == "warning":
                    st.warning(f"🟡 **{alert.get('rule_name')}**: {alert.get('message')}")
                else:
                    st.info(f"🔵 **{alert.get('rule_name')}**: {alert.get('message')}")

        # Schema drift
        if report.schema_drift and report.schema_drift.get("has_drift"):
            st.markdown("### Schema Drift Detectado")
            st.warning(report.schema_drift.get("summary"))
            for change in report.schema_drift.get("changes", [])[:5]:
                severity = change.get("severity", "info")
                icon = "🔴" if severity == "critical" else "🟡"
                st.markdown(f"- {icon} {change.get('message')}")

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


def _initialize_classification_agent():
    """Initialize the Data Classification Agent."""
    if not CLASSIFICATION_AVAILABLE:
        return None

    try:
        return DataClassificationAgent()
    except Exception:
        return None


def render_classification_tab() -> None:
    """UI for the Data Classification Agent."""
    st.subheader("🛡️ Data Classification Agent")
    st.markdown(
        "Classifique dados automaticamente por níveis de sensibilidade: "
        "PII (dados pessoais), PHI (dados de saúde), PCI (dados de pagamento) e dados financeiros."
    )

    if not CLASSIFICATION_AVAILABLE:
        st.error(
            "⚠️ Data Classification Agent não disponível. "
            "Verifique se as dependências estão instaladas: `pip install -r data_classification/requirements.txt`"
        )
        return

    # Initialize agent in session state
    if "classification_agent" not in st.session_state:
        with st.spinner("Inicializando agente de classificação..."):
            st.session_state.classification_agent = _initialize_classification_agent()

    agent = st.session_state.get("classification_agent")
    if not agent:
        st.error("Não foi possível inicializar o agente.")
        return

    # Data source selection
    st.markdown("### Classificar Dados")

    # Use the warehouse selector component
    source_config = render_warehouse_selector("classification")

    uploaded_file = None
    file_type = None
    encoding = "utf-8"
    separator = ","

    if source_config["source_type"] == "file":
        file_type = st.selectbox("Tipo de arquivo", ["CSV", "Parquet"], key="classification_file_type")

        uploaded_file = st.file_uploader(
            f"Selecione o arquivo {file_type}",
            type=["csv"] if file_type == "CSV" else ["parquet"],
            key="classification_file"
        )

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.number_input(
            "Tamanho da amostra",
            min_value=100,
            max_value=100000,
            value=1000,
            key="classification_sample_size"
        )

    if source_config["source_type"] == "file" and file_type == "CSV":
        with col2:
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], key="class_encoding")
            separator = st.selectbox("Separador", [",", ";", "|"], key="class_sep")

    # Determine if we can run analysis
    can_run = False
    if source_config["source_type"] == "file" and uploaded_file:
        can_run = True
    elif source_config["source_type"] == "warehouse" and source_config.get("table"):
        can_run = True

    if can_run and st.button("🔍 Classificar Dados", type="primary", key="run_classification"):
        with st.spinner("Analisando dados e detectando padrões sensíveis..."):
            try:
                if source_config["source_type"] == "file":
                    # File-based classification
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
                    finally:
                        os.unlink(temp_path)
                else:
                    # Warehouse-based classification
                    connector = get_warehouse_connector(
                        source_config["warehouse"],
                        source_config["warehouse_config"]
                    )

                    if connector:
                        with st.status("Conectando ao Data Warehouse...", expanded=True) as status:
                            status.write(f"Conectando ao {source_config['warehouse']}...")
                            connector.connect()
                            status.write(f"Lendo amostra de {source_config['table']}...")

                            result = connector.read_sample(
                                source_config["table"],
                                schema=source_config.get("schema"),
                                database=source_config.get("database"),
                                n_rows=sample_size
                            )

                            status.write(f"Lidas {result.row_count} linhas. Classificando...")

                            import pandas as pd
                            df = pd.DataFrame(result.rows)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
                                df.to_parquet(f.name)
                                temp_path = f.name

                            try:
                                report = agent.classify_from_parquet(temp_path, sample_size=sample_size)
                                report.source_name = f"{source_config['warehouse']}.{source_config.get('schema', '')}.{source_config['table']}"
                            finally:
                                os.unlink(temp_path)

                            connector.disconnect()
                            status.update(label="Classificação concluída!", state="complete")
                    else:
                        st.error("Não foi possível criar conexão com o Data Warehouse.")
                        return

                st.session_state.classification_report = report
                st.success("✅ Classificação concluída!")

            except Exception as exc:
                st.error(f"Erro ao classificar: {exc}")

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


def _initialize_value_agent():
    """Initialize the Data Asset Value Agent."""
    if not VALUE_AVAILABLE:
        return None

    try:
        return DataAssetValueAgent(
            weights={
                'usage': 0.30,
                'joins': 0.25,
                'lineage': 0.25,
                'data_product': 0.20
            },
            time_range_days=30,
            persist_dir=str(BASE_DIR / "data_asset_value" / ".value_data")
        )
    except Exception:
        return None


def render_value_tab() -> None:
    """UI for the Data Asset Value Agent."""
    st.subheader("💎 Data Asset Value Scanner")
    st.markdown(
        "Analise o valor dos ativos de dados baseado em uso, JOINs, linhagem e impacto em data products. "
        "Identifique ativos críticos, hubs de dados e ativos subutilizados."
    )

    if not VALUE_AVAILABLE:
        st.error(
            "⚠️ Data Asset Value Agent não disponível. "
            "Verifique se as dependências estão instaladas."
        )
        return

    # Initialize agent in session state
    if "value_agent" not in st.session_state:
        with st.spinner("Inicializando agente de valor..."):
            st.session_state.value_agent = _initialize_value_agent()

    agent = st.session_state.get("value_agent")
    if not agent:
        st.error("Não foi possível inicializar o agente.")
        return

    # Check for configured warehouses
    configured_warehouses = list(st.session_state.get("warehouse_connections", {}).keys())

    # Source selection
    st.markdown("### Fonte dos Logs de Queries")

    source_options = ["Arquivo JSON enviado", "Dados de exemplo"]
    if configured_warehouses:
        source_options.insert(1, "Query History do Data Warehouse")

    value_source = st.radio(
        "Selecione a fonte dos logs de queries",
        source_options,
        horizontal=True,
        key="value_source_mode"
    )

    # File upload section
    st.markdown("### Analisar Valor dos Ativos")

    query_logs = None
    query_logs_file = None
    data_products_file = None
    asset_metadata_file = None
    lineage_output_file = None
    dw_query_logs = None

    if value_source == "Query History do Data Warehouse":
        st.markdown("**Extrair histórico de queries diretamente do Data Warehouse**")

        selected_wh = st.selectbox(
            "Data Warehouse",
            configured_warehouses,
            key="value_dw_select"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            dw_days = st.number_input(
                "Período (dias)",
                min_value=1,
                max_value=90,
                value=30,
                key="value_dw_days"
            )
        with col2:
            dw_limit = st.number_input(
                "Limite de queries",
                min_value=100,
                max_value=10000,
                value=1000,
                key="value_dw_limit"
            )
        with col3:
            dw_user_filter = st.text_input(
                "Filtrar por usuário (opcional)",
                key="value_dw_user"
            )

        col4, col5 = st.columns(2)
        with col4:
            dw_database_filter = st.text_input(
                "Filtrar por database (opcional)",
                key="value_dw_database"
            )
        with col5:
            dw_table_filter = st.text_input(
                "Filtrar por tabela (opcional)",
                key="value_dw_table"
            )

        # Additional optional files
        st.markdown("**Arquivos adicionais (opcional)**")
        col_dp, col_meta, col_lin = st.columns(3)

        with col_dp:
            data_products_file = st.file_uploader(
                "📦 Config Data Products",
                type=["json"],
                help="Configuração de data products",
                key="value_dw_data_products"
            )

        with col_meta:
            asset_metadata_file = st.file_uploader(
                "📊 Metadados de Ativos",
                type=["json"],
                help="Metadados adicionais",
                key="value_dw_asset_metadata"
            )

        with col_lin:
            lineage_output_file = st.file_uploader(
                "🔗 Saída do Lineage",
                type=["json"],
                help="Output do Lineage Agent",
                key="value_dw_lineage_output"
            )

        if st.button("🚀 Extrair Logs e Analisar Valor", type="primary", key="run_dw_value_analysis"):
            if selected_wh:
                wh_config = st.session_state.warehouse_connections.get(selected_wh, {})

                with st.status("Conectando ao Data Warehouse...", expanded=True) as status:
                    try:
                        dw_connector = get_warehouse_connector(selected_wh, wh_config)
                        dw_connector.connect()
                        status.write(f"Conectado a {selected_wh}")

                        status.write("Extraindo histórico de queries...")
                        query_history = dw_connector.get_query_history(
                            days=dw_days,
                            limit=dw_limit,
                            database_filter=dw_database_filter or None,
                            table_filter=dw_table_filter or None,
                            user_filter=dw_user_filter or None
                        )

                        status.write(f"{len(query_history)} queries encontradas")

                        # Convert to format expected by the agent
                        dw_query_logs = []
                        for qh in query_history:
                            if qh.get("query_text"):
                                dw_query_logs.append({
                                    "query_id": qh.get("query_id"),
                                    "query_text": qh.get("query_text"),
                                    "user_name": qh.get("user_name"),
                                    "timestamp": str(qh.get("start_time")) if qh.get("start_time") else None,
                                    "execution_time_ms": qh.get("execution_time_ms"),
                                    "rows_produced": qh.get("rows_produced"),
                                    "bytes_scanned": qh.get("bytes_scanned"),
                                    "status": qh.get("status", "success")
                                })

                        if not dw_query_logs:
                            st.warning("Nenhuma query válida encontrada no período.")
                        else:
                            # Load optional files
                            data_product_config = None
                            if data_products_file:
                                data_product_config = json.load(data_products_file)

                            asset_metadata = None
                            if asset_metadata_file:
                                asset_metadata = json.load(asset_metadata_file)

                            lineage_data = None
                            if lineage_output_file:
                                lineage_data = json.load(lineage_output_file)

                            status.write("Analisando valor dos ativos...")

                            report = agent.analyze_from_query_logs(
                                query_logs=dw_query_logs,
                                lineage_data=lineage_data,
                                data_product_config=data_product_config,
                                asset_metadata=asset_metadata,
                                time_range_days=dw_days
                            )

                            st.session_state.value_report = report
                            status.update(label="Análise concluída", state="complete")
                            st.success(f"✅ Análise concluída: {report.assets_analyzed} ativos analisados")

                        dw_connector.disconnect()

                    except Exception as e:
                        st.error(f"Erro ao conectar ao Data Warehouse: {e}")

    elif value_source == "Arquivo JSON enviado":
        col1, col2 = st.columns(2)

        with col1:
            query_logs_file = st.file_uploader(
                "📋 Logs de Queries (JSON)",
                type=["json"],
                help="Arquivo JSON com logs de queries SQL executadas",
                key="value_query_logs"
            )

        with col2:
            data_products_file = st.file_uploader(
                "📦 Config Data Products (JSON, opcional)",
                type=["json"],
                help="Configuração de data products com impacto de negócio",
                key="value_data_products"
            )

        col3, col4 = st.columns(2)

        with col3:
            asset_metadata_file = st.file_uploader(
                "📊 Metadados de Ativos (JSON, opcional)",
                type=["json"],
                help="Metadados adicionais: criticidade, custo, risco",
                key="value_asset_metadata"
            )

        with col4:
            lineage_output_file = st.file_uploader(
                "🔗 Saída do Lineage Agent (JSON, opcional)",
                type=["json"],
                help="Output do Data Lineage Agent para análise de impacto",
                key="value_lineage_output"
            )

        time_range = st.slider(
            "Período de análise (dias)",
            min_value=7,
            max_value=90,
            value=30,
            key="value_time_range"
        )

        if st.button("🚀 Analisar Valor", type="primary", key="run_value_analysis"):
            if not query_logs_file:
                st.warning("Envie um arquivo de logs de queries.")
                return

            with st.spinner("Analisando valor dos ativos..."):
                try:
                    query_logs = json.load(query_logs_file)

                    data_product_config = None
                    if data_products_file:
                        data_product_config = json.load(data_products_file)

                    asset_metadata = None
                    if asset_metadata_file:
                        asset_metadata = json.load(asset_metadata_file)

                    lineage_data = None
                    if lineage_output_file:
                        lineage_data = json.load(lineage_output_file)

                    report = agent.analyze_from_query_logs(
                        query_logs=query_logs,
                        lineage_data=lineage_data,
                        data_product_config=data_product_config,
                        asset_metadata=asset_metadata,
                        time_range_days=time_range
                    )

                    st.session_state.value_report = report
                    st.success(f"✅ Análise concluída: {report.assets_analyzed} ativos analisados")

                except Exception as exc:
                    st.error(f"Erro ao analisar: {exc}")

    elif value_source == "Dados de exemplo":
        st.info("Usando dados de exemplo para demonstração.")

        time_range = st.slider(
            "Período de análise (dias)",
            min_value=7,
            max_value=90,
            value=30,
            key="value_time_range_sample"
        )

        if st.button("🚀 Analisar com Dados de Exemplo", type="primary", key="run_sample_value_analysis"):
            with st.spinner("Analisando valor dos ativos..."):
                try:
                    sample_path = BASE_DIR / "data_asset_value" / "examples" / "sample_query_logs.json"
                    with open(sample_path, 'r', encoding='utf-8') as f:
                        query_logs = json.load(f)

                    dp_path = BASE_DIR / "data_asset_value" / "examples" / "data_products_config.json"
                    with open(dp_path, 'r', encoding='utf-8') as f:
                        data_product_config = json.load(f)

                    meta_path = BASE_DIR / "data_asset_value" / "examples" / "asset_metadata.json"
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        asset_metadata = json.load(f)

                    report = agent.analyze_from_query_logs(
                        query_logs=query_logs,
                        lineage_data=None,
                        data_product_config=data_product_config,
                        asset_metadata=asset_metadata,
                        time_range_days=time_range
                    )

                    st.session_state.value_report = report
                    st.success(f"✅ Análise concluída: {report.assets_analyzed} ativos analisados")

                except Exception as exc:
                    st.error(f"Erro ao analisar: {exc}")

    # Display results
    if "value_report" in st.session_state:
        report = st.session_state.value_report
        st.divider()

        # Summary metrics
        summary = report.summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Ativos Analisados", summary.get("total_assets", 0))
        with col2:
            st.metric("Ativos Críticos", summary.get("critical_assets_count", 0))
        with col3:
            st.metric("Alto Valor", summary.get("high_value_assets_count", 0))
        with col4:
            st.metric("Score Médio", f"{summary.get('average_value_score', 0):.1f}")

        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric("Órfãos", summary.get("orphan_assets_count", 0))
        with col6:
            st.metric("Em Declínio", summary.get("declining_assets_count", 0))
        with col7:
            st.metric("JOINs", summary.get("total_join_relationships", 0))
        with col8:
            st.metric("Data Products", summary.get("data_products_analyzed", 0))

        # Top Value Assets
        st.markdown("### 🏆 Top 10 Ativos por Valor")

        top_assets = []
        for score in report.asset_scores[:10]:
            category_icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
                "unknown": "⚪"
            }
            icon = category_icons.get(score.value_category, "⚪")

            trend_icons = {
                "increasing": "📈",
                "stable": "➡️",
                "decreasing": "📉"
            }
            trend = trend_icons.get(score.access_trend, "➡️")

            top_assets.append({
                "Ativo": score.asset_name,
                "Valor": f"{score.overall_value_score:.1f}",
                "Categoria": f"{icon} {score.value_category}",
                "Uso": f"{score.usage_score:.1f}",
                "JOINs": f"{score.join_score:.1f}",
                "Linhagem": f"{score.lineage_score:.1f}",
                "Data Products": f"{score.data_product_score:.1f}",
                "Impacto": f"{score.business_impact_score:.1f}",
                "Tendência": trend,
                "Queries": score.total_queries,
                "Usuários": score.unique_users
            })

        st.dataframe(top_assets, use_container_width=True, hide_index=True)

        # Tabs for different views
        insights_tab, hub_tab, orphan_tab, recommendations_tab = st.tabs([
            "📊 Insights",
            "🔗 Hub Assets",
            "⚠️ Órfãos/Declínio",
            "💡 Recomendações"
        ])

        with insights_tab:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Ativos Críticos")
                if report.critical_assets:
                    for asset in report.critical_assets[:10]:
                        st.markdown(f"- `{asset}`")
                else:
                    st.info("Nenhum ativo crítico identificado")

            with col2:
                st.markdown("#### Top Value por Data Product")
                if report.data_product_impacts:
                    dp_assets = {}
                    for impact in report.data_product_impacts:
                        if impact.data_product_name not in dp_assets:
                            dp_assets[impact.data_product_name] = []
                        dp_assets[impact.data_product_name].append(impact.asset_name)

                    for dp, assets in list(dp_assets.items())[:5]:
                        st.markdown(f"**{dp}**: {', '.join(assets[:3])}")
                else:
                    st.info("Sem data products configurados")

        with hub_tab:
            st.markdown("#### Hub Assets (Alta Conectividade)")
            st.markdown("Ativos com muitos relacionamentos de JOIN - pontos críticos de integração.")

            if report.hub_assets:
                for asset in report.hub_assets:
                    score = agent.get_asset_value(asset, report)
                    if score:
                        st.markdown(
                            f"- **{asset}**: {score.join_count} JOINs, "
                            f"conectado a {len([j for j in report.join_relationships if j.left_asset == asset or j.right_asset == asset])} ativos"
                        )
            else:
                st.info("Nenhum hub asset identificado")

            # Join relationships visualization
            if report.join_relationships:
                st.markdown("#### Relacionamentos de JOIN Frequentes")
                join_data = []
                for j in sorted(report.join_relationships, key=lambda x: x.frequency, reverse=True)[:15]:
                    join_data.append({
                        "Esquerda": j.left_asset,
                        "Direita": j.right_asset,
                        "Tipo": j.join_type,
                        "Frequência": j.frequency,
                        "Data Products": ", ".join(j.data_products[:2]) if j.data_products else "-"
                    })
                st.dataframe(join_data, use_container_width=True, hide_index=True)

        with orphan_tab:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Ativos Órfãos (Sem Uso)")
                if report.orphan_assets:
                    st.warning(f"{len(report.orphan_assets)} ativos sem uso detectado")
                    for asset in report.orphan_assets[:10]:
                        st.markdown(f"- `{asset}`")
                else:
                    st.success("Todos os ativos têm uso registrado")

            with col2:
                st.markdown("#### Ativos em Declínio")
                if report.declining_assets:
                    st.warning(f"{len(report.declining_assets)} ativos com uso em queda")
                    for asset in report.declining_assets[:10]:
                        score = agent.get_asset_value(asset, report)
                        if score:
                            st.markdown(f"- `{asset}` (score: {score.overall_value_score:.1f})")
                else:
                    st.success("Nenhum ativo com uso em declínio")

        with recommendations_tab:
            st.markdown("#### Recomendações de Governança")
            if report.recommendations:
                for i, rec in enumerate(report.recommendations, 1):
                    st.info(f"**{i}.** {rec}")
            else:
                st.success("Nenhuma recomendação específica no momento")

            # Asset-level recommendations
            st.markdown("#### Recomendações por Ativo")
            assets_with_recs = [s for s in report.asset_scores if s.recommendations]
            if assets_with_recs:
                for score in assets_with_recs[:5]:
                    with st.expander(f"{score.asset_name} ({score.value_category})"):
                        for rec in score.recommendations:
                            st.markdown(f"- {rec}")
            else:
                st.info("Sem recomendações específicas por ativo")

        # Export
        st.divider()
        st.markdown("### Exportar Relatório")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📥 JSON",
                report.to_json(),
                file_name="asset_value_report.json",
                mime="application/json"
            )

        with col2:
            st.download_button(
                "📥 Markdown",
                report.to_markdown(),
                file_name="asset_value_report.md",
                mime="text/markdown"
            )


def render_ner_tab() -> None:
    """UI for the Sensitive Data NER Agent."""
    st.subheader("🔒 Sensitive Data NER Agent")
    st.markdown(
        "Detecte e anonimize dados sensíveis em texto livre antes de enviar para LLMs. "
        "Proteja PII, PHI, PCI, dados financeiros, informações estratégicas e **credenciais (API keys, tokens, senhas)**."
    )

    if not NER_AVAILABLE:
        st.error(
            "⚠️ Sensitive Data NER Agent não disponível. "
            "Verifique se as dependências estão instaladas."
        )
        return

    # Sidebar-like configuration in expander
    with st.expander("⚙️ Configurações de Política", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ações por Categoria**")
            pii_action = st.selectbox(
                "PII",
                options=["anonymize", "block", "warn", "allow"],
                index=0,
                key="ner_pii_action"
            )
            phi_action = st.selectbox(
                "PHI (Saúde)",
                options=["block", "anonymize", "warn", "allow"],
                index=0,
                key="ner_phi_action"
            )
            pci_action = st.selectbox(
                "PCI (Cartões)",
                options=["block", "anonymize", "warn", "allow"],
                index=0,
                key="ner_pci_action"
            )
            financial_action = st.selectbox(
                "Financeiro",
                options=["anonymize", "block", "warn", "allow"],
                index=0,
                key="ner_financial_action"
            )
            business_action = st.selectbox(
                "Negócios",
                options=["block", "anonymize", "warn", "allow"],
                index=0,
                key="ner_business_action"
            )
            credentials_action = st.selectbox(
                "Credenciais (API Keys)",
                options=["block", "anonymize", "warn", "allow"],
                index=0,
                key="ner_credentials_action"
            )

        with col2:
            st.markdown("**Configurações Gerais**")
            anon_strategy = st.selectbox(
                "Estratégia de Anonimização",
                options=["redact", "mask", "hash", "partial", "pseudonymize"],
                index=0,
                key="ner_anon_strategy"
            )
            min_confidence = st.slider(
                "Confiança Mínima",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="ner_min_confidence"
            )
            strict_mode = st.checkbox(
                "Modo Estrito",
                value=False,
                help="Requer validação de checksum quando disponível",
                key="ner_strict_mode"
            )

        st.markdown("**Termos de Negócio Sensíveis**")
        business_terms_input = st.text_area(
            "Termos sensíveis (um por linha)",
            placeholder="Projeto Confidencial\nAquisição XYZ\nParceria ABC",
            height=80,
            key="ner_business_terms"
        )

    # Main input area
    st.markdown("### 📝 Texto para Análise")

    input_text = st.text_area(
        "Cole o texto que deseja analisar",
        height=200,
        placeholder="""Exemplo:
O cliente João da Silva, CPF 123.456.789-09, solicitou atualização.
Email: joao.silva@email.com, telefone (11) 98765-4321.

Diagnóstico: CID-10 J45.0 - Asma predominantemente alérgica.

Pagamento via cartão 4532 1234 5678 9010.

Referente ao Projeto Confidencial para aquisição da empresa XYZ.""",
        key="ner_input_text"
    )

    if st.button("🔍 Analisar e Proteger", type="primary", key="run_ner"):
        if not input_text:
            st.warning("Digite ou cole o texto para análise.")
            return

        with st.spinner("Analisando dados sensíveis..."):
            try:
                # Parse business terms
                business_terms = [
                    t.strip() for t in business_terms_input.split("\n")
                    if t.strip()
                ] if business_terms_input else None

                # Create policy
                policy = FilterPolicy(
                    pii_action=FilterAction(pii_action),
                    phi_action=FilterAction(phi_action),
                    pci_action=FilterAction(pci_action),
                    financial_action=FilterAction(financial_action),
                    business_action=FilterAction(business_action),
                    credentials_action=FilterAction(credentials_action),
                    min_confidence=min_confidence,
                    anonymization_strategy=AnonymizationStrategy(anon_strategy),
                )

                # Create agent and analyze
                agent = SensitiveDataNERAgent(
                    business_terms=business_terms,
                    filter_policy=policy,
                    strict_mode=strict_mode,
                )

                result = agent.analyze(input_text, anonymize=True)
                st.session_state.ner_result = result
                st.success("✅ Análise concluída!")

            except Exception as exc:
                st.error(f"Erro na análise: {exc}")

    # Display results
    if "ner_result" in st.session_state:
        result = st.session_state.ner_result
        st.divider()

        # Action badge
        action_colors = {
            FilterAction.ALLOW: ("🟢", "success"),
            FilterAction.WARN: ("🟡", "warning"),
            FilterAction.ANONYMIZE: ("🟠", "info"),
            FilterAction.BLOCK: ("🔴", "error"),
        }
        icon, _ = action_colors.get(result.filter_action, ("⚪", "info"))

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ação", f"{icon} {result.filter_action.value.upper()}")
        with col2:
            st.metric("Entidades", result.statistics["total"])
        with col3:
            st.metric("Score de Risco", f"{result.risk_score:.0%}")
        with col4:
            st.metric("Tempo", f"{result.processing_time_ms:.1f}ms")

        if result.blocked_reason:
            st.error(f"🚫 Motivo do bloqueio: {result.blocked_reason}")

        for warning in result.warnings:
            st.warning(warning)

        # Tabs for results
        entities_tab, anon_tab, stats_tab, json_tab = st.tabs([
            "📊 Entidades",
            "🔐 Texto Anonimizado",
            "📈 Estatísticas",
            "📋 JSON"
        ])

        with entities_tab:
            if result.entities:
                st.markdown("### Entidades Detectadas")

                entity_data = []
                for e in result.entities:
                    category_icons = {
                        EntityCategory.PII: "🔵",
                        EntityCategory.PHI: "🟢",
                        EntityCategory.PCI: "🟡",
                        EntityCategory.FINANCIAL: "🟠",
                        EntityCategory.BUSINESS: "🟣",
                        EntityCategory.CREDENTIALS: "🔴",
                    }
                    icon = category_icons.get(e.category, "⚪")

                    entity_data.append({
                        "Valor": e.value[:40] + "..." if len(e.value) > 40 else e.value,
                        "Tipo": e.entity_type,
                        "Categoria": f"{icon} {e.category.value.upper()}",
                        "Confiança": f"{e.confidence:.0%}",
                        "Validado": "✓" if e.is_validated else "✗",
                        "Linha": e.line_number,
                    })

                st.dataframe(entity_data, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Nenhuma entidade sensível detectada!")

        with anon_tab:
            if result.anonymized_text:
                st.markdown("### Texto Seguro para LLM")
                st.code(result.anonymized_text, language=None)

                st.download_button(
                    "📋 Baixar texto anonimizado",
                    result.anonymized_text,
                    file_name="texto_anonimizado.txt",
                    mime="text/plain"
                )
            else:
                st.info("Nenhuma anonimização necessária")

        with stats_tab:
            st.markdown("### Estatísticas por Categoria")

            cat_cols = st.columns(6)
            categories = [
                ("PII", result.statistics.get("pii", 0), "🔵"),
                ("PHI", result.statistics.get("phi", 0), "🟢"),
                ("PCI", result.statistics.get("pci", 0), "🟡"),
                ("Financeiro", result.statistics.get("financial", 0), "🟠"),
                ("Negócios", result.statistics.get("business", 0), "🟣"),
                ("Credenciais", result.statistics.get("credentials", 0), "🔴"),
            ]

            for i, (name, count, icon) in enumerate(categories):
                with cat_cols[i]:
                    st.metric(f"{icon} {name}", count)

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

        with json_tab:
            st.markdown("### Resultado em JSON")
            st.json(result.to_dict())

            st.download_button(
                "📥 Baixar JSON",
                result.to_json(),
                file_name="ner_analysis.json",
                mime="application/json"
            )


def render_vault_tab() -> None:
    """UI for the Secure Vault management."""
    st.subheader("🔐 Secure Vault")
    st.markdown(
        "Gerencie o cofre seguro para armazenamento criptografado de dados sensíveis. "
        "Permite armazenar mapeamentos originais/anonimizados e recuperar dados posteriormente."
    )

    if not VAULT_AVAILABLE:
        st.error(
            "⚠️ Secure Vault não disponível. "
            "Instale o pacote cryptography: `pip install cryptography`"
        )
        return

    # Initialize vault in session state
    if "vault" not in st.session_state:
        st.session_state.vault = None
        st.session_state.vault_token = None
        st.session_state.vault_user = None

    # Vault initialization section
    with st.expander("🔧 Configuração do Vault", expanded=st.session_state.vault is None):
        if st.session_state.vault is None:
            st.markdown("### Inicializar Vault")
            st.warning(
                "⚠️ **Importante**: A senha mestra é usada para derivar as chaves de criptografia. "
                "**Guarde-a em local seguro** - sem ela, os dados não podem ser recuperados."
            )

            col1, col2 = st.columns(2)
            with col1:
                master_password = st.text_input(
                    "Senha Mestra",
                    type="password",
                    key="vault_master_password",
                    help="Senha para derivar chaves de criptografia"
                )
            with col2:
                admin_password = st.text_input(
                    "Senha do Admin",
                    type="password",
                    key="vault_admin_password",
                    help="Senha do usuário administrador"
                )

            retention_mode = st.selectbox(
                "Política após decriptação",
                [
                    "Apagar imediatamente (padrão)",
                    "Reter por um período",
                    "Reter indefinidamente",
                ],
                help=(
                    "Escolha se as sessões devem ser excluídas após a decriptação "
                    "ou mantidas por um período configurável."
                ),
            )

            retention_days = None
            delete_after_decrypt = True

            if retention_mode == "Reter por um período":
                retention_days = int(
                    st.number_input(
                        "Dias de retenção após decriptar",
                        min_value=1,
                        max_value=3650,
                        value=30,
                        help="Quantidade de dias para manter a sessão decriptada antes de removê-la.",
                    )
                )
                delete_after_decrypt = False
            elif retention_mode == "Reter indefinidamente":
                delete_after_decrypt = False

            if st.button("🚀 Inicializar Vault", type="primary"):
                if not master_password or not admin_password:
                    st.error("Preencha ambas as senhas para inicializar o vault.")
                else:
                    try:
                        vault_config = VaultConfig(
                            storage_path=str(BASE_DIR / ".vault_data"),
                            require_authentication=True,
                            delete_after_decrypt=delete_after_decrypt,
                            decryption_retention_days=retention_days,
                        )
                        vault = SecureVault(vault_config)
                        vault.initialize(master_password)

                        # Create admin user and authenticate
                        try:
                            vault.create_user("admin", admin_password, AccessLevel.SUPER_ADMIN)
                        except ValueError:
                            pass  # User already exists

                        token = vault.authenticate("admin", admin_password)

                        st.session_state.vault = vault
                        st.session_state.vault_token = token
                        st.session_state.vault_user = "admin"

                        st.success("✅ Vault inicializado e autenticado com sucesso!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erro ao inicializar vault: {exc}")

        else:
            st.success(f"✅ Vault ativo | Usuário: **{st.session_state.vault_user}**")

            if st.button("🚪 Logout"):
                if st.session_state.vault and st.session_state.vault_token:
                    st.session_state.vault.logout(st.session_state.vault_token)
                st.session_state.vault = None
                st.session_state.vault_token = None
                st.session_state.vault_user = None
                st.rerun()

    if st.session_state.vault is None:
        st.info("Inicialize o vault para acessar as funcionalidades.")
        return

    vault = st.session_state.vault
    token = st.session_state.vault_token

    # Main vault features
    sessions_tab, decrypt_tab, stats_tab, audit_tab, users_tab = st.tabs([
        "📁 Sessões",
        "🔓 Decriptografar",
        "📊 Estatísticas",
        "📋 Auditoria",
        "👥 Usuários"
    ])

    with sessions_tab:
        st.markdown("### Sessões Armazenadas")

        try:
            sessions = vault.list_sessions(token, limit=50)

            if sessions:
                session_data = []
                for s in sessions:
                    retention_info = s.get("metadata", {}).get("decryption_retention", {})
                    retention_label = "Apagar após decriptar"
                    if retention_info.get("mode") == "retain":
                        days = retention_info.get("retention_days")
                        if days:
                            retention_label = f"Reter {days}d"
                        else:
                            retention_label = "Reter indefinido"

                    session_data.append({
                        "ID": s["session_id"][:20] + "...",
                        "Entidades": s["entity_count"],
                        "Risco": f"{s['risk_score']:.0%}",
                        "Criado": s["created_at"][:19],
                        "Acessos": s["access_count"],
                        "Retenção": retention_label,
                        "Preview": s.get("anonymized_preview", "")[:50]
                    })

                st.dataframe(session_data, use_container_width=True, hide_index=True)

                # Session selection for operations
                selected_id = st.selectbox(
                    "Selecione uma sessão",
                    options=[s["session_id"] for s in sessions],
                    format_func=lambda x: x[:40] + "..."
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Ver Anonimizado"):
                        try:
                            anon_text, anon_entities = vault.get_anonymized_session(
                                token, selected_id
                            )
                            st.markdown("#### Texto Anonimizado")
                            st.code(anon_text)

                            if anon_entities:
                                st.markdown("#### Entidades")
                                st.json(anon_entities)
                        except Exception as exc:
                            st.error(f"Erro: {exc}")

                with col2:
                    if st.button("🗑️ Excluir Sessão", type="secondary"):
                        try:
                            if vault.delete_session(token, selected_id):
                                st.success("Sessão excluída!")
                                st.rerun()
                            else:
                                st.warning("Sessão não encontrada")
                        except PermissionError:
                            st.error("Sem permissão para excluir")
            else:
                st.info("Nenhuma sessão armazenada ainda.")
                st.markdown(
                    "Use o **NER Module** com vault habilitado para armazenar sessões automaticamente."
                )

        except Exception as exc:
            st.error(f"Erro ao listar sessões: {exc}")

    with decrypt_tab:
        st.markdown("### Decriptografar Sessão")
        st.warning(
            "⚠️ Esta operação revela os dados originais. "
            "Apenas usuários autorizados devem ter acesso."
        )

        retention_message = "As sessões são apagadas imediatamente após a decriptação."
        if not vault.config.delete_after_decrypt:
            if vault.config.decryption_retention_days:
                retention_message = (
                    f"As sessões permanecem armazenadas por "
                    f"{vault.config.decryption_retention_days} dia(s) após a decriptação."
                )
            else:
                retention_message = (
                    "As sessões permanecem armazenadas mesmo após a decriptação "
                    "(retenção perene)."
                )

        st.info(retention_message)

        session_id_input = st.text_input(
            "ID da Sessão",
            placeholder="sess_...",
            key="decrypt_session_id"
        )

        if st.button("🔓 Decriptografar", type="primary", key="decrypt_btn"):
            if not session_id_input:
                st.warning("Digite o ID da sessão")
            else:
                try:
                    original_text, entities = vault.decrypt_session(
                        token, session_id_input
                    )

                    st.markdown("#### Texto Original")
                    st.text_area(
                        "Conteúdo decriptografado",
                        value=original_text,
                        height=200,
                        disabled=True
                    )

                    if entities:
                        st.markdown("#### Entidades Originais")
                        entity_data = []
                        for e in entities:
                            entity_data.append({
                                "Valor Original": e.get("value", "")[:40],
                                "Anonimizado": e.get("anonymized", ""),
                                "Tipo": e.get("entity_type", ""),
                                "Categoria": e.get("category", ""),
                            })
                        st.dataframe(entity_data, use_container_width=True, hide_index=True)

                    st.download_button(
                        "📥 Baixar Original",
                        original_text,
                        file_name="original_decrypted.txt",
                        mime="text/plain"
                    )

                except PermissionError:
                    st.error("Sem permissão para decriptografar esta sessão")
                except ValueError as ve:
                    st.error(f"Sessão não encontrada: {ve}")
                except Exception as exc:
                    st.error(f"Erro ao decriptografar: {exc}")

    with stats_tab:
        st.markdown("### Estatísticas do Vault")

        try:
            stats = vault.get_stats(token)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sessões", stats.get("total_sessions", 0))
            with col2:
                st.metric("Mapeamentos", stats.get("total_mappings", 0))
            with col3:
                st.metric("Chaves", stats.get("key_count", 0))
            with col4:
                size_mb = stats.get("storage_size_bytes", 0) / (1024 * 1024)
                st.metric("Tamanho", f"{size_mb:.2f} MB")

            st.markdown("#### Entidades por Tipo")
            entities_by_type = stats.get("entities_by_type", {})
            if entities_by_type:
                st.bar_chart(entities_by_type)

            st.markdown("#### Entidades por Categoria")
            entities_by_cat = stats.get("entities_by_category", {})
            if entities_by_cat:
                cat_cols = st.columns(len(entities_by_cat))
                for i, (cat, count) in enumerate(entities_by_cat.items()):
                    with cat_cols[i]:
                        st.metric(cat.upper(), count)

            st.markdown("#### Versões de Chave")
            key_versions = stats.get("key_versions", [])
            if key_versions:
                key_data = []
                for k in key_versions:
                    key_data.append({
                        "ID": k["key_id"][:20] + "...",
                        "Versão": k["version"],
                        "Ativa": "✅" if k["is_active"] else "❌",
                        "Criada": k["created_at"][:19],
                        "Expira": k["expires_at"][:19] if k.get("expires_at") else "Nunca"
                    })
                st.dataframe(key_data, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Erro ao obter estatísticas: {exc}")

    with audit_tab:
        st.markdown("### Logs de Auditoria")
        st.markdown("Registro imutável de todas as operações no vault.")

        col1, col2 = st.columns(2)
        with col1:
            audit_limit = st.number_input(
                "Limite de registros",
                min_value=10,
                max_value=500,
                value=50,
                key="audit_limit"
            )
        with col2:
            filter_user = st.text_input(
                "Filtrar por usuário (opcional)",
                key="audit_filter_user"
            )

        if st.button("🔄 Carregar Logs"):
            try:
                logs = vault.get_audit_logs(
                    token,
                    limit=audit_limit,
                    user_id=filter_user if filter_user else None
                )

                if logs:
                    log_data = []
                    for log in logs:
                        event_icons = {
                            "auth.login.success": "🔑",
                            "auth.login.failure": "🚫",
                            "data.stored": "💾",
                            "data.decrypted": "🔓",
                            "session.deleted": "🗑️",
                            "key.rotated": "🔄",
                            "security.access.denied": "⛔",
                        }
                        icon = event_icons.get(log["event_type"], "📝")

                        log_data.append({
                            "": icon,
                            "Evento": log["event_type"],
                            "Usuário": log.get("username") or log.get("user_id", "-")[:10],
                            "Sessão": (log.get("session_id") or "")[:15],
                            "Timestamp": log["timestamp"][:19],
                        })

                    st.dataframe(log_data, use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhum log encontrado")

            except PermissionError:
                st.error("Sem permissão para acessar logs de auditoria")
            except Exception as exc:
                st.error(f"Erro ao carregar logs: {exc}")

    with users_tab:
        st.markdown("### Gerenciamento de Usuários")

        st.markdown("#### Criar Novo Usuário")

        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Nome de usuário", key="new_user_name")
            new_password = st.text_input("Senha", type="password", key="new_user_pass")
        with col2:
            new_level = st.selectbox(
                "Nível de Acesso",
                options=[
                    ("Somente Leitura", AccessLevel.READ_ONLY),
                    ("Decriptografar", AccessLevel.DECRYPT),
                    ("Decriptografar Tudo", AccessLevel.FULL_DECRYPT),
                    ("Administrador", AccessLevel.ADMIN),
                    ("Super Admin", AccessLevel.SUPER_ADMIN),
                ],
                format_func=lambda x: x[0],
                key="new_user_level"
            )

        if st.button("➕ Criar Usuário"):
            if not new_username or not new_password:
                st.warning("Preencha nome de usuário e senha")
            else:
                try:
                    vault.create_user(
                        new_username,
                        new_password,
                        new_level[1],
                        token
                    )
                    st.success(f"Usuário '{new_username}' criado com sucesso!")
                except PermissionError:
                    st.error("Sem permissão para criar usuários")
                except ValueError as ve:
                    st.error(f"Erro: {ve}")
                except Exception as exc:
                    st.error(f"Erro ao criar usuário: {exc}")

        st.markdown("#### Níveis de Acesso")
        st.markdown("""
        | Nível | Permissões |
        |-------|------------|
        | **Somente Leitura** | Ver dados anonimizados |
        | **Decriptografar** | Decriptografar sessões específicas |
        | **Decriptografar Tudo** | Decriptografar todas as sessões |
        | **Administrador** | Gerenciar sessões e chaves |
        | **Super Admin** | Gerenciar usuários e auditoria |
        """)


def render_settings_tab() -> None:
    """Centralized settings tab for API keys, Data Warehouse connections, and LLM configuration."""
    section_header(
        "⚙️ Configurações",
        "Gerencie conexões com Data Warehouses, chaves de API e provedores de LLM em um único lugar.",
    )

    settings = st.session_state.connection_settings

    # Model lists for LLM providers
    OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-3.5-turbo", "gpt-4-turbo"]
    GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.0-pro"]
    DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
    CLAUDE_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]

    # Store model lists in session state for other tabs to use
    if "available_models" not in st.session_state:
        st.session_state.available_models = {
            "openai": OPENAI_MODELS,
            "gemini": GEMINI_MODELS,
            "deepseek": DEEPSEEK_MODELS,
            "claude": CLAUDE_MODELS,
        }

    # Create sub-tabs for different configuration categories
    llm_tab, warehouse_tab, catalog_tab = st.tabs([
        "🤖 Provedores LLM",
        "🏢 Data Warehouses",
        "📚 Catálogos"
    ])

    # ==================== LLM PROVIDERS TAB ====================
    with llm_tab:
        st.markdown("### Configuração de Provedores de IA")

        # Current provider status
        current_provider = settings.get("llm_provider", "openai")
        current_model = settings.get("llm_model", "gpt-4o-mini")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Provedor Ativo", current_provider.upper())
        with col2:
            st.metric("Modelo Padrão", current_model)
        with col3:
            configured_count = sum([
                1 if settings.get("openai_api_key") else 0,
                1 if settings.get("gemini_api_key") else 0,
                1 if settings.get("deepseek_api_key") else 0,
                1 if settings.get("anthropic_api_key") else 0,
            ])
            st.metric("Provedores Configurados", f"{configured_count}/4")

        st.divider()

        # Provider selection
        openai_tab, gemini_tab, deepseek_tab, claude_tab = st.tabs(
            ["OpenAI", "Google Gemini", "DeepSeek", "Anthropic Claude"]
        )

        with openai_tab:
            st.markdown("##### OpenAI Configuration")
            with st.form("settings_openai_form"):
                openai_key = st.text_input(
                    "API Key",
                    value=settings.get("openai_api_key", ""),
                    type="password",
                    help="Sua chave da API OpenAI"
                )
                openai_url = st.text_input(
                    "API URL (Base)",
                    value=settings.get("openai_api_url", "https://api.openai.com/v1"),
                    help="URL base da API (útil para Azure OpenAI ou proxies)"
                )
                openai_model = st.selectbox(
                    "Modelo Padrão",
                    options=OPENAI_MODELS,
                    index=OPENAI_MODELS.index(settings.get("llm_model", "gpt-4o-mini"))
                    if settings.get("llm_model") in OPENAI_MODELS else 0,
                )
                set_openai_default = st.checkbox(
                    "Definir OpenAI como provedor padrão",
                    value=current_provider == "openai"
                )
                if st.form_submit_button("💾 Salvar OpenAI", use_container_width=True):
                    _apply_connection_settings({
                        "openai_api_key": openai_key.strip(),
                        "openai_api_url": openai_url.strip(),
                        "llm_model": openai_model if set_openai_default else settings.get("llm_model"),
                        "llm_provider": "openai" if set_openai_default else settings.get("llm_provider"),
                    })
                    st.success("✅ Configuração OpenAI salva!")
                    st.rerun()

        with gemini_tab:
            st.markdown("##### Google Gemini Configuration")
            with st.form("settings_gemini_form"):
                gemini_key = st.text_input(
                    "API Key",
                    value=settings.get("gemini_api_key", ""),
                    type="password",
                    help="Sua chave da API Google (Gemini/Vertex AI)"
                )
                gemini_model = st.selectbox(
                    "Modelo Padrão",
                    options=GEMINI_MODELS,
                    index=GEMINI_MODELS.index(settings.get("gemini_model", "gemini-1.5-flash"))
                    if settings.get("gemini_model") in GEMINI_MODELS else 0,
                )
                set_gemini_default = st.checkbox(
                    "Definir Gemini como provedor padrão",
                    value=current_provider == "gemini"
                )
                if st.form_submit_button("💾 Salvar Gemini", use_container_width=True):
                    _apply_connection_settings({
                        "gemini_api_key": gemini_key.strip(),
                        "gemini_model": gemini_model,
                        "llm_model": gemini_model if set_gemini_default else settings.get("llm_model"),
                        "llm_provider": "gemini" if set_gemini_default else settings.get("llm_provider"),
                    })
                    st.success("✅ Configuração Gemini salva!")
                    st.rerun()

        with deepseek_tab:
            st.markdown("##### DeepSeek Configuration")
            with st.form("settings_deepseek_form"):
                deepseek_key = st.text_input(
                    "API Key",
                    value=settings.get("deepseek_api_key", ""),
                    type="password",
                    help="Sua chave da API DeepSeek"
                )
                deepseek_url = st.text_input(
                    "API URL",
                    value=settings.get("deepseek_api_url", "https://api.deepseek.com"),
                    help="URL da API DeepSeek"
                )
                deepseek_model = st.selectbox(
                    "Modelo Padrão",
                    options=DEEPSEEK_MODELS,
                    index=DEEPSEEK_MODELS.index(settings.get("deepseek_model", "deepseek-chat"))
                    if settings.get("deepseek_model") in DEEPSEEK_MODELS else 0,
                )
                set_deepseek_default = st.checkbox(
                    "Definir DeepSeek como provedor padrão",
                    value=current_provider == "deepseek"
                )
                if st.form_submit_button("💾 Salvar DeepSeek", use_container_width=True):
                    _apply_connection_settings({
                        "deepseek_api_key": deepseek_key.strip(),
                        "deepseek_api_url": deepseek_url.strip(),
                        "deepseek_model": deepseek_model,
                        "llm_model": deepseek_model if set_deepseek_default else settings.get("llm_model"),
                        "llm_provider": "deepseek" if set_deepseek_default else settings.get("llm_provider"),
                    })
                    st.success("✅ Configuração DeepSeek salva!")
                    st.rerun()

        with claude_tab:
            st.markdown("##### Anthropic Claude Configuration")
            with st.form("settings_claude_form"):
                claude_key = st.text_input(
                    "API Key",
                    value=settings.get("anthropic_api_key", ""),
                    type="password",
                    help="Sua chave da API Anthropic"
                )
                claude_model = st.selectbox(
                    "Modelo Padrão",
                    options=CLAUDE_MODELS,
                    index=CLAUDE_MODELS.index(settings.get("anthropic_model", "claude-3-5-sonnet-20240620"))
                    if settings.get("anthropic_model") in CLAUDE_MODELS else 0,
                )
                set_claude_default = st.checkbox(
                    "Definir Claude como provedor padrão",
                    value=current_provider == "claude"
                )
                if st.form_submit_button("💾 Salvar Claude", use_container_width=True):
                    _apply_connection_settings({
                        "anthropic_api_key": claude_key.strip(),
                        "anthropic_model": claude_model,
                        "llm_model": claude_model if set_claude_default else settings.get("llm_model"),
                        "llm_provider": "claude" if set_claude_default else settings.get("llm_provider"),
                    })
                    st.success("✅ Configuração Claude salva!")
                    st.rerun()

    # ==================== DATA WAREHOUSES TAB ====================
    with warehouse_tab:
        st.markdown("### Conexões com Data Warehouses")
        st.markdown("Configure conexões com seus data warehouses para uso nos módulos de governança.")

        # Initialize warehouse settings in session state if not exists
        if "warehouse_connections" not in st.session_state:
            st.session_state.warehouse_connections = {
                "snowflake": {
                    "account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
                    "username": os.environ.get("SNOWFLAKE_USERNAME", ""),
                    "password": os.environ.get("SNOWFLAKE_PASSWORD", ""),
                    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
                    "database": os.environ.get("SNOWFLAKE_DATABASE", ""),
                    "schema": os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
                    "role": os.environ.get("SNOWFLAKE_ROLE", ""),
                    "enabled": False,
                },
                "redshift": {
                    "host": os.environ.get("REDSHIFT_HOST", ""),
                    "port": os.environ.get("REDSHIFT_PORT", "5439"),
                    "database": os.environ.get("REDSHIFT_DATABASE", ""),
                    "username": os.environ.get("REDSHIFT_USERNAME", ""),
                    "password": os.environ.get("REDSHIFT_PASSWORD", ""),
                    "schema": os.environ.get("REDSHIFT_SCHEMA", "public"),
                    "enabled": False,
                },
                "bigquery": {
                    "project_id": os.environ.get("BIGQUERY_PROJECT_ID", ""),
                    "dataset": os.environ.get("BIGQUERY_DATASET", ""),
                    "credentials_path": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
                    "location": os.environ.get("BIGQUERY_LOCATION", "US"),
                    "enabled": False,
                },
                "synapse": {
                    "server": os.environ.get("SYNAPSE_SERVER", ""),
                    "database": os.environ.get("SYNAPSE_DATABASE", ""),
                    "username": os.environ.get("SYNAPSE_USERNAME", ""),
                    "password": os.environ.get("SYNAPSE_PASSWORD", ""),
                    "authentication": os.environ.get("SYNAPSE_AUTHENTICATION", "sql"),
                    "enabled": False,
                },
            }

        wh_connections = st.session_state.warehouse_connections

        # Status overview
        st.markdown("#### Status das Conexões")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sf_status = "✅ Configurado" if wh_connections["snowflake"].get("account") else "⚪ Não configurado"
            st.markdown(f"**Snowflake**: {sf_status}")
        with col2:
            rs_status = "✅ Configurado" if wh_connections["redshift"].get("host") else "⚪ Não configurado"
            st.markdown(f"**Redshift**: {rs_status}")
        with col3:
            bq_status = "✅ Configurado" if wh_connections["bigquery"].get("project_id") else "⚪ Não configurado"
            st.markdown(f"**BigQuery**: {bq_status}")
        with col4:
            sy_status = "✅ Configurado" if wh_connections["synapse"].get("server") else "⚪ Não configurado"
            st.markdown(f"**Synapse**: {sy_status}")

        st.divider()

        # Warehouse configuration tabs
        snowflake_tab, redshift_tab, bigquery_tab, synapse_tab = st.tabs([
            "❄️ Snowflake",
            "🔴 Amazon Redshift",
            "🔵 Google BigQuery",
            "🟣 Azure Synapse"
        ])

        with snowflake_tab:
            st.markdown("##### Snowflake Connection")
            with st.form("snowflake_config_form"):
                sf = wh_connections["snowflake"]
                sf_account = st.text_input("Account", value=sf.get("account", ""), help="Ex: xy12345.us-east-1")
                sf_username = st.text_input("Username", value=sf.get("username", ""))
                sf_password = st.text_input("Password", value=sf.get("password", ""), type="password")
                col1, col2 = st.columns(2)
                with col1:
                    sf_warehouse = st.text_input("Warehouse", value=sf.get("warehouse", ""))
                    sf_database = st.text_input("Database", value=sf.get("database", ""))
                with col2:
                    sf_schema = st.text_input("Schema", value=sf.get("schema", "PUBLIC"))
                    sf_role = st.text_input("Role (opcional)", value=sf.get("role", ""))
                sf_enabled = st.checkbox("Habilitar conexão Snowflake", value=sf.get("enabled", False))

                if st.form_submit_button("💾 Salvar Snowflake", use_container_width=True):
                    wh_connections["snowflake"] = {
                        "account": sf_account, "username": sf_username, "password": sf_password,
                        "warehouse": sf_warehouse, "database": sf_database, "schema": sf_schema,
                        "role": sf_role, "enabled": sf_enabled
                    }
                    # Set environment variables
                    os.environ["SNOWFLAKE_ACCOUNT"] = sf_account
                    os.environ["SNOWFLAKE_USERNAME"] = sf_username
                    os.environ["SNOWFLAKE_PASSWORD"] = sf_password
                    os.environ["SNOWFLAKE_WAREHOUSE"] = sf_warehouse
                    os.environ["SNOWFLAKE_DATABASE"] = sf_database
                    os.environ["SNOWFLAKE_SCHEMA"] = sf_schema
                    if sf_role:
                        os.environ["SNOWFLAKE_ROLE"] = sf_role
                    st.success("✅ Configuração Snowflake salva!")

        with redshift_tab:
            st.markdown("##### Amazon Redshift Connection")
            with st.form("redshift_config_form"):
                rs = wh_connections["redshift"]
                rs_host = st.text_input("Host", value=rs.get("host", ""), help="Endpoint do cluster Redshift")
                col1, col2 = st.columns(2)
                with col1:
                    rs_port = st.text_input("Port", value=rs.get("port", "5439"))
                    rs_database = st.text_input("Database", value=rs.get("database", ""))
                with col2:
                    rs_username = st.text_input("Username", value=rs.get("username", ""))
                    rs_password = st.text_input("Password", value=rs.get("password", ""), type="password")
                rs_schema = st.text_input("Schema", value=rs.get("schema", "public"))
                rs_enabled = st.checkbox("Habilitar conexão Redshift", value=rs.get("enabled", False))

                if st.form_submit_button("💾 Salvar Redshift", use_container_width=True):
                    wh_connections["redshift"] = {
                        "host": rs_host, "port": rs_port, "database": rs_database,
                        "username": rs_username, "password": rs_password, "schema": rs_schema,
                        "enabled": rs_enabled
                    }
                    os.environ["REDSHIFT_HOST"] = rs_host
                    os.environ["REDSHIFT_PORT"] = rs_port
                    os.environ["REDSHIFT_DATABASE"] = rs_database
                    os.environ["REDSHIFT_USERNAME"] = rs_username
                    os.environ["REDSHIFT_PASSWORD"] = rs_password
                    os.environ["REDSHIFT_SCHEMA"] = rs_schema
                    st.success("✅ Configuração Redshift salva!")

        with bigquery_tab:
            st.markdown("##### Google BigQuery Connection")
            with st.form("bigquery_config_form"):
                bq = wh_connections["bigquery"]
                bq_project = st.text_input("Project ID", value=bq.get("project_id", ""))
                bq_dataset = st.text_input("Dataset", value=bq.get("dataset", ""))
                bq_creds = st.text_input("Credentials Path", value=bq.get("credentials_path", ""),
                                         help="Caminho para o arquivo JSON de credenciais de service account")
                bq_location = st.selectbox("Location", options=["US", "EU", "asia-northeast1", "us-central1"],
                                           index=0 if bq.get("location", "US") == "US" else 1)
                bq_enabled = st.checkbox("Habilitar conexão BigQuery", value=bq.get("enabled", False))

                if st.form_submit_button("💾 Salvar BigQuery", use_container_width=True):
                    wh_connections["bigquery"] = {
                        "project_id": bq_project, "dataset": bq_dataset,
                        "credentials_path": bq_creds, "location": bq_location, "enabled": bq_enabled
                    }
                    os.environ["BIGQUERY_PROJECT_ID"] = bq_project
                    os.environ["BIGQUERY_DATASET"] = bq_dataset
                    if bq_creds:
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_creds
                    os.environ["BIGQUERY_LOCATION"] = bq_location
                    st.success("✅ Configuração BigQuery salva!")

        with synapse_tab:
            st.markdown("##### Azure Synapse Analytics Connection")
            with st.form("synapse_config_form"):
                sy = wh_connections["synapse"]
                sy_server = st.text_input("Server", value=sy.get("server", ""),
                                          help="Ex: your-workspace.sql.azuresynapse.net")
                sy_database = st.text_input("Database", value=sy.get("database", ""))
                col1, col2 = st.columns(2)
                with col1:
                    sy_username = st.text_input("Username", value=sy.get("username", ""))
                    sy_password = st.text_input("Password", value=sy.get("password", ""), type="password")
                with col2:
                    sy_auth = st.selectbox("Authentication", options=["sql", "ad", "msi"],
                                           index=["sql", "ad", "msi"].index(sy.get("authentication", "sql")))
                sy_enabled = st.checkbox("Habilitar conexão Synapse", value=sy.get("enabled", False))

                if st.form_submit_button("💾 Salvar Synapse", use_container_width=True):
                    wh_connections["synapse"] = {
                        "server": sy_server, "database": sy_database,
                        "username": sy_username, "password": sy_password,
                        "authentication": sy_auth, "enabled": sy_enabled
                    }
                    os.environ["SYNAPSE_SERVER"] = sy_server
                    os.environ["SYNAPSE_DATABASE"] = sy_database
                    os.environ["SYNAPSE_USERNAME"] = sy_username
                    os.environ["SYNAPSE_PASSWORD"] = sy_password
                    os.environ["SYNAPSE_AUTHENTICATION"] = sy_auth
                    st.success("✅ Configuração Synapse salva!")

    # ==================== CATALOGS TAB ====================
    with catalog_tab:
        st.markdown("### Conexões com Catálogos de Metadados")
        st.markdown("Configure conexões com catálogos corporativos de metadados.")

        # OpenMetadata
        st.markdown("#### OpenMetadata")
        with st.form("openmetadata_config_form"):
            om_host = st.text_input(
                "Host URL",
                value=settings.get("openmetadata_host", ""),
                help="Ex: http://localhost:8585/api"
            )
            om_token = st.text_input(
                "API Token",
                value=settings.get("openmetadata_api_token", ""),
                type="password",
                help="Token JWT para autenticação"
            )
            if st.form_submit_button("💾 Salvar OpenMetadata", use_container_width=True):
                _apply_connection_settings({
                    "openmetadata_host": om_host.strip(),
                    "openmetadata_api_token": om_token.strip(),
                })
                st.success("✅ Configuração OpenMetadata salva!")

        st.divider()

        # Apache Atlas
        st.markdown("#### Apache Atlas")
        with st.form("atlas_config_form"):
            atlas_host = st.text_input(
                "Host URL",
                value=settings.get("atlas_host", ""),
                help="Ex: http://atlas-host:21000"
            )
            if st.form_submit_button("💾 Salvar Atlas", use_container_width=True):
                _apply_connection_settings({"atlas_host": atlas_host.strip()})
                st.success("✅ Configuração Atlas salva!")

        st.divider()

        # AWS Glue
        st.markdown("#### AWS Glue Data Catalog")
        with st.form("glue_config_form"):
            glue_region = st.text_input(
                "AWS Region",
                value=settings.get("glue_region", ""),
                help="Ex: us-east-1"
            )
            st.info("As credenciais AWS devem estar configuradas no ambiente (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
            if st.form_submit_button("💾 Salvar AWS Glue", use_container_width=True):
                _apply_connection_settings({"glue_region": glue_region.strip()})
                st.success("✅ Configuração AWS Glue salva!")


def get_available_llm_models() -> Dict[str, List[str]]:
    """Return available LLM models for use in other tabs."""
    if "available_models" in st.session_state:
        return st.session_state.available_models
    return {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-3.5-turbo"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "claude": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
    }


def get_configured_warehouses() -> List[str]:
    """Return list of configured and enabled warehouses."""
    if "warehouse_connections" not in st.session_state:
        return []
    warehouses = []
    wh = st.session_state.warehouse_connections
    if wh.get("snowflake", {}).get("enabled") and wh["snowflake"].get("account"):
        warehouses.append("snowflake")
    if wh.get("redshift", {}).get("enabled") and wh["redshift"].get("host"):
        warehouses.append("redshift")
    if wh.get("bigquery", {}).get("enabled") and wh["bigquery"].get("project_id"):
        warehouses.append("bigquery")
    if wh.get("synapse", {}).get("enabled") and wh["synapse"].get("server"):
        warehouses.append("synapse")
    return warehouses


def render_warehouse_selector(key_prefix: str, show_schema: bool = True, show_table: bool = True) -> Dict[str, Any]:
    """
    Render a reusable warehouse/table selector component.

    Args:
        key_prefix: Unique prefix for widget keys
        show_schema: Whether to show schema selector
        show_table: Whether to show table selector

    Returns:
        Dict with selected warehouse config and table info, or None if using file upload
    """
    configured_warehouses = get_configured_warehouses()

    # Data source selection
    source_options = ["📁 Upload de arquivo"]
    if configured_warehouses:
        source_options.append("🏢 Data Warehouse conectado")

    data_source = st.radio(
        "Fonte de dados",
        options=source_options,
        horizontal=True,
        key=f"{key_prefix}_data_source"
    )

    if data_source == "📁 Upload de arquivo" or not configured_warehouses:
        return {"source_type": "file", "warehouse": None}

    # Warehouse selection
    with st.container(border=True):
        st.markdown("##### Selecione a fonte de dados")

        warehouse_labels = {
            "snowflake": "❄️ Snowflake",
            "redshift": "🔴 Amazon Redshift",
            "bigquery": "🔵 Google BigQuery",
            "synapse": "🟣 Azure Synapse"
        }

        col1, col2 = st.columns(2)

        with col1:
            selected_wh = st.selectbox(
                "Data Warehouse",
                options=configured_warehouses,
                format_func=lambda x: warehouse_labels.get(x, x),
                key=f"{key_prefix}_warehouse"
            )

        # Get warehouse connection details
        wh_config = st.session_state.warehouse_connections.get(selected_wh, {})

        with col2:
            # Database/Project selector based on warehouse type
            if selected_wh == "snowflake":
                database = st.text_input(
                    "Database",
                    value=wh_config.get("database", ""),
                    key=f"{key_prefix}_database"
                )
            elif selected_wh == "redshift":
                database = st.text_input(
                    "Database",
                    value=wh_config.get("database", ""),
                    key=f"{key_prefix}_database"
                )
            elif selected_wh == "bigquery":
                database = st.text_input(
                    "Project ID",
                    value=wh_config.get("project_id", ""),
                    key=f"{key_prefix}_database"
                )
            elif selected_wh == "synapse":
                database = st.text_input(
                    "Database",
                    value=wh_config.get("database", ""),
                    key=f"{key_prefix}_database"
                )
            else:
                database = ""

        schema = None
        table = None

        if show_schema:
            col3, col4 = st.columns(2)
            with col3:
                if selected_wh == "bigquery":
                    schema = st.text_input(
                        "Dataset",
                        value=wh_config.get("dataset", ""),
                        key=f"{key_prefix}_schema"
                    )
                else:
                    schema = st.text_input(
                        "Schema",
                        value=wh_config.get("schema", "public"),
                        key=f"{key_prefix}_schema"
                    )

            with col4:
                if show_table:
                    table = st.text_input(
                        "Tabela",
                        placeholder="nome_da_tabela",
                        key=f"{key_prefix}_table"
                    )
        elif show_table:
            table = st.text_input(
                "Tabela",
                placeholder="nome_da_tabela",
                key=f"{key_prefix}_table"
            )

        # Connection status indicator
        st.caption(f"✅ Conectado via configurações salvas em Settings")

    return {
        "source_type": "warehouse",
        "warehouse": selected_wh,
        "warehouse_config": wh_config,
        "database": database,
        "schema": schema,
        "table": table
    }


def get_warehouse_connector(warehouse_type: str, config: Dict[str, Any]):
    """
    Create a warehouse connector based on type and config.

    Returns connector instance or None if creation fails.
    """
    try:
        from warehouse.connectors import create_warehouse_connector

        if warehouse_type == "snowflake":
            return create_warehouse_connector(
                "snowflake",
                account=config.get("account"),
                username=config.get("username"),
                password=config.get("password"),
                warehouse=config.get("warehouse"),
                database=config.get("database"),
                schema=config.get("schema"),
                role=config.get("role")
            )
        elif warehouse_type == "redshift":
            return create_warehouse_connector(
                "redshift",
                host=config.get("host"),
                port=int(config.get("port", 5439)),
                database=config.get("database"),
                username=config.get("username"),
                password=config.get("password"),
                schema=config.get("schema")
            )
        elif warehouse_type == "bigquery":
            return create_warehouse_connector(
                "bigquery",
                project_id=config.get("project_id"),
                dataset=config.get("dataset"),
                credentials_path=config.get("credentials_path"),
                location=config.get("location")
            )
        elif warehouse_type == "synapse":
            return create_warehouse_connector(
                "synapse",
                server=config.get("server"),
                database=config.get("database"),
                username=config.get("username"),
                password=config.get("password"),
                authentication=config.get("authentication", "sql")
            )
    except ImportError:
        st.warning("⚠️ Módulo warehouse não disponível. Instale com: `pip install -r warehouse/requirements.txt`")
        return None
    except Exception as e:
        st.error(f"Erro ao criar conector: {e}")
        return None

    return None


init_session_state()
hero_section()
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Lineage", "Discovery", "Enrichment", "Classification", "Quality", "Asset Value", "NER Module", "Vault", "⚙️ Settings"
])
with tab1:
    render_lineage_tab()
with tab2:
    render_rag_tab()
with tab3:
    render_enrichment_tab()
with tab4:
    render_classification_tab()
with tab5:
    render_quality_tab()
with tab6:
    render_value_tab()
with tab7:
    render_ner_tab()
with tab8:
    render_vault_tab()
with tab9:
    render_settings_tab()

