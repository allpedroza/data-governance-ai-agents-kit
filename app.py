"""Unified Streamlit UI to orchestrate the two available agents."""
import csv
import json
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


st.set_page_config(
    page_title="Data Governance AI Agents",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
        .metric-card {background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 12px;}
        .section-card {background: #0b1220; padding: 1.25rem; border-radius: 16px; border: 1px solid #1f2937;}
        .pill {display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; background: #1e293b; color: #e2e8f0; margin-right: 0.5rem;}
        .callout {border-left: 4px solid #22c55e; background: #0f172a; padding: 0.75rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


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

    if "rag_agent" not in st.session_state:
        try:
            st.session_state.rag_agent = DataDiscoveryRAGAgent(
                persist_directory=str(BASE_DIR / "rag_discovery" / ".chroma_ui"),
                collection_name="ui_catalog",
            )
            st.session_state.rag_agent_error = None
        except Exception as exc:  # noqa: BLE001 - surface initialization issues
            st.session_state.rag_agent = None
            st.session_state.rag_agent_error = str(exc)


def hero_section() -> None:
    """Top banner with quick context."""
    left, right = st.columns([3, 2])
    with left:
        st.title("Data Governance AI Agents")
        st.markdown(
            "**Interface unificada para trabalhar com o Data Lineage Agent e o Data Discovery RAG Agent.**\n"
            "Envie pipelines, documente tabelas e faça perguntas em um único lugar."
        )
        st.markdown(
            "<div class='callout'>Configuração rápida: defina sua `OPENAI_API_KEY` para habilitar todos os recursos.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.image(
            "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=800&q=80",
            caption="Governança de dados assistida por IA",
            use_column_width=True,
        )


def render_lineage_tab() -> None:
    """UI for running the Data Lineage Agent."""
    st.subheader("🔗 Data Lineage Agent")
    st.markdown("Analise rapidamente pipelines e visualize o estado da análise em tempo real.")

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
    st.subheader("🔍 Data Discovery RAG Agent")
    st.markdown(
        "Construa um catálogo rápido de tabelas e faça perguntas em linguagem natural. "
        "Se a `OPENAI_API_KEY` estiver configurada, a busca usa embeddings e RAG; caso contrário, aplica busca textual simples."
    )

    if st.session_state.get("rag_agent_error"):
        st.warning(
            "Não foi possível inicializar o agente RAG automaticamente. "
            f"Detalhes: {st.session_state['rag_agent_error']}"
        )

    connection_tab, chat_tab = st.tabs(["Conectar catálogo", "Conversar com o agente"])

    with connection_tab:
        st.markdown("Escolha como quer conectar seu catálogo antes de iniciar a conversa.")
        with st.form("catalog_connection_form"):
            mode = st.radio(
                "Origem do catálogo",
                options=["Arquivo de catálogo", "Conectores disponíveis"],
                horizontal=True,
            )
            catalog_name = st.text_input("Nome do catálogo", placeholder="Data Lake Principal")

            if mode == "Arquivo de catálogo":
                uploaded = st.file_uploader(
                    "Envie um arquivo JSON ou CSV com as tabelas",
                    type=["json", "csv", "txt"],
                    accept_multiple_files=False,
                )
            else:
                connector = st.selectbox(
                    "Selecione um conector existente",
                    [
                        "Apache Atlas",
                        "Glue Data Catalog",
                        "BigQuery",
                        "Snowflake",
                    ],
                )
                dataset_hint = st.text_input(
                    "Escopo ou dataset", placeholder="ex: projeto-analytics"
                )
                uploaded = None

            connect = st.form_submit_button("Conectar catálogo")

        if connect:
            if mode == "Arquivo de catálogo" and not uploaded:
                st.warning("Envie um arquivo de metadados para conectar.")
            else:
                catalog_label = catalog_name or (
                    uploaded.name if uploaded else connector  # type: ignore[misc]
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

                st.session_state.connected_catalogs.append(
                    {
                        "name": catalog_label,
                        "source": "Arquivo" if uploaded else connector,  # type: ignore[misc]
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


init_session_state()
hero_section()
tab1, tab2 = st.tabs(["Lineage", "Discovery"])
with tab1:
    render_lineage_tab()
with tab2:
    render_rag_tab()

