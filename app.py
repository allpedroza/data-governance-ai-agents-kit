"""Unified Streamlit UI to orchestrate the two available agents."""
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
            "atlas_host": os.environ.get("ATLAS_HOST", ""),
            "glue_region": os.environ.get("AWS_REGION", ""),
            "chroma_persist": str(BASE_DIR / "rag_discovery" / ".chroma_ui"),
        }

    if "rag_agent" not in st.session_state:
        _initialize_rag_agent()


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
            use_container_width=True,
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
    st.markdown("### Guia de conexões essenciais")
    settings: Dict[str, str] = st.session_state.connection_settings

    status_labels = []
    status_labels.append(
        "✅ OPENAI_API_KEY configurada" if settings.get("openai_api_key") else "⚠️ OPENAI_API_KEY ausente"
    )
    status_labels.append(
        "✅ Persistência local do Chroma pronta" if settings.get("chroma_persist") else "⚠️ Revise o diretório do Chroma"
    )
    st.markdown("; ".join(status_labels))

    with st.expander("Configurar IA (OpenAI) e vetorização", expanded=not settings.get("openai_api_key")):
        st.markdown(
            "- Defina a `OPENAI_API_KEY` para habilitar embeddings e evitar erros 401 como o exibido na indexação.\n"
            "- Opcionalmente ajuste a `OPENAI_API_URL` para provedores compatíveis.\n"
            "- O catálogo vetorial usa ChromaDB local em `{}`; nenhum serviço externo é necessário.".format(
                settings.get("chroma_persist")
            )
        )
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
            apply_credentials = st.form_submit_button("Salvar e reiniciar o agente RAG")

        if apply_credentials:
            _apply_connection_settings(
                {
                    "openai_api_key": openai_key.strip(),
                    "openai_api_url": openai_url.strip(),
                    "atlas_host": settings.get("atlas_host", ""),
                    "glue_region": settings.get("glue_region", ""),
                }
            )
            if st.session_state.get("rag_agent"):
                st.success("Credenciais aplicadas. O agente RAG foi reiniciado com a nova configuração.")
            else:
                st.error(
                    "Atualizamos as credenciais, mas o agente ainda não inicializou. "
                    "Revise os valores e tente novamente."
                )

    with st.expander("Conexões de catálogos corporativos"):
        st.markdown(
            "- **Apache Atlas**: informe o host em `ATLAS_HOST` (ex.: `http://atlas-host:21000`).\n"
            "- **Glue Data Catalog**: use a região em `AWS_REGION` e credenciais AWS no ambiente.\n"
            "- **BigQuery/Snowflake**: a conexão é guiada na seção de conectores; inclua o projeto/warehouse ao selecionar.\n"
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

    _render_connection_guide()

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
            dataset_hint = ""

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

    # File upload for enrichment
    st.markdown("### Enriquecer Metadados")

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
        if file_type == "CSV":
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], key="csv_encoding")
            separator = st.selectbox("Separador", [",", ";", "|"], key="csv_sep")

    additional_context = st.text_area(
        "Contexto adicional (opcional)",
        placeholder="Informações adicionais sobre a tabela, domínio de negócio, etc.",
        key="enrich_context"
    )

    if uploaded_file and st.button("🚀 Gerar Metadados", type="primary", key="run_enrichment"):
        with st.spinner("Analisando dados e gerando metadados..."):
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

                st.session_state.enrichment_result = result
                st.success("✓ Metadados gerados com sucesso!")

            except Exception as exc:
                st.error(f"Erro ao processar: {exc}")
            finally:
                os.unlink(temp_path)

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


init_session_state()
hero_section()
tab1, tab2, tab3 = st.tabs(["Lineage", "Discovery", "Enrichment"])
with tab1:
    render_lineage_tab()
with tab2:
    render_rag_tab()
with tab3:
    render_enrichment_tab()

