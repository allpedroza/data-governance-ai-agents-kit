"""
Metadata Enrichment Wizard
Fluxo guiado de 5 etapas para enriquecimento de metadados com IA.

Etapas:
  1. Conectores — Cloud DWH, MDM, LLM provider
  2. Base de Conhecimento — indexação de normativos
  3. Seleção de Dataset — arquivo ou warehouse
  4. Validação & Edição — revisão coluna a coluna
  5. Publicação — DDL SQL, BQ CLI, AWS Glue, Synapse, JSON, Markdown, OpenMetadata
"""

import os
import json
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Path setup
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # Navigation
    "wizard_step": 1,
    "step_completed": {1: False, 2: False, 3: False, 4: False, 5: False},
    # Step 1 — LLM / Embeddings / Vector Store
    "llm_provider_name": "OpenAI",
    "llm_model": "gpt-4o-mini",
    "llm_api_key": "",
    "llm_provider_obj": None,
    "embedding_name": "SentenceTransformer",
    "embedding_model": "all-MiniLM-L6-v2",
    "vectorstore_name": "Chroma",
    "vectorstore_persist_dir": "./chroma_metadata_wizard",
    "vectorstore_collection": "metadata_standards",
    "agent": None,
    # Step 1 — Warehouse
    "warehouse_type": "None",
    "warehouse_connector": None,
    "sf_account": "", "sf_username": "", "sf_password": "",
    "sf_database": "", "sf_schema": "PUBLIC", "sf_warehouse": "", "sf_role": "",
    "bq_project": "", "bq_dataset": "",
    "rs_host": "", "rs_port": "5439", "rs_database": "", "rs_username": "", "rs_password": "",
    "syn_server": "", "syn_database": "", "syn_username": "", "syn_password": "",
    # Step 1 — OpenMetadata
    "openmetadata_enabled": False,
    "openmetadata_url": "",
    "openmetadata_token": "",
    # Step 2 — Knowledge Base
    "standards_indexed": False,
    "standards_stats": {},
    "indexed_files": [],
    # Step 3 — Dataset
    "source_type_radio": "CSV",
    "uploaded_file_bytes": None,
    "uploaded_file_name": "",
    "uploaded_file_ext": ".csv",
    "csv_encoding": "utf-8",
    "csv_separator": ",",
    "delta_path": "",
    "delta_version": None,
    "wh_selected_db": "",
    "wh_selected_schema": "",
    "selected_table": "",
    "sample_size": 100,
    "additional_context": "",
    # Step 4 — Validation
    "enrichment_result": None,
    "sample_result": None,
    "edited_table_fields": {},
    "edited_columns": {},
    "regenerating_column": None,
    "validation_dirty": False,
    # Step 5 — Publish config
    "pub_ddl_schema": "dbo",
    "pub_bq_project": "",
    "pub_bq_dataset": "",
    "pub_aws_region": "us-east-1",
    "pub_aws_db": "",
    "pub_synapse_schema": "dbo",
    "publish_result": None,
}


def init_session_state() -> None:
    for key, val in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

STEP_LABELS = {1: "Conectores", 2: "Base de Conhecimento", 3: "Dataset",
               4: "Validar & Editar", 5: "Publicar"}


def render_progress_bar(current_step: int) -> None:
    completed = st.session_state.get("step_completed", {})
    cols = st.columns(5)
    for col, num in zip(cols, range(1, 6)):
        label = STEP_LABELS[num]
        with col:
            if completed.get(num) and num != current_step:
                if st.button(f"✓ {num}. {label}", key=f"nav_{num}",
                             use_container_width=True):
                    st.session_state["wizard_step"] = num
                    st.rerun()
            elif num == current_step:
                st.markdown(
                    f"<div style='text-align:center;font-weight:bold;"
                    f"background:#1f77b4;color:white;padding:6px;"
                    f"border-radius:6px'>{num}. {label}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;color:#999;"
                    f"padding:6px'>{num}. {label}</div>",
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def _build_llm_provider():
    name = st.session_state["llm_provider_name"]
    model = st.session_state["llm_model"]
    key = st.session_state["llm_api_key"]
    if name == "OpenAI":
        from rag_discovery.providers.llm.openai_llm import OpenAILLM
        return OpenAILLM(model=model, api_key=key or None)
    elif name == "Anthropic":
        from rag_discovery.providers.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(model=model, api_key=key or None)
    elif name == "VertexAI":
        from rag_discovery.providers.llm.vertexai_llm import VertexAILLM
        return VertexAILLM(model=model,
                           project_id=st.session_state.get("llm_project_id"),
                           location=st.session_state.get("llm_location", "us-central1"))
    elif name == "DeepSeek":
        from rag_discovery.providers.llm.deepseek_llm import DeepSeekLLM
        return DeepSeekLLM(model=model, api_key=key or None)
    raise ValueError(f"Unknown LLM provider: {name}")


def _build_embedding_provider():
    name = st.session_state["embedding_name"]
    model = st.session_state["embedding_model"]
    if name == "SentenceTransformer":
        from rag_discovery.providers.embeddings.sentence_transformer import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model_name=model)
    else:
        from rag_discovery.providers.embeddings.openai_embeddings import OpenAIEmbeddings
        key = st.session_state["llm_api_key"]
        return OpenAIEmbeddings(model=model, api_key=key or None)


def _build_vector_store():
    name = st.session_state["vectorstore_name"]
    persist = st.session_state.get("vectorstore_persist_dir", "./chroma_metadata_wizard")
    collection = st.session_state.get("vectorstore_collection", "metadata_standards")
    if name == "Chroma":
        from rag_discovery.providers.vectorstore.chroma_store import ChromaStore
        return ChromaStore(collection_name=collection, persist_directory=persist)
    else:
        from rag_discovery.providers.vectorstore.faiss_store import FAISSStore
        return FAISSStore(collection_name=collection, persist_directory=persist)


def _build_warehouse_connector():
    wt = st.session_state["warehouse_type"]
    if wt == "Snowflake":
        from warehouse.connectors.snowflake import SnowflakeConnector
        return SnowflakeConnector(
            account=st.session_state["sf_account"],
            username=st.session_state["sf_username"],
            password=st.session_state["sf_password"],
            database=st.session_state["sf_database"],
            schema=st.session_state["sf_schema"],
            warehouse=st.session_state["sf_warehouse"],
            role=st.session_state.get("sf_role") or None,
        )
    elif wt == "BigQuery":
        from warehouse.connectors.bigquery import BigQueryConnector
        return BigQueryConnector(
            project_id=st.session_state["bq_project"],
            dataset=st.session_state["bq_dataset"],
        )
    elif wt == "Redshift":
        from warehouse.connectors.redshift import RedshiftConnector
        return RedshiftConnector(
            host=st.session_state["rs_host"],
            port=int(st.session_state.get("rs_port", 5439)),
            database=st.session_state["rs_database"],
            username=st.session_state["rs_username"],
            password=st.session_state["rs_password"],
        )
    elif wt == "Synapse":
        from warehouse.connectors.synapse import SynapseConnector
        return SynapseConnector(
            server=st.session_state["syn_server"],
            database=st.session_state["syn_database"],
            username=st.session_state["syn_username"],
            password=st.session_state["syn_password"],
        )
    return None


# ---------------------------------------------------------------------------
# Step 1 — Conectores
# ---------------------------------------------------------------------------

def render_step1_connectors() -> None:
    st.header("Etapa 1 de 5 — Conectores")
    st.info(
        "Configure o provedor de IA e, opcionalmente, uma conexão com Data Warehouse e catálogo MDM."
    )

    with st.expander("LLM Provider", expanded=True):
        _section_llm()

    with st.expander("Embeddings & Vector Store", expanded=True):
        _section_embeddings()

    with st.expander("Data Warehouse (opcional)", expanded=False):
        _section_warehouse()

    with st.expander("OpenMetadata — alvo de publicação (opcional)", expanded=False):
        _section_openmetadata()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Testar & Salvar Conexões", type="primary", use_container_width=True):
            _handle_test_save()
    with c2:
        if st.button("Próximo →", use_container_width=True,
                     disabled=not st.session_state["step_completed"].get(1)):
            st.session_state["wizard_step"] = 2
            st.rerun()


def _section_llm() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Provedor", ["OpenAI", "Anthropic", "VertexAI", "DeepSeek"],
                     key="llm_provider_name")
    with c2:
        name = st.session_state["llm_provider_name"]
        models = {
            "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            "Anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
            "VertexAI": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
        }
        st.selectbox("Modelo", models.get(name, []), key="llm_model")

    if name in ("OpenAI", "Anthropic", "DeepSeek"):
        env_map = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY",
                   "DeepSeek": "DEEPSEEK_API_KEY"}
        default_key = os.getenv(env_map.get(name, ""), "")
        st.text_input("API Key", type="password", key="llm_api_key",
                      value=st.session_state.get("llm_api_key") or default_key)
    elif name == "VertexAI":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("GCP Project ID", key="llm_project_id",
                          value=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        with c2:
            st.text_input("Location", value="us-central1", key="llm_location")


def _section_embeddings() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Embedding Provider", ["SentenceTransformer", "OpenAI"],
                     key="embedding_name")
        if st.session_state["embedding_name"] == "SentenceTransformer":
            st.selectbox(
                "Modelo",
                ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2",
                 "all-mpnet-base-v2"],
                key="embedding_model",
            )
        else:
            st.text_input("Modelo", value="text-embedding-3-small", key="embedding_model")
    with c2:
        st.selectbox("Vector Store", ["Chroma", "FAISS"], key="vectorstore_name")
        st.text_input("Diretório de persistência",
                      value="./chroma_metadata_wizard", key="vectorstore_persist_dir")
        st.text_input("Nome da coleção",
                      value="metadata_standards", key="vectorstore_collection")


def _section_warehouse() -> None:
    wt = st.selectbox(
        "Tipo de Warehouse",
        ["None", "Snowflake", "BigQuery", "Redshift", "Synapse"],
        key="warehouse_type",
    )

    if wt == "Snowflake":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Account", key="sf_account")
            st.text_input("Usuário", key="sf_username")
            st.text_input("Senha", type="password", key="sf_password")
        with c2:
            st.text_input("Database", key="sf_database")
            st.text_input("Schema", value="PUBLIC", key="sf_schema")
            st.text_input("Warehouse", key="sf_warehouse")
        st.text_input("Role (opcional)", key="sf_role")

    elif wt == "BigQuery":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Project ID", key="bq_project",
                          value=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        with c2:
            st.text_input("Dataset", key="bq_dataset")

    elif wt == "Redshift":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Host", key="rs_host")
            st.text_input("Porta", value="5439", key="rs_port")
        with c2:
            st.text_input("Database", key="rs_database")
            st.text_input("Usuário", key="rs_username")
        st.text_input("Senha", type="password", key="rs_password")

    elif wt == "Synapse":
        st.text_input("Server",
                      placeholder="myworkspace.sql.azuresynapse.net",
                      key="syn_server")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Database", key="syn_database")
            st.text_input("Usuário", key="syn_username")
        with c2:
            st.text_input("Senha", type="password", key="syn_password")


def _section_openmetadata() -> None:
    st.toggle("Habilitar publicação via OpenMetadata", key="openmetadata_enabled")
    if st.session_state.get("openmetadata_enabled"):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("URL do servidor",
                          placeholder="http://localhost:8585",
                          key="openmetadata_url")
        with c2:
            st.text_input("Token Bearer", type="password", key="openmetadata_token")
        if st.button("Testar conexão OpenMetadata"):
            _test_openmetadata_connection()


def _test_openmetadata_connection() -> None:
    import requests
    url = st.session_state.get("openmetadata_url", "").rstrip("/")
    token = st.session_state.get("openmetadata_token", "")
    if not url:
        st.error("Informe a URL do servidor.")
        return
    try:
        r = requests.get(f"{url}/api/v1/tables",
                         headers={"Authorization": f"Bearer {token}"},
                         params={"limit": 1}, timeout=10)
        if r.ok:
            st.success("Conexão OpenMetadata OK.")
        else:
            st.error(f"Falha ({r.status_code}): {r.text[:200]}")
    except Exception as e:
        st.error(f"Erro ao conectar: {e}")


def _handle_test_save() -> None:
    errors: List[str] = []
    try:
        llm = _build_llm_provider()
        emb = _build_embedding_provider()
        vs = _build_vector_store()
        from metadata_enrichment.agent import MetadataEnrichmentAgent
        agent = MetadataEnrichmentAgent(
            embedding_provider=emb,
            llm_provider=llm,
            vector_store=vs,
            standards_persist_dir=st.session_state.get(
                "vectorstore_persist_dir", "./chroma_metadata_wizard"),
            language="pt-br",
        )
        st.session_state["agent"] = agent
    except Exception as e:
        errors.append(f"Falha ao inicializar agente: {e}")

    if st.session_state.get("warehouse_type", "None") != "None":
        try:
            connector = _build_warehouse_connector()
            connector.test_connection()
            st.session_state["warehouse_connector"] = connector
            # Pre-fill Step 5 publish params for BQ / AWS
            wt = st.session_state["warehouse_type"]
            if wt == "BigQuery":
                st.session_state["pub_bq_project"] = st.session_state.get("bq_project", "")
                st.session_state["pub_bq_dataset"] = st.session_state.get("bq_dataset", "")
            elif wt == "Redshift":
                st.session_state["pub_aws_db"] = st.session_state.get("rs_database", "")
        except Exception as e:
            errors.append(f"Falha na conexão warehouse: {e}")

    if not errors:
        st.session_state["step_completed"][1] = True
        st.success("Conexões validadas. Avance para a Etapa 2.")
    else:
        for err in errors:
            st.error(err)


# ---------------------------------------------------------------------------
# Step 2 — Base de Conhecimento
# ---------------------------------------------------------------------------

def render_step2_knowledge_base() -> None:
    st.header("Etapa 2 de 5 — Base de Conhecimento")
    st.info(
        "Indexe normativos, convenções de nomenclatura e glossários para guiar "
        "a geração de metadados com IA. Esta etapa é opcional mas recomendada."
    )

    tab_upload, tab_dir = st.tabs(["Upload de Arquivos", "Indexar Diretório"])

    with tab_upload:
        _upload_standards()

    with tab_dir:
        _dir_standards()

    _render_standards_stats()

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["wizard_step"] = 1
            st.rerun()
    with c2:
        if st.button("Pular etapa (sem normativos)", use_container_width=True):
            st.session_state["step_completed"][2] = True
            st.session_state["wizard_step"] = 3
            st.rerun()
    with c3:
        if st.button("Próximo →", use_container_width=True,
                     disabled=not st.session_state["step_completed"].get(2)):
            st.session_state["wizard_step"] = 3
            st.rerun()


def _upload_standards() -> None:
    st.markdown(
        "**Formatos suportados:** JSON (array de documentos), Markdown (.md), texto (.txt)  \n"
        "**Schema JSON:** `[{\"id\", \"title\", \"content\", \"category\", \"tags\"}]`"
    )
    uploaded = st.file_uploader(
        "Selecione os arquivos de normativo",
        type=["json", "md", "txt"],
        accept_multiple_files=True,
        key="standards_uploader",
    )
    default_cat = st.selectbox(
        "Categoria padrão (para .md / .txt)",
        ["naming_convention", "classification", "security", "glossary", "quality"],
        key="standards_default_category",
    )

    if uploaded and st.button("Indexar arquivos carregados", type="primary"):
        agent = st.session_state.get("agent")
        if not agent:
            st.error("Configure e teste as conexões na Etapa 1 primeiro.")
            return
        total = 0
        names: List[str] = []
        for f in uploaded:
            suffix = Path(f.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                if suffix == ".json":
                    count = agent.index_standards_from_json(tmp_path)
                else:
                    from metadata_enrichment.standards.standards_rag import StandardDocument
                    with open(tmp_path, encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                    doc = StandardDocument(
                        id=f.name,
                        title=Path(f.name).stem,
                        content=content,
                        category=default_cat,
                    )
                    count = agent.index_standards([doc])
                total += count
                names.append(f.name)
            except Exception as e:
                st.error(f"Erro ao indexar {f.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if total > 0:
            st.session_state["standards_indexed"] = True
            st.session_state["indexed_files"].extend(names)
            st.session_state["step_completed"][2] = True
            st.success(f"{total} documento(s) indexados de {len(names)} arquivo(s).")
            _refresh_standards_stats()


def _dir_standards() -> None:
    st.text_input(
        "Caminho absoluto do diretório de normativos",
        placeholder="/home/user/standards/",
        key="standards_dir_path",
    )
    if st.button("Indexar diretório"):
        path = st.session_state.get("standards_dir_path", "")
        if not path or not Path(path).is_dir():
            st.error("Diretório não encontrado.")
            return
        agent = st.session_state.get("agent")
        if not agent:
            st.error("Configure e teste as conexões na Etapa 1 primeiro.")
            return
        with st.spinner("Indexando..."):
            count = agent.index_standards_from_directory(path)
        st.session_state["standards_indexed"] = True
        st.session_state["step_completed"][2] = True
        st.success(f"{count} documentos indexados.")
        _refresh_standards_stats()


def _refresh_standards_stats() -> None:
    agent = st.session_state.get("agent")
    if agent:
        st.session_state["standards_stats"] = agent.standards_rag.get_statistics()


def _render_standards_stats() -> None:
    stats = st.session_state.get("standards_stats", {})
    if not stats:
        _refresh_standards_stats()
        stats = st.session_state.get("standards_stats", {})
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Documentos indexados", stats.get("total_documents", 0))
        c2.metric("Vetores no store", stats.get("vector_store_count", 0))
        cats = stats.get("categories", {})
        if cats:
            st.caption("Categorias: " + ", ".join(f"{k} ({v})" for k, v in cats.items()))
    indexed = st.session_state.get("indexed_files", [])
    if indexed:
        with st.expander(f"Arquivos indexados ({len(indexed)})"):
            for name in indexed:
                st.write(f"- {name}")


# ---------------------------------------------------------------------------
# Step 3 — Seleção de Dataset
# ---------------------------------------------------------------------------

def render_step3_dataset_selection() -> None:
    st.header("Etapa 3 de 5 — Seleção de Dataset")
    st.info("Escolha a fonte de dados a enriquecer.")

    source = st.radio(
        "Tipo de fonte",
        ["CSV", "Parquet", "Warehouse", "Delta Lake"],
        horizontal=True,
        key="source_type_radio",
    )

    st.divider()

    if source == "CSV":
        _form_file("csv")
    elif source == "Parquet":
        _form_file("parquet")
    elif source == "Warehouse":
        _form_warehouse()
    elif source == "Delta Lake":
        _form_delta()

    st.divider()
    with st.expander("Opções de amostragem"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Tamanho da amostra (linhas)", 10, 5000, 100, key="sample_size")
        with c2:
            st.text_area(
                "Contexto adicional (opcional)",
                placeholder="Domínio de negócio, instruções especiais...",
                key="additional_context",
                height=80,
            )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["wizard_step"] = 2
            st.rerun()
    with c3:
        if st.button("Enriquecer Dataset →", type="primary", use_container_width=True):
            _handle_enrich()


def _form_file(ext: str) -> None:
    accepted = [ext]
    label = "CSV" if ext == "csv" else "Parquet"
    uploaded = st.file_uploader(
        f"Selecione o arquivo {label}",
        type=accepted,
        key="dataset_file_uploader",
    )
    if uploaded:
        st.session_state["uploaded_file_bytes"] = uploaded.getvalue()
        st.session_state["uploaded_file_name"] = uploaded.name
        st.session_state["uploaded_file_ext"] = f".{ext}"

    if ext == "csv":
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], key="csv_encoding")
        with c2:
            st.selectbox(
                "Separador",
                [(",", "Vírgula  ,"), (";", "Ponto-e-vírgula ;"),
                 ("|", "Pipe |"), ("\t", "Tab \\t")],
                format_func=lambda x: x[1],
                key="_csv_sep_tuple",
            )


def _form_warehouse() -> None:
    connector = st.session_state.get("warehouse_connector")
    if not connector:
        st.warning(
            "Nenhum conector de warehouse configurado. "
            "Configure na Etapa 1 e teste a conexão."
        )
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            dbs = connector.list_databases()
            st.selectbox("Database", dbs, key="wh_selected_db")
        except Exception:
            st.text_input("Database", key="wh_selected_db")
    with c2:
        try:
            schemas = connector.list_schemas(
                database=st.session_state.get("wh_selected_db")) or []
            st.selectbox("Schema", schemas, key="wh_selected_schema")
        except Exception:
            st.text_input("Schema", key="wh_selected_schema")
    with c3:
        try:
            tables = connector.list_tables(
                schema=st.session_state.get("wh_selected_schema"),
                database=st.session_state.get("wh_selected_db"),
            ) or []
            table_names = [getattr(t, "full_name", str(t)) for t in tables]
            st.selectbox("Tabela", table_names, key="selected_table")
        except Exception:
            st.text_input("Tabela (schema.tabela)", key="selected_table")


def _form_delta() -> None:
    st.text_input(
        "Caminho da tabela Delta Lake",
        placeholder="/mnt/delta/my_table ou s3://bucket/prefix",
        key="delta_path",
    )
    st.number_input(
        "Versão (opcional — em branco = latest)",
        min_value=0,
        value=None,
        key="delta_version",
    )


def _handle_enrich() -> None:
    agent = st.session_state.get("agent")
    if not agent:
        st.error("Configure e teste as conexões na Etapa 1 primeiro.")
        return

    source = st.session_state["source_type_radio"]
    sample_size = int(st.session_state.get("sample_size", 100))
    context = st.session_state.get("additional_context") or None

    with st.spinner("Amostrando dados e gerando metadados com IA..."):
        try:
            sample_result = None

            if source in ("CSV", "Parquet"):
                file_bytes = st.session_state.get("uploaded_file_bytes")
                file_name = st.session_state.get("uploaded_file_name", "table")
                if not file_bytes:
                    st.error("Nenhum arquivo carregado.")
                    return
                ext = st.session_state.get("uploaded_file_ext", ".csv")
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    f.write(file_bytes)
                    tmp_path = f.name
                try:
                    if source == "CSV":
                        sep_tuple = st.session_state.get("_csv_sep_tuple", (",", ""))
                        sep = sep_tuple[0] if isinstance(sep_tuple, tuple) else ","
                        from metadata_enrichment.sampling.data_sampler import CSVSampler
                        sampler = CSVSampler()
                        sample_result = sampler.sample(
                            tmp_path,
                            sample_size=sample_size,
                            encoding=st.session_state.get("csv_encoding", "utf-8"),
                            separator=sep,
                        )
                    else:
                        from metadata_enrichment.sampling.data_sampler import ParquetSampler
                        sampler = ParquetSampler()
                        sample_result = sampler.sample(tmp_path, sample_size=sample_size)
                finally:
                    os.unlink(tmp_path)

            elif source == "Warehouse":
                table = st.session_state.get("selected_table", "")
                if not table:
                    st.error("Selecione uma tabela.")
                    return
                conn_str = _build_connection_string()
                schema = st.session_state.get("wh_selected_schema") or None
                tbl_name = table.split(".")[-1] if "." in table else table
                from metadata_enrichment.sampling.data_sampler import SQLSampler
                sampler = SQLSampler(conn_str)
                sample_result = sampler.sample(tbl_name, sample_size=sample_size,
                                               schema=schema)

            elif source == "Delta Lake":
                delta_path = st.session_state.get("delta_path", "")
                if not delta_path:
                    st.error("Informe o caminho Delta Lake.")
                    return
                from metadata_enrichment.sampling.data_sampler import DeltaSampler
                sampler = DeltaSampler()
                sample_result = sampler.sample(
                    delta_path,
                    sample_size=sample_size,
                    version=st.session_state.get("delta_version") or None,
                )

            result = agent.enrich(sample_result, additional_context=context)

            st.session_state["sample_result"] = sample_result
            st.session_state["enrichment_result"] = result
            st.session_state["edited_table_fields"] = {
                "description": result.description,
                "business_name": result.business_name,
                "domain": result.domain,
                "tags": ", ".join(result.tags),
                "classification": result.classification,
                "owner_suggestion": result.owner_suggestion,
            }
            st.session_state["edited_columns"] = {
                col.name: {
                    "description": col.description,
                    "business_name": col.business_name,
                    "tags": ", ".join(col.tags),
                    "classification": col.classification,
                    "is_pii": col.is_pii,
                }
                for col in result.columns
            }
            st.session_state["validation_dirty"] = False
            st.session_state["step_completed"][3] = True
            st.session_state["wizard_step"] = 4
            st.rerun()

        except Exception as e:
            st.error(f"Erro durante o enriquecimento: {e}")
            import traceback
            st.code(traceback.format_exc())


def _build_connection_string() -> str:
    wt = st.session_state.get("warehouse_type", "None")
    if wt == "Snowflake":
        return (
            f"snowflake://{st.session_state['sf_username']}:"
            f"{st.session_state['sf_password']}@{st.session_state['sf_account']}/"
            f"{st.session_state['sf_database']}/{st.session_state.get('sf_schema','PUBLIC')}"
            f"?warehouse={st.session_state.get('sf_warehouse','')}"
        )
    elif wt == "BigQuery":
        return f"bigquery://{st.session_state['bq_project']}/{st.session_state['bq_dataset']}"
    elif wt == "Redshift":
        return (
            f"redshift+redshift_connector://{st.session_state['rs_username']}:"
            f"{st.session_state['rs_password']}@{st.session_state['rs_host']}:"
            f"{st.session_state.get('rs_port', 5439)}/{st.session_state['rs_database']}"
        )
    elif wt == "Synapse":
        return (
            f"mssql+pyodbc://{st.session_state['syn_username']}:"
            f"{st.session_state['syn_password']}@{st.session_state['syn_server']}/"
            f"{st.session_state['syn_database']}?driver=ODBC+Driver+17+for+SQL+Server"
        )
    return ""


# ---------------------------------------------------------------------------
# Step 4 — Validação & Edição
# ---------------------------------------------------------------------------

def render_step4_validation() -> None:
    st.header("Etapa 4 de 5 — Validação & Edição")

    result = st.session_state.get("enrichment_result")
    if not result:
        st.error("Nenhum resultado disponível. Volte para a Etapa 3.")
        return

    c_back, c_regen, c_save, c_next = st.columns(4)
    with c_back:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["wizard_step"] = 3
            st.rerun()
    with c_regen:
        if st.button("Regerar Tudo", use_container_width=True):
            _handle_regenerate_all()
    with c_save:
        if st.button("Salvar edições", use_container_width=True):
            _apply_edits_to_result()
            st.success("Edições salvas.")
    with c_next:
        if st.button("Próximo →", type="primary", use_container_width=True):
            _apply_edits_to_result()
            st.session_state["step_completed"][4] = True
            st.session_state["wizard_step"] = 5
            st.rerun()

    st.divider()
    _render_table_editor(result)

    st.subheader("Colunas")
    st.caption(
        f"{len(result.columns)} coluna(s) — "
        f"{sum(1 for c in result.columns if c.is_pii)} com PII"
    )
    for idx, col in enumerate(result.columns):
        _render_column_editor(col, idx)


def _render_table_editor(result) -> None:
    edits = st.session_state["edited_table_fields"]

    st.subheader("Metadados da Tabela")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tabela", result.table_name)
    c2.metric("Linhas", f"{result.row_count:,}")
    c3.metric("Confiança", f"{result.confidence:.0%}")
    if result.has_pii:
        c4.error("Contém PII")
    else:
        c4.success("Sem PII")

    c_left, c_right = st.columns(2)
    with c_left:
        edits["business_name"] = st.text_input(
            "Nome de negócio", value=edits.get("business_name", ""),
            key="edit_tbl_bname")
        domain_options = list(result.__class__.__module__ and
                              ["customer", "sales", "finance", "product",
                               "marketing", "hr", "operations", "analytics", "general"])
        current_domain = edits.get("domain", "general")
        if current_domain not in domain_options:
            domain_options.append(current_domain)
        edits["domain"] = st.selectbox(
            "Domínio", domain_options,
            index=domain_options.index(current_domain),
            key="edit_tbl_domain")
        edits["owner_suggestion"] = st.text_input(
            "Proprietário / Time",
            value=edits.get("owner_suggestion", ""),
            key="edit_tbl_owner")

    with c_right:
        cls_opts = ["public", "internal", "confidential", "restricted"]
        cur_cls = edits.get("classification", "internal")
        if cur_cls not in cls_opts:
            cur_cls = "internal"
        edits["classification"] = st.selectbox(
            "Classificação", cls_opts,
            index=cls_opts.index(cur_cls),
            key="edit_tbl_cls")
        edits["tags"] = st.text_input(
            "Tags (separadas por vírgula)",
            value=edits.get("tags", ""),
            key="edit_tbl_tags")

    edits["description"] = st.text_area(
        "Descrição (PT-BR)",
        value=edits.get("description", ""),
        height=120,
        key="edit_tbl_desc")

    if result.standards_used:
        st.caption("Normativos utilizados: " + ", ".join(result.standards_used))


def _render_column_editor(col, idx: int) -> None:
    edits = st.session_state["edited_columns"].get(col.name, {})
    pii_badge = " ⚠️ PII" if col.is_pii else ""
    conf = col.confidence
    conf_color = "green" if conf >= 0.8 else "orange" if conf >= 0.6 else "red"
    label = f"`{col.name}` — {col.original_type}{pii_badge} — :{conf_color}[{conf:.0%}]"

    expanded = col.is_pii or conf < 0.6
    with st.expander(label, expanded=expanded):
        st.markdown(
            f"**Tipo semântico:** `{col.semantic_type or 'N/A'}` &nbsp;|&nbsp; "
            f"**Nullable:** {'Sim' if col.is_nullable else 'Não'}"
        )
        if col.sample_values:
            st.caption("Amostra: " + ", ".join(str(v)[:40] for v in col.sample_values[:5]))

        c_left, c_right = st.columns(2)
        with c_left:
            edits["description"] = st.text_area(
                "Descrição (PT-BR)",
                value=edits.get("description", col.description),
                height=80,
                key=f"col_desc_{idx}")
            edits["business_name"] = st.text_input(
                "Nome de negócio",
                value=edits.get("business_name", col.business_name),
                key=f"col_bname_{idx}")
        with c_right:
            cls_opts = ["public", "internal", "confidential", "restricted"]
            cur_cls = edits.get("classification", col.classification)
            if cur_cls not in cls_opts:
                cur_cls = "internal"
            edits["classification"] = st.selectbox(
                "Classificação", cls_opts,
                index=cls_opts.index(cur_cls),
                key=f"col_cls_{idx}")
            edits["tags"] = st.text_input(
                "Tags",
                value=edits.get("tags", ", ".join(col.tags)),
                key=f"col_tags_{idx}")
            edits["is_pii"] = st.checkbox(
                "É PII",
                value=edits.get("is_pii", col.is_pii),
                key=f"col_pii_{idx}")

        if col.reasoning:
            with st.popover("Raciocínio do LLM"):
                st.write(col.reasoning)

        if st.button(f"Regerar esta coluna", key=f"regen_col_{idx}"):
            _handle_regenerate_column(col.name)

        st.session_state["edited_columns"][col.name] = edits


def _apply_edits_to_result() -> None:
    result = st.session_state.get("enrichment_result")
    if not result:
        return
    edits = st.session_state["edited_table_fields"]
    result.description = edits.get("description", result.description)
    result.business_name = edits.get("business_name", result.business_name)
    result.domain = edits.get("domain", result.domain)
    result.tags = [t.strip() for t in edits.get("tags", "").split(",") if t.strip()]
    result.classification = edits.get("classification", result.classification)
    result.owner_suggestion = edits.get("owner_suggestion", result.owner_suggestion)

    col_edits = st.session_state["edited_columns"]
    for col in result.columns:
        if col.name in col_edits:
            e = col_edits[col.name]
            col.description = e.get("description", col.description)
            col.business_name = e.get("business_name", col.business_name)
            col.tags = [t.strip() for t in e.get("tags", "").split(",") if t.strip()]
            col.classification = e.get("classification", col.classification)
            col.is_pii = e.get("is_pii", col.is_pii)

    result.pii_columns = [c.name for c in result.columns if c.is_pii]
    result.has_pii = bool(result.pii_columns)
    st.session_state["validation_dirty"] = False


def _handle_regenerate_all() -> None:
    with st.spinner("Regenerando todos os metadados..."):
        _handle_enrich()


def _handle_regenerate_column(col_name: str) -> None:
    agent = st.session_state.get("agent")
    sample_result = st.session_state.get("sample_result")
    result = st.session_state.get("enrichment_result")
    if not agent or not sample_result or not result:
        st.error("Estado insuficiente para regenerar coluna.")
        return

    table_ctx = {
        "domain": st.session_state["edited_table_fields"].get("domain", "general"),
        "description": st.session_state["edited_table_fields"].get("description", ""),
    }
    additional = st.session_state.get("additional_context") or None

    with st.spinner(f"Regenerando '{col_name}'..."):
        try:
            new_col = agent.regenerate_column(
                column_name=col_name,
                sample_result=sample_result,
                table_context=table_ctx,
                additional_context=additional,
            )
            for i, col in enumerate(result.columns):
                if col.name == col_name:
                    result.columns[i] = new_col
                    break
            st.session_state["edited_columns"][col_name] = {
                "description": new_col.description,
                "business_name": new_col.business_name,
                "tags": ", ".join(new_col.tags),
                "classification": new_col.classification,
                "is_pii": new_col.is_pii,
            }
            st.session_state["validation_dirty"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao regenerar '{col_name}': {e}")


# ---------------------------------------------------------------------------
# Step 5 — Publicação
# ---------------------------------------------------------------------------

# ---- Generators ------------------------------------------------------------

def generate_ddl_sql(result, schema: str) -> str:
    """Gera COMMENT ON TABLE/COLUMN (ANSI SQL — compatível com PostgreSQL, Snowflake)."""
    esc = lambda s: s.replace("'", "''")
    lines = [
        f"-- Metadados gerados automaticamente pelo Metadata Enrichment Wizard",
        f"-- Tabela: {result.table_name}  |  Gerado em: {result.enriched_at}",
        "",
        f"COMMENT ON TABLE {schema}.{result.table_name}",
        f"    IS '{esc(result.description)}';",
        "",
    ]
    for col in result.columns:
        lines.append(
            f"COMMENT ON COLUMN {schema}.{result.table_name}.{col.name}"
        )
        lines.append(f"    IS '{esc(col.description)}';")
    return "\n".join(lines)


def generate_bq_cli(result, project: str, dataset: str) -> str:
    """Gera comandos `bq` CLI para BigQuery."""
    esc = lambda s: s.replace('"', '\\"')
    fqn = f"{project}:{dataset}.{result.table_name}"
    label_flags = " ".join(
        f'--set_label {t.lower().replace(" ", "_")}:true'
        for t in result.tags[:8]
    )
    schema_cols = ",\n".join(
        f'  {{"name":"{col.name}","type":"{col.original_type.upper()}","description":"{esc(col.description)}"}}'
        for col in result.columns
    )
    lines = [
        f"# Comandos bq CLI — gerados pelo Metadata Enrichment Wizard",
        f"# Projeto: {project}  Dataset: {dataset}  Tabela: {result.table_name}",
        "",
        f"# 1. Atualizar descrição e labels da tabela",
        f'bq update --description "{esc(result.description)}" \\',
        f"  {label_flags} \\",
        f"  {fqn}",
        "",
        f"# 2. Atualizar descrições das colunas via schema JSON",
        f"# Salve o JSON abaixo em schema.json e execute o comando seguinte:",
        f"",
        f"# cat > /tmp/{result.table_name}_schema.json << 'EOF'",
        f"[",
        schema_cols,
        f"]",
        f"# EOF",
        f"",
        f"# bq update --schema /tmp/{result.table_name}_schema.json {fqn}",
        "",
        f"# 3. Verificar resultado",
        f"# bq show --format=prettyjson {fqn}",
    ]
    return "\n".join(lines)


def generate_aws_glue_boto3(result, region: str, db_name: str) -> str:
    """Gera código Python boto3 para AWS Glue Data Catalog."""
    esc = lambda s: s.replace("'", "\\'").replace('"', '\\"')
    cols_repr = ",\n        ".join(
        f'{{"Name": "{col.name}", "Type": "{col.original_type}", '
        f'"Comment": "{esc(col.description)}"}}'
        for col in result.columns
    )
    lines = [
        "# Código boto3 — gerado pelo Metadata Enrichment Wizard",
        f"# Região: {region}  Database Glue: {db_name}  Tabela: {result.table_name}",
        "",
        "import boto3",
        "",
        f'client = boto3.client("glue", region_name="{region}")',
        "",
        "# Obter definição atual da tabela (necessário para preservar StorageDescriptor)",
        f'response = client.get_table(DatabaseName="{db_name}", Name="{result.table_name}")',
        "table_def = response[\"Table\"]",
        "",
        "# Montar TableInput com metadados enriquecidos",
        "table_input = {",
        f'    "Name": "{result.table_name}",',
        f'    "Description": "{esc(result.description)}",',
        "    \"Parameters\": {",
        f'        **table_def.get("Parameters", {{}}),',
        f'        "classification": "{result.classification}",',
        f'        "domain": "{result.domain}",',
        f'        "business_name": "{esc(result.business_name)}",',
        f'        "tags": "{", ".join(result.tags)}",',
        f'        "owner": "{esc(result.owner_suggestion)}",',
        "    },",
        "    \"StorageDescriptor\": {",
        "        **table_def[\"StorageDescriptor\"],",
        "        \"Columns\": [",
        f"        {cols_repr}",
        "        ],",
        "    },",
        "}",
        "",
        "client.update_table(",
        f'    DatabaseName="{db_name}",',
        "    TableInput=table_input,",
        ")",
        f'print("Tabela {result.table_name} atualizada no Glue Data Catalog.")',
    ]
    return "\n".join(lines)


def generate_synapse_sql(result, schema: str) -> str:
    """Gera T-SQL com sp_addextendedproperty para Azure Synapse / SQL Server."""
    esc = lambda s: s.replace("'", "''")
    lines = [
        f"-- T-SQL Extended Properties — Azure Synapse / SQL Server",
        f"-- Gerado pelo Metadata Enrichment Wizard",
        f"-- Schema: {schema}  Tabela: {result.table_name}",
        "",
        f"-- Remover propriedade existente (se houver)",
        f"IF EXISTS (",
        f"    SELECT 1 FROM sys.extended_properties",
        f"    WHERE major_id = OBJECT_ID(N'{schema}.{result.table_name}')",
        f"    AND name = N'MS_Description' AND minor_id = 0",
        f")",
        f"    EXEC sys.sp_dropextendedproperty",
        f"        @name = N'MS_Description',",
        f"        @level0type = N'SCHEMA', @level0name = N'{schema}',",
        f"        @level1type = N'TABLE',  @level1name = N'{result.table_name}';",
        "",
        f"-- Descrição da tabela",
        f"EXEC sys.sp_addextendedproperty",
        f"    @name      = N'MS_Description',",
        f"    @value     = N'{esc(result.description)}',",
        f"    @level0type = N'SCHEMA', @level0name = N'{schema}',",
        f"    @level1type = N'TABLE',  @level1name = N'{result.table_name}';",
        "",
        f"-- Classificação e domínio da tabela",
        f"EXEC sys.sp_addextendedproperty",
        f"    @name      = N'Classification',",
        f"    @value     = N'{result.classification}',",
        f"    @level0type = N'SCHEMA', @level0name = N'{schema}',",
        f"    @level1type = N'TABLE',  @level1name = N'{result.table_name}';",
        "",
        "-- Descrições das colunas",
    ]
    for col in result.columns:
        lines += [
            f"IF EXISTS (",
            f"    SELECT 1 FROM sys.extended_properties",
            f"    WHERE major_id = OBJECT_ID(N'{schema}.{result.table_name}')",
            f"    AND name = N'MS_Description'",
            f"    AND minor_id = COLUMNPROPERTY(",
            f"        OBJECT_ID(N'{schema}.{result.table_name}'), N'{col.name}', 'ColumnId')",
            f")",
            f"    EXEC sys.sp_dropextendedproperty",
            f"        @name = N'MS_Description',",
            f"        @level0type = N'SCHEMA', @level0name = N'{schema}',",
            f"        @level1type = N'TABLE',  @level1name = N'{result.table_name}',",
            f"        @level2type = N'COLUMN', @level2name = N'{col.name}';",
            "",
            f"EXEC sys.sp_addextendedproperty",
            f"    @name      = N'MS_Description',",
            f"    @value     = N'{esc(col.description)}',",
            f"    @level0type = N'SCHEMA', @level0name = N'{schema}',",
            f"    @level1type = N'TABLE',  @level1name = N'{result.table_name}',",
            f"    @level2type = N'COLUMN', @level2name = N'{col.name}';",
            "",
        ]
    return "\n".join(lines)


# ---- Step 5 renderer -------------------------------------------------------

def render_step5_publish() -> None:
    st.header("Etapa 5 de 5 — Publicação")

    result = st.session_state.get("enrichment_result")
    if not result:
        st.error("Nenhum resultado disponível. Volte para a Etapa 4.")
        return

    _apply_edits_to_result()

    _render_publish_summary(result)

    st.divider()
    st.subheader("Formatos de Saída")

    tabs = st.tabs([
        "JSON", "Markdown",
        "DDL SQL", "BQ CLI", "AWS Glue (boto3)", "Azure Synapse SQL",
        "OpenMetadata"
    ])

    # ---- JSON ----
    with tabs[0]:
        st.caption("Download do catálogo completo em JSON.")
        st.download_button(
            "Baixar JSON",
            data=result.to_json(),
            file_name=f"{result.table_name}_metadata.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Preview"):
            st.json(json.loads(result.to_json()))

    # ---- Markdown ----
    with tabs[1]:
        st.caption("Download da documentação em Markdown (pronta para Data Catalog / wiki).")
        md_content = result.to_markdown()
        st.download_button(
            "Baixar Markdown",
            data=md_content,
            file_name=f"{result.table_name}_metadata.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander("Preview"):
            st.markdown(md_content)

    # ---- DDL SQL ----
    with tabs[2]:
        st.caption(
            "Instruções `COMMENT ON TABLE/COLUMN` (ANSI SQL — "
            "compatível com PostgreSQL, Snowflake, DuckDB)."
        )
        schema = st.text_input("Schema", value="public", key="pub_ddl_schema")
        ddl = generate_ddl_sql(result, schema)
        st.code(ddl, language="sql")
        st.download_button(
            "Baixar DDL SQL",
            data=ddl,
            file_name=f"{result.table_name}_metadata_comments.sql",
            mime="text/plain",
            use_container_width=True,
        )

    # ---- BQ CLI ----
    with tabs[3]:
        st.caption("Comandos `bq update` para atualizar descrições no Google BigQuery.")
        c1, c2 = st.columns(2)
        with c1:
            project = st.text_input(
                "GCP Project ID",
                value=st.session_state.get("pub_bq_project") or
                      st.session_state.get("bq_project", "my-project"),
                key="pub_bq_project_input",
            )
        with c2:
            dataset = st.text_input(
                "Dataset",
                value=st.session_state.get("pub_bq_dataset") or
                      st.session_state.get("bq_dataset", "my_dataset"),
                key="pub_bq_dataset_input",
            )
        bq_script = generate_bq_cli(result, project or "my-project", dataset or "my_dataset")
        st.code(bq_script, language="bash")
        st.download_button(
            "Baixar script BQ CLI (.sh)",
            data=bq_script,
            file_name=f"{result.table_name}_bq_update.sh",
            mime="text/plain",
            use_container_width=True,
        )

    # ---- AWS Glue ----
    with tabs[4]:
        st.caption(
            "Código Python boto3 para atualizar o AWS Glue Data Catalog "
            "(compatível com Athena, Redshift Spectrum, Lake Formation)."
        )
        c1, c2 = st.columns(2)
        with c1:
            region = st.text_input(
                "Região AWS",
                value=st.session_state.get("pub_aws_region", "us-east-1"),
                key="pub_aws_region_input",
            )
        with c2:
            db_name = st.text_input(
                "Database no Glue Catalog",
                value=st.session_state.get("pub_aws_db") or
                      st.session_state.get("rs_database", "my_database"),
                key="pub_aws_db_input",
            )
        aws_code = generate_aws_glue_boto3(
            result, region or "us-east-1", db_name or "my_database"
        )
        st.code(aws_code, language="python")
        st.download_button(
            "Baixar script AWS Glue (.py)",
            data=aws_code,
            file_name=f"{result.table_name}_aws_glue_update.py",
            mime="text/plain",
            use_container_width=True,
        )

    # ---- Synapse SQL ----
    with tabs[5]:
        st.caption(
            "T-SQL com `sp_addextendedproperty` para Azure Synapse Dedicated Pool "
            "e SQL Server (compatível com SSMS, Azure Data Studio)."
        )
        syn_schema = st.text_input(
            "Schema",
            value=st.session_state.get("pub_synapse_schema", "dbo"),
            key="pub_synapse_schema_input",
        )
        syn_sql = generate_synapse_sql(result, syn_schema or "dbo")
        st.code(syn_sql, language="sql")
        st.download_button(
            "Baixar T-SQL Synapse (.sql)",
            data=syn_sql,
            file_name=f"{result.table_name}_synapse_metadata.sql",
            mime="text/plain",
            use_container_width=True,
        )

    # ---- OpenMetadata ----
    with tabs[6]:
        if not st.session_state.get("openmetadata_enabled"):
            st.info(
                "OpenMetadata não configurado. Habilite na Etapa 1 → "
                "seção 'OpenMetadata — alvo de publicação'."
            )
        else:
            st.caption("Publicar metadados diretamente via REST API do OpenMetadata.")
            st.text(f"Servidor: {st.session_state.get('openmetadata_url', '')}")
            st.text(f"Tabela: {result.table_name}")
            if st.button("Publicar no OpenMetadata", type="primary"):
                _publish_openmetadata(result)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Voltar para Validação", use_container_width=True):
            st.session_state["wizard_step"] = 4
            st.rerun()
    with c2:
        if st.button("Novo Enriquecimento", use_container_width=True):
            _reset_wizard()


def _render_publish_summary(result) -> None:
    st.subheader("Resumo")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tabela", result.table_name)
    c2.metric("Nome de negócio", result.business_name)
    c3.metric("Domínio", result.domain)
    c4.metric("Classificação", result.classification)
    c5.metric("Colunas", len(result.columns))

    if result.has_pii:
        st.warning(f"Contém PII em: {', '.join(result.pii_columns)}")

    with st.expander("Descrição completa"):
        st.write(result.description)

    with st.expander(f"Resumo das {len(result.columns)} colunas"):
        import pandas as pd
        rows = []
        for col in result.columns:
            rows.append({
                "Coluna": col.name,
                "Tipo": col.original_type,
                "Descrição": col.description[:70] + ("..." if len(col.description) > 70 else ""),
                "Classificação": col.classification,
                "PII": "⚠️" if col.is_pii else "",
                "Confiança": f"{col.confidence:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _publish_openmetadata(result) -> None:
    try:
        import requests
    except ImportError:
        st.error("Pacote 'requests' não disponível.")
        return

    url = st.session_state.get("openmetadata_url", "").rstrip("/")
    token = st.session_state.get("openmetadata_token", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with st.spinner("Publicando no OpenMetadata..."):
        try:
            r = requests.get(
                f"{url}/api/v1/tables/name/{result.table_name}",
                headers=headers, timeout=15,
            )
            if r.status_code == 404:
                st.error(f"Tabela '{result.table_name}' não encontrada no OpenMetadata.")
                return

            table_id = r.json().get("id")
            if not table_id:
                st.error("Não foi possível obter o ID da tabela.")
                return

            patch_body = [
                {"op": "add", "path": "/description",
                 "value": result.description},
                {"op": "add", "path": "/displayName",
                 "value": result.business_name},
                {"op": "add", "path": "/tags",
                 "value": [{"tagFQN": t} for t in result.tags]},
            ]
            r2 = requests.patch(
                f"{url}/api/v1/tables/{table_id}",
                json=patch_body, headers=headers, timeout=15,
            )
            if r2.ok:
                st.success(f"Publicado com sucesso: {result.table_name}")
                st.session_state["step_completed"][5] = True
            else:
                st.error(f"Erro PATCH ({r2.status_code}): {r2.text[:300]}")

        except Exception as e:
            st.error(f"Erro ao publicar: {e}")


def _reset_wizard() -> None:
    for key in list(st.session_state.keys()):
        if key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
    st.session_state["wizard_step"] = 1
    st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Metadata Enrichment Wizard",
        page_icon="🧙",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()

    st.title("🧙 Metadata Enrichment Wizard")
    st.markdown(
        "Fluxo guiado para enriquecer metadados com IA — "
        "conecte fontes, indexe normativos, selecione datasets, "
        "valide os resultados e publique no catálogo."
    )

    render_progress_bar(st.session_state["wizard_step"])
    st.divider()

    step = st.session_state["wizard_step"]
    if step == 1:
        render_step1_connectors()
    elif step == 2:
        render_step2_knowledge_base()
    elif step == 3:
        render_step3_dataset_selection()
    elif step == 4:
        render_step4_validation()
    elif step == 5:
        render_step5_publish()


if __name__ == "__main__":
    main()
