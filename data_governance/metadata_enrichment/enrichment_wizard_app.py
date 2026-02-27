"""
Metadata Enrichment Wizard
Fluxo guiado de 6 etapas para enriquecimento de metadados com IA.

Etapas:
  1. Conectores — Cloud DWH, MDM, LLM provider
  2. Base de Conhecimento — indexação de normativos
  3. Diagnóstico — leitura e avaliação semântica dos metadados existentes
  4. Seleção de Dataset — arquivo ou warehouse (enriquecimento)
  5. Validação & Edição — revisão coluna a coluna com comparação antes/depois
  6. Publicação — DDL SQL, BQ CLI, AWS Glue, Synapse, JSON, Markdown, OpenMetadata
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
    "step_completed": {1: False, 2: False, 3: False, 4: False, 5: False, 6: False},
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
    # Step 3 — Diagnosis
    "diagnosis_results": [],        # List[TableMetadataDiagnosis]
    "selected_for_enrichment": [],  # List[str] table names selected by user
    "original_metadata": {},        # Dict[table_name, {description, owner, tags, classification, columns}]
    "pii_diagnosis_results": [],    # List[TableClassification]
    "wh_scan_db": "",
    "wh_scan_schema": "",
    # Step 4 — Dataset (file or manual warehouse selection)
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
    # Step 5 — Validation (multi-table)
    "enrichment_result": None,
    "enrichment_results_list": [],  # List[EnrichmentResult] for multi-table
    "current_enrichment_idx": 0,
    "sample_result": None,
    "edited_table_fields": {},
    "edited_columns": {},
    "regenerating_column": None,
    "validation_dirty": False,
    # Step 6 — Publish config
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

STEP_LABELS = {
    1: "Conectores",
    2: "Base de Conhecimento",
    3: "Diagnóstico",
    4: "Dataset",
    5: "Validar & Editar",
    6: "Publicar",
}


def render_progress_bar(current_step: int) -> None:
    completed = st.session_state.get("step_completed", {})
    cols = st.columns(6)
    for col, num in zip(cols, range(1, 7)):
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
    st.header("Etapa 1 de 6 — Conectores")
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
    st.header("Etapa 2 de 6 — Base de Conhecimento")
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
# Step 3 — Diagnóstico de Metadados (NEW)
# ---------------------------------------------------------------------------

def render_step3_diagnosis() -> None:
    st.header("Etapa 3 de 6 — Diagnóstico de Metadados")
    st.info(
        "O sistema lê os metadados existentes no seu catálogo ou warehouse e avalia "
        "semanticamente a qualidade de cada descrição. Tabelas com metadados ausentes "
        "ou pobres são identificadas e priorizadas para enriquecimento."
    )

    has_warehouse = st.session_state.get("warehouse_connector") is not None
    has_openmetadata = (
        st.session_state.get("openmetadata_enabled")
        and st.session_state.get("openmetadata_url", "")
    )

    c_back, _, c_skip = st.columns([1, 2, 1])
    with c_back:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["wizard_step"] = 2
            st.rerun()
    with c_skip:
        if st.button("Usar arquivo →", use_container_width=True,
                     help="Pula o diagnóstico e vai para seleção manual de arquivo/tabela"):
            st.session_state["step_completed"][3] = True
            st.session_state["wizard_step"] = 4
            st.rerun()

    st.divider()

    if not has_warehouse and not has_openmetadata:
        st.warning(
            "Nenhum warehouse ou OpenMetadata configurado. "
            "O diagnóstico automático requer uma fonte conectada. "
            "Use o botão **'Usar arquivo →'** para selecionar um dataset manualmente."
        )
        return

    # --- Source selection for scan ---
    scan_sources = []
    if has_warehouse:
        scan_sources.append("Warehouse")
    if has_openmetadata:
        scan_sources.append("OpenMetadata")

    scan_source = st.radio(
        "Fonte para diagnóstico",
        scan_sources,
        horizontal=True,
        key="diagnosis_scan_source",
    )

    if scan_source == "Warehouse":
        _render_diagnosis_warehouse_scan()
    else:
        _render_diagnosis_openmetadata_scan()


def _render_diagnosis_warehouse_scan() -> None:
    """Scan warehouse tables, read existing metadata, run quality evaluation."""
    connector = st.session_state.get("warehouse_connector")
    if not connector:
        st.error("Conector de warehouse não disponível.")
        return

    c1, c2 = st.columns(2)
    with c1:
        try:
            dbs = connector.list_databases()
            st.selectbox("Database", dbs, key="wh_scan_db")
        except Exception:
            st.text_input("Database", key="wh_scan_db")
    with c2:
        try:
            schemas = connector.list_schemas(
                database=st.session_state.get("wh_scan_db")) or []
            st.selectbox("Schema", schemas, key="wh_scan_schema")
        except Exception:
            st.text_input("Schema", key="wh_scan_schema")

    with st.expander("Opções de enriquecimento", expanded=False):
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

    if st.button("Escanear Catálogo e Diagnosticar", type="primary"):
        _run_warehouse_diagnosis()

    _render_diagnosis_panel()


def _run_warehouse_diagnosis() -> None:
    """Read metadata from warehouse and run LLM quality evaluation."""
    connector = st.session_state.get("warehouse_connector")
    agent = st.session_state.get("agent")
    if not agent or not connector:
        st.error("Configure e teste as conexões na Etapa 1.")
        return

    scan_schema = st.session_state.get("wh_scan_schema") or None
    scan_db = st.session_state.get("wh_scan_db") or None

    with st.spinner("Listando tabelas..."):
        try:
            table_list = connector.list_tables(schema=scan_schema, database=scan_db) or []
        except Exception as e:
            st.error(f"Erro ao listar tabelas: {e}")
            return

    if not table_list:
        st.warning("Nenhuma tabela encontrada no schema selecionado.")
        return

    progress = st.progress(0)
    status_text = st.empty()
    table_dicts = []

    for i, tbl in enumerate(table_list):
        tbl_name = getattr(tbl, "name", str(tbl))
        tbl_schema = getattr(tbl, "schema", scan_schema) or scan_schema or ""
        tbl_db = getattr(tbl, "database", scan_db) or scan_db or ""
        status_text.text(f"Lendo metadados: {tbl_name} ({i+1}/{len(table_list)})")
        progress.progress((i + 1) / len(table_list) * 0.5)

        description = ""
        row_count = getattr(tbl, "row_count", None)
        last_modified = getattr(tbl, "last_modified", None)
        columns = []

        try:
            # Read table-level description (BigQuery exposes it via get_table_info)
            info = connector.get_table_info(tbl_name, tbl_schema, tbl_db)
            if info:
                meta = getattr(info, "metadata", {}) or {}
                description = meta.get("description", "") or ""
                row_count = row_count or getattr(info, "row_count", None)
                last_modified = last_modified or getattr(info, "last_modified", None)
        except Exception:
            pass

        try:
            schema_cols = connector.get_table_schema(tbl_name, tbl_schema, tbl_db) or []
            for col in schema_cols:
                columns.append({
                    "name": col.get("name", ""),
                    "description": col.get("comment", "") or "",
                })
        except Exception:
            pass

        table_dicts.append({
            "name": tbl_name,
            "schema": tbl_schema,
            "database": tbl_db,
            "description": description,
            "owner": "",
            "tags": [],
            "classification": "",
            "columns": columns,
            "row_count": row_count,
            "last_modified": str(last_modified) if last_modified else None,
            "source": "warehouse",
        })

    status_text.text("Avaliando qualidade dos metadados com IA...")
    from metadata_enrichment.scorer import MetadataQualityEvaluator

    evaluator = MetadataQualityEvaluator(agent.llm_provider)
    results = []
    total = len(table_dicts)
    for i, td in enumerate(table_dicts):
        status_text.text(f"Avaliando: {td['name']} ({i+1}/{total})")
        progress.progress(0.5 + (i + 1) / total * 0.5)
        results.append(evaluator.evaluate(td))

    progress.progress(1.0)
    status_text.text(f"Diagnóstico concluído — {total} tabelas avaliadas.")

    # PII estimation from column names (fast, no sampling)
    try:
        from classification.data_classification_agent import DataClassificationAgent
    except ImportError:
        from data_governance.classification.data_classification_agent import DataClassificationAgent
    pii_clf = DataClassificationAgent(llm_provider=agent.llm_provider)
    pii_results = pii_clf.classify_batch_from_dicts(table_dicts)

    # Pre-select absent + poor tables
    pre_selected = [r.table_name for r in results if r.status in ("absent", "poor")]
    st.session_state["diagnosis_results"] = results
    st.session_state["pii_diagnosis_results"] = pii_results
    st.session_state["selected_for_enrichment"] = pre_selected


def _render_diagnosis_openmetadata_scan() -> None:
    """Fetch tables from OpenMetadata and run quality evaluation."""
    with st.expander("Opções de enriquecimento", expanded=False):
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

    if st.button("Escanear OpenMetadata e Diagnosticar", type="primary"):
        _run_openmetadata_diagnosis()

    _render_diagnosis_panel()


def _run_openmetadata_diagnosis() -> None:
    """Fetch tables from OpenMetadata and evaluate metadata quality."""
    agent = st.session_state.get("agent")
    url = st.session_state.get("openmetadata_url", "").rstrip("/")
    token = st.session_state.get("openmetadata_token", "")
    if not agent or not url:
        st.error("Configure o agente e o OpenMetadata na Etapa 1.")
        return

    try:
        from rag_discovery.openmetadata_connector import OpenMetadataConnector
    except ImportError:
        from data_governance.rag_discovery.openmetadata_connector import OpenMetadataConnector

    with st.spinner("Buscando tabelas no OpenMetadata..."):
        try:
            omd = OpenMetadataConnector(server_url=url, bearer_token=token)
            tables = omd.fetch_tables()
        except Exception as e:
            st.error(f"Erro ao buscar tabelas: {e}")
            return

    if not tables:
        st.warning("Nenhuma tabela encontrada no OpenMetadata.")
        return

    table_dicts = []
    for tm in tables:
        columns = []
        for col in (tm.columns or []):
            if isinstance(col, dict):
                columns.append({
                    "name": col.get("name", ""),
                    "description": col.get("description", "") or "",
                })
            else:
                columns.append({
                    "name": getattr(col, "name", ""),
                    "description": getattr(col, "description", "") or "",
                })
        table_dicts.append({
            "name": tm.name,
            "schema": tm.schema,
            "database": tm.database,
            "description": tm.description or "",
            "owner": tm.owner or "",
            "tags": list(tm.tags or []),
            "classification": "",
            "columns": columns,
            "row_count": tm.row_count,
            "last_modified": tm.updated_at,
            "source": "openmetadata",
        })

    from metadata_enrichment.scorer import MetadataQualityEvaluator
    evaluator = MetadataQualityEvaluator(agent.llm_provider)

    progress = st.progress(0)
    status_text = st.empty()
    results = []
    total = len(table_dicts)
    for i, td in enumerate(table_dicts):
        status_text.text(f"Avaliando: {td['name']} ({i+1}/{total})")
        progress.progress((i + 1) / total)
        results.append(evaluator.evaluate(td))

    progress.progress(1.0)
    status_text.text(f"Diagnóstico concluído — {total} tabelas avaliadas.")

    # PII estimation from column names (fast, no sampling)
    try:
        from classification.data_classification_agent import DataClassificationAgent
    except ImportError:
        from data_governance.classification.data_classification_agent import DataClassificationAgent
    pii_clf = DataClassificationAgent(llm_provider=agent.llm_provider)
    pii_results = pii_clf.classify_batch_from_dicts(table_dicts)

    pre_selected = [r.table_name for r in results if r.status in ("absent", "poor")]
    st.session_state["diagnosis_results"] = results
    st.session_state["pii_diagnosis_results"] = pii_results
    st.session_state["selected_for_enrichment"] = pre_selected


def _render_diagnosis_panel() -> None:
    """Render the triage dashboard: KPIs, filterable table list, selection controls, PII audit."""
    results = st.session_state.get("diagnosis_results", [])
    if not results:
        return

    pii_results = st.session_state.get("pii_diagnosis_results", [])
    pii_by_name = {p.table_name: p for p in pii_results}

    # --- KPIs ---
    n_absent = sum(1 for r in results if r.status == "absent")
    n_poor = sum(1 for r in results if r.status == "poor")
    n_ok = sum(1 for r in results if r.status == "sufficient")
    avg_score = sum(r.quality_score for r in results) / len(results) if results else 0.0
    n_pii_high = sum(1 for p in pii_results if p.risk_level == "high")
    n_pii_any = sum(1 for p in pii_results if p.has_pii)

    st.divider()
    st.subheader("Resultado do Diagnóstico")

    tab_quality, tab_pii = st.tabs(["Qualidade de Metadados", "Auditoria PII / LGPD"])

    with tab_quality:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total de tabelas", len(results))
        c2.metric("🔴 Ausente", n_absent)
        c3.metric("🟡 Pobre", n_poor)
        c4.metric("🟢 Suficiente", n_ok)
        c5.metric("Score médio", f"{avg_score:.0%}")

        st.divider()

        # Filters
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            search = st.text_input("Buscar tabela", placeholder="nome...", key="diag_search")
        with fc2:
            filter_status = st.multiselect(
                "Filtrar por status",
                ["🔴 Ausente", "🟡 Pobre", "🟢 Suficiente"],
                default=["🔴 Ausente", "🟡 Pobre"],
                key="diag_filter_status",
            )

        status_map = {"🔴 Ausente": "absent", "🟡 Pobre": "poor", "🟢 Suficiente": "sufficient"}
        allowed = {status_map[s] for s in filter_status}
        filtered = [
            r for r in results
            if r.status in allowed
            and (not search or search.lower() in r.table_name.lower())
        ]
        filtered.sort(key=lambda r: r.quality_score)

        # Selection buttons
        sel_state = st.session_state.get("selected_for_enrichment", [])
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if st.button("Selecionar Ausentes", use_container_width=True):
                st.session_state["selected_for_enrichment"] = [
                    r.table_name for r in results if r.status == "absent"
                ]
                st.rerun()
        with bc2:
            if st.button("Selecionar Pobres", use_container_width=True):
                st.session_state["selected_for_enrichment"] = [
                    r.table_name for r in results if r.status == "poor"
                ]
                st.rerun()
        with bc3:
            if st.button("Selecionar Tudo", use_container_width=True):
                st.session_state["selected_for_enrichment"] = [r.table_name for r in results]
                st.rerun()
        with bc4:
            if st.button("Limpar seleção", use_container_width=True):
                st.session_state["selected_for_enrichment"] = []
                st.rerun()

        # Table list with PII column
        _STATUS_ICON = {"absent": "🔴", "poor": "🟡", "sufficient": "🟢"}
        _PII_ICON = {"high": "🔒 Alto", "medium": "🟠 Médio", "low": "🔵 Baixo", "none": ""}
        for r in filtered:
            is_selected = r.table_name in sel_state
            pii_d = pii_by_name.get(r.table_name)
            c_check, c_name, c_score, c_status, c_cols, c_pii = st.columns(
                [0.5, 2.5, 1.2, 1.5, 1.5, 1.5]
            )
            with c_check:
                checked = st.checkbox(
                    "", value=is_selected, key=f"diag_sel_{r.table_name}",
                    label_visibility="collapsed",
                )
                if checked and r.table_name not in sel_state:
                    sel_state = sel_state + [r.table_name]
                    st.session_state["selected_for_enrichment"] = sel_state
                elif not checked and r.table_name in sel_state:
                    sel_state = [t for t in sel_state if t != r.table_name]
                    st.session_state["selected_for_enrichment"] = sel_state
            with c_name:
                st.markdown(f"`{r.table_name}`")
                if r.schema:
                    st.caption(r.schema)
            with c_score:
                st.markdown(f"**{r.quality_score:.0%}**")
            with c_status:
                st.markdown(_STATUS_ICON.get(r.status, "") + f" {r.status.capitalize()}")
            with c_cols:
                n_enrich = len(r.columns_to_enrich)
                n_total = len(r.column_qualities)
                st.caption(f"{n_enrich}/{n_total} colunas")
            with c_pii:
                if pii_d and pii_d.has_pii:
                    icon = _PII_ICON.get(pii_d.risk_level, "")
                    st.markdown(icon)
                    if pii_d.lgpd_sensitive_columns:
                        st.caption("Sensível LGPD")

    with tab_pii:
        _render_pii_audit_panel(pii_results, n_pii_high, n_pii_any)

    # --- Enrich button ---
    st.divider()
    n_sel = len(st.session_state.get("selected_for_enrichment", []))
    if n_sel:
        if st.button(
            f"Enriquecer Selecionadas ({n_sel} tabelas) →",
            type="primary",
            use_container_width=True,
        ):
            _handle_enrich_from_diagnosis()
    else:
        st.info("Selecione ao menos uma tabela para enriquecer.")
        if st.button("Ir para Seleção Manual →", use_container_width=True):
            st.session_state["step_completed"][3] = True
            st.session_state["wizard_step"] = 4
            st.rerun()


def _render_pii_audit_panel(pii_results: list, n_high: int, n_any: int) -> None:
    """Render the PII / LGPD audit tab content."""
    if not pii_results:
        st.info("Execute o diagnóstico para ver a auditoria PII.")
        return

    _RISK_ICON = {"high": "🔒", "medium": "🟠", "low": "🔵", "none": "⬜"}
    _LGPD_LABEL = {
        "pessoal_ordinario": "Pessoal ordinário",
        "pessoal_sensivel": "Pessoal sensível (Art. 5 IX)",
        "nao_pessoal": "Não pessoal",
    }

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tabelas analisadas", len(pii_results))
    c2.metric("🔒 Risco alto", n_high)
    c3.metric("Com PII detectado", n_any)
    c4.metric("Sem PII estimado", len(pii_results) - n_any)

    if n_high:
        st.error(
            f"**{n_high} tabela(s) com risco alto** — contêm dados pessoais sensíveis (LGPD Art. 5 IX) "
            "ou identificadores diretos (CPF, cartão). Revise a classificação antes de publicar."
        )
    elif n_any:
        st.warning(
            f"**{n_any} tabela(s) com PII detectado** — verifique classificação e controles de acesso."
        )
    else:
        st.success("Nenhuma coluna com indicadores PII detectada nos nomes de coluna.")

    st.caption(
        "Detecção baseada em nomes de coluna (sem acesso aos dados). "
        "Execute o enriquecimento para confirmação por amostragem."
    )

    # Per-table PII detail
    pii_tables = [p for p in pii_results if p.has_pii]
    pii_tables.sort(key=lambda p: {"high": 0, "medium": 1, "low": 2, "none": 3}.get(p.risk_level, 3))

    for p in pii_tables:
        risk_icon = _RISK_ICON.get(p.risk_level, "")
        label = f"{risk_icon} `{p.table_name}` — risco {p.risk_level}"
        if p.lgpd_sensitive_columns:
            label += " — ⚠️ Sensível LGPD"
        with st.expander(label, expanded=(p.risk_level == "high")):
            st.caption(
                f"Classificação recomendada: **{p.recommended_classification.upper()}**  |  "
                f"Método de detecção: {p.detection_method}"
            )

            # Per-column PII details
            for cd in p.pii_columns:
                col_risk_icon = _RISK_ICON.get(cd.risk_level, "")
                lgpd_labels = [_LGPD_LABEL.get(cat, cat) for cat in cd.lgpd_categories]
                st.markdown(
                    f"- **`{cd.column_name}`** — "
                    f"{col_risk_icon} {', '.join(cd.pii_labels)}  "
                    f"| LGPD: _{', '.join(lgpd_labels)}_"
                )
                st.caption(f"  Evidência: {cd.evidence}")

    if not pii_tables:
        st.info("Nenhuma tabela com PII detectado.")


def _handle_enrich_from_diagnosis() -> None:
    """Enrich all selected tables from diagnosis results (warehouse source)."""
    agent = st.session_state.get("agent")
    if not agent:
        st.error("Configure e teste as conexões na Etapa 1.")
        return

    selected = st.session_state.get("selected_for_enrichment", [])
    if not selected:
        st.error("Selecione ao menos uma tabela.")
        return

    diagnosis_results = st.session_state.get("diagnosis_results", [])
    diag_by_name = {d.table_name: d for d in diagnosis_results}

    sample_size = int(st.session_state.get("sample_size", 100))
    base_context = st.session_state.get("additional_context") or None
    conn_str = _build_connection_string()

    enrichment_results = []
    original_metadata: Dict[str, Any] = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, table_name in enumerate(selected):
        status_text.text(f"Enriquecendo: {table_name} ({i+1}/{len(selected)})")
        diag = diag_by_name.get(table_name)

        # Build context enriched with existing metadata so LLM improves rather than replaces
        existing_ctx_parts = []
        if diag:
            if diag.existing_description:
                existing_ctx_parts.append(f'Descrição atual: "{diag.existing_description}"')
            if diag.existing_owner:
                existing_ctx_parts.append(f'Owner: "{diag.existing_owner}"')
            if diag.existing_tags:
                existing_ctx_parts.append(f"Tags existentes: {diag.existing_tags}")
            cols_with_desc = [cq.name for cq in diag.column_qualities if cq.existing_value]
            if cols_with_desc:
                existing_ctx_parts.append(
                    f"Colunas que já têm descrição (melhore apenas se necessário): "
                    f"{', '.join(cols_with_desc)}"
                )

        if existing_ctx_parts:
            existing_ctx = (
                "Metadados existentes encontrados — melhore-os sem substituir o que já está adequado:\n"
                + "\n".join(f"  - {p}" for p in existing_ctx_parts)
            )
        else:
            existing_ctx = None

        full_context = "\n\n".join(filter(None, [base_context, existing_ctx])) or None

        # Save original metadata snapshot
        original_metadata[table_name] = {
            "description": diag.existing_description if diag else "",
            "owner": diag.existing_owner if diag else "",
            "tags": list(diag.existing_tags) if diag else [],
            "classification": diag.existing_classification if diag else "",
            "columns": {
                cq.name: cq.existing_value for cq in diag.column_qualities
            } if diag else {},
        }

        try:
            schema = diag.schema if diag else st.session_state.get("wh_scan_schema") or None
            tbl_name = table_name.split(".")[-1] if "." in table_name else table_name
            from metadata_enrichment.sampling.data_sampler import SQLSampler
            sampler = SQLSampler(conn_str)
            sample_result = sampler.sample(tbl_name, sample_size=sample_size, schema=schema)

            # Upgrade PII diagnosis with data-based evidence and append to context
            try:
                from classification.data_classification_agent import DataClassificationAgent
            except ImportError:
                from data_governance.classification.data_classification_agent import DataClassificationAgent
            pii_clf = DataClassificationAgent(llm_provider=agent.llm_provider)
            name_diag = next(
                (p for p in st.session_state.get("pii_diagnosis_results", [])
                 if p.table_name == table_name),
                None,
            )
            data_pii_diag = pii_clf.classify_from_sample(sample_result)
            final_pii_diag = (
                pii_clf.merge_with_sample(name_diag, data_pii_diag)
                if name_diag else data_pii_diag
            )

            # Update session state with refined PII diagnosis
            pii_list = st.session_state.get("pii_diagnosis_results", [])
            pii_list = [p for p in pii_list if p.table_name != table_name]
            pii_list.append(final_pii_diag)
            st.session_state["pii_diagnosis_results"] = pii_list

            # Inject confirmed PII columns into enrichment context
            if final_pii_diag.has_pii:
                pii_col_strs = [
                    f"  - {cd.column_name}: {', '.join(cd.pii_labels)}"
                    for cd in final_pii_diag.pii_columns
                ]
                pii_ctx = (
                    f"Colunas com PII confirmado por amostragem (classifique como is_pii=true):\n"
                    + "\n".join(pii_col_strs)
                )
                full_context = "\n\n".join(filter(None, [full_context, pii_ctx]))

            result = agent.enrich(sample_result, additional_context=full_context)
            enrichment_results.append(result)
        except Exception as e:
            st.error(f"Erro ao enriquecer {table_name}: {e}")
            import traceback
            st.code(traceback.format_exc())

        progress_bar.progress((i + 1) / len(selected))

    if enrichment_results:
        st.session_state["enrichment_results_list"] = enrichment_results
        st.session_state["enrichment_result"] = enrichment_results[0]
        st.session_state["original_metadata"] = original_metadata
        st.session_state["current_enrichment_idx"] = 0
        _populate_edited_fields(enrichment_results[0])
        st.session_state["sample_result"] = None  # not available for multi-table from diagnosis
        st.session_state["validation_dirty"] = False
        st.session_state["step_completed"][3] = True
        st.session_state["step_completed"][4] = True  # dataset selection was done via diagnosis
        st.session_state["wizard_step"] = 5
        st.rerun()


def _populate_edited_fields(result) -> None:
    """Populate session state edit dicts from an EnrichmentResult."""
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


# ---------------------------------------------------------------------------
# Step 4 — Seleção de Dataset (previously Step 3)
# ---------------------------------------------------------------------------

def render_step4_dataset_selection() -> None:
    st.header("Etapa 4 de 6 — Seleção de Dataset")
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
            st.session_state["wizard_step"] = 3
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

            # Snapshot existing metadata as "before" for Before/After comparison in Step 5
            # (for file sources there are no pre-existing descriptions)
            orig_meta: Dict[str, Any] = st.session_state.get("original_metadata", {})
            if result.table_name not in orig_meta:
                orig_meta[result.table_name] = {
                    "description": "",
                    "owner": "",
                    "tags": [],
                    "classification": "",
                    "columns": {},
                }
            st.session_state["original_metadata"] = orig_meta

            st.session_state["sample_result"] = sample_result
            st.session_state["enrichment_result"] = result
            st.session_state["enrichment_results_list"] = [result]
            st.session_state["current_enrichment_idx"] = 0
            _populate_edited_fields(result)
            st.session_state["validation_dirty"] = False
            st.session_state["step_completed"][4] = True
            st.session_state["wizard_step"] = 5
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
# Step 5 — Validação & Edição (previously Step 4) — with Before/After badges
# ---------------------------------------------------------------------------

def render_step5_validation() -> None:
    st.header("Etapa 5 de 6 — Validação & Edição")

    results_list = st.session_state.get("enrichment_results_list", [])
    result = st.session_state.get("enrichment_result")
    if not result:
        st.error("Nenhum resultado disponível. Volte para a Etapa 4.")
        return

    # Multi-table selector
    if len(results_list) > 1:
        table_names = [r.table_name for r in results_list]
        idx = st.session_state.get("current_enrichment_idx", 0)
        selected_name = st.selectbox(
            "Tabela em revisão",
            table_names,
            index=idx,
            key="validation_table_selector",
        )
        new_idx = table_names.index(selected_name)
        if new_idx != idx:
            _apply_edits_to_result()
            st.session_state["current_enrichment_idx"] = new_idx
            st.session_state["enrichment_result"] = results_list[new_idx]
            _populate_edited_fields(results_list[new_idx])
            st.rerun()
        result = results_list[new_idx]

    c_back, c_regen, c_save, c_next = st.columns(4)
    with c_back:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["wizard_step"] = 4
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
            st.session_state["step_completed"][5] = True
            st.session_state["wizard_step"] = 6
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


def _before_after_caption(original: str, ai_value: str) -> None:
    """Display a before/after indicator below an edited field."""
    if not (original or "").strip():
        st.caption("🆕 Novo — campo não existia antes do enriquecimento")
    elif (original or "").strip() != (ai_value or "").strip():
        preview = original[:120] + ("..." if len(original) > 120 else "")
        st.caption(f"✨ Melhorado — anterior: _{preview}_")
    else:
        st.caption("Inalterado — mesmo valor do original")


def _render_table_editor(result) -> None:
    edits = st.session_state["edited_table_fields"]
    orig = st.session_state.get("original_metadata", {}).get(result.table_name, {})

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
        _before_after_caption(orig.get("owner", ""), edits.get("owner_suggestion", ""))

    with c_right:
        cls_opts = ["public", "internal", "confidential", "restricted"]
        cur_cls = edits.get("classification", "internal")
        if cur_cls not in cls_opts:
            cur_cls = "internal"
        edits["classification"] = st.selectbox(
            "Classificação", cls_opts,
            index=cls_opts.index(cur_cls),
            key="edit_tbl_cls")
        _before_after_caption(orig.get("classification", ""), edits.get("classification", ""))
        edits["tags"] = st.text_input(
            "Tags (separadas por vírgula)",
            value=edits.get("tags", ""),
            key="edit_tbl_tags")
        orig_tags_str = ", ".join(orig.get("tags", []))
        _before_after_caption(orig_tags_str, edits.get("tags", ""))

    edits["description"] = st.text_area(
        "Descrição (PT-BR)",
        value=edits.get("description", ""),
        height=120,
        key="edit_tbl_desc")
    _before_after_caption(orig.get("description", ""), edits.get("description", ""))

    if result.standards_used:
        st.caption("Normativos utilizados: " + ", ".join(result.standards_used))


def _render_column_editor(col, idx: int) -> None:
    edits = st.session_state["edited_columns"].get(col.name, {})
    result = st.session_state.get("enrichment_result")
    orig_col_desc = (
        st.session_state.get("original_metadata", {})
        .get(result.table_name if result else "", {})
        .get("columns", {})
        .get(col.name, "")
    ) if result else ""

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
            _before_after_caption(orig_col_desc, edits.get("description", col.description))
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


# ---- Step 6 renderer -------------------------------------------------------

def render_step6_publish() -> None:
    st.header("Etapa 6 de 6 — Publicação")

    result = st.session_state.get("enrichment_result")
    if not result:
        st.error("Nenhum resultado disponível. Volte para a Etapa 5.")
        return

    _apply_edits_to_result()

    _render_impact_report()
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
            st.session_state["wizard_step"] = 5
            st.rerun()
    with c2:
        if st.button("Novo Diagnóstico", use_container_width=True):
            _reset_wizard()


def _render_impact_report() -> None:
    """Show an impact report comparing metadata before and after enrichment."""
    results_list = st.session_state.get("enrichment_results_list", [])
    orig_meta = st.session_state.get("original_metadata", {})

    if not results_list:
        return

    # Compute delta stats across all enriched tables
    tables_enriched = len(results_list)
    cols_new = 0       # column descriptions that didn't exist before
    cols_improved = 0  # column descriptions that existed but were enriched
    cols_total = 0
    tables_new_desc = 0
    tables_improved_desc = 0
    pii_found = sum(1 for r in results_list if r.has_pii)

    for r in results_list:
        orig = orig_meta.get(r.table_name, {})
        orig_desc = orig.get("description", "")
        if not orig_desc:
            tables_new_desc += 1
        elif orig_desc.strip() != r.description.strip():
            tables_improved_desc += 1

        orig_cols = orig.get("columns", {})
        for col in r.columns:
            cols_total += 1
            orig_col_desc = orig_cols.get(col.name, "")
            if not orig_col_desc:
                cols_new += 1
            elif orig_col_desc.strip() != col.description.strip():
                cols_improved += 1

    pii_results = st.session_state.get("pii_diagnosis_results", [])
    pii_high = sum(1 for p in pii_results if p.risk_level == "high")
    pii_sensitive_cols = sum(
        len(p.lgpd_sensitive_columns) for p in pii_results if p.lgpd_sensitive_columns
    )

    with st.expander("Relatório de Impacto", expanded=True):
        st.subheader("Impacto do Enriquecimento")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tabelas enriquecidas", tables_enriched)
        c2.metric("Descrições novas (tabelas)", tables_new_desc + tables_improved_desc)
        c3.metric("Descrições novas (colunas)", cols_new)
        c4.metric("Descrições melhoradas (colunas)", cols_improved)

        if pii_found or pii_high:
            st.divider()
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Tabelas com PII", pii_found)
            pc2.metric("🔒 Risco alto (LGPD)", pii_high)
            pc3.metric("Colunas sensíveis LGPD", pii_sensitive_cols)
            if pii_high:
                st.warning(
                    "Tabelas com risco alto contêm dados pessoais sensíveis (LGPD Art. 5 IX). "
                    "Verifique a classificação **restricted** e os controles de acesso antes de publicar."
                )

        if pii_found:
            st.warning(f"PII detectado em {pii_found} tabela(s). Verifique a classificação antes de publicar.")

        if tables_enriched > 1:
            import pandas as pd
            rows = []
            for r in results_list:
                orig = orig_meta.get(r.table_name, {})
                had_desc = bool((orig.get("description", "") or "").strip())
                rows.append({
                    "Tabela": r.table_name,
                    "Descrição antes": "✓" if had_desc else "—",
                    "Descrição depois": "✓",
                    "Colunas enriquecidas": sum(
                        1 for col in r.columns
                        if not (orig.get("columns", {}).get(col.name, "") or "").strip()
                        or (orig.get("columns", {}).get(col.name, "") or "").strip() != col.description.strip()
                    ),
                    "Total colunas": len(r.columns),
                    "PII": "⚠️" if r.has_pii else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
                st.session_state["step_completed"][6] = True
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
        render_step3_diagnosis()
    elif step == 4:
        render_step4_dataset_selection()
    elif step == 5:
        render_step5_validation()
    elif step == 6:
        render_step6_publish()


if __name__ == "__main__":
    main()
