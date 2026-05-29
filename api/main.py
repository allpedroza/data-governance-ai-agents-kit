import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any

# Add the parent directory to the python path so we can import our agents
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_governance.rag_discovery.data_discovery_rag_agent import DataDiscoveryRAGAgent
from data_governance.lineage.data_lineage_agent import DataLineageAgent
from data_governance.data_quality.agent import DataQualityAgent
from data_governance.data_contracts.agent import DataContractAgent
from data_governance.data_classification.agent import DataClassificationAgent
import tempfile

app = FastAPI(title="Data Gov AI API", version="1.0.0")

# Setup CORS for the Next.js frontend (default port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sensitive Data Imports
try:
    from ai_governance.sensitive_data_ner.agent import SensitiveDataNERAgent
    from ai_governance.sensitive_data_ner.vault import SecureVault
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    SensitiveDataNERAgent = None
    SecureVault = None

# Global Agents
rag_agent = None
lineage_agent = None
quality_agent = None
contract_agent = None
classification_agent = None
value_agent = None
steward_agent = None
ner_agent = None
vault = None

@app.on_event("startup")
def startup_event():
    global rag_agent, lineage_agent, quality_agent, contract_agent, classification_agent, value_agent, steward_agent, ner_agent, vault
    try:
        rag_agent = DataDiscoveryRAGAgent(
            persist_directory=str(BASE_DIR / "data_governance" / "rag_discovery" / ".chroma_api"),
            collection_name="api_catalog",
        )
        print("DataDiscoveryRAGAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize RAG Agent: {e}")
        
    try:
        lineage_agent = DataLineageAgent()
        print("DataLineageAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Lineage Agent: {e}")
        
    try:
        quality_agent = DataQualityAgent(use_llm=True)
        print("DataQualityAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Quality Agent: {e}")

    try:
        contract_agent = DataContractAgent(use_llm=True)
        print("DataContractAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Contract Agent: {e}")
        
    try:
        classification_agent = DataClassificationAgent()
        print("DataClassificationAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Classification Agent: {e}")

    try:
        value_agent = DataAssetValueAgent()
        print("DataAssetValueAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Value Agent: {e}")

    try:
        steward_agent = DataStewardAgent(persist_dir="./steward_data")
        print("DataStewardAgent initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Steward Agent: {e}")

    if NER_AVAILABLE:
        try:
            ner_agent = SensitiveDataNERAgent()
            from ai_governance.sensitive_data_ner.vault import VaultConfig
            v_config = VaultConfig(require_authentication=False)
            vault = SecureVault(config=v_config)
            vault.initialize(master_password="test-password-123")
            print("SensitiveDataNERAgent and Vault initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize NER/Vault Agent: {e}")

# Request/Response Models
class DiscoveryRequest(BaseModel):
    query: str
    limit: int = 5
    exact_match: bool = False

@app.post("/api/v1/discovery")
async def discover_data(request: Request, req: DiscoveryRequest):
    apply_dynamic_settings(request)
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not available. Check API keys.")
    try:
        results = rag_agent.search(
            query=req.query, 
            n_results=req.limit
        )
        
        # Ensure we can serialize the results to JSON
        serialized_results = []
        for res in results:
            if hasattr(res, "model_dump"):
                serialized_results.append(res.model_dump())
            elif hasattr(res, "__dict__"):
                serialized_results.append(res.__dict__)
            else:
                serialized_results.append(str(res))
                
        return {"results": serialized_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form, Header
from typing import List, Optional
import shutil

_current_settings_hash = None
taxonomy_agent = None

def apply_dynamic_settings(request: Request):
    global _current_settings_hash, rag_agent, lineage_agent, quality_agent, contract_agent, classification_agent, value_agent, steward_agent, taxonomy_agent
    
    headers = request.headers
    
    # Map headers to env vars
    env_mapping = {
        "x-openai-key": "OPENAI_API_KEY",
        "x-gemini-key": "GEMINI_API_KEY",
        "x-anthropic-key": "ANTHROPIC_API_KEY",
        "x-deepseek-key": "DEEPSEEK_API_KEY",
        "x-llm-provider": "LLM_PROVIDER",
        "x-llm-model": "LLM_MODEL",
        "x-warehouse-type": "WAREHOUSE_TYPE",
        "x-warehouse-host": "WAREHOUSE_HOST",
        "x-warehouse-user": "WAREHOUSE_USER",
        "x-warehouse-password": "WAREHOUSE_PASSWORD",
        "x-catalog-type": "CATALOG_TYPE",
        "x-catalog-host": "CATALOG_HOST",
        "x-catalog-token": "CATALOG_TOKEN"
    }
    
    current_hash = ""
    for header_key, env_key in env_mapping.items():
        val = headers.get(header_key)
        if val:
            os.environ[env_key] = val
            current_hash += val
            
    if current_hash and current_hash != _current_settings_hash:
        _current_settings_hash = current_hash
        
        # Reinitialize agents that depend on OpenAI
        rag_agent = DataDiscoveryRAGAgent(
            persist_directory=str(BASE_DIR / "data_governance" / "rag_discovery" / ".chroma_api"),
            collection_name="api_catalog",
        )
        lineage_agent = DataLineageAgent()
        quality_agent = DataQualityAgent()
        contract_agent = DataContractAgent()
        classification_agent = DataClassificationAgent()
        
        from data_governance.data_asset_value.agent import DataAssetValueAgent
        value_agent = DataAssetValueAgent()
        
        from data_governance.data_steward.agent import DataStewardAgent
        steward_agent = DataStewardAgent(persist_dir="./steward_data")
        
        from data_governance.taxonomy.agent import TaxonomyAgent
        taxonomy_agent = TaxonomyAgent()

@app.post("/api/v1/quality/evaluate")
async def evaluate_quality(request: Request, file: UploadFile = File(...)):
    apply_dynamic_settings(request)
    if not quality_agent:
        raise HTTPException(status_code=503, detail="Quality Agent not available.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        report = quality_agent.evaluate_file(tmp_path)
        return report.model_dump() if hasattr(report, "model_dump") else report.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/api/v1/classification/classify")
async def classify_data(request: Request, file: UploadFile = File(...)):
    apply_dynamic_settings(request)
    if not classification_agent:
        raise HTTPException(status_code=503, detail="Classification Agent not available.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        report = classification_agent.classify_from_csv(tmp_path)
        return report.model_dump() if hasattr(report, "model_dump") else report.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/api/v1/contracts/validate")
async def validate_contract(request: Request, file: UploadFile = File(...), contract_yaml: str = Form(...)):
    apply_dynamic_settings(request)
    if not contract_agent:
        raise HTTPException(status_code=503, detail="Contract Agent not available.")
        
    import pandas as pd
    from data_governance.data_contracts.models import DataContract
    import yaml
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        df = pd.read_csv(tmp_path)
        contract_dict = yaml.safe_load(contract_yaml)
        contract = DataContract(**contract_dict)
        report = contract_agent.validate_dataframe(contract, df)
        return report.model_dump() if hasattr(report, "model_dump") else report.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

from data_governance.lineage.graph_adapter import generate_cytoscape_json

@app.post("/api/v1/lineage/analyze")
async def analyze_lineage(request: Request, files: List[UploadFile] = File(...)):
    apply_dynamic_settings(request)
    if not lineage_agent:
        raise HTTPException(status_code=503, detail="Lineage Agent not available.")
    
    tmp_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for upload in files:
            dest = Path(temp_dir) / upload.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            tmp_paths.append(str(dest))
            
        try:
            results = lineage_agent.analyze_pipeline(tmp_paths)
            cytoscape_data = generate_cytoscape_json(results)
            return {"cytoscape_data": cytoscape_data, "raw_results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class ValueAnalysisRequest(BaseModel):
    query_logs: List[dict]

@app.post("/api/v1/value/analyze")
async def analyze_value(request: Request, req: ValueAnalysisRequest):
    apply_dynamic_settings(request)
    if not value_agent:
        raise HTTPException(status_code=503, detail="Value Agent not available.")
    try:
        report = value_agent.analyze_from_query_logs(query_logs=req.query_logs)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StewardRequest(BaseModel):
    asset_name: str
    metadata: dict

@app.post("/api/v1/steward/assign")
async def assign_steward(request: Request, req: StewardRequest):
    apply_dynamic_settings(request)
    if not steward_agent:
        raise HTTPException(status_code=503, detail="Steward Agent not available.")
    try:
        assignment = steward_agent.assign_steward(asset_name=req.asset_name, metadata=req.metadata)
        return assignment.to_dict() if hasattr(assignment, "to_dict") else assignment.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Taxonomy: Explore → Generate → Evaluate ──────────────────────────

class TaxonomyExploreRequest(BaseModel):
    warehouse_type: str = "snowflake"
    database: Optional[str] = None
    schema_name: Optional[str] = None

@app.post("/api/v1/taxonomy/explore")
async def explore_lake(request: Request, req: TaxonomyExploreRequest):
    """Step 1: Connect to warehouse and extract schema metadata."""
    apply_dynamic_settings(request)
    
    wh_type = req.warehouse_type.lower()
    wh_host = os.environ.get("WAREHOUSE_HOST", "")
    wh_user = os.environ.get("WAREHOUSE_USER", "")
    wh_pass = os.environ.get("WAREHOUSE_PASSWORD", "")
    
    if not wh_host or not wh_user:
        raise HTTPException(
            status_code=400,
            detail="Configure as credenciais de Warehouse em Settings antes de explorar."
        )
    
    try:
        from data_governance.warehouse.connectors import create_warehouse_connector
        
        connector_kwargs = {
            "host": wh_host,
            "username": wh_user,
            "password": wh_pass,
        }
        if req.database:
            connector_kwargs["database"] = req.database
        if req.schema_name:
            connector_kwargs["schema"] = req.schema_name
        
        # For Snowflake, map host → account
        if wh_type == "snowflake":
            connector_kwargs["account"] = wh_host
            del connector_kwargs["host"]
        
        connector = create_warehouse_connector(wh_type, **connector_kwargs)
        connector.connect()
        
        schemas = connector.list_schemas(req.database)
        
        lake_meta = {"warehouse_type": wh_type, "database": req.database, "schemas": []}
        
        target_schemas = [req.schema_name] if req.schema_name else schemas[:5]  # limit to 5 schemas
        
        for schema in target_schemas:
            schema_info = {"name": schema, "tables": []}
            tables = connector.list_tables(schema=schema, database=req.database)
            
            for table in tables[:30]:  # limit to 30 tables per schema
                cols = connector.get_table_schema(table.name, schema=schema, database=req.database)
                schema_info["tables"].append({
                    "name": table.name,
                    "full_name": table.full_name,
                    "type": table.table_type,
                    "columns": [
                        {
                            "name": c.get("name", ""),
                            "type": c.get("type", ""),
                            "nullable": c.get("nullable", True),
                            "comment": c.get("comment", ""),
                        }
                        for c in cols
                    ]
                })
            
            lake_meta["schemas"].append(schema_info)
        
        connector.disconnect()
        return lake_meta
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao explorar o warehouse: {str(e)}")


class TaxonomyGenerateRequest(BaseModel):
    lake_metadata: dict

@app.post("/api/v1/taxonomy/generate")
async def generate_taxonomy(request: Request, req: TaxonomyGenerateRequest):
    """Step 2: Use LLM to generate taxonomy YAML from lake metadata."""
    apply_dynamic_settings(request)
    
    llm_provider = os.environ.get("LLM_PROVIDER", "openai")
    
    # Build a concise summary of the lake metadata for the LLM prompt
    meta_summary_parts = []
    for schema_info in req.lake_metadata.get("schemas", []):
        schema_name = schema_info.get("name", "unknown")
        for table in schema_info.get("tables", []):
            cols_str = ", ".join([
                f"{c['name']} ({c['type']})" 
                for c in table.get("columns", [])[:20]
            ])
            meta_summary_parts.append(f"Table: {schema_name}.{table['name']} → Columns: [{cols_str}]")
    
    meta_summary = "\n".join(meta_summary_parts[:50])  # limit context
    
    prompt = f"""You are a Data Governance expert. Based on the following data lake metadata, generate a comprehensive Taxonomy YAML document.

The YAML must follow this structure:
- metadata: (title, version, domain, platform, owner, steward)
- naming_conventions: (casing_rules, canonical_acronyms, forbidden_forms, full_words_required)
- concept_groups: (list of groups, each with name, icon, description, concepts with name, data_type, definition, accepted_types, entity_qualified_forms, aliases)
- context_rules: (single_entity vs multi_entity rules with examples and anti_patterns)
- ambiguous_aliases: (aliases that could refer to different entities, with resolution_rules)
- datetime_standards: (timezone, date_format, timestamp_format)
- lake_standards: (zones with name, alias, description, naming_pattern)
- validation_rules: (id, name, severity, scope, rule_type, message)

Analyze the column names, types, and table structures to:
1. Identify concept groups (e.g., Customer, Product, Order, Address)
2. Map aliases (same concept with different column names across tables)
3. Define naming conventions based on patterns you observe
4. Create context rules for single vs multi-entity tables
5. Flag ambiguous column names that appear in multiple contexts

DATA LAKE METADATA:
{meta_summary}

Return ONLY valid YAML, no markdown fences, no explanations."""

    try:
        if llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000,
            )
            yaml_content = response.choices[0].message.content.strip()
            
        elif llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
            response = model.generate_content(prompt)
            yaml_content = response.text.strip()
            
        elif llm_provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
            model = os.environ.get("LLM_MODEL", "deepseek-chat")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000,
            )
            yaml_content = response.choices[0].message.content.strip()
            
        else:
            raise HTTPException(status_code=400, detail=f"LLM provider '{llm_provider}' not supported.")
        
        # Clean markdown fences if LLM wrapped them
        if yaml_content.startswith("```"):
            yaml_content = yaml_content.split("\n", 1)[1]
        if yaml_content.endswith("```"):
            yaml_content = yaml_content.rsplit("```", 1)[0]
        yaml_content = yaml_content.strip()
        
        return {"yaml_content": yaml_content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar taxonomia via LLM: {str(e)}")


class TaxonomyEvaluateRequest(BaseModel):
    yaml_content: str

@app.post("/api/v1/taxonomy/evaluate")
async def evaluate_taxonomy(request: Request, req: TaxonomyEvaluateRequest):
    """Step 3: Score the taxonomy YAML and return results + HTML artifact."""
    apply_dynamic_settings(request)
    if not taxonomy_agent:
        raise HTTPException(status_code=503, detail="Taxonomy Agent not available.")
    
    from data_governance.taxonomy.models import TaxonomyDocument
    from data_governance.taxonomy.html_generator import generate_taxonomy_html
    import yaml
    try:
        data = yaml.safe_load(req.yaml_content)
        taxonomy_doc = TaxonomyDocument.from_dict(data)
        score = taxonomy_agent.score_taxonomy(taxonomy_doc)
        
        # Generate standalone HTML artifact
        html_artifact = generate_taxonomy_html(taxonomy_doc.to_dict(), score.to_dict())
        
        result = score.to_dict()
        result["html_artifact"] = html_artifact
        result["taxonomy_data"] = taxonomy_doc.to_dict()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NerRequest(BaseModel):
    text: str
    anonymize: bool = True

@app.post("/api/v1/vault/anonymize")
async def anonymize_text(request: NerRequest):
    if not NER_AVAILABLE or not ner_agent:
        raise HTTPException(status_code=503, detail="NER/Vault Agent not available (spaCy/cryptography missing).")
    try:
        # Use analyze method instead of analyze_text, and let it anonymize
        result = ner_agent.analyze(request.text, anonymize=request.anonymize)
        
        return {
            "entities": [e.to_dict() for e in result.entities],
            "anonymized_text": result.anonymized_text if request.anonymize else request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Metadata Enrichment ──────────────────────────────────────────────

@app.post("/api/v1/enrichment/enrich")
async def enrich_metadata(request: Request, file: UploadFile = File(...)):
    """Enrich metadata from a CSV file upload using LLM analysis."""
    apply_dynamic_settings(request)
    
    import pandas as pd
    import json as json_mod
    
    llm_provider = os.environ.get("LLM_PROVIDER", "openai")
    
    # Read CSV into pandas
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        df = pd.read_csv(tmp_path, nrows=200)
        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {str(e)}")
    
    table_name = file.filename.rsplit(".", 1)[0] if file.filename else "unknown_table"
    
    # Build column profiles
    columns_profile = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        distinct = int(series.nunique())
        total = len(series)
        samples = [str(v) for v in series.dropna().head(5).tolist()]
        dtype = str(series.dtype)
        
        patterns = []
        non_null = series.dropna().astype(str)
        if non_null.str.match(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$").any():
            patterns.append("cpf")
        if non_null.str.match(r"^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$").any():
            patterns.append("cnpj")
        if non_null.str.contains(r"@.*\.", regex=True).any():
            patterns.append("email")
        if non_null.str.match(r"^\+?\d{10,13}$").any():
            patterns.append("phone")
        
        columns_profile.append({
            "name": col, "type": dtype, "samples": samples,
            "null_ratio": f"{null_count / total:.1%}" if total > 0 else "0%",
            "distinct_count": distinct, "patterns": patterns,
        })
    
    sample_rows = df.head(5).fillna("").to_dict(orient="records")
    sample_rows_str = json_mod.dumps(sample_rows, ensure_ascii=False, default=str)[:3000]
    columns_str = json_mod.dumps(columns_profile, ensure_ascii=False, indent=2)[:5000]
    
    system_prompt = """Você é um especialista em governança de dados e catalogação de metadados.
Gere descrições, tags e classificações para uma tabela e suas colunas.
Responda APENAS em JSON válido, sem markdown.

Formato EXATO:
{
    "table": {
        "description": "Descrição em português",
        "description_en": "Description in English",
        "business_name": "Nome amigável",
        "domain": "customer|sales|finance|product|marketing|hr|operations|analytics|general",
        "tags": ["tag1", "tag2"],
        "classification": "public|internal|confidential|restricted",
        "owner_suggestion": "Time/área sugerido",
        "confidence": 0.0-1.0
    },
    "columns": [
        {
            "name": "column_name",
            "description": "Descrição em português",
            "description_en": "Description in English",
            "business_name": "Nome amigável",
            "tags": ["tag1"],
            "classification": "public|internal|confidential|restricted",
            "semantic_type": "pii|email|phone|cpf|cnpj|date|currency|id|flag|name|address|text|numeric|null",
            "is_pii": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "breve explicação"
        }
    ]
}

Regras PII: CPF, CNPJ, RG, email, telefone, nome completo, endereço = PII → classification='restricted'."""

    user_prompt = f"""Analise a tabela '{table_name}' e gere metadados enriquecidos.

## Colunas (perfil)
{columns_str}

## Amostra de dados
{sample_rows_str}

Total de linhas: {len(df)}

Gere o JSON com table + columns:"""

    try:
        if llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1, max_tokens=4000,
            )
            llm_content = resp.choices[0].message.content.strip()
        elif llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            gmodel = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
            resp = gmodel.generate_content(f"{system_prompt}\n\n{user_prompt}")
            llm_content = resp.text.strip()
        elif llm_provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
            model = os.environ.get("LLM_MODEL", "deepseek-chat")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1, max_tokens=4000,
            )
            llm_content = resp.choices[0].message.content.strip()
        else:
            raise HTTPException(status_code=400, detail=f"LLM provider '{llm_provider}' not supported.")
        
        if llm_content.startswith("```"):
            llm_content = llm_content.split("\n", 1)[1]
        if llm_content.endswith("```"):
            llm_content = llm_content.rsplit("```", 1)[0]
        
        enrichment = json_mod.loads(llm_content.strip())
    except json_mod.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM retornou JSON inválido. Tente novamente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao enriquecer metadados: {str(e)}")
    
    table_data = enrichment.get("table", {})
    columns_data = enrichment.get("columns", [])
    pii_columns = [c["name"] for c in columns_data if c.get("is_pii")]
    has_pii = len(pii_columns) > 0
    if has_pii:
        table_data["classification"] = "restricted"
    
    return {
        "table_name": table_name,
        "source": "csv_upload",
        "description": table_data.get("description", ""),
        "description_en": table_data.get("description_en", ""),
        "business_name": table_data.get("business_name", table_name),
        "domain": table_data.get("domain", "general"),
        "tags": table_data.get("tags", []),
        "classification": table_data.get("classification", "internal"),
        "owner_suggestion": table_data.get("owner_suggestion", ""),
        "columns": columns_data,
        "row_count": len(df),
        "column_count": len(df.columns),
        "has_pii": has_pii,
        "pii_columns": pii_columns,
        "confidence": table_data.get("confidence", 0.7),
    }

