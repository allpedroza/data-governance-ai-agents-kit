# Data Governance AI Agents Kit

## Resumo
Framework completo de agentes de IA para governança de dados, cobrindo linhagem, descoberta semântica, enriquecimento de metadados, classificação, qualidade e valor de ativos.

## Summary
Complete AI agent framework for data governance, covering lineage, semantic discovery, metadata enrichment, classification, quality, and asset value.

*English details available at the end of the file.*

## Visão Geral
- Seis agentes especializados funcionam de forma integrada ou independente.
- Todos operam somente com metadados ou arquivos fornecidos (sem exigir acesso direto a bancos de produção).
- Exemplos prontos em Python e interface unificada via Streamlit.

## Início Rápido
```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie ambiente virtual com uv (recomendado) ou pip padrão
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
# pip install -r requirements.txt

# Configure variáveis sensíveis (por exemplo, modelos comerciais)
export OPENAI_API_KEY="sua-chave-aqui"

# Inicie a interface unificada
streamlit run app.py
```

## Agentes e Documentação
- **Data Lineage Agent** – Mapeamento de dependências, análise de impacto e grafos interativos. [lineage/README.md](lineage/README.md)
- **Data Discovery RAG Agent** – Busca semântica com RAG híbrido e validação contra catálogo. [rag_discovery/README.md](rag_discovery/README.md)
- **Metadata Enrichment Agent** – Geração automática de descrições, tags e classificação. [metadata_enrichment/README.md](metadata_enrichment/README.md)
- **Data Classification Agent** – Identificação de PII/PHI/dados financeiros apenas com schemas. [data_classification/README.md](data_classification/README.md)
- **Data Quality Agent** – Métricas multidimensionais, SLA de frescor e schema drift. [data_quality/README.md](data_quality/README.md)
- **Data Asset Value Agent** – Score de valor considerando uso, JOINs, linhagem e data products. [data_asset_value/README.md](data_asset_value/README.md)

## Exemplo de Integração
```python
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery.agent import DataDiscoveryAgent
from metadata_enrichment.agent import MetadataEnrichmentAgent
from data_classification import DataClassificationAgent
from data_quality.agent import DataQualityAgent
from data_asset_value import DataAssetValueAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# 1) Linhagem
lineage_agent = DataLineageAgent()
lineage_result = lineage_agent.analyze_pipeline(["etl/*.sql", "etl/*.py"])

# 2) Classificação
classification_agent = DataClassificationAgent()
classification_report = classification_agent.classify_from_csv("data/customers.csv")

# 3) Qualidade
quality_agent = DataQualityAgent()
quality_report = quality_agent.evaluate_file(
    "data/customers.csv",
    freshness_config={"timestamp_column": "updated_at", "sla_hours": 24}
)

# 4) Enriquecimento
embeddings = SentenceTransformerEmbeddings()
enrichment_agent = MetadataEnrichmentAgent(
    embedding_provider=embeddings,
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards")
)
enrichment_result = enrichment_agent.enrich_from_csv("data/customers.csv")

# 5) Descoberta
vector_store = ChromaStore(collection_name="catalog")
discovery_agent = DataDiscoveryAgent(
    embedding_provider=embeddings,
    llm_provider=OpenAILLM(),
    vector_store=vector_store
)
discovery_agent.index_metadata([enrichment_result.to_table_metadata()])
discovery_answer = discovery_agent.discover("tabelas com dados de clientes e boa qualidade")

# 6) Valor de ativo
value_agent = DataAssetValueAgent()
value_report = value_agent.analyze_from_query_logs(
    query_logs=[],
    lineage_data=lineage_result,
    data_product_config={},
    asset_metadata={}
)
```

Este fluxo mostra como cada agente compartilha insumos (linhagem, metadados enriquecidos, classificação, qualidade) para entregar governança ponta a ponta.

## Interface Unificada
Execute `streamlit run app.py` para navegar pelos seis agentes com configurações rápidas e exemplos pré-carregados.

## Instalação Seletiva
Use os requirements específicos se precisar apenas de um módulo isolado (ex.: `pip install -r metadata_enrichment/requirements.txt`).

---

## Summary
Complete AI agent framework for data governance, covering lineage, semantic discovery, metadata enrichment, classification, quality, and asset value.

## Overview
- Six specialized agents work together or independently.
- All operate only with provided metadata/files without requiring direct access to production databases.
- Ready-to-run Python examples and a unified Streamlit interface.

## Quickstart
```bash
# Clone the repository
git clone <repo-url>
cd data-governance-ai-agents-kit

# Create a virtual environment with uv (recommended) or plain pip
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
# pip install -r requirements.txt

# Configure sensitive variables (e.g., commercial model keys)
export OPENAI_API_KEY="your-key-here"

# Launch the unified interface
streamlit run app.py
```

## Agents and Documentation
- **Data Lineage Agent** – Dependency mapping, impact analysis, and interactive graphs. [lineage/README.md](lineage/README.md)
- **Data Discovery RAG Agent** – Semantic search with hybrid RAG and catalog validation. [rag_discovery/README.md](rag_discovery/README.md)
- **Metadata Enrichment Agent** – Automatic descriptions, tags, and classification generation. [metadata_enrichment/README.md](metadata_enrichment/README.md)
- **Data Classification Agent** – PII/PHI/financial detection using schemas only. [data_classification/README.md](data_classification/README.md)
- **Data Quality Agent** – Multi-dimensional metrics, freshness SLAs, and schema drift detection. [data_quality/README.md](data_quality/README.md)
- **Data Asset Value Agent** – Value scoring based on usage, joins, lineage, and data products. [data_asset_value/README.md](data_asset_value/README.md)

## Integration Example
```python
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery.agent import DataDiscoveryAgent
from metadata_enrichment.agent import MetadataEnrichmentAgent
from data_classification import DataClassificationAgent
from data_quality.agent import DataQualityAgent
from data_asset_value import DataAssetValueAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# 1) Lineage
lineage_agent = DataLineageAgent()
lineage_result = lineage_agent.analyze_pipeline(["etl/*.sql", "etl/*.py"])

# 2) Classification
classification_agent = DataClassificationAgent()
classification_report = classification_agent.classify_from_csv("data/customers.csv")

# 3) Quality
quality_agent = DataQualityAgent()
quality_report = quality_agent.evaluate_file(
    "data/customers.csv",
    freshness_config={"timestamp_column": "updated_at", "sla_hours": 24}
)

# 4) Enrichment
embeddings = SentenceTransformerEmbeddings()
enrichment_agent = MetadataEnrichmentAgent(
    embedding_provider=embeddings,
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards")
)
enrichment_result = enrichment_agent.enrich_from_csv("data/customers.csv")

# 5) Discovery
vector_store = ChromaStore(collection_name="catalog")
discovery_agent = DataDiscoveryAgent(
    embedding_provider=embeddings,
    llm_provider=OpenAILLM(),
    vector_store=vector_store
)
discovery_agent.index_metadata([enrichment_result.to_table_metadata()])
discovery_answer = discovery_agent.discover("tables with customer data and good quality")

# 6) Asset value
value_agent = DataAssetValueAgent()
value_report = value_agent.analyze_from_query_logs(
    query_logs=[],
    lineage_data=lineage_result,
    data_product_config={},
    asset_metadata={}
)
```

This flow illustrates how each agent shares inputs (lineage, enriched metadata, classification, quality) to deliver end-to-end governance.

## Unified Interface
Run `streamlit run app.py` to explore all six agents with quick settings and preloaded examples.

## Selective Installation
Use module-specific requirement files if you only need one component (e.g., `pip install -r metadata_enrichment/requirements.txt`).
