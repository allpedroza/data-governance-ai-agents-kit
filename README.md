# Data Governance AI Agents Kit

**Framework completo de agentes de IA para governança de dados**, fornecendo análise de linhagem, descoberta semântica, enriquecimento de metadados e monitoramento de qualidade.

## Visão Geral

Este projeto fornece **4 agentes de IA especializados** que trabalham de forma integrada para resolver desafios de governança de dados:

| Agente | Propósito |
|--------|-----------|
| **Data Lineage Agent** | Mapear dependências e analisar impacto de mudanças |
| **Data Discovery RAG Agent** | Descoberta semântica de dados com busca em linguagem natural |
| **Metadata Enrichment Agent** | Geração automática de descrições, tags e classificações |
| **Data Quality Agent** | Monitoramento de qualidade com SLA e detecção de schema drift |

## Início Rápido

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Instale as dependências
pip install -r requirements.txt

# Configure a API key (necessária para alguns agentes)
export OPENAI_API_KEY="sua-chave-aqui"

# Inicie a interface unificada
streamlit run app.py
```

---

## Agentes Disponíveis

### 1. Data Lineage Agent

Sistema de IA para **análise automática de linhagem de dados** em pipelines complexos.

**Características**:
- Análise de múltiplos formatos (Python, SQL, Terraform, Databricks, Airflow, Scala)
- Extração automática de dependências entre assets
- Visualização interativa de grafos (Force, Hierarchical, Sankey, 3D)
- Análise de impacto de mudanças
- Identificação de componentes críticos e ciclos
- Integração com Apache Atlas

**Documentação**: [lineage/README.md](lineage/README.md)

**Exemplo**:
```python
from lineage.data_lineage_agent import DataLineageAgent

agent = DataLineageAgent()
analysis = agent.analyze_pipeline([
    "etl/extract.sql",
    "etl/transform.py",
    "etl/load.sql"
])

# Análise de impacto
impact = agent.analyze_change_impact(["customers_table"])
print(f"Risk Level: {impact['risk_level']}")
```

---

### 2. Data Discovery RAG Agent

Sistema de IA para **descoberta de dados** usando **RAG (Retrieval-Augmented Generation)** com busca híbrida (semântica + lexical).

**Características**:
- Busca semântica em linguagem natural
- Dartboard Ranking (semântica + lexical + importância)
- Validação de tabelas contra catálogo
- Providers plugáveis (OpenAI, SentenceTransformers, VertexAI)
- Vector stores: ChromaDB, FAISS
- Integração com Apache Atlas e Lineage Agent

**Documentação**: [rag_discovery/README.md](rag_discovery/README.md)

**Exemplo**:
```python
from rag_discovery.agent import DataDiscoveryAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# Inicializa com providers
agent = DataDiscoveryAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(),
    vector_store=ChromaStore(collection_name="my_catalog")
)

# Indexa metadados
agent.index_from_json("catalog.json")

# Busca semântica
result = agent.discover("Onde estão os dados de clientes?")
print(result.answer)
```

---

### 3. Metadata Enrichment Agent

Sistema de IA para **geração automática de metadados** usando RAG sobre padrões de arquitetura e sampling de dados.

**Características**:
- Geração de descrições para tabelas e colunas (PT-BR e EN)
- Classificação automática de dados (public, internal, confidential, restricted)
- Detecção de PII (CPF, CNPJ, email, telefone, etc.)
- RAG sobre normativos e padrões de nomenclatura
- Sugestão de domínio e proprietário
- Suporte a CSV, Parquet, SQL, Delta Lake

**Documentação**: [metadata_enrichment/README.md](metadata_enrichment/README.md)

**Exemplo**:
```python
from metadata_enrichment.agent import MetadataEnrichmentAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

agent = MetadataEnrichmentAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards")
)

# Indexar padrões de nomenclatura
agent.index_standards_from_json("standards.json")

# Enriquecer metadados
result = agent.enrich_from_csv("customers.csv")

print(f"Descrição: {result.description}")
print(f"PII detectado: {result.has_pii}")
print(f"Colunas PII: {result.pii_columns}")
```

---

### 4. Data Quality Agent

Sistema de IA para **monitoramento de qualidade de dados** com métricas multi-dimensionais, SLA e detecção de schema drift.

**Características**:
- 6 dimensões de qualidade: Completeness, Uniqueness, Validity, Consistency, Freshness, Schema
- Monitoramento de Freshness com SLA configurável
- Detecção de schema drift com versionamento
- Sistema de regras e alertas configuráveis
- Suporte a CSV, Parquet, SQL, Delta Lake
- Exportação de relatórios (JSON, Markdown)

**Documentação**: [data_quality/README.md](data_quality/README.md)

**Exemplo**:
```python
from data_quality.agent import DataQualityAgent
from data_quality.rules import QualityRule, AlertLevel

agent = DataQualityAgent(enable_schema_tracking=True)

# Avaliar qualidade com SLA de freshness
report = agent.evaluate_file(
    "orders.parquet",
    freshness_config={
        "timestamp_column": "updated_at",
        "sla_hours": 4
    },
    validity_configs=[{
        "column": "email",
        "pattern_name": "email",
        "threshold": 0.95
    }]
)

print(f"Score: {report.overall_score:.0%}")
print(f"Status: {report.overall_status}")
print(f"Schema Drift: {report.schema_drift}")
```

---

## Integração entre Agentes

Os 4 agentes podem ser **integrados** para um framework completo de governança:

```python
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery.agent import DataDiscoveryAgent
from metadata_enrichment.agent import MetadataEnrichmentAgent
from data_quality.agent import DataQualityAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# 1. LINEAGE: Mapear dependências do pipeline
lineage_agent = DataLineageAgent()
lineage_result = lineage_agent.analyze_pipeline(["etl/*.sql", "etl/*.py"])
print(f"Assets mapeados: {lineage_result['metrics']['total_assets']}")

# 2. QUALITY: Avaliar qualidade dos dados
quality_agent = DataQualityAgent()
quality_report = quality_agent.evaluate_file(
    "data/customers.csv",
    freshness_config={"timestamp_column": "updated_at", "sla_hours": 24}
)
print(f"Qualidade: {quality_report.overall_score:.0%}")

# 3. ENRICHMENT: Gerar metadados automaticamente
enrichment_agent = MetadataEnrichmentAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards")
)
enrichment_result = enrichment_agent.enrich_from_csv("data/customers.csv")
print(f"PII detectado: {enrichment_result.has_pii}")

# 4. DISCOVERY: Indexar para busca semântica
discovery_agent = DataDiscoveryAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(),
    vector_store=ChromaStore(collection_name="catalog")
)

# Criar metadados enriquecidos com qualidade e linhagem
from rag_discovery.agent import TableMetadata
table = TableMetadata(
    name=enrichment_result.table_name,
    description=enrichment_result.description,
    columns=[{"name": c.name, "type": c.original_type, "description": c.description}
             for c in enrichment_result.columns],
    tags=enrichment_result.tags + [f"quality:{quality_report.overall_status}"]
)
discovery_agent.index_metadata([table])

# Buscar com contexto completo
result = discovery_agent.discover("tabelas com dados de clientes e boa qualidade")
print(result.answer)
```

### Workflows Recomendados

| Workflow | Agentes | Caso de Uso |
|----------|---------|-------------|
| **Catalogação Automática** | Enrichment → Discovery | Documentar data lake automaticamente |
| **Análise de Impacto** | Lineage + Discovery | Avaliar mudanças antes de deploy |
| **Compliance LGPD** | Enrichment + Lineage | Rastrear dados pessoais |
| **Monitoramento Contínuo** | Quality + Discovery | Alertas de qualidade no catálogo |
| **Onboarding** | Discovery + Lineage | Entender o data lake rapidamente |

---

## Interface Unificada (Streamlit)

O projeto inclui uma interface web unificada com todos os 4 agentes:

```bash
streamlit run app.py
```

**Tabs disponíveis**:
- **Lineage**: Upload de arquivos e visualização de grafos
- **Discovery**: Chat com o catálogo de dados
- **Enrichment**: Geração automática de metadados
- **Quality**: Avaliação de qualidade e alertas

Cada agente também possui interface standalone:
```bash
streamlit run lineage/app.py          # Apenas Lineage
streamlit run rag_discovery/app.py    # Apenas Discovery (se disponível)
streamlit run metadata_enrichment/streamlit_app.py  # Apenas Enrichment
streamlit run data_quality/streamlit_app.py         # Apenas Quality
```

---

## Instalação

### Pré-requisitos

- Python 3.8+
- OpenAI API Key (para agentes que usam LLM)

### Instalação Completa

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Instale todas as dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
export OPENAI_API_KEY="sua-chave-aqui"
```

### Instalação Individual

```bash
# Apenas Lineage Agent (sem LLM)
pip install -r lineage/requirements.txt

# Apenas RAG Discovery Agent
pip install -r rag_discovery/requirements.txt

# Apenas Metadata Enrichment Agent
pip install -r metadata_enrichment/requirements.txt

# Apenas Data Quality Agent (sem LLM)
pip install -r data_quality/requirements.txt
```

---

## Arquitetura

```
data-governance-ai-agents-kit/
│
├── app.py                           # Interface Streamlit unificada (4 tabs)
├── requirements.txt                 # Dependências globais
├── README.md                        # Este arquivo
│
├── lineage/                         # Data Lineage Agent
│   ├── data_lineage_agent.py        # Agente principal
│   ├── visualization_engine.py      # Visualizações Plotly
│   ├── parsers/                     # Parsers (SQL, Python, Terraform, etc.)
│   ├── examples/
│   ├── app.py                       # Streamlit standalone
│   └── README.md
│
├── rag_discovery/                   # Data Discovery RAG Agent
│   ├── agent.py                     # Agente v2 (Dartboard Ranking)
│   ├── data_discovery_rag_agent.py  # Agente v1 (compatibilidade)
│   ├── providers/                   # Providers plugáveis
│   │   ├── embeddings/              # OpenAI, SentenceTransformers
│   │   ├── llm/                     # OpenAI, VertexAI
│   │   └── vectorstore/             # ChromaDB, FAISS
│   ├── retrieval/                   # Busca híbrida
│   ├── examples/
│   └── README.md
│
├── metadata_enrichment/             # Metadata Enrichment Agent
│   ├── agent.py                     # Agente principal
│   ├── standards/                   # RAG para normativos
│   │   └── standards_rag.py
│   ├── sampling/                    # Conectores de sampling
│   │   └── data_sampler.py
│   ├── providers/                   # Reusa rag_discovery/providers
│   ├── examples/
│   ├── streamlit_app.py
│   └── README.md
│
└── data_quality/                    # Data Quality Agent
    ├── agent.py                     # Agente principal
    ├── metrics/                     # Métricas de qualidade
    │   ├── quality_metrics.py       # 5 dimensões
    │   └── schema_drift.py          # Detecção de drift
    ├── rules/                       # Sistema de regras
    │   └── quality_rules.py
    ├── connectors/                  # Conectores de dados
    │   └── data_connector.py
    ├── examples/
    ├── streamlit_app.py
    └── README.md
```

---

## Configuração

### Variáveis de Ambiente

```bash
# OpenAI (para RAG, Discovery e Enrichment)
export OPENAI_API_KEY="sk-..."
export OPENAI_API_URL="https://api.openai.com/v1"  # Opcional

# Modelo para Lineage (opcional)
export DATA_LINEAGE_LLM_MODEL="gpt-4o"

# Apache Atlas (opcional)
export ATLAS_HOST="http://atlas-host:21000"
export ATLAS_USERNAME="admin"
export ATLAS_PASSWORD="admin"
```

---

## Comparação de Agentes

| Característica | Lineage | Discovery | Enrichment | Quality |
|---------------|---------|-----------|------------|---------|
| **Objetivo** | Mapear dependências | Busca semântica | Gerar metadados | Monitorar qualidade |
| **Input** | Código (SQL, Python) | Query em LN | Dados (CSV, Parquet) | Dados (CSV, Parquet) |
| **Output** | Grafo + Impacto | Respostas + Tabelas | Descrições + Tags | Score + Alertas |
| **LLM** | Opcional | Requerido | Requerido | Não |
| **Embeddings** | Não | Sim | Sim | Não |
| **Principais Features** | Impact analysis, Ciclos | Híbrido search, RAG | PII detection, Standards | SLA, Schema drift |

---

## Casos de Uso

### 1. Catalogação Automática de Data Lake

```python
# Processar todos os arquivos do data lake
from pathlib import Path

for file in Path("data_lake/").glob("**/*.parquet"):
    # Avaliar qualidade
    quality = quality_agent.evaluate_file(str(file))

    # Enriquecer metadados
    enriched = enrichment_agent.enrich_from_parquet(str(file))

    # Indexar no catálogo
    discovery_agent.index_metadata([create_table_metadata(enriched, quality)])

print("Catálogo criado com metadados enriquecidos e scores de qualidade!")
```

### 2. Compliance LGPD/GDPR

```python
# Identificar e rastrear dados pessoais
results = []
for file in data_files:
    enriched = enrichment_agent.enrich_from_csv(file)
    if enriched.has_pii:
        # Rastrear linhagem dos dados PII
        lineage = lineage_agent.analyze_pipeline([file])
        results.append({
            "file": file,
            "pii_columns": enriched.pii_columns,
            "downstream_impact": lineage["metrics"]["total_assets"]
        })

print(f"Encontrados {len(results)} arquivos com dados pessoais")
```

### 3. Monitoramento de SLA

```python
# Verificar freshness diariamente
from data_quality.rules import QualityRule, AlertLevel

# Adicionar regra de SLA
agent.add_rule(QualityRule(
    name="orders_freshness_sla",
    dimension="freshness",
    table_name="orders",
    threshold=0.95,
    alert_level=AlertLevel.CRITICAL,
    params={"sla_hours": 4}
))

# Avaliar
report = agent.evaluate_file("orders.parquet")

# Verificar alertas
for alert in agent.get_active_alerts():
    print(f"[{alert.level}] {alert.message}")
```

---

## Roadmap

### Concluído
- [x] Data Lineage Agent com múltiplos parsers
- [x] Data Discovery RAG Agent com busca híbrida
- [x] Metadata Enrichment Agent com PII detection
- [x] Data Quality Agent com SLA monitoring
- [x] Interface Streamlit unificada
- [x] Integração com Apache Atlas
- [x] Providers plugáveis (embeddings, LLM, vectorstore)

### Em Desenvolvimento
- [ ] Column-level lineage
- [ ] Integração com dbt
- [ ] Integração com AWS Glue Data Catalog
- [ ] Integração com Databricks Unity Catalog
- [ ] API REST para integração com outras ferramentas
- [ ] Dashboard de métricas de governança
- [ ] Suporte a modelos locais (Ollama)

---

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/amazing-feature`)
3. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
4. Push para a branch (`git push origin feature/amazing-feature`)
5. Abra um Pull Request

---

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

---

## Agradecimentos

- **Apache Atlas** - Catálogo de metadados
- **ChromaDB** - Banco vetorial
- **OpenAI** - Embeddings e LLM
- **SentenceTransformers** - Embeddings locais
- **NetworkX** - Análise de grafos
- **Plotly** - Visualizações interativas
- **Streamlit** - Interface web
