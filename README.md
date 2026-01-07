# Data Governance AI Agents Kit

**Framework completo de agentes de IA para governança de dados**, fornecendo análise de linhagem, descoberta semântica, enriquecimento de metadados, classificação de dados, monitoramento de qualidade, análise de valor de ativos e proteção contra vazamento de dados sensíveis.

## Visão Geral

Este projeto fornece **7 agentes de IA especializados** que trabalham de forma integrada para resolver desafios de governança de dados:

| Agente | Propósito |
|--------|-----------|
| **Data Lineage Agent** | Mapear dependências e analisar impacto de mudanças |
| **Data Discovery RAG Agent** | Descoberta semântica de dados com busca em linguagem natural |
| **Metadata Enrichment Agent** | Geração automática de descrições, tags e classificações |
| **Data Classification Agent** | Classificação de PII/PHI/Financeiro/ e termos sensíveis à estratégia dos negócios a partir de metadados |
| **Data Quality Agent** | Monitoramento de qualidade com SLA e detecção de schema drift |
| **Data Asset Value Agent** | Análise de valor de ativos baseado em uso, JOINs e impacto em data products |
| **Sensitive Data NER Agent** | Detecção e anonimização de dados sensíveis em texto livre para proteção de LLMs |

## Início Rápido

### Com uv (Recomendado)

[uv](https://docs.astral.sh/uv/) é um gerenciador de pacotes Python extremamente rápido (10-100x mais rápido que pip).

```bash
# Instale o uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie ambiente virtual e instale dependências
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -r requirements.txt

# Configure a API key (necessária para alguns agentes)
export OPENAI_API_KEY="sua-chave-aqui"

# Inicie a interface unificada
streamlit run app.py
```

### Com pip (Alternativo)

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie ambiente virtual (opcional, mas recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

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

### 4. Data Classification Agent

Sistema de IA para **classificação automática de dados** por níveis de sensibilidade, detectando PII, PHI, PCI, dados financeiros e termos estratégicos proprietários.

**Características**:
- Detecção de PII (CPF, CNPJ, SSN, email, telefone, IP, etc.)
- Detecção de PHI (CID-10, CNS, CRM, prontuário médico)
- Detecção de PCI (cartão de crédito, CVV, IBAN, SWIFT)
- Detecção de dados financeiros (contas, transações, valores)
- Detecção de termos estratégicos proprietários via **dicionário customizável** (nomes de projetos, iniciativas, roadmaps)
- Níveis de sensibilidade: public, internal, confidential, restricted
- Flags de compliance: LGPD, GDPR, HIPAA, PCI-DSS, SOX
- Suporte a CSV, Parquet, SQL, Delta Lake

**Documentação**: [data_classification/README.md](data_classification/README.md)

**Exemplo**:
```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()

# Classificar dados
report = agent.classify_from_csv("customers.csv")

print(f"Sensibilidade: {report.overall_sensitivity}")
print(f"PII: {report.pii_columns}")
print(f"PHI: {report.phi_columns}")
print(f"Proprietário: {report.proprietary_columns}")
print(f"Compliance: {report.compliance_flags}")
```

---

### 5. Data Quality Agent

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

### 6. Data Asset Value Agent

Sistema de IA para **análise de valor de ativos de dados** baseado em padrões de uso, relacionamentos de JOIN, linhagem e impacto em data products.

**Características**:
- Análise de uso direto em queries SQL
- Mapeamento de relacionamentos de JOIN (hub assets)
- Integração com saída do Data Lineage Agent
- Scoring multi-dimensional de valor (uso, JOINs, linhagem, data products)
- Identificação de ativos críticos e órfãos
- Detecção de tendências de uso (increasing/stable/decreasing)
- Recomendações de governança automatizadas
- Suporte a configuração de data products com impacto de negócio

**Documentação**: [data_asset_value/README.md](data_asset_value/README.md)

**Exemplo**:
```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent(
    weights={
        'usage': 0.30,       # Frequência de uso
        'joins': 0.25,       # Relacionamentos de JOIN
        'lineage': 0.25,     # Impacto na linhagem
        'data_product': 0.20 # Importância em data products
    },
    time_range_days=30
)

# Analisar a partir de logs de queries
report = agent.analyze_from_query_logs(
    query_logs=query_logs,
    lineage_data=lineage_agent_output,  # Integração com Lineage Agent
    data_product_config=data_products,
    asset_metadata=asset_metadata
)

# Top assets por valor
for asset in report.asset_scores[:5]:
    print(f"{asset.asset_name}: {asset.overall_value_score:.1f} ({asset.value_category})")

# Ativos críticos e hubs
print(f"Críticos: {report.critical_assets}")
print(f"Hubs: {report.hub_assets}")
print(f"Órfãos: {report.orphan_assets}")

# Exportar relatório
print(report.to_markdown())
```

**Formato do Query Log**:
```json
[
  {
    "query": "SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id",
    "timestamp": "2024-12-15T10:30:00Z",
    "user": "analyst_maria",
    "duration_ms": 1250,
    "rows_scanned": 150000,
    "data_product": "customer_360"
  }
]
```

---

### 7. Sensitive Data NER Agent

Sistema de IA para **detecção e anonimização de dados sensíveis** em texto livre, projetado como gateway de proteção para requisições a LLMs terceiras.

**Características**:
- Detecção de PII (CPF, CNPJ, SSN, email, telefone, nomes)
- Detecção de PHI (CID-10, CNS, CRM, prontuário médico)
- Detecção de PCI (cartão de crédito, CVV, IBAN, SWIFT)
- Detecção de FINANCIAL (contas bancárias, PIX, criptomoedas)
- Detecção de BUSINESS (projetos confidenciais, termos estratégicos)
- Detecção de CREDENTIALS (API keys, tokens, secrets, senhas)
- Detecção determinística (90+ padrões regex) + preditiva (heurísticas, checksum)
- **🆕 Integração SpaCy**: NER com modelos treinados em português para maior precisão
- **🆕 POS Tagging**: Distinção automática entre nomes próprios e verbos
- Múltiplas estratégias de anonimização (REDACT, MASK, HASH, PARTIAL, ENCRYPT)
- **Secure Vault**: Armazenamento criptografado AES-256 para mapeamentos originais/anonimizados
- **Controle de Acesso**: 5 níveis de permissão (READ_ONLY, DECRYPT, FULL_DECRYPT, ADMIN, SUPER_ADMIN)
- **Política de Retenção**: DELETE_ON_DECRYPT, RETAIN_DAYS, RETAIN_FOREVER
- **Audit Log**: Trilha de auditoria tamper-evident com hash chain

**🆕 Integração SpaCy (Opcional)**:

O agente agora suporta SpaCy para maior precisão na detecção de nomes de pessoas, evitando falsos positivos como verbos e termos técnicos.

```bash
# Instalar SpaCy e modelo português
pip install spacy
python -m spacy download pt_core_news_md  # modelo médio (recomendado)
# ou
python -m spacy download pt_core_news_lg  # modelo grande (mais preciso)
```

Com SpaCy, o sistema:
- Classifica entidades como PER (Pessoa), ORG (Organização), LOC (Localização)
- Usa POS tagging para distinguir PROPN (nome próprio) de VERB (verbo)
- Evita falsos positivos como "Critical Error", "Carlos comprou", "Sistema Falhou"

**Documentação**: [sensitive_data_ner/README.md](sensitive_data_ner/README.md)

**Exemplo**:
```python
from sensitive_data_ner import SensitiveDataNERAgent, FilterPolicy, FilterAction

# Configurar política de filtro
policy = FilterPolicy(
    pii_action=FilterAction.ANONYMIZE,
    phi_action=FilterAction.BLOCK,
    pci_action=FilterAction.BLOCK,
    credentials_action=FilterAction.BLOCK,  # Bloquear API keys e secrets
    business_action=FilterAction.BLOCK,
)

agent = SensitiveDataNERAgent(
    filter_policy=policy,
    business_terms=["Projeto Arara Azul", "Operação Fênix"]
)

# Analisar texto
text = """
O cliente João Silva, CPF 123.456.789-09, solicitou acesso.
Use a API key: sk-abc123xyz para autenticação.
"""

result = agent.analyze(text)
print(f"Entidades: {result.statistics['total']}")
print(f"Risco: {result.risk_score:.1%}")
print(f"Texto seguro:\n{result.anonymized_text}")
```

**Com Secure Vault (armazenamento criptografado)**:
```python
from sensitive_data_ner import SecureVault, VaultConfig, RetentionPolicy

# Configurar vault com política de retenção
config = VaultConfig(
    storage_path=".secure_vault",
    default_retention_policy=RetentionPolicy.RETAIN_DAYS,
    default_retention_days=30,  # Manter por 30 dias após decrypt
)

vault = SecureVault(config)

# Criar sessão e armazenar mapeamento
session = vault.create_session(
    user_id="user123",
    original_text=text,
    anonymized_text=result.anonymized_text,
    mappings=[{"original": "123.456.789-09", "anonymized": "[CPF]"}]
)

# Posteriormente: recuperar texto original (requer permissão DECRYPT)
original = vault.decrypt_session(
    session_id=session.session_id,
    user_id="admin_user"
)
```

**Apenas Data Classification Agent**:
```bash
pip install -r classification/requirements.txt
```

**Apenas Metadata Enrichment Agent**:
```bash
pip install -r metadata_enrichment/requirements.txt
export OPENAI_API_KEY="sua-chave-aqui"
```

---

## Integração entre Agentes

Os 7 agentes podem ser **integrados** para um framework completo de governança:

```python
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery.agent import DataDiscoveryAgent
from metadata_enrichment.agent import MetadataEnrichmentAgent
from data_classification import DataClassificationAgent
from data_quality.agent import DataQualityAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# 1. LINEAGE: Mapear dependências do pipeline
lineage_agent = DataLineageAgent()
lineage_result = lineage_agent.analyze_pipeline(["etl/*.sql", "etl/*.py"])
print(f"Assets mapeados: {lineage_result['metrics']['total_assets']}")

# 2. CLASSIFICATION: Classificar dados por sensibilidade
classification_agent = DataClassificationAgent()
classification_report = classification_agent.classify_from_csv("data/customers.csv")
print(f"Sensibilidade: {classification_report.overall_sensitivity}")
print(f"PII detectado: {classification_report.pii_columns}")
print(f"Compliance: {classification_report.compliance_flags}")

# 3. QUALITY: Avaliar qualidade dos dados
quality_agent = DataQualityAgent()
quality_report = quality_agent.evaluate_file(
    "data/customers.csv",
    freshness_config={"timestamp_column": "updated_at", "sla_hours": 24}
)
print(f"Qualidade: {quality_report.overall_score:.0%}")

# 4. ENRICHMENT: Gerar metadados automaticamente
enrichment_agent = MetadataEnrichmentAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards")
)
enrichment_result = enrichment_agent.enrich_from_csv("data/customers.csv")
print(f"Descrição: {enrichment_result.description}")

# 5. DISCOVERY: Indexar para busca semântica
discovery_agent = DataDiscoveryAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(),
    vector_store=ChromaStore(collection_name="catalog")
)

# Criar metadados enriquecidos com classificação e qualidade
from rag_discovery.agent import TableMetadata
table = TableMetadata(
    name=enrichment_result.table_name,
    description=enrichment_result.description,
    columns=[{"name": c.name, "type": c.original_type, "description": c.description}
             for c in enrichment_result.columns],
    tags=enrichment_result.tags + [
        f"quality:{quality_report.overall_status}",
        f"sensitivity:{classification_report.overall_sensitivity}"
    ]
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
| **Compliance LGPD/GDPR** | Classification + Lineage | Rastrear e classificar dados pessoais |
| **Detecção de PII/PHI** | Classification | Identificar dados sensíveis automaticamente |
| **Monitoramento Contínuo** | Quality + Discovery | Alertas de qualidade no catálogo |
| **Onboarding** | Discovery + Lineage | Entender o data lake rapidamente |
| **Valor de Ativos** | Value + Lineage | Identificar ativos críticos e subutilizados |
| **Priorização de Investimentos** | Value + Classification | Focar em ativos de alto valor e alto risco |
| **Gateway de LLM Seguro** | NER + Vault | Filtrar dados sensíveis antes de enviar a LLMs |
| **Proteção de Credenciais** | NER | Detectar vazamento de API keys e secrets |

---

## Interface Unificada (Streamlit)

O projeto inclui uma interface web unificada com todos os 7 agentes:

```bash
streamlit run app.py
```

**Tabs disponíveis**:
- **Lineage**: Upload de arquivos e visualização de grafos
- **Discovery**: Chat com o catálogo de dados
- **Enrichment**: Geração automática de metadados
- **Classification**: Detecção de PII/PHI/Financeiro
- **Quality**: Avaliação de qualidade e alertas
- **Asset Value**: Análise de valor de ativos de dados
- **Sensitive Data NER**: Detecção e anonimização de dados sensíveis com Vault

Cada agente também possui interface standalone:
```bash
streamlit run lineage/app.py          # Apenas Lineage
streamlit run rag_discovery/app.py    # Apenas Discovery (se disponível)
streamlit run metadata_enrichment/streamlit_app.py  # Apenas Enrichment
streamlit run data_classification/streamlit_app.py  # Apenas Classification
streamlit run data_quality/streamlit_app.py         # Apenas Quality
streamlit run sensitive_data_ner/streamlit_app.py   # Apenas NER
```

---

## Instalação

### Pré-requisitos

- Python 3.8+
- OpenAI API Key (para agentes que usam LLM)
- [uv](https://docs.astral.sh/uv/) (recomendado) ou pip

### Instalando o uv

O `uv` é um gerenciador de pacotes Python extremamente rápido (escrito em Rust), que substitui pip, pip-tools e virtualenv.

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip (alternativo)
pip install uv

# Via Homebrew (macOS)
brew install uv
```

### Instalação Completa com uv (Recomendado)

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie e ative o ambiente virtual
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instale todas as dependências
uv pip install -r requirements.txt

# Configure variáveis de ambiente
export OPENAI_API_KEY="sua-chave-aqui"
```

### Instalação Completa com pip

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instale todas as dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
export OPENAI_API_KEY="sua-chave-aqui"
```

### Instalação Individual

Instale apenas os agentes que você precisa:

**Com uv:**
```bash
# Apenas Lineage Agent (sem LLM)
uv pip install -r lineage/requirements.txt

# Apenas RAG Discovery Agent
uv pip install -r rag_discovery/requirements.txt

# Apenas Metadata Enrichment Agent
uv pip install -r metadata_enrichment/requirements.txt

# Apenas Data Classification Agent (sem LLM)
uv pip install -r data_classification/requirements.txt

# Apenas Data Quality Agent (sem LLM)
uv pip install -r data_quality/requirements.txt

# Apenas Sensitive Data NER Agent (sem LLM)
uv pip install -r sensitive_data_ner/requirements.txt
```

**Com pip:**
```bash
# Apenas Lineage Agent (sem LLM)
pip install -r lineage/requirements.txt

# Apenas RAG Discovery Agent
pip install -r rag_discovery/requirements.txt

# Apenas Metadata Enrichment Agent
pip install -r metadata_enrichment/requirements.txt

# Apenas Data Classification Agent (sem LLM)
pip install -r data_classification/requirements.txt

# Apenas Data Quality Agent (sem LLM)
pip install -r data_quality/requirements.txt

# Apenas Sensitive Data NER Agent (sem LLM)
pip install -r sensitive_data_ner/requirements.txt
```

---

## Arquitetura

```
data-governance-ai-agents-kit/
│
├── app.py                           # Interface Streamlit unificada (6 tabs)
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
├── data_classification/             # Data Classification Agent
│   ├── agent.py                     # Agente principal
│   ├── classifiers/                 # Classificadores por categoria
│   ├── rules/                       # Regras de classificação
│   ├── examples/
│   ├── streamlit_app.py
│   └── README.md
│
├── data_quality/                    # Data Quality Agent
│   ├── agent.py                     # Agente principal
│   ├── metrics/                     # Métricas de qualidade
│   │   ├── quality_metrics.py       # 5 dimensões
│   │   └── schema_drift.py          # Detecção de drift
│   ├── rules/                       # Sistema de regras
│   │   └── quality_rules.py
│   ├── connectors/                  # Conectores de dados
│   │   └── data_connector.py
│   ├── examples/
│   ├── streamlit_app.py
│   └── README.md
│
├── data_asset_value/                # Data Asset Value Agent
│   ├── agent.py                     # Agente principal com parser e calculator
│   ├── __init__.py                  # Exports do módulo
│   └── examples/
│       ├── sample_query_logs.json   # Logs de queries de exemplo
│       ├── data_products_config.json # Config de data products
│       ├── asset_metadata.json      # Metadados de ativos (criticidade, custo, risco)
│       └── usage_example.py         # Script de exemplo de uso
│
└── sensitive_data_ner/              # Sensitive Data NER Agent
    ├── __init__.py                  # Exports principais
    ├── agent.py                     # SensitiveDataNERAgent (principal)
    ├── anonymizers.py               # Estratégias de anonimização
    ├── streamlit_app.py             # Interface visual
    ├── patterns/                    # Padrões de detecção
    │   ├── __init__.py
    │   └── entity_patterns.py       # 50+ padrões regex por categoria
    ├── predictive/                  # Detecção preditiva
    │   ├── __init__.py
    │   ├── validators.py            # Validação de checksum (CPF, cartão, etc.)
    │   └── heuristics.py            # Análise de contexto e confiança
    ├── vault/                       # Armazenamento seguro
    │   ├── __init__.py
    │   ├── vault.py                 # SecureVault principal
    │   ├── storage.py               # SQLite/PostgreSQL backends
    │   ├── key_manager.py           # Gestão de chaves AES-256
    │   ├── access_control.py        # RBAC (5 níveis)
    │   └── audit.py                 # Audit logging tamper-evident
    └── examples/
        └── usage_example.py
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

| Característica | Lineage | Discovery | Enrichment | Classification | Quality | Asset Value | NER |
|---------------|---------|-----------|------------|----------------|---------|-------------|-----|
| **Objetivo** | Mapear dependências | Busca semântica | Gerar metadados | Classificar sensibilidade | Monitorar qualidade | Analisar valor | Anonimizar textos |
| **Input** | Código (SQL, Python) | Query em LN | Dados (CSV, Parquet) | Dados (CSV, Parquet) | Dados (CSV, Parquet) | Query logs (JSON) | Texto livre |
| **Output** | Grafo + Impacto | Respostas + Tabelas | Descrições + Tags | PII/PHI/PCI + Compliance | Score + Alertas | Value Score + Insights | Texto anonimizado |
| **LLM** | Opcional | Requerido | Requerido | Não | Não | Não | Não |
| **Embeddings** | Não | Sim | Sim | Não | Não | Não | Não |
| **Principais Features** | Impact analysis, Ciclos | Híbrido search, RAG | Standards, Owner | LGPD, HIPAA, PCI-DSS | SLA, Schema drift | Usage, JOINs, Hubs | Vault, Audit, Retention |

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
# Identificar e rastrear dados pessoais com Classification Agent
from data_classification import DataClassificationAgent

classifier = DataClassificationAgent()
results = []

for file in data_files:
    # Classificar dados por sensibilidade
    classification = classifier.classify_from_csv(file)

    if classification.pii_columns or classification.phi_columns:
        # Rastrear linhagem dos dados sensíveis
        lineage = lineage_agent.analyze_pipeline([file])
        results.append({
            "file": file,
            "sensitivity": classification.overall_sensitivity,
            "pii_columns": classification.pii_columns,
            "phi_columns": classification.phi_columns,
            "compliance_flags": classification.compliance_flags,
            "downstream_impact": lineage["metrics"]["total_assets"]
        })

print(f"Encontrados {len(results)} arquivos com dados sensíveis")
for r in results:
    print(f"  {r['file']}: {r['sensitivity']} - {r['compliance_flags']}")
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
- [x] Data Classification Agent com LGPD/HIPAA/PCI-DSS
- [x] Data Quality Agent com SLA monitoring
- [x] Data Asset Value Agent com análise de uso e impacto
- [x] Sensitive Data NER Agent com detecção de 6 categorias (PII, PHI, PCI, Financial, Business, Credentials)
- [x] Secure Vault com criptografia AES-256 e controle de acesso
- [x] Política de retenção (DELETE_ON_DECRYPT, RETAIN_DAYS, RETAIN_FOREVER)
- [x] Interface Streamlit unificada (7 agentes)
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

### Classification Agent
- [x] Regras de PII/PHI/Financeiro baseadas em metadados
- [x] Níveis de severidade e recomendações LGPD/GDPR
- [ ] Validação multilíngue com LLM
- [ ] Biblioteca ampliada de regras setoriais

### Metadata Enrichment Agent
- [x] RAG sobre normativos internos
- [x] Suporte a sampling (CSV, Parquet, SQL, Delta)
- [x] Exportação em JSON/Markdown/HTML
- [ ] Conectores adicionais (BigQuery, S3 inventories)
- [ ] Templates personalizáveis de catálogo

---

## Contribuindo

Contribuições são muito bem-vindas! Este guia explica passo a passo como contribuir com o projeto.

### Pré-requisitos

Antes de contribuir, certifique-se de ter:
- Conta no GitHub
- Git instalado localmente
- Python 3.8+ instalado
- [uv](https://docs.astral.sh/uv/) instalado (recomendado)

### Passo 1: Fazer Fork do Repositório

1. Acesse o repositório original no GitHub
2. Clique no botão **"Fork"** no canto superior direito
3. Selecione sua conta como destino do fork
4. Aguarde a criação do fork (cópia do repositório na sua conta)

### Passo 2: Clonar seu Fork

```bash
# Clone SEU fork (substitua <seu-usuario> pelo seu username do GitHub)
git clone https://github.com/<seu-usuario>/data-governance-ai-agents-kit.git

# Entre no diretório
cd data-governance-ai-agents-kit

# Configure o repositório original como "upstream"
git remote add upstream https://github.com/allpedroza/data-governance-ai-agents-kit.git

# Verifique os remotes configurados
git remote -v
# Deve mostrar:
# origin    https://github.com/<seu-usuario>/data-governance-ai-agents-kit.git (fetch)
# origin    https://github.com/<seu-usuario>/data-governance-ai-agents-kit.git (push)
# upstream  https://github.com/allpedroza/data-governance-ai-agents-kit.git (fetch)
# upstream  https://github.com/allpedroza/data-governance-ai-agents-kit.git (push)
```

### Passo 3: Manter seu Fork Atualizado

Antes de criar uma nova branch, sempre sincronize com o repositório original:

```bash
# Busque as atualizações do repositório original
git fetch upstream

# Mude para a branch principal
git checkout main

# Mescle as atualizações
git merge upstream/main

# Atualize seu fork no GitHub
git push origin main
```

### Passo 4: Criar uma Branch para sua Contribuição

**Importante**: Nunca faça commits diretamente na branch `main`. Sempre crie uma branch específica.

```bash
# Crie e mude para uma nova branch
git checkout -b feature/nome-da-sua-feature

# Exemplos de nomes de branch:
# feature/add-bigquery-connector     (para novas funcionalidades)
# fix/lineage-parser-error           (para correções de bugs)
# docs/update-readme                 (para documentação)
# refactor/improve-quality-metrics   (para refatorações)
```

### Passo 5: Fazer suas Alterações

1. Faça as alterações necessárias no código
2. Adicione ou atualize testes se aplicável
3. Verifique se o código segue os padrões do projeto
4. Teste suas alterações localmente

```bash
# Configure o ambiente
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Execute os testes (se houver)
pytest

# Verifique se a aplicação funciona
streamlit run app.py
```

### Passo 6: Commit das Mudanças

```bash
# Adicione os arquivos modificados
git add .

# Faça o commit com mensagem descritiva
git commit -m "feat: adiciona conector para BigQuery"

# Padrões de mensagem de commit:
# feat: nova funcionalidade
# fix: correção de bug
# docs: alterações em documentação
# refactor: refatoração de código
# test: adição ou modificação de testes
# chore: tarefas de manutenção
```

### Passo 7: Enviar para seu Fork

```bash
# Envie sua branch para seu fork no GitHub
git push origin feature/nome-da-sua-feature
```

### Passo 8: Criar um Pull Request

1. Acesse seu fork no GitHub
2. Você verá uma mensagem sugerindo criar um Pull Request - clique em **"Compare & pull request"**
3. Ou clique na aba **"Pull requests"** → **"New pull request"**
4. Certifique-se de que:
   - **base repository**: `allpedroza/data-governance-ai-agents-kit`
   - **base**: `main`
   - **head repository**: `<seu-usuario>/data-governance-ai-agents-kit`
   - **compare**: `feature/nome-da-sua-feature`
5. Preencha o título e descrição do PR:
   - Descreva o que foi alterado
   - Explique o motivo da alteração
   - Mencione issues relacionadas (se houver)
6. Clique em **"Create pull request"**

### Boas Práticas

- **Mantenha PRs pequenos e focados**: Um PR deve resolver uma única issue ou adicionar uma única feature
- **Escreva testes**: Sempre que possível, adicione testes para novas funcionalidades
- **Documente**: Atualize a documentação se sua alteração afetar o uso do projeto
- **Siga o estilo do código**: Mantenha consistência com o código existente
- **Responda feedback**: Esteja disponível para responder comentários e fazer ajustes

### Fluxo Visual

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUXO DE CONTRIBUIÇÃO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FORK                    2. CLONE                            │
│  ┌──────────┐              ┌──────────┐                         │
│  │ Original │──fork───────▶│ Seu Fork │                         │
│  │  (GitHub)│              │ (GitHub) │                         │
│  └──────────┘              └────┬─────┘                         │
│                                 │clone                          │
│                                 ▼                               │
│                            ┌──────────┐                         │
│  3. BRANCH                 │  Local   │                         │
│  main ────────────────────▶│   Repo   │                         │
│    └── feature/xyz         └────┬─────┘                         │
│                                 │                               │
│  4. COMMIT & PUSH               │                               │
│  ┌──────────────────────────────┘                               │
│  │                                                              │
│  ▼                                                              │
│  ┌──────────┐    push     ┌──────────┐    PR     ┌──────────┐  │
│  │  Local   │────────────▶│ Seu Fork │──────────▶│ Original │  │
│  │  Branch  │             │ (GitHub) │           │ (GitHub) │  │
│  └──────────┘             └──────────┘           └──────────┘  │
│                                                                 │
│  5. REVIEW: Aguarde revisão e faça ajustes se necessário       │
│  6. MERGE: Após aprovação, sua contribuição será incorporada!  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dúvidas?

Se tiver dúvidas sobre como contribuir, abra uma [issue](https://github.com/allpedroza/data-governance-ai-agents-kit/issues) no repositório.

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
