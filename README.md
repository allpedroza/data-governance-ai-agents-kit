# Data Governance AI Agents Kit

**Kit completo de agentes de IA para governança de dados**, incluindo análise de linhagem e descoberta de dados com RAG.

## 📋 Visão Geral

Este projeto fornece **agentes de IA especializados** para resolver desafios comuns de governança de dados:

1. **🔗 Data Lineage Agent**: Análise automática de linhagem de dados
2. **🔍 Data Discovery RAG Agent**: Descoberta de dados usando RAG com banco vetorizado
3. **🛡️ Data Classification Agent**: Classificação de PII/PHI/Financeiro a partir de metadados
4. **🧠 Metadata Enrichment Agent**: Geração automática de descrições, tags e classificações para ativos de dados

## 🚀 Agentes Disponíveis

### 1. Data Lineage Agent

Sistema de IA para **análise automática de linhagem de dados** em pipelines complexos.

**Características**:
- ✅ Análise de múltiplos formatos (Python, SQL, Terraform, Databricks, Airflow)
- ✅ Extração automática de dependências
- ✅ Visualização interativa de grafos
- ✅ Análise de impacto de mudanças
- ✅ Identificação de componentes críticos
- ✅ Integração com Apache Atlas

**Documentação**: [lineage/README.md](lineage/README.md)

**Casos de Uso**:
- Mapeamento de dependências em pipelines
- Análise de impacto antes de mudanças
- Identificação de pontos únicos de falha
- Auditoria e compliance

**Exemplo Rápido**:
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

Sistema de IA para **descoberta de dados** usando **RAG (Retrieval-Augmented Generation)** com banco vetorizado.

**Características**:
- ✅ Busca semântica em linguagem natural
- ✅ Banco vetorizado (ChromaDB) para metadados
- ✅ Perguntas e respostas com contexto completo
- ✅ Integração com Apache Atlas
- ✅ Integração com Data Lineage Agent
- ✅ Suporte a múltiplos formatos (Parquet, Delta, CSV)

**Documentação**: [rag_discovery/README.md](rag_discovery/README.md)

**Casos de Uso**:
- Descoberta de dados em data lakes complexos
- Onboarding de novos membros
- Identificação de dados sensíveis (PII)
- Documentação automática
- Recomendação de datasets

**Exemplo Rápido**:
```python
from rag_discovery import DataDiscoveryRAGAgent, TableMetadata

# Inicializa o agente
agent = DataDiscoveryRAGAgent(
    collection_name="my_data_lake"
)

# Indexa uma tabela
table = TableMetadata(
    name="customers",
    database="production",
    description="Dados de clientes",
    columns=[
        {"name": "id", "type": "bigint"},
        {"name": "name", "type": "varchar"}
    ],
    tags=["pii", "critical"]
)
agent.index_table(table)

# Busca semântica
results = agent.search("Onde estão os dados de clientes?")

# Pergunta com RAG
response = agent.ask(
    "Quais tabelas devo usar para análise de vendas?"
)
print(response['answer'])
```

---

## 🔗 Integração entre Agentes

Os agentes podem ser **integrados** para governança completa:

```python
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery import DataDiscoveryRAGAgent
from metadata_enrichment.agent import MetadataEnrichmentAgent
from rag_discovery.examples.lineage_integration import convert_lineage_assets_to_metadata

# 1. Analisa linhagem
lineage_agent = DataLineageAgent()
lineage_agent.analyze_pipeline(["pipeline.sql", "etl.py"])

# 2. Converte para metadados RAG (com contexto de linhagem)
tables = convert_lineage_assets_to_metadata(lineage_agent)

# 3. Indexa com contexto de dependências
rag_agent = DataDiscoveryRAGAgent()
rag_agent.index_tables_batch(tables)

# 4. Enriquecimento automático de metadados
enrichment_agent = MetadataEnrichmentAgent(...)
enriched_tables = [
    enrichment_agent.enrich_from_sql(table.name, connection_string="...")
    for table in tables
]

# 5. Classificação de sensibilidade (usando schemas enriquecidos)
# ... montar TableSchema a partir dos metadados e usar DataClassificationAgent

# 6. Busca considerando impacto e sensibilidade
results = rag_agent.search("tabelas críticas com PII e alto impacto downstream")

# 7. Análise de impacto enriquecida
response = rag_agent.ask("Se eu modificar a tabela customers, qual o impacto?")
```

**Benefícios da Integração**:
- 🎯 Descoberta de dados com contexto de linhagem
- 📊 Análise de impacto enriquecida com IA
- 🔍 Busca semântica considerando dependências e sensibilidade
- 📝 Documentação automática e enriquecimento de catálogos

---

### 3. Data Classification Agent

Agente para **classificar automaticamente dados sensíveis (PII, PHI e financeiros)** usando apenas schemas e metadados, garantindo alinhamento com **LGPD/GDPR** sem acessar os dados brutos.

**Características**:
- ✅ Identificação de PII/PHI/Financeiro via nomes, tipos, descrições e tags
- ✅ Níveis de severidade (LOW, MEDIUM, HIGH, CRITICAL)
- ✅ Recomendações de compliance (DPIA, minimização, mascaramento/tokenização)
- ✅ Extensível com regras customizadas (`SensitiveDataRule`)

**Documentação**: [classification/README.md](classification/README.md)

**Exemplo Rápido**:
```python
from classification import (
    ColumnMetadata,
    DataClassificationAgent,
    TableSchema,
)

table = TableSchema(
    name="payments",
    schema="finance",
    description="Transações com cartão e CPF do pagador",
    columns=[
        ColumnMetadata(name="payment_id", type="bigint"),
        ColumnMetadata(name="cpf", type="varchar", tags=["pii"]),
        ColumnMetadata(name="credit_card_number", type="varchar"),
    ],
)

agent = DataClassificationAgent()
classification = agent.classify_table(table)
print(classification.sensitivity_level)  # HIGH
print(classification.detected_categories)  # ['FINANCIAL', 'PII']
```

---

### 4. Metadata Enrichment Agent

Agente de IA para **gerar descrições, tags, classificação e detecção de PII** a partir de schemas, amostras de dados e normativos.

**Características**:
- ✅ Geração automática de descrições PT/EN para tabelas e colunas
- ✅ Classificação de dados (public, internal, confidential, restricted) com detecção de PII
- ✅ Sugestão de domínio e proprietário, além de tags de organização
- ✅ RAG sobre normativos internos (nomenclatura, governança, segurança)
- ✅ Data sampling para CSV, Parquet, SQL e Delta Lake
- ✅ Exportação em JSON, Markdown e HTML

**Documentação**: [metadata_enrichment/README.md](metadata_enrichment/README.md)

**Casos de Uso**:
- Documentação automática de tabelas de data lakes/warehouses
- Criação rápida de catálogos de dados com sugestões consistentes
- Enriquecimento de metadados para onboarding e descoberta
- Padronização baseada em normativos internos

**Exemplo Rápido**:
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

agent.index_standards_from_json("./examples/sample_standards.json")
result = agent.enrich_from_csv("./data/customers.csv")

print(result.classification)  # ex.: confidential
print(result.has_pii)
```

---

## 📦 Instalação

### Pré-requisitos

- Python 3.8+
- OpenAI API Key (para RAG Agent)

### Instalação Completa

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Instale todas as dependências da UI + agentes usando o MESMO Python do Streamlit
python -m pip install -r requirements.txt

# Configure variáveis de ambiente
export OPENAI_API_KEY="sua-chave-aqui"
```

> Se ainda aparecer `ModuleNotFoundError: No module named 'openai'`, confirme que o
> `python -m pip` acima corresponde ao Python que executará `streamlit run app.py`.
> Você pode verificar com `python -V` e `python -m pip -V`.

### Instalação Individual

**Apenas Lineage Agent**:
```bash
pip install -r lineage/requirements.txt
```

**Apenas RAG Agent**:
```bash
pip install -r rag_discovery/requirements.txt
export OPENAI_API_KEY="sua-chave-aqui"
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

## 🎯 Casos de Uso Combinados

### 1. Governança Completa de Data Lake

**Cenário**: Empresa precisa de visibilidade completa do data lake

**Solução**:
1. Use **Lineage Agent** para mapear dependências
2. Use **Metadata Enrichment Agent** para gerar descrições e classificação
3. Use **Classification Agent** para confirmar sensibilidade
4. Use **RAG Agent** para descoberta semântica com contexto completo
5. Combine para análise de impacto contextualizada

### 2. Migração de Plataforma

**Cenário**: Migração de on-premise para cloud

**Solução**:
1. **Lineage Agent** identifica todas as dependências
2. **RAG Agent** documenta e organiza metadados
3. Análise de impacto previne quebras

### 3. Compliance e Auditoria

**Cenário**: Atender LGPD/GDPR

**Solução**:
1. **Metadata Enrichment Agent** sugere domínios, donos e detecta PII
2. **Classification Agent** consolida níveis de sensibilidade e controles
3. **Lineage Agent** rastreia fluxo de dados sensíveis
4. **RAG Agent** facilita buscas contextualizadas para auditoria

### 4. Onboarding de Equipe

**Cenário**: Novos data engineers precisam entender o data lake

**Solução**:
1. **RAG Agent** responde perguntas em linguagem natural
2. **Lineage Agent** mostra dependências visualmente
3. Documentação contextualizada automática

---

## 📚 Exemplos

### Lineage Agent

```bash
# Exemplo básico
cd lineage
python examples/basic_usage.py

# Análise de impacto
python examples/impact_analysis.py

# Visualização Atlas
python examples/atlas_visualization.py
```

### RAG Agent

```bash
# Exemplo básico
cd rag_discovery
python examples/basic_usage.py

# Integração com Atlas
python examples/atlas_integration.py

# Integração com Lineage
python examples/lineage_integration.py
```

---

## 🏗️ Arquitetura

```
data-governance-ai-agents-kit/
│
├── lineage/                          # Data Lineage Agent
│   ├── data_lineage_agent.py         # Agente principal
│   ├── parsers/                      # Parsers (SQL, Python, etc)
│   ├── examples/                     # Exemplos de uso
│   ├── requirements.txt
│   └── README.md
│
├── rag_discovery/                    # Data Discovery RAG Agent
│   ├── data_discovery_rag_agent.py   # Agente principal
│   ├── examples/                     # Exemplos de uso
│   │   ├── basic_usage.py
│   │   ├── atlas_integration.py
│   │   └── lineage_integration.py
│   ├── requirements.txt
│   ├── .gitignore
│   └── README.md
│
├── classification/                   # Data Classification Agent
│   ├── data_classification_agent.py  # Agente principal
│   ├── requirements.txt
│   └── README.md
│
├── metadata_enrichment/              # Metadata Enrichment Agent
│   ├── agent.py                      # Agente principal
│   ├── standards/                    # RAG para normativos
│   ├── sampling/                     # Coletores de amostras de dados
│   ├── examples/                     # Exemplos e normativos
│   ├── streamlit_app.py              # UI dedicada
│   └── README.md
│
└── README.md                         # Este arquivo
```

---

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# OpenAI (para RAG Agent)
export OPENAI_API_KEY="sk-..."
export OPENAI_API_URL="https://api.openai.com/v1"  # Opcional

# Data Lineage LLM (opcional - para fallback parsing)
export DATA_LINEAGE_LLM_MODEL="gpt-5.1"

# Apache Atlas (opcional)
export ATLAS_HOST="http://atlas-host:21000"
export ATLAS_USERNAME="admin"
export ATLAS_PASSWORD="admin"
```

---

## 📊 Comparação de Agentes

| Característica | Lineage Agent | RAG Agent | Classification Agent | Metadata Enrichment Agent |
|---------------|---------------|-----------|----------------------|---------------------------|
| **Objetivo** | Mapear dependências | Descobrir dados | Classificar PII/PHI/Financeiro | Enriquecer descrições, tags e classificação |
| **Input** | Código (SQL, Python) | Metadados | Schemas e metadados | Schemas, amostras e normativos |
| **Output** | Grafo de linhagem | Respostas em LN | Nível de sensibilidade e controles | Descrições PT/EN, tags, classificação |
| **Técnica** | AST parsing + Graph | Embeddings + RAG | Regras semânticas + (opcional) LLM | RAG sobre normativos + sampling |
| **LLM** | Opcional (fallback) | Requerido | Opcional (validação) | Recomendado |
| **Casos de Uso** | Análise de impacto | Busca semântica | Compliance LGPD/GDPR | Catálogo e documentação automática |

---

## 🛣️ Roadmap

### Lineage Agent
- [x] Parsers básicos (SQL, Python, Terraform)
- [x] Visualização de grafos
- [x] Análise de impacto
- [x] Integração com Apache Atlas
- [ ] Suporte a dbt
- [ ] Suporte a Airflow nativo
- [ ] Column-level lineage

### RAG Agent
- [x] Busca semântica básica
- [x] Integração com Atlas
- [x] Integração com Lineage Agent
- [ ] Suporte a modelos locais (sentence-transformers)
- [ ] Interface web interativa
- [ ] Integração com AWS Glue
- [ ] Integração com Databricks Unity Catalog
- [ ] Cache de embeddings

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

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/amazing-feature`)
3. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
4. Push para a branch (`git push origin feature/amazing-feature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

---

## 📧 Suporte

Para dúvidas, sugestões ou suporte:

- 🐛 **Issues**: Abra uma issue no GitHub
- 💬 **Discussões**: Use a seção de Discussions
- 📧 **Email**: [seu-email]

---

## 🙏 Agradecimentos

- **Apache Atlas** - Integração de catálogo
- **ChromaDB** - Banco vetorizado
- **OpenAI** - Embeddings e LLM
- **NetworkX** - Análise de grafos
- **Plotly** - Visualizações interativas

---

## ⭐ Star History

Se este projeto foi útil para você, considere dar uma ⭐!

---
