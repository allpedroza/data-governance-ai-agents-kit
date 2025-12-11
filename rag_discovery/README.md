# 🔍 Data Discovery RAG Agent

Sistema de IA para **descoberta de dados** usando **RAG (Retrieval-Augmented Generation)** com banco vetorizado. Permite busca semântica em metadados de data lakes usando linguagem natural.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Instalação](#instalação)
- [Guia Rápido](#guia-rápido)
- [Casos de Uso](#casos-de-uso)
- [Exemplos](#exemplos)
- [Arquitetura](#arquitetura)
- [Integração com Apache Atlas](#integração-com-apache-atlas)
- [Integração com Data Lineage Agent](#integração-com-data-lineage-agent)
- [API Reference](#api-reference)
- [FAQ](#faq)

## 🎯 Visão Geral

O **Data Discovery RAG Agent** resolve o problema de **descoberta de dados** em data lakes complexos. Usando técnicas de RAG (Retrieval-Augmented Generation), o agente:

1. **Indexa metadados** de tabelas em um banco vetorizado (ChromaDB)
2. Permite **busca semântica** usando linguagem natural
3. **Responde perguntas** sobre os dados com contexto completo
4. **Integra** com ferramentas de catálogo como Apache Atlas
5. **Combina** com análise de linhagem para governança completa

### Por que usar RAG para descoberta de dados?

- ✅ **Busca natural**: "Onde estão os dados de clientes?" ao invés de queries SQL complexas
- ✅ **Contextual**: Entende sinônimos, conceitos relacionados e intenção
- ✅ **Escalável**: Funciona com milhares de tabelas
- ✅ **Inteligente**: Usa LLM para explicar e recomendar datasets

## ✨ Características

### Core Features

- 🔎 **Busca Semântica**: Encontre tabelas usando linguagem natural
- 🤖 **RAG Completo**: Perguntas e respostas com contexto de metadados
- 💾 **Banco Vetorizado**: ChromaDB para embedding storage eficiente
- 📊 **Metadados Ricos**: Suporta colunas, tags, descrições, estatísticas
- 🔄 **Integração Atlas**: Importa metadados do Apache Atlas
- 🌐 **Multi-formato**: Suporta Parquet, Delta, CSV, etc.

### Funcionalidades Avançadas

- 🎯 **Relevance Scoring**: Ranking inteligente de resultados
- 🏷️ **Tag-based Filtering**: Filtre por PII, critical, etc.
- 📈 **Estatísticas**: Insights sobre o catálogo indexado
- 💾 **Export/Import**: Backup e portabilidade de metadados
- 🔗 **Lineage Integration**: Combine com análise de linhagem

## 📦 Instalação

### Pré-requisitos

- Python 3.8+
- Chave de API da OpenAI (para embeddings e LLM)

### Instalação básica

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit/rag_discovery

# Instale as dependências
pip install -r requirements.txt

# Configure a API key da OpenAI
export OPENAI_API_KEY="sua-chave-aqui"

# Opcional: configure URL customizada (ex: Azure OpenAI)
export OPENAI_API_URL="https://api.openai.com/v1"
```

### Instalação com Docker

```bash
docker build -t data-discovery-rag .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY data-discovery-rag
```

## 🚀 Guia Rápido

### 1. Inicializar o Agente

```python
from data_discovery_rag_agent import DataDiscoveryRAGAgent

# Inicializa o agente
agent = DataDiscoveryRAGAgent(
    collection_name="my_data_lake",
    persist_directory="./chroma_db"
)
```

### 2. Indexar Metadados

```python
from data_discovery_rag_agent import TableMetadata

# Cria metadados de uma tabela
table = TableMetadata(
    name="customers",
    database="production",
    schema="public",
    description="Tabela de clientes com dados cadastrais",
    columns=[
        {"name": "id", "type": "bigint", "description": "ID único"},
        {"name": "name", "type": "varchar", "description": "Nome do cliente"},
        {"name": "email", "type": "varchar", "description": "Email"}
    ],
    owner="data-team",
    tags=["pii", "critical"],
    location="s3://lake/customers/",
    format="delta"
)

# Indexa a tabela
agent.index_table(table)
```

### 3. Buscar Tabelas

```python
# Busca usando linguagem natural
results = agent.search("Onde estão os dados de clientes?", n_results=5)

for result in results:
    print(f"{result.table.name}: {result.relevance_score:.1%}")
    print(f"  {result.table.description}")
```

### 4. Fazer Perguntas

```python
# Pergunta com RAG completo (LLM + busca vetorizada)
response = agent.ask(
    "Quais tabelas devo usar para análise de vendas por cliente?"
)

print(response['answer'])
print(f"Confiança: {response['confidence']:.1%}")
```

## 💡 Casos de Uso

### 1. Descoberta de Dados

**Problema**: Data engineers perdem tempo procurando tabelas relevantes

**Solução**:
```python
# Ao invés de navegar por centenas de tabelas...
results = agent.search("dados de transações financeiras do último ano")

# Obtém exatamente o que precisa com contexto
```

### 2. Onboarding de Novos Membros

**Problema**: Novos membros não conhecem o data lake

**Solução**:
```python
response = agent.ask(
    "Como funcionam os dados de usuários neste data lake? "
    "Quais tabelas existem e para que servem?"
)
# Documentação automática e contextual
```

### 3. Compliance e Governança

**Problema**: Identificar todos os dados sensíveis (PII)

**Solução**:
```python
# Busca com filtros
results = agent.search(
    "dados pessoais",
    filter_metadata={"tags": "pii"}
)

# Auditoria facilitada
```

### 4. Análise de Impacto

**Problema**: Entender o impacto de mudanças em tabelas

**Solução**:
```python
# Combina com Data Lineage Agent
response = agent.ask(
    "Se eu modificar a tabela customers, "
    "quais outras tabelas serão impactadas?"
)
```

### 5. Data Quality Monitoring

**Problema**: Identificar tabelas que precisam de atenção

**Solução**:
```python
results = agent.search(
    "tabelas grandes sem documentação ou com poucos metadados"
)
```

## 📚 Exemplos

### Exemplo 1: Uso Básico

```bash
python examples/basic_usage.py
```

Demonstra:
- Inicialização do agente
- Indexação de metadados
- Buscas semânticas
- Perguntas com RAG

### Exemplo 2: Integração com Apache Atlas

```bash
python examples/atlas_integration.py
```

Demonstra:
- Import de metadados do Atlas
- Conversão de entidades Atlas
- Busca em catálogo corporativo

### Exemplo 3: Integração com Data Lineage

```bash
python examples/lineage_integration.py
```

Demonstra:
- Análise de linhagem + descoberta
- Contexto enriquecido com dependências
- Análise de impacto combinada

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
│          "Onde estão os dados de clientes?"            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAI Embeddings API                      │
│           (text-embedding-3-small)                      │
└────────────────────┬────────────────────────────────────┘
                     │ Vector (1536 dimensions)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  ChromaDB                               │
│            (Vector Database)                            │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Table 1   │  │   Table 2   │  │   Table N   │   │
│  │  Embedding  │  │  Embedding  │  │  Embedding  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │ Top K Results
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Context Builder                            │
│     (Prepares metadata for LLM)                         │
└────────────────────┬────────────────────────────────────┘
                     │ Context + Query
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 OpenAI LLM                              │
│                 (GPT-5.1)                               │
│        Generates natural language answer               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Response to User                          │
│  "A tabela 'customers' em production.public             │
│   contém dados de clientes..."                         │
└─────────────────────────────────────────────────────────┘
```

### Componentes

1. **TableMetadata**: Dataclass para metadados estruturados
2. **Embedding Generation**: OpenAI text-embedding-3-small
3. **Vector Database**: ChromaDB com persistência local
4. **Retrieval**: Busca por similaridade de cosseno
5. **Generation**: GPT-5.1 para respostas contextualizadas

## 🔗 Integração com Apache Atlas

### Exportar do Atlas

```python
from apache_atlas.client.base_client import AtlasClient

# Conecta ao Atlas
client = AtlasClient(
    'http://atlas-host:21000',
    ('admin', 'admin')
)

# Busca tabelas
entities = client.search_entities('hive_table')

# Converte e indexa
from examples.atlas_integration import extract_metadata_from_atlas_entity

for entity in entities:
    table = extract_metadata_from_atlas_entity(entity)
    agent.index_table(table)
```

### Via REST API

```bash
curl -u admin:admin \
  http://atlas-host:21000/api/atlas/v2/search/basic \
  -d '{"typeName": "hive_table"}' \
  -H 'Content-Type: application/json' \
  > atlas_export.json
```

## 🔗 Integração com Data Lineage Agent

```python
from data_lineage_agent import DataLineageAgent
from examples.lineage_integration import convert_lineage_assets_to_metadata

# 1. Analisa linhagem
lineage_agent = DataLineageAgent()
lineage_agent.analyze_pipeline(["pipeline.sql", "etl.py"])

# 2. Converte para metadados RAG
tables = convert_lineage_assets_to_metadata(lineage_agent)

# 3. Indexa com contexto de linhagem
rag_agent.index_tables_batch(tables)

# 4. Busca com contexto de dependências
results = rag_agent.search("tabelas críticas com alto impacto")
```

## 📖 API Reference

### DataDiscoveryRAGAgent

#### `__init__(collection_name, persist_directory, embedding_model, llm_model)`

Inicializa o agente.

**Parâmetros**:
- `collection_name` (str): Nome da coleção ChromaDB
- `persist_directory` (str): Diretório de persistência
- `embedding_model` (str): Modelo OpenAI para embeddings
- `llm_model` (str): Modelo OpenAI para geração

#### `index_table(table, force_update=False)`

Indexa uma tabela no banco vetorizado.

**Parâmetros**:
- `table` (TableMetadata): Metadados da tabela
- `force_update` (bool): Atualiza se já existir

#### `index_tables_batch(tables, force_update=False)`

Indexa múltiplas tabelas em batch.

**Parâmetros**:
- `tables` (List[TableMetadata]): Lista de metadados
- `force_update` (bool): Atualiza se já existirem

#### `search(query, n_results=5, filter_metadata=None)`

Busca tabelas usando query natural.

**Parâmetros**:
- `query` (str): Query em linguagem natural
- `n_results` (int): Número de resultados
- `filter_metadata` (Dict): Filtros de metadados

**Retorna**: `List[SearchResult]`

#### `ask(question, n_context=3, include_reasoning=True)`

Responde pergunta usando RAG.

**Parâmetros**:
- `question` (str): Pergunta em linguagem natural
- `n_context` (int): Número de tabelas como contexto
- `include_reasoning` (bool): Inclui raciocínio do LLM

**Retorna**: `Dict` com answer, relevant_tables, confidence

#### `get_statistics()`

Retorna estatísticas do índice.

**Retorna**: `Dict` com total_tables, databases, formats

#### `export_metadata(output_file)`

Exporta metadados para JSON.

#### `import_from_json(json_file)`

Importa metadados de JSON.

#### `reset_index()`

Reseta o índice (USE COM CUIDADO!).

## ❓ FAQ

### Como funciona a busca semântica?

A busca converte sua query e os metadados em vetores (embeddings) usando o modelo da OpenAI. Então, usa similaridade de cosseno para encontrar as tabelas mais relevantes.

### Preciso da OpenAI API?

Sim, atualmente o agente usa OpenAI para embeddings e geração de respostas. Você pode adaptar para usar modelos locais (sentence-transformers) se necessário.

### Quanto custa usar o agente?

Custos típicos:
- Embedding: ~$0.00002 por 1000 tokens (~$0.02 por 1000 tabelas)
- LLM (perguntas): ~$0.15 por 1M tokens de input

Para 1000 tabelas + 100 perguntas/dia: ~$0.50/dia

### Como lidar com milhares de tabelas?

- Use indexação em batch
- Configure `n_results` apropriadamente
- Use filtros de metadados para refinar buscas
- ChromaDB é otimizado para milhões de vetores

### Como atualizar metadados?

```python
# Re-indexa com force_update=True
agent.index_table(updated_table, force_update=True)
```

### Como integrar com meu catálogo existente?

Adapte a função `extract_metadata_from_atlas_entity` para seu catálogo (AWS Glue, Databricks Unity Catalog, etc).

### Como melhorar a qualidade das respostas?

1. **Enriqueça metadados**: Adicione descrições detalhadas
2. **Use tags**: Facilita filtragem e contexto
3. **Ajuste n_context**: Mais contexto = respostas melhores
4. **Atualize embeddings**: Re-indexe quando mudar descrições

## 📝 Próximos Passos

- [ ] Suporte a modelos locais (sentence-transformers)
- [ ] Interface web para descoberta interativa
- [ ] Integração com AWS Glue Catalog
- [ ] Integração com Databricks Unity Catalog
- [ ] Cache de embeddings para reduzir custos
- [ ] Suporte a busca híbrida (keyword + semantic)
- [ ] Fine-tuning de embeddings para domínio específico

## 📄 Licença

Este projeto é parte do Data Governance AI Agents Kit.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, abra issues e pull requests.

## 📧 Suporte

Para dúvidas e suporte, abra uma issue no repositório.

---

**Desenvolvido com ❤️ usando Claude AI**
