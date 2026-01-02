# Metadata Enrichment Agent

Agente de IA para geração automática de descrições, tags e classificações de metadados para tabelas de dados.

## Visão Geral

O **Metadata Enrichment Agent** analisa tabelas de dados e gera automaticamente:

- **Descrições** detalhadas para tabelas e colunas (PT-BR e EN)
- **Tags** relevantes para busca e organização
- **Classificação de dados** (public, internal, confidential, restricted)
- **Detecção de PII** (dados pessoais identificáveis)
- **Sugestão de domínio** (customer, sales, finance, etc.)
- **Sugestão de proprietário** (área/time responsável)

### Características

- 🔍 **RAG sobre Normativos**: Usa padrões de arquitetura e nomenclatura como contexto
- 📊 **Data Sampling**: Coleta amostras de dados para inferir a natureza das informações
- 🏷️ **Classificação Automática**: Detecta PII, tipos semânticos e níveis de sensibilidade
- 🌐 **Multi-fonte**: Suporta Parquet, CSV, SQL e Delta Lake
- 📝 **Exportação**: JSON, Markdown e HTML

## Instalação

```bash
# A partir do diretório raiz do projeto
pip install -r metadata_enrichment/requirements.txt
```

## Uso Rápido

### Via Python

```python
from metadata_enrichment.agent import MetadataEnrichmentAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

# Inicializar providers
embedding_provider = SentenceTransformerEmbeddings()
llm_provider = OpenAILLM(model="gpt-4o-mini")
vector_store = ChromaStore(collection_name="standards")

# Criar agente
agent = MetadataEnrichmentAgent(
    embedding_provider=embedding_provider,
    llm_provider=llm_provider,
    vector_store=vector_store
)

# Indexar normativos (opcional, mas recomendado)
agent.index_standards_from_json("./examples/sample_standards.json")

# Enriquecer metadados de um arquivo CSV
result = agent.enrich_from_csv("./data/customers.csv")

# Exportar resultados
print(result.to_markdown())
```

### Via Streamlit UI

```bash
# Interface standalone
streamlit run metadata_enrichment/streamlit_app.py

# Ou via interface unificada (inclui todos os agentes)
streamlit run app.py
```

## Arquitetura

```
metadata_enrichment/
├── agent.py                    # Agente principal
├── standards/
│   └── standards_rag.py        # RAG para normativos
├── sampling/
│   └── data_sampler.py         # Conectores de sampling
├── providers/
│   └── __init__.py             # Reusa providers do rag_discovery
├── examples/
│   ├── basic_usage.py          # Exemplo de uso
│   └── sample_standards.json   # Normativos de exemplo
├── streamlit_app.py            # UI Streamlit
├── requirements.txt
└── README.md
```

## RAG de Normativos

O agente usa um sistema RAG para buscar padrões relevantes ao gerar metadados:

### Categorias de Normativos

| Categoria | Descrição |
|-----------|-----------|
| `naming_convention` | Convenções de nomenclatura |
| `data_classification` | Classificação de dados (PII, LGPD) |
| `glossary` | Glossário de termos de negócio |
| `architecture` | Padrões de arquitetura |
| `governance` | Políticas de governança |
| `quality` | Padrões de qualidade |
| `security` | Padrões de segurança |

### Formato de Normativos (JSON)

```json
[
  {
    "title": "Convenção de Nomenclatura de Tabelas",
    "content": "Tabelas devem seguir o padrão...",
    "category": "naming_convention",
    "tags": ["nomenclatura", "tabelas"]
  }
]
```

## Data Sampling

O agente coleta amostras de dados para inferir tipos semânticos e detectar PII:

### Fontes Suportadas

| Fonte | Classe | Exemplo |
|-------|--------|---------|
| CSV | `CSVSampler` | `agent.enrich_from_csv("data.csv")` |
| Parquet | `ParquetSampler` | `agent.enrich_from_parquet("data.parquet")` |
| SQL | `SQLSampler` | `agent.enrich_from_sql("table", connection_string="...")` |
| Delta Lake | `DeltaSampler` | `agent.enrich_from_delta("/path/to/delta")` |

### Padrões Detectados

O sampler detecta automaticamente:

- **Email**: `user@domain.com`
- **CPF**: `123.456.789-00`
- **CNPJ**: `12.345.678/0001-90`
- **Telefone**: `(11) 98765-4321`
- **UUID**: `550e8400-e29b-41d4-a716-446655440000`
- **Datas**: `2024-01-15`, `15/01/2024`
- **Moeda**: `R$ 1.234,56`
- **IP**: `192.168.1.1`
- **URL**: `https://example.com`
- **Cartão de crédito**: `4111 1111 1111 1111`

## Resultado do Enriquecimento

### EnrichmentResult

```python
@dataclass
class EnrichmentResult:
    table_name: str
    description: str           # Descrição em PT-BR
    description_en: str        # Descrição em inglês
    business_name: str         # Nome amigável
    domain: str                # Domínio de dados
    tags: List[str]
    classification: str        # public, internal, confidential, restricted
    owner_suggestion: str
    columns: List[ColumnEnrichment]
    has_pii: bool
    pii_columns: List[str]
    confidence: float
```

### Exportação

```python
# JSON
result.to_json()

# Markdown
result.to_markdown()

# Dict
result.to_dict()
```

## Processamento em Lote

```python
sources = [
    {"type": "csv", "path": "./customers.csv"},
    {"type": "parquet", "path": "./orders.parquet"},
    {"type": "sql", "table_name": "products", "connection_string": "postgresql://..."}
]

results = agent.enrich_batch(sources, output_dir="./catalog_output")

# Exportar catálogo completo
agent.export_catalog(results, "./data_catalog.json", format="json")
agent.export_catalog(results, "./data_catalog.md", format="markdown")
agent.export_catalog(results, "./data_catalog.html", format="html")
```

## Configuração

### Variáveis de Ambiente

```bash
# Obrigatório para LLM
export OPENAI_API_KEY="sk-..."

# Opcional: endpoint customizado
export OPENAI_API_URL="https://api.openai.com/v1"
```

### Modelos Recomendados

| Uso | Modelo | Custo |
|-----|--------|-------|
| Produção | `gpt-4o` | Alto |
| Desenvolvimento | `gpt-4o-mini` | Baixo |
| Embeddings | `all-MiniLM-L6-v2` | Local (gratuito) |

## Integração com Outros Agentes

O Metadata Enrichment Agent se integra com os outros agentes do kit:

```python
# Descobrir tabelas com RAG Discovery
from rag_discovery.agent import DataDiscoveryAgent
discovery_agent = DataDiscoveryAgent(...)
tables = discovery_agent.search("customer data")

# Enriquecer metadados
for table in tables:
    result = enrichment_agent.enrich_from_sql(table.name, connection_string)

# Analisar linhagem com Lineage Agent
from lineage.data_lineage_agent import DataLineageAgent
lineage_agent = DataLineageAgent()
# ... usar resultados enriquecidos para documentar linhagem
```

## Exemplos

Veja o diretório `examples/` para exemplos completos:

- `basic_usage.py`: Uso básico do agente
- `sample_standards.json`: Normativos de exemplo

## Contribuindo

1. Siga os padrões existentes (Provider Pattern)
2. Adicione testes para novas funcionalidades
3. Documente novas categorias de normativos
