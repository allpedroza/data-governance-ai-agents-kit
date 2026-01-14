# Data Discovery RAG Agent

## Resumo
Agente de IA para descoberta semântica de dados com RAG híbrido (semântico + lexical), validação de catálogo e ranking contextual.

## Summary
AI agent for semantic data discovery with hybrid RAG (semantic + lexical), catalog validation, and contextual ranking.

*English details available at the end of the file.*

## Características
- **Busca em linguagem natural** sobre tabelas e colunas.
- **Dartboard ranking** combinando semântica, correspondência lexical e importância.
- **Providers plugáveis** para embeddings, LLM e vector stores (ChromaDB, FAISS, etc.).
- **Validação de tabelas** contra catálogos existentes.
- **Integração** com Lineage, Enrichment e Classification para enriquecer respostas.
- **Discovery de modelos** com cards, endpoints e owners no mesmo índice.

## Uso Rápido
```python
from rag_discovery.agent import DataDiscoveryAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

agent = DataDiscoveryAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(),
    vector_store=ChromaStore(collection_name="catalog"),
)

agent.index_from_json("catalog.json")
result = agent.discover("Onde estão os dados de clientes?")
print(result.answer)
```

Combine com metadados enriquecidos para melhorar contexto e relevância.

## Discovery de Modelos
```python
from rag_discovery.agent import DataDiscoveryAgent, ModelMetadata
from rag_discovery.providers.model_catalog import ModelCatalogProvider

agent.index_models([
    ModelMetadata(
        name="churn_predictor",
        use_case="retencao",
        owner="ml-team",
        endpoints=["/v1/churn"],
        input_datasets=["warehouse.crm.customers"],
        features=["tenure", "plan_type"]
    )
])

provider = ModelCatalogProvider("./model_cards")
agent.index_model_catalog(provider)

result = agent.discover("Quais modelos usam dados de clientes?")
print(result.answer)
```

---

## Summary
AI agent for semantic data discovery with hybrid RAG (semantic + lexical), catalog validation, and contextual ranking.

## Features
- **Natural language search** across tables and columns.
- **Dartboard ranking** blending semantic, lexical, and importance scores.
- **Pluggable providers** for embeddings, LLMs, and vector stores (ChromaDB, FAISS, etc.).
- **Table validation** against existing catalogs.
- **Integration** with Lineage, Enrichment, and Classification to enrich answers.
- **Model discovery** with model cards, endpoints, and owners in the same index.

## Quickstart
```python
from rag_discovery.agent import DataDiscoveryAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

agent = DataDiscoveryAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(),
    vector_store=ChromaStore(collection_name="catalog"),
)

agent.index_from_json("catalog.json")
result = agent.discover("Onde estão os dados de clientes?")
print(result.answer)
```

Combine with enriched metadata to improve context and relevance.

## Model Discovery
```python
from rag_discovery.agent import DataDiscoveryAgent, ModelMetadata
from rag_discovery.providers.model_catalog import ModelCatalogProvider

agent.index_models([
    ModelMetadata(
        name="churn_predictor",
        use_case="retention",
        owner="ml-team",
        endpoints=["/v1/churn"],
        input_datasets=["warehouse.crm.customers"],
        features=["tenure", "plan_type"]
    )
])

provider = ModelCatalogProvider("./model_cards")
agent.index_model_catalog(provider)

result = agent.discover("Which models use customer data?")
print(result.answer)
```
