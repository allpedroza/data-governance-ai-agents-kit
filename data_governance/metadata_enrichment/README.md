# Metadata Enrichment Agent

## Resumo
Agente de IA para gerar descrições, tags e classificação de tabelas/colunas usando RAG sobre padrões de arquitetura e amostras opcionais.

## Summary
AI agent to generate descriptions, tags, and classification for tables/columns using RAG over architecture standards and optional samples.

*English details available at the end of the file.*

## Características
- **Geração bilíngue** (PT/EN) de descrições e colunas.
- **Classificação automática** de sensibilidade (public, internal, confidential, restricted).
- **Detecção de PII** e aplicação de tags de compliance.
- **Suporte a múltiplos formatos**: CSV, Parquet, SQL, Delta Lake.
- **RAG** sobre padrões e dicionários corporativos.

## Uso Rápido
```python
from metadata_enrichment.agent import MetadataEnrichmentAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

agent = MetadataEnrichmentAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards"),
)

result = agent.enrich_from_csv("customers.csv")
print(result.description)
print(result.tags)
```

Carregue padrões corporativos com `index_standards_from_json` para contextualizar as respostas.

## Integrações
- Envie metadados para o **Data Discovery RAG Agent** para busca semântica.
- Combine com **Data Classification** para reforçar etiquetas de sensibilidade.
- Utilize **Data Quality** para incluir status de qualidade nas tags.

---

## Summary
AI agent to generate descriptions, tags, and classification for tables/columns using RAG over architecture standards and optional samples.

## Features
- **Bilingual generation** (PT/EN) for table and column descriptions.
- **Automatic sensitivity classification** (public, internal, confidential, restricted).
- **PII detection** with compliance tags.
- **Supports multiple formats**: CSV, Parquet, SQL, Delta Lake.
- **RAG** over standards and corporate dictionaries.

## Quickstart
```python
from metadata_enrichment.agent import MetadataEnrichmentAgent
from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
from rag_discovery.providers.llm import OpenAILLM
from rag_discovery.providers.vectorstore import ChromaStore

agent = MetadataEnrichmentAgent(
    embedding_provider=SentenceTransformerEmbeddings(),
    llm_provider=OpenAILLM(model="gpt-4o-mini"),
    vector_store=ChromaStore(collection_name="standards"),
)

result = agent.enrich_from_csv("customers.csv")
print(result.description)
print(result.tags)
```

Load corporate standards with `index_standards_from_json` to contextualize responses.

## Integrations
- Send metadata to the **Data Discovery RAG Agent** for semantic search.
- Combine with **Data Classification** to strengthen sensitivity labels.
- Use **Data Quality** to attach quality status in tags.
