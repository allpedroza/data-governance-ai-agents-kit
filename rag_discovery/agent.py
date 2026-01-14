"""
Data Discovery Agent - Main class for hybrid RAG-based data discovery
Combines semantic search, lexical matching, and table validation
"""

import time
import json
import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import EmbeddingProvider, LLMProvider, VectorStoreProvider
from .retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig, RetrievalResult
from .validation.table_validator import TableValidator
from .utils.logger import StructuredLogger

if TYPE_CHECKING:
    pass


@dataclass
class TableMetadata:
    """Structured table metadata"""
    name: str
    database: str = ""
    schema: str = ""
    description: str = ""
    columns: List[Dict[str, Any]] = field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    location: str = ""
    format: str = ""
    partition_keys: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert to text representation for embedding"""
        parts = []

        # Full name
        full_name = f"{self.database}.{self.schema}.{self.name}" if self.database and self.schema else self.name
        parts.append(f"Tabela: {full_name}")

        if self.description:
            parts.append(f"Descrição: {self.description}")

        if self.columns:
            parts.append("Colunas:")
            for col in self.columns:
                col_info = f"  - {col.get('name', 'unknown')} ({col.get('type', 'unknown')})"
                if col.get('description'):
                    col_info += f": {col['description']}"
                parts.append(col_info)

        if self.owner:
            parts.append(f"Proprietário: {self.owner}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.format:
            parts.append(f"Formato: {self.format}")

        if self.partition_keys:
            parts.append(f"Particionado por: {', '.join(self.partition_keys)}")

        if self.location:
            parts.append(f"Localização: {self.location}")

        if self.row_count:
            parts.append(f"Linhas: {self.row_count:,}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'database': self.database,
            'schema': self.schema,
            'description': self.description,
            'columns': self.columns,
            'row_count': self.row_count,
            'size_bytes': self.size_bytes,
            'owner': self.owner,
            'tags': self.tags,
            'location': self.location,
            'format': self.format,
            'partition_keys': self.partition_keys,
            'asset_type': 'table'
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableMetadata":
        """Create from dictionary"""
        return cls(
            name=data.get('name', ''),
            database=data.get('database', ''),
            schema=data.get('schema', ''),
            description=data.get('description', ''),
            columns=data.get('columns', []),
            row_count=data.get('row_count'),
            size_bytes=data.get('size_bytes'),
            owner=data.get('owner', ''),
            tags=data.get('tags', []),
            location=data.get('location', ''),
            format=data.get('format', ''),
            partition_keys=data.get('partition_keys', [])
        )


@dataclass
class ModelMetadata:
    """Structured model metadata for discovery."""
    name: str
    description: str = ""
    owner: str = ""
    use_case: str = ""
    endpoints: List[str] = field(default_factory=list)
    input_datasets: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = ""

    def to_text(self) -> str:
        parts = [f"Modelo: {self.name}"]

        if self.description:
            parts.append(f"Descrição: {self.description}")
        if self.use_case:
            parts.append(f"Caso de uso: {self.use_case}")
        if self.owner:
            parts.append(f"Owner: {self.owner}")
        if self.endpoints:
            parts.append(f"Endpoints: {', '.join(self.endpoints)}")
        if self.input_datasets:
            parts.append(f"Dados de entrada: {', '.join(self.input_datasets)}")
        if self.features:
            parts.append(f"Features: {', '.join(self.features)}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if self.version:
            parts.append(f"Versão: {self.version}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "use_case": self.use_case,
            "endpoints": self.endpoints,
            "input_datasets": self.input_datasets,
            "features": self.features,
            "tags": self.tags,
            "version": self.version,
            "asset_type": "model"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            use_case=data.get("use_case", ""),
            endpoints=data.get("endpoints", []),
            input_datasets=data.get("input_datasets", []),
            features=data.get("features", []),
            tags=data.get("tags", []),
            version=data.get("version", "")
        )


@dataclass
class DiscoveryResult:
    """Result of a discovery query"""
    query: str
    answer: str
    validated_tables: List[Dict[str, Any]]
    invalid_tables: List[Dict[str, Any]]
    validated_models: List[Dict[str, Any]] = field(default_factory=list)
    invalid_models: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_results: List[RetrievalResult]
    confidence: float
    latency_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataDiscoveryAgent:
    """
    Hybrid Data Discovery Agent

    Features:
    - Dartboard Ranking (semantic + lexical + importance)
    - Table validation against catalog
    - Pluggable providers (no vendor lock-in)
    - Structured logging
    - Diversity filtering

    Architecture:
    1. Query → Hybrid Retrieval (Dartboard) → Relevant chunks
    2. Extract table names from chunks
    3. Validate tables against catalog
    4. Generate response with LLM

    Usage:
        agent = DataDiscoveryAgent(
            embedding_provider=SentenceTransformerEmbeddings(),
            llm_provider=OpenAILLM(),
            vector_store=ChromaStore(),
            catalog_source="./catalog.txt"
        )
        agent.index_metadata(tables)
        result = agent.discover("Where are customer data?")
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        vector_store: VectorStoreProvider,
        catalog_source: Optional[str] = None,
        log_dir: str = "./logs",
        # Dartboard weights
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        diversity_threshold: float = 0.95
    ):
        """
        Initialize Data Discovery Agent

        Args:
            embedding_provider: Provider for embeddings (local or API)
            llm_provider: Provider for LLM responses
            vector_store: Vector database provider
            catalog_source: Path to table catalog for validation
            log_dir: Directory for logs
            alpha: Semantic search weight (0-1)
            beta: Lexical search weight (0-1)
            gamma: Importance weight (0-1)
            diversity_threshold: Similarity threshold for diversity filter
        """
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.vector_store = vector_store

        # Hybrid retriever config
        retriever_config = HybridRetrieverConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            diversity_threshold=diversity_threshold
        )

        # Initialize retriever
        self.retriever = HybridRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            config=retriever_config
        )

        # Table validator
        self.validator = None
        if catalog_source:
            self.validator = TableValidator(catalog_source)

        # Logger
        self.logger = StructuredLogger(log_dir=log_dir)

        # Print initialization info
        print("=" * 60)
        print("Data Discovery Agent Initialized")
        print("=" * 60)
        print(f"  Embedding: {type(embedding_provider).__name__}")
        print(f"  LLM: {type(llm_provider).__name__}")
        print(f"  VectorStore: {type(vector_store).__name__}")
        print(f"  Catalog: {self.validator.catalog_size if self.validator else 'Not loaded'} tables")
        print(f"  Weights: α={alpha}, β={beta}, γ={gamma}")
        print("=" * 60)

    def index_metadata(
        self,
        tables: List[TableMetadata],
        show_progress: bool = True
    ) -> int:
        """
        Index table metadata

        Args:
            tables: List of TableMetadata objects
            show_progress: Show progress during indexing

        Returns:
            Number of tables indexed
        """
        if not tables:
            return 0

        documents = []
        for table in tables:
            full_name = f"{table.database}.{table.schema}.{table.name}" if table.database else table.name
            doc = {
                "id": full_name,
                "text": table.to_text(),
                "metadata": table.to_dict()
            }
            documents.append(doc)

        count = self.retriever.index_documents(documents, show_progress=show_progress)

        self.logger.info(f"Indexed {count} tables")

        return count

    def index_models(
        self,
        models: List[ModelMetadata],
        show_progress: bool = True
    ) -> int:
        """
        Index model metadata for discovery.

        Args:
            models: List of ModelMetadata objects
            show_progress: Show progress during indexing

        Returns:
            Number of models indexed
        """
        if not models:
            return 0

        documents = []
        for model in models:
            doc = {
                "id": f"model::{model.name}",
                "text": model.to_text(),
                "metadata": model.to_dict()
            }
            documents.append(doc)

        count = self.retriever.index_documents(documents, show_progress=show_progress)

        self.logger.info(f"Indexed {count} models")
        return count

    def index_models_from_json(
        self,
        json_path: str,
        show_progress: bool = True
    ) -> int:
        """
        Index model metadata from JSON file.

        Args:
            json_path: Path to JSON file with model metadata

        Returns:
            Number of models indexed
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        models = [ModelMetadata.from_dict(item) for item in data]
        return self.index_models(models, show_progress=show_progress)

    def index_model_catalog(self, provider: Any, show_progress: bool = True) -> int:
        """
        Index model metadata using a provider/connector.

        Args:
            provider: Provider with load_models() returning a list of dicts
            show_progress: Show progress during indexing

        Returns:
            Number of models indexed
        """
        model_cards = provider.load_models()
        models = [ModelMetadata.from_dict(item) for item in model_cards]
        return self.index_models(models, show_progress=show_progress)

    def index_from_json(
        self,
        json_path: str,
        show_progress: bool = True
    ) -> int:
        """
        Index metadata from JSON file

        Args:
            json_path: Path to JSON file with table metadata

        Returns:
            Number of tables indexed
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tables = []
        for item in data:
            tables.append(TableMetadata.from_dict(item))

        return self.index_metadata(tables, show_progress=show_progress)

    def _extract_tables_from_text(self, text: str) -> List[str]:
        """Extract table names from text using regex"""
        # BigQuery format: project.dataset.table
        pattern = r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)'
        return re.findall(pattern, text)

    def _extract_tables_from_results(
        self,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """Extract table info from retrieval results"""
        tables = []
        seen_tables = set()

        for r in results:
            # Get table name from metadata
            table_name = r.metadata.get("name", "")
            if r.metadata.get("database"):
                full_name = f"{r.metadata.get('database')}.{r.metadata.get('schema', '')}.{table_name}"
            else:
                full_name = table_name

            # Also extract from text
            text_tables = self._extract_tables_from_text(r.text)

            # Use metadata table or first extracted
            if full_name and full_name not in seen_tables:
                seen_tables.add(full_name)
                tables.append({
                    "table_name": full_name,
                    "description": r.metadata.get("description", ""),
                    "relevance_score": r.combined_score,
                    "semantic_score": r.semantic_score,
                    "lexical_score": r.lexical_score,
                    "confidence": "high" if r.combined_score > 0.7 else "medium" if r.combined_score > 0.4 else "low"
                })
            elif text_tables:
                for t in text_tables:
                    if t not in seen_tables:
                        seen_tables.add(t)
                        tables.append({
                            "table_name": t,
                            "description": "",
                            "relevance_score": r.combined_score,
                            "confidence": "medium"
                        })

        return tables

    def _extract_models_from_results(
        self,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """Extract model info from retrieval results."""
        models = []
        seen_models = set()

        for r in results:
            if r.metadata.get("asset_type") != "model":
                continue

            model_name = r.metadata.get("name", "")
            if not model_name or model_name in seen_models:
                continue

            seen_models.add(model_name)
            models.append({
                "model_name": model_name,
                "description": r.metadata.get("description", ""),
                "owner": r.metadata.get("owner", ""),
                "use_case": r.metadata.get("use_case", ""),
                "endpoints": r.metadata.get("endpoints", []),
                "input_datasets": r.metadata.get("input_datasets", []),
                "features": r.metadata.get("features", []),
                "relevance_score": r.combined_score,
                "semantic_score": r.semantic_score,
                "lexical_score": r.lexical_score,
                "confidence": "high" if r.combined_score > 0.7 else "medium" if r.combined_score > 0.4 else "low"
            })

        return models

    def _generate_response(
        self,
        query: str,
        validated_tables: List[Dict[str, Any]],
        invalid_tables: List[Dict[str, Any]],
        validated_models: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate response using LLM"""
        validated_models = validated_models or []
        if not validated_tables and not validated_models:
            return "Não foram encontradas tabelas ou modelos relevantes para sua consulta no catálogo disponível."

        # Build context from validated tables
        context_parts = []
        for t in validated_tables:
            context_parts.append(f"- {t['table_name']}: {t.get('description', 'Sem descrição')}")

        context = "\n".join(context_parts)

        model_context = ""
        if validated_models:
            model_lines = []
            for model in validated_models:
                model_lines.append(
                    f"- {model.get('model_name')}: {model.get('description', 'Sem descrição')} "
                    f"(owner: {model.get('owner', 'N/A')})"
                )
                if model.get("input_datasets"):
                    model_lines.append(
                        f"  Dados consumidos: {', '.join(model.get('input_datasets', []))}"
                    )
            model_context = "\n".join(model_lines)

        system_prompt = """Você é um Assistente de Descoberta de Dados especializado em data lakes.
Sua tarefa é recomendar tabelas e modelos baseado na consulta do usuário.

Formate cada recomendação assim:
- [nome_completo_da_tabela ou modelo]
  Descrição: Breve descrição
  Relevância: Por que este item é relevante para a consulta

Seja conciso e informativo. Use português brasileiro.
Inclua apenas tabelas ou modelos que foram validados no contexto fornecido."""

        prompt = f"""Consulta do usuário: {query}

Tabelas validadas encontradas:
{context}

Modelos validados encontrados:
{model_context if model_context else 'Nenhum modelo relevante encontrado.'}

Gere uma resposta recomendando as tabelas e/ou modelos mais relevantes."""

        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0
        )

        return response.content

    def discover(
        self,
        query: str,
        top_k: int = 5,
        validate: bool = True,
        user: Optional[str] = None
    ) -> DiscoveryResult:
        """
        Discover relevant tables for a query

        Args:
            query: Natural language query
            top_k: Number of results to return
            validate: Whether to validate against catalog
            user: User identifier for logging

        Returns:
            DiscoveryResult with answer and table details
        """
        start_time = time.time()

        # 1. Hybrid retrieval
        retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=top_k * 2,  # Get more for validation filtering
            apply_diversity=True
        )

        # 2. Extract tables from results
        discovered_tables = self._extract_tables_from_results(retrieval_results)
        discovered_models = self._extract_models_from_results(retrieval_results)

        # 3. Validate tables
        if validate and self.validator:
            validated_tables, invalid_tables = self.validator.validate(discovered_tables)
        else:
            validated_tables = discovered_tables
            invalid_tables = []

        validated_models = discovered_models[:top_k]
        invalid_models: List[Dict[str, Any]] = []

        # Limit to top_k validated
        validated_tables = validated_tables[:top_k]

        # 4. Generate response
        answer = self._generate_response(query, validated_tables, invalid_tables, validated_models)

        # 5. Calculate confidence
        if retrieval_results:
            avg_score = sum(r.combined_score for r in retrieval_results) / len(retrieval_results)
            validation_rate = len(validated_tables) / len(discovered_tables) if discovered_tables else 0
            confidence = (avg_score * 0.7) + (validation_rate * 0.3)
        else:
            confidence = 0.0

        latency_ms = int((time.time() - start_time) * 1000)

        # 6. Log
        self.logger.log_query(
            query=query,
            discovered_tables=len(discovered_tables),
            validated_tables=len(validated_tables),
            invalid_tables=len(invalid_tables),
            latency_ms=latency_ms,
            model=self.llm_provider.model_name,
            embedding_model=self.embedding_provider.model_name,
            response=answer,
            user=user
        )

        return DiscoveryResult(
            query=query,
            answer=answer,
            validated_tables=validated_tables,
            invalid_tables=invalid_tables,
            validated_models=validated_models,
            invalid_models=invalid_models,
            retrieval_results=retrieval_results,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "embedding_model": self.embedding_provider.model_name,
                "llm_model": self.llm_provider.model_name,
                "retriever_stats": self.retriever.get_statistics()
            }
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Simple search without LLM generation

        Args:
            query: Search query
            n_results: Number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievalResult
        """
        return self.retriever.retrieve(
            query=query,
            top_k=n_results,
            filter_metadata=filter_metadata,
            apply_diversity=True
        )

    def ask(
        self,
        question: str,
        n_context: int = 3,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question about the data lake (RAG)

        Args:
            question: Question in natural language
            n_context: Number of tables to use as context
            user: User identifier

        Returns:
            Dictionary with answer and metadata
        """
        result = self.discover(
            query=question,
            top_k=n_context,
            validate=True,
            user=user
        )

        return {
            "question": question,
            "answer": result.answer,
            "relevant_tables": result.validated_tables,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "retriever": self.retriever.get_statistics(),
            "vector_store_count": self.vector_store.count(),
        }

        if self.validator:
            stats["catalog"] = self.validator.get_statistics()

        stats["logs"] = self.logger.get_session_stats()

        return stats

    def adjust_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> None:
        """Adjust Dartboard ranking weights"""
        self.retriever.adjust_weights(alpha, beta, gamma)

    def load_catalog(self, catalog_source: str) -> int:
        """Load or reload table catalog"""
        if self.validator is None:
            self.validator = TableValidator()

        count = self.validator.load_catalog(catalog_source)
        self.logger.info(f"Loaded catalog with {count} tables")
        return count
