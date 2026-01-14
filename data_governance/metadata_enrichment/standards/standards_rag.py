# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Standards RAG - Retrieval Augmented Generation for Architecture Standards

Indexes and retrieves relevant standards, naming conventions, and
architectural guidelines to inform metadata generation.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from rag_discovery.providers.base import EmbeddingProvider, VectorStoreProvider
except ImportError:
    from data_governance.rag_discovery.providers.base import EmbeddingProvider, VectorStoreProvider


@dataclass
class StandardDocument:
    """A standards/normative document"""
    id: str
    title: str
    content: str
    category: str  # naming_convention, architecture, data_classification, governance
    source: str = ""  # file path or URL
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert to text for embedding"""
        parts = [
            f"Título: {self.title}",
            f"Categoria: {self.category}",
            f"Conteúdo: {self.content}"
        ]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "version": self.version,
            "tags": self.tags,
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardDocument":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            source=data.get("source", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            metadata={k: v for k, v in data.items()
                     if k not in ["id", "title", "content", "category", "source", "version", "tags"]}
        )


@dataclass
class StandardsSearchResult:
    """Result from standards search"""
    document: StandardDocument
    score: float
    relevance: str  # high, medium, low


class StandardsRAG:
    """
    RAG system for architecture standards and naming conventions

    Features:
    - Index standards documents (JSON, Markdown, plain text)
    - Semantic search for relevant standards
    - Category-based filtering
    - Version tracking

    Usage:
        rag = StandardsRAG(
            embedding_provider=SentenceTransformerEmbeddings(),
            vector_store=ChromaStore(collection_name="standards")
        )
        rag.index_standards(standards_list)
        results = rag.search("convenção de nomenclatura para tabelas de cliente")
    """

    # Default categories for standards
    CATEGORIES = {
        "naming_convention": "Convenções de nomenclatura para tabelas, colunas e assets",
        "architecture": "Padrões de arquitetura de dados e pipelines",
        "data_classification": "Classificação de dados sensíveis e PII",
        "governance": "Políticas de governança e compliance",
        "quality": "Padrões de qualidade de dados",
        "security": "Padrões de segurança e acesso",
        "glossary": "Glossário de termos de negócio"
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        persist_dir: str = "./standards_index"
    ):
        """
        Initialize Standards RAG

        Args:
            embedding_provider: Provider for embeddings
            vector_store: Vector store for semantic search
            persist_dir: Directory to persist index metadata
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Track indexed documents
        self._indexed_docs: Dict[str, StandardDocument] = {}
        self._load_index_metadata()

        print(f"StandardsRAG initialized with {len(self._indexed_docs)} documents")

    def _load_index_metadata(self) -> None:
        """Load index metadata from disk"""
        metadata_file = self.persist_dir / "index_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for doc_data in data.get("documents", []):
                    doc = StandardDocument.from_dict(doc_data)
                    self._indexed_docs[doc.id] = doc

    def _save_index_metadata(self) -> None:
        """Save index metadata to disk"""
        metadata_file = self.persist_dir / "index_metadata.json"
        data = {
            "documents": [doc.to_dict() for doc in self._indexed_docs.values()]
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content hash"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def index_standards(
        self,
        standards: List[StandardDocument],
        show_progress: bool = True
    ) -> int:
        """
        Index standards documents

        Args:
            standards: List of StandardDocument objects
            show_progress: Show progress during indexing

        Returns:
            Number of documents indexed
        """
        if not standards:
            return 0

        # Prepare for batch embedding
        ids = []
        texts = []
        metadatas = []

        for doc in standards:
            # Generate ID if not provided
            if not doc.id:
                doc.id = self._generate_id(doc.content)

            # Skip if already indexed
            if doc.id in self._indexed_docs:
                continue

            ids.append(doc.id)
            texts.append(doc.to_text())
            metadatas.append(doc.to_dict())
            self._indexed_docs[doc.id] = doc

        if not ids:
            print("All documents already indexed")
            return 0

        # Generate embeddings
        if show_progress:
            print(f"Generating embeddings for {len(ids)} standards...")

        embeddings_results = self.embedding_provider.embed_batch(texts)
        embeddings = [r.vector for r in embeddings_results]

        # Add to vector store
        self.vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # Persist metadata
        self._save_index_metadata()

        if show_progress:
            print(f"Indexed {len(ids)} standards documents")

        return len(ids)

    def index_from_json(self, json_path: str) -> int:
        """
        Index standards from JSON file

        Expected format:
        [
            {
                "title": "Convenção de Nomenclatura de Tabelas",
                "content": "Tabelas devem seguir o padrão...",
                "category": "naming_convention",
                "tags": ["nomenclatura", "tabelas"]
            }
        ]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        standards = [StandardDocument.from_dict(item) for item in data]
        return self.index_standards(standards)

    def index_from_markdown(self, md_path: str, category: str = "general") -> int:
        """
        Index standards from Markdown file

        Splits by headers (##) to create separate documents
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by ## headers
        sections = content.split("\n## ")
        standards = []

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            lines = section.strip().split("\n")
            title = lines[0].replace("#", "").strip() if lines else f"Section {i}"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section

            if body:
                standards.append(StandardDocument(
                    id="",
                    title=title,
                    content=body,
                    category=category,
                    source=md_path
                ))

        return self.index_standards(standards)

    def index_from_directory(
        self,
        directory: str,
        extensions: List[str] = [".json", ".md", ".txt"]
    ) -> int:
        """
        Index all standards files from a directory
        """
        dir_path = Path(directory)
        total_indexed = 0

        for ext in extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                try:
                    if ext == ".json":
                        count = self.index_from_json(str(file_path))
                    elif ext == ".md":
                        # Infer category from directory name
                        category = file_path.parent.name
                        if category not in self.CATEGORIES:
                            category = "general"
                        count = self.index_from_markdown(str(file_path), category)
                    else:
                        # Plain text - single document
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc = StandardDocument(
                            id="",
                            title=file_path.stem,
                            content=content,
                            category="general",
                            source=str(file_path)
                        )
                        count = self.index_standards([doc])

                    total_indexed += count
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")

        return total_indexed

    def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.3
    ) -> List[StandardsSearchResult]:
        """
        Search for relevant standards

        Args:
            query: Search query
            n_results: Number of results to return
            category: Filter by category
            min_score: Minimum relevance score

        Returns:
            List of StandardsSearchResult
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query).vector

        # Build filter
        filter_metadata = None
        if category:
            filter_metadata = {"category": category}

        # Search
        results = self.vector_store.search(
            embedding=query_embedding,
            n_results=n_results * 2,  # Get more for filtering
            filter_metadata=filter_metadata
        )

        # Process results
        search_results = []

        for i, doc_id in enumerate(results.ids[0] if results.ids else []):
            if doc_id not in self._indexed_docs:
                continue

            # Calculate score (convert distance to similarity)
            distance = results.distances[0][i] if results.distances else 1.0
            score = 1.0 / (1.0 + distance)  # Convert to 0-1 score

            if score < min_score:
                continue

            # Determine relevance level
            if score >= 0.7:
                relevance = "high"
            elif score >= 0.5:
                relevance = "medium"
            else:
                relevance = "low"

            search_results.append(StandardsSearchResult(
                document=self._indexed_docs[doc_id],
                score=score,
                relevance=relevance
            ))

        # Sort by score and limit
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:n_results]

    def get_context_for_enrichment(
        self,
        table_name: str,
        column_names: List[str],
        sample_data: Optional[Dict[str, List[Any]]] = None,
        n_results: int = 3
    ) -> str:
        """
        Get relevant standards context for metadata enrichment

        Args:
            table_name: Name of the table
            column_names: List of column names
            sample_data: Optional sample data per column
            n_results: Number of standards to retrieve per query

        Returns:
            Formatted context string with relevant standards
        """
        context_parts = []

        # Search for naming conventions
        naming_query = f"convenção nomenclatura tabela {table_name}"
        naming_results = self.search(naming_query, n_results=n_results, category="naming_convention")

        if naming_results:
            context_parts.append("### Convenções de Nomenclatura Relevantes:")
            for r in naming_results:
                context_parts.append(f"- {r.document.title}: {r.document.content[:500]}...")

        # Search for data classification
        classification_query = f"classificação dados colunas {' '.join(column_names[:5])}"
        class_results = self.search(classification_query, n_results=n_results, category="data_classification")

        if class_results:
            context_parts.append("\n### Padrões de Classificação de Dados:")
            for r in class_results:
                context_parts.append(f"- {r.document.title}: {r.document.content[:500]}...")

        # Search glossary for business terms
        glossary_query = f"glossário termos {table_name} {' '.join(column_names[:3])}"
        glossary_results = self.search(glossary_query, n_results=n_results, category="glossary")

        if glossary_results:
            context_parts.append("\n### Termos do Glossário de Negócios:")
            for r in glossary_results:
                context_parts.append(f"- {r.document.title}: {r.document.content[:500]}...")

        return "\n".join(context_parts) if context_parts else "Nenhum padrão relevante encontrado."

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        categories_count = {}
        for doc in self._indexed_docs.values():
            cat = doc.category
            categories_count[cat] = categories_count.get(cat, 0) + 1

        return {
            "total_documents": len(self._indexed_docs),
            "categories": categories_count,
            "vector_store_count": self.vector_store.count()
        }

    def list_categories(self) -> Dict[str, str]:
        """List available categories with descriptions"""
        return self.CATEGORIES.copy()

    def clear(self) -> None:
        """Clear all indexed standards"""
        self.vector_store.reset()
        self._indexed_docs.clear()
        self._save_index_metadata()
        print("Standards index cleared")
