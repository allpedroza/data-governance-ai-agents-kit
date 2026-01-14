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
Metadata Enrichment Agent

Generates table and column descriptions, tags, and classifications
using RAG over architecture standards and data sampling.

Features:
- RAG over architecture standards and naming conventions
- Data sampling for inference
- Automatic PII detection
- Business glossary integration
- Multi-language support (PT-BR / EN)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from rag_discovery.providers.base import EmbeddingProvider, LLMProvider, VectorStoreProvider, LLMResponse
except ImportError:
    from data_governance.rag_discovery.providers.base import EmbeddingProvider, LLMProvider, VectorStoreProvider, LLMResponse
from .standards.standards_rag import StandardsRAG, StandardDocument
from .sampling.data_sampler import DataSampler, SampleResult, ColumnProfile


@dataclass
class ColumnEnrichment:
    """Enriched metadata for a column"""
    name: str
    original_type: str
    description: str
    description_en: str  # English version
    business_name: str  # Friendly name
    tags: List[str]
    classification: str  # public, internal, confidential, restricted
    semantic_type: Optional[str]  # pii, date, currency, id, etc.
    is_pii: bool
    is_nullable: bool
    sample_values: List[str]
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_type": self.original_type,
            "description": self.description,
            "description_en": self.description_en,
            "business_name": self.business_name,
            "tags": self.tags,
            "classification": self.classification,
            "semantic_type": self.semantic_type,
            "is_pii": self.is_pii,
            "is_nullable": self.is_nullable,
            "sample_values": self.sample_values,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class EnrichmentResult:
    """Result of metadata enrichment for a table"""
    table_name: str
    source: str
    description: str
    description_en: str
    business_name: str
    domain: str  # customer, sales, finance, etc.
    tags: List[str]
    classification: str
    owner_suggestion: str
    columns: List[ColumnEnrichment]
    row_count: int
    has_pii: bool
    pii_columns: List[str]
    confidence: float
    processing_time_ms: int
    enriched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    standards_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "source": self.source,
            "description": self.description,
            "description_en": self.description_en,
            "business_name": self.business_name,
            "domain": self.domain,
            "tags": self.tags,
            "classification": self.classification,
            "owner_suggestion": self.owner_suggestion,
            "columns": [col.to_dict() for col in self.columns],
            "row_count": self.row_count,
            "has_pii": self.has_pii,
            "pii_columns": self.pii_columns,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "enriched_at": self.enriched_at,
            "standards_used": self.standards_used,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown documentation"""
        lines = [
            f"# {self.business_name}",
            f"**Nome técnico:** `{self.table_name}`",
            f"",
            f"## Descrição",
            self.description,
            f"",
            f"**Domínio:** {self.domain}",
            f"**Classificação:** {self.classification}",
            f"**Tags:** {', '.join(self.tags)}",
            f"**Proprietário sugerido:** {self.owner_suggestion}",
            f"",
            f"## Colunas",
            "",
            "| Coluna | Tipo | Descrição | Classificação | Tags |",
            "|--------|------|-----------|---------------|------|"
        ]

        for col in self.columns:
            tags_str = ", ".join(col.tags[:3]) if col.tags else "-"
            pii_marker = " ⚠️" if col.is_pii else ""
            lines.append(
                f"| `{col.name}`{pii_marker} | {col.original_type} | {col.description[:50]}... | {col.classification} | {tags_str} |"
            )

        if self.pii_columns:
            lines.extend([
                "",
                "## ⚠️ Dados Sensíveis (PII)",
                f"Esta tabela contém {len(self.pii_columns)} coluna(s) com dados pessoais:",
                ""
            ])
            for pii_col in self.pii_columns:
                lines.append(f"- `{pii_col}`")

        lines.extend([
            "",
            "---",
            f"*Gerado automaticamente em {self.enriched_at} | Confiança: {self.confidence:.0%}*"
        ])

        return "\n".join(lines)


class MetadataEnrichmentAgent:
    """
    Metadata Enrichment Agent

    Generates descriptions, tags, and classifications for tables and columns
    using RAG over architecture standards and data sampling.

    Usage:
        agent = MetadataEnrichmentAgent(
            embedding_provider=SentenceTransformerEmbeddings(),
            llm_provider=OpenAILLM(),
            vector_store=ChromaStore(collection_name="standards")
        )

        # Index standards
        agent.index_standards_from_directory("./standards")

        # Enrich from file
        result = agent.enrich_from_parquet("./data/customers.parquet")

        # Enrich from SQL
        result = agent.enrich_from_sql("customers", connection_string="...")

        # Export
        print(result.to_markdown())
    """

    # Domain keywords for classification
    DOMAINS = {
        "customer": ["customer", "cliente", "client", "user", "usuario", "account", "conta"],
        "sales": ["sale", "venda", "order", "pedido", "transaction", "transacao", "payment", "pagamento"],
        "finance": ["finance", "financeiro", "invoice", "fatura", "billing", "cobranca", "revenue", "receita"],
        "product": ["product", "produto", "item", "sku", "catalog", "catalogo", "inventory", "estoque"],
        "marketing": ["campaign", "campanha", "lead", "marketing", "promotion", "promocao"],
        "hr": ["employee", "funcionario", "salary", "salario", "hiring", "contratacao"],
        "operations": ["operation", "operacao", "logistics", "logistica", "shipping", "envio"],
        "analytics": ["metric", "metrica", "kpi", "report", "relatorio", "dashboard", "analytics"]
    }

    # Classification levels
    CLASSIFICATIONS = {
        "public": "Dados públicos sem restrição de acesso",
        "internal": "Dados internos da organização",
        "confidential": "Dados confidenciais com acesso restrito",
        "restricted": "Dados altamente restritos (PII, financeiro sensível)"
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        vector_store: VectorStoreProvider,
        standards_persist_dir: str = "./standards_index",
        language: str = "pt-br"
    ):
        """
        Initialize Metadata Enrichment Agent

        Args:
            embedding_provider: Provider for embeddings
            llm_provider: Provider for LLM
            vector_store: Vector store for standards RAG
            standards_persist_dir: Directory to persist standards index
            language: Primary language (pt-br or en)
        """
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.language = language

        # Initialize Standards RAG
        self.standards_rag = StandardsRAG(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            persist_dir=standards_persist_dir
        )

        print("=" * 60)
        print("Metadata Enrichment Agent Initialized")
        print("=" * 60)
        print(f"  Embedding: {type(embedding_provider).__name__}")
        print(f"  LLM: {type(llm_provider).__name__}")
        print(f"  Language: {language}")
        print(f"  Standards indexed: {self.standards_rag.get_statistics()['total_documents']}")
        print("=" * 60)

    def index_standards(self, standards: List[StandardDocument]) -> int:
        """Index standards documents"""
        return self.standards_rag.index_standards(standards)

    def index_standards_from_json(self, json_path: str) -> int:
        """Index standards from JSON file"""
        return self.standards_rag.index_from_json(json_path)

    def index_standards_from_directory(self, directory: str) -> int:
        """Index all standards from a directory"""
        return self.standards_rag.index_from_directory(directory)

    def _infer_domain(self, table_name: str, column_names: List[str]) -> str:
        """Infer data domain from table and column names"""
        text = f"{table_name} {' '.join(column_names)}".lower()

        scores = {}
        for domain, keywords in self.DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _generate_table_enrichment(
        self,
        sample_result: SampleResult,
        standards_context: str
    ) -> Dict[str, Any]:
        """Generate table-level enrichment using LLM"""

        system_prompt = """Você é um especialista em governança de dados e catalogação de metadados.
Sua tarefa é gerar descrições detalhadas e tags para tabelas de dados.

Você deve responder APENAS em JSON válido, sem markdown ou texto adicional.

Formato de resposta:
{
    "description": "Descrição detalhada da tabela em português",
    "description_en": "Detailed table description in English",
    "business_name": "Nome amigável para negócio",
    "domain": "domínio de dados (customer, sales, finance, product, marketing, hr, operations, analytics, general)",
    "tags": ["tag1", "tag2", "tag3"],
    "classification": "public|internal|confidential|restricted",
    "owner_suggestion": "Área/time sugerido como proprietário",
    "reasoning": "Explicação do raciocínio usado",
    "confidence": 0.0 a 1.0
}

Regras:
- Use os padrões e normativos fornecidos como contexto
- Infira o propósito da tabela pelos nomes das colunas e dados de amostra
- Classifique como 'restricted' se houver PII
- Seja específico nas descrições, não genérico
- Tags devem ser relevantes para busca e organização"""

        # Build prompt with sample data
        prompt = f"""Analise a seguinte tabela e gere metadados enriquecidos:

## Informações da Tabela
{sample_result.to_text_summary()}

## Padrões e Normativos Relevantes
{standards_context}

## Amostra de Dados (primeiras linhas)
{json.dumps(sample_result.sample_rows[:5], ensure_ascii=False, default=str)}

Gere os metadados enriquecidos em JSON:"""

        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1500
        )

        # Parse JSON response
        try:
            # Clean response (remove markdown if present)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback
            return {
                "description": f"Tabela {sample_result.table_name}",
                "description_en": f"Table {sample_result.table_name}",
                "business_name": sample_result.table_name,
                "domain": self._infer_domain(sample_result.table_name, sample_result.get_column_names()),
                "tags": [],
                "classification": "internal",
                "owner_suggestion": "A definir",
                "reasoning": "Falha ao processar resposta do LLM",
                "confidence": 0.3
            }

    def _generate_column_enrichments(
        self,
        sample_result: SampleResult,
        table_context: Dict[str, Any],
        standards_context: str
    ) -> List[Dict[str, Any]]:
        """Generate column-level enrichments using LLM"""

        system_prompt = """Você é um especialista em governança de dados.
Sua tarefa é gerar descrições e classificações para colunas de dados.

Responda APENAS em JSON válido (array), sem markdown ou texto adicional.

Para cada coluna, forneça:
{
    "name": "nome_da_coluna",
    "description": "Descrição em português",
    "description_en": "Description in English",
    "business_name": "Nome amigável",
    "tags": ["tag1", "tag2"],
    "classification": "public|internal|confidential|restricted",
    "semantic_type": "pii|email|phone|date|currency|id|flag|name|address|null",
    "is_pii": true/false,
    "confidence": 0.0 a 1.0,
    "reasoning": "breve explicação"
}

Regras para PII:
- CPF, CNPJ, RG, passaporte = PII
- Email, telefone = PII
- Nome completo = PII
- Endereço = PII
- Dados de saúde = PII
- Dados financeiros pessoais = PII"""

        # Prepare column info
        columns_info = []
        for col in sample_result.columns:
            columns_info.append({
                "name": col.name,
                "type": col.data_type,
                "samples": col.sample_values[:5],
                "null_ratio": f"{col.null_ratio:.1%}",
                "distinct_count": col.distinct_count,
                "detected_patterns": col.patterns,
                "inferred_type": col.inferred_semantic_type
            })

        prompt = f"""Analise as seguintes colunas e gere metadados:

## Contexto da Tabela
Tabela: {sample_result.table_name}
Domínio: {table_context.get('domain', 'general')}
Descrição: {table_context.get('description', 'N/A')}

## Colunas para Análise
{json.dumps(columns_info, ensure_ascii=False, indent=2)}

## Padrões Relevantes
{standards_context}

Gere o array JSON com enriquecimento para cada coluna:"""

        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=3000
        )

        # Parse JSON response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback - generate basic enrichments
            return [
                {
                    "name": col.name,
                    "description": f"Coluna {col.name}",
                    "description_en": f"Column {col.name}",
                    "business_name": col.name,
                    "tags": [],
                    "classification": "internal",
                    "semantic_type": col.inferred_semantic_type,
                    "is_pii": col.inferred_semantic_type == "pii",
                    "confidence": 0.3,
                    "reasoning": "Gerado automaticamente (fallback)"
                }
                for col in sample_result.columns
            ]

    def enrich(
        self,
        sample_result: SampleResult,
        additional_context: Optional[str] = None
    ) -> EnrichmentResult:
        """
        Enrich metadata for a sampled table

        Args:
            sample_result: Result from data sampling
            additional_context: Optional additional context

        Returns:
            EnrichmentResult with descriptions, tags, classifications
        """
        start_time = time.time()

        # Get relevant standards
        standards_context = self.standards_rag.get_context_for_enrichment(
            table_name=sample_result.table_name,
            column_names=sample_result.get_column_names(),
            n_results=3
        )

        if additional_context:
            standards_context = f"{standards_context}\n\n## Contexto Adicional\n{additional_context}"

        # Generate table-level enrichment
        table_enrichment = self._generate_table_enrichment(sample_result, standards_context)

        # Generate column-level enrichments
        column_enrichments_raw = self._generate_column_enrichments(
            sample_result, table_enrichment, standards_context
        )

        # Build column enrichments
        columns = []
        pii_columns = []

        for col_data in column_enrichments_raw:
            col_name = col_data.get("name", "")
            original_col = sample_result.get_column(col_name)

            col_enrichment = ColumnEnrichment(
                name=col_name,
                original_type=original_col.data_type if original_col else "unknown",
                description=col_data.get("description", ""),
                description_en=col_data.get("description_en", ""),
                business_name=col_data.get("business_name", col_name),
                tags=col_data.get("tags", []),
                classification=col_data.get("classification", "internal"),
                semantic_type=col_data.get("semantic_type"),
                is_pii=col_data.get("is_pii", False),
                is_nullable=original_col.null_ratio > 0 if original_col else True,
                sample_values=original_col.sample_values[:5] if original_col else [],
                confidence=col_data.get("confidence", 0.5),
                reasoning=col_data.get("reasoning", "")
            )
            columns.append(col_enrichment)

            if col_enrichment.is_pii:
                pii_columns.append(col_name)

        # Determine table classification based on columns
        has_pii = len(pii_columns) > 0
        if has_pii:
            table_classification = "restricted"
        else:
            table_classification = table_enrichment.get("classification", "internal")

        # Get standards used
        standards_used = []
        search_results = self.standards_rag.search(sample_result.table_name, n_results=3)
        for r in search_results:
            standards_used.append(r.document.title)

        processing_time = int((time.time() - start_time) * 1000)

        return EnrichmentResult(
            table_name=sample_result.table_name,
            source=sample_result.source,
            description=table_enrichment.get("description", ""),
            description_en=table_enrichment.get("description_en", ""),
            business_name=table_enrichment.get("business_name", sample_result.table_name),
            domain=table_enrichment.get("domain", "general"),
            tags=table_enrichment.get("tags", []),
            classification=table_classification,
            owner_suggestion=table_enrichment.get("owner_suggestion", ""),
            columns=columns,
            row_count=sample_result.row_count,
            has_pii=has_pii,
            pii_columns=pii_columns,
            confidence=table_enrichment.get("confidence", 0.5),
            processing_time_ms=processing_time,
            standards_used=standards_used,
            metadata=sample_result.metadata
        )

    def enrich_from_parquet(
        self,
        file_path: str,
        sample_size: int = 100,
        additional_context: Optional[str] = None
    ) -> EnrichmentResult:
        """Enrich metadata from a Parquet file"""
        from .sampling.data_sampler import ParquetSampler

        sampler = ParquetSampler()
        sample = sampler.sample(file_path, sample_size=sample_size)
        return self.enrich(sample, additional_context)

    def enrich_from_csv(
        self,
        file_path: str,
        sample_size: int = 100,
        encoding: str = "utf-8",
        separator: str = ",",
        additional_context: Optional[str] = None
    ) -> EnrichmentResult:
        """Enrich metadata from a CSV file"""
        from .sampling.data_sampler import CSVSampler

        sampler = CSVSampler()
        sample = sampler.sample(file_path, sample_size=sample_size, encoding=encoding, separator=separator)
        return self.enrich(sample, additional_context)

    def enrich_from_sql(
        self,
        table_name: str,
        connection_string: str,
        schema: Optional[str] = None,
        sample_size: int = 100,
        additional_context: Optional[str] = None
    ) -> EnrichmentResult:
        """Enrich metadata from a SQL table"""
        from .sampling.data_sampler import SQLSampler

        sampler = SQLSampler(connection_string)
        sample = sampler.sample(table_name, sample_size=sample_size, schema=schema)
        return self.enrich(sample, additional_context)

    def enrich_from_delta(
        self,
        table_path: str,
        sample_size: int = 100,
        version: Optional[int] = None,
        additional_context: Optional[str] = None
    ) -> EnrichmentResult:
        """Enrich metadata from a Delta Lake table"""
        from .sampling.data_sampler import DeltaSampler

        sampler = DeltaSampler()
        sample = sampler.sample(table_path, sample_size=sample_size, version=version)
        return self.enrich(sample, additional_context)

    def enrich_batch(
        self,
        sources: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> List[EnrichmentResult]:
        """
        Enrich multiple tables in batch

        Args:
            sources: List of dicts with keys: type, path, and optional params
                     e.g., [{"type": "parquet", "path": "./data.parquet"}]
            output_dir: Optional directory to save results

        Returns:
            List of EnrichmentResult
        """
        results = []

        for source in sources:
            source_type = source.get("type", "parquet")
            path = source.get("path", "")

            try:
                if source_type == "parquet":
                    result = self.enrich_from_parquet(path, **source.get("params", {}))
                elif source_type == "csv":
                    result = self.enrich_from_csv(path, **source.get("params", {}))
                elif source_type == "sql":
                    result = self.enrich_from_sql(
                        table_name=source.get("table_name", path),
                        connection_string=source.get("connection_string", ""),
                        **source.get("params", {})
                    )
                elif source_type == "delta":
                    result = self.enrich_from_delta(path, **source.get("params", {}))
                else:
                    print(f"Unknown source type: {source_type}")
                    continue

                results.append(result)

                # Save to output directory
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    # Save JSON
                    json_file = output_path / f"{result.table_name}_metadata.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        f.write(result.to_json())

                    # Save Markdown
                    md_file = output_path / f"{result.table_name}_metadata.md"
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(result.to_markdown())

                print(f"✓ Enriched: {result.table_name} ({result.processing_time_ms}ms)")

            except Exception as e:
                print(f"✗ Error processing {path}: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "standards": self.standards_rag.get_statistics(),
            "embedding_model": self.embedding_provider.model_name,
            "llm_model": self.llm_provider.model_name,
            "language": self.language
        }

    def export_catalog(
        self,
        results: List[EnrichmentResult],
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export enrichment results as a data catalog

        Args:
            results: List of EnrichmentResult
            output_path: Output file path
            format: Output format (json, markdown, html)

        Returns:
            Path to exported file
        """
        if format == "json":
            catalog = {
                "catalog_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "tables": [r.to_dict() for r in results]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, ensure_ascii=False, indent=2)

        elif format == "markdown":
            lines = [
                "# Data Catalog",
                f"*Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
                "",
                "## Sumário",
                ""
            ]

            # Table of contents
            for r in results:
                pii_marker = " ⚠️" if r.has_pii else ""
                lines.append(f"- [{r.business_name}](#{r.table_name.replace('.', '')}){pii_marker}")

            lines.append("")

            # Each table
            for r in results:
                lines.append(r.to_markdown())
                lines.append("\n---\n")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

        elif format == "html":
            # Simple HTML export
            html_content = self._generate_html_catalog(results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return output_path

    def _generate_html_catalog(self, results: List[EnrichmentResult]) -> str:
        """Generate HTML catalog"""
        tables_html = []
        for r in results:
            pii_badge = '<span class="badge pii">PII</span>' if r.has_pii else ''

            columns_rows = "\n".join([
                f"""<tr>
                    <td><code>{col.name}</code>{'⚠️' if col.is_pii else ''}</td>
                    <td>{col.original_type}</td>
                    <td>{col.description}</td>
                    <td><span class="badge {col.classification}">{col.classification}</span></td>
                    <td>{', '.join(col.tags[:3])}</td>
                </tr>"""
                for col in r.columns
            ])

            tables_html.append(f"""
            <div class="table-card" id="{r.table_name.replace('.', '')}">
                <h2>{r.business_name} {pii_badge}</h2>
                <p class="technical-name"><code>{r.table_name}</code></p>
                <p>{r.description}</p>
                <div class="meta">
                    <span class="badge {r.classification}">{r.classification}</span>
                    <span class="domain">{r.domain}</span>
                    <span class="tags">{', '.join(r.tags)}</span>
                </div>
                <table class="columns-table">
                    <thead>
                        <tr>
                            <th>Coluna</th>
                            <th>Tipo</th>
                            <th>Descrição</th>
                            <th>Classificação</th>
                            <th>Tags</th>
                        </tr>
                    </thead>
                    <tbody>
                        {columns_rows}
                    </tbody>
                </table>
            </div>
            """)

        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Data Catalog</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .table-card {{ background: white; border-radius: 8px; padding: 24px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .technical-name {{ color: #666; font-size: 14px; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
        .badge.public {{ background: #d4edda; color: #155724; }}
        .badge.internal {{ background: #cce5ff; color: #004085; }}
        .badge.confidential {{ background: #fff3cd; color: #856404; }}
        .badge.restricted {{ background: #f8d7da; color: #721c24; }}
        .badge.pii {{ background: #f8d7da; color: #721c24; margin-left: 8px; }}
        .meta {{ margin: 16px 0; }}
        .meta span {{ margin-right: 12px; }}
        .columns-table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        .columns-table th, .columns-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        .columns-table th {{ background: #f8f9fa; font-weight: 600; }}
        code {{ background: #f1f1f1; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
    </style>
</head>
<body>
    <h1>📊 Data Catalog</h1>
    <p>Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    {''.join(tables_html)}
</body>
</html>"""
