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
Data Discovery RAG Agent - Sistema de IA para Descoberta de Dados com RAG
Utiliza banco vetorizado para busca semântica em metadados de data lakes
Autor: Claude AI Assistant
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

# Shared models (imported from separate module to avoid chromadb dependency chain)
try:
    from rag_discovery.models import TableMetadata, SearchResult
except ImportError:
    from .models import TableMetadata, SearchResult

# Vector database - lazy import to avoid requiring chromadb at module load time
# chromadb and Settings are imported in _initialize_chroma() when actually needed

# Embeddings
from openai import OpenAI

# Utilidades
import pandas as pd
from tqdm import tqdm

# Re-export for backwards compatibility
__all__ = ['TableMetadata', 'SearchResult', 'DataDiscoveryRAGAgent', 'create_sample_metadata']


class DataDiscoveryRAGAgent:
    """
    Agente de IA para descoberta de dados usando RAG (Retrieval-Augmented Generation)
    Indexa metadados em banco vetorizado e permite busca semântica
    """

    def __init__(
        self,
        collection_name: str = "data_lake_metadata",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-5.1"
    ):
        """
        Inicializa o agente RAG

        Args:
            collection_name: Nome da coleção no ChromaDB
            persist_directory: Diretório para persistir o banco vetorizado
            embedding_model: Modelo para gerar embeddings (OpenAI)
            llm_model: Modelo LLM para geração de respostas
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Cliente OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        self.openai_client = None

        # ChromaDB
        self.chroma_client = None
        self.collection = None
        self._initialize_chroma()

        # Cache de metadados
        self.metadata_cache: Dict[str, TableMetadata] = {}

        print(f"✅ Data Discovery RAG Agent inicializado")
        print(f"   📊 Coleção: {collection_name}")
        print(f"   💾 Persistência: {persist_directory}")
        print(f"   🧠 Modelo de embedding: {embedding_model}")

    def _initialize_chroma(self):
        """Inicializa o cliente ChromaDB"""
        try:
            # Lazy import of chromadb to avoid requiring it at module load time
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb is required for DataDiscoveryRAGAgent. "
                    "Install it with: pip install chromadb"
                )

            # Cria diretório se não existir
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            # Inicializa ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Obtém ou cria coleção
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Data Lake metadata for semantic search"}
            )

            print(f"✅ ChromaDB inicializado com {self.collection.count()} documentos")

        except ImportError:
            raise
        except Exception as e:
            print(f"❌ Erro ao inicializar ChromaDB: {e}")
            raise

    def _get_openai_client(self) -> OpenAI:
        """Obtém cliente OpenAI (lazy initialization)"""
        if self.openai_client is None:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY não configurada. "
                    "Defina a variável de ambiente OPENAI_API_KEY"
                )
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
        return self.openai_client

    def _generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para um texto usando OpenAI"""
        try:
            client = self._get_openai_client()
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            raise

    def _generate_table_id(self, table: TableMetadata) -> str:
        """Gera ID único para uma tabela"""
        full_name = f"{table.database}.{table.schema}.{table.name}"
        return hashlib.md5(full_name.encode()).hexdigest()

    def index_table(self, table: TableMetadata, force_update: bool = False):
        """
        Indexa uma tabela no banco vetorizado

        Args:
            table: Metadados da tabela
            force_update: Se True, atualiza mesmo se já existir
        """
        table_id = self._generate_table_id(table)

        # Verifica se já existe
        if not force_update:
            try:
                existing = self.collection.get(ids=[table_id])
                if existing['ids']:
                    print(f"⏭️  Tabela {table.name} já indexada (use force_update=True para atualizar)")
                    return
            except:
                pass

        # Gera representação textual
        text_repr = table.to_text_representation()

        # Gera embedding
        print(f"🔄 Gerando embedding para {table.name}...")
        embedding = self._generate_embedding(text_repr)

        # Prepara metadados para ChromaDB
        metadata = {
            'name': table.name,
            'database': table.database,
            'schema': table.schema,
            'description': table.description[:500] if table.description else "",  # Limita tamanho
            'owner': table.owner,
            'format': table.format,
            'location': table.location,
            'num_columns': len(table.columns),
            'row_count': table.row_count or 0,
            'tags': ','.join(table.tags) if table.tags else "",
            'indexed_at': datetime.now().isoformat()
        }

        # Adiciona ao ChromaDB
        self.collection.upsert(
            ids=[table_id],
            embeddings=[embedding],
            documents=[text_repr],
            metadatas=[metadata]
        )

        # Adiciona ao cache
        self.metadata_cache[table_id] = table

        print(f"✅ Tabela {table.name} indexada com sucesso")

    def index_tables_batch(self, tables: List[TableMetadata], force_update: bool = False):
        """
        Indexa múltiplas tabelas em batch para melhor performance

        Args:
            tables: Lista de metadados de tabelas
            force_update: Se True, atualiza mesmo se já existirem
        """
        print(f"📚 Indexando {len(tables)} tabelas em batch...")

        for table in tqdm(tables, desc="Indexando tabelas"):
            try:
                self.index_table(table, force_update=force_update)
            except Exception as e:
                print(f"❌ Erro ao indexar {table.name}: {e}")

        print(f"✅ Indexação batch concluída. Total de documentos: {self.collection.count()}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Busca tabelas usando query em linguagem natural

        Args:
            query: Query em linguagem natural
            n_results: Número de resultados a retornar
            filter_metadata: Filtros de metadados (ex: {'database': 'production'})

        Returns:
            Lista de SearchResult ordenada por relevância
        """
        print(f"🔍 Buscando: '{query}'")

        # Gera embedding da query
        query_embedding = self._generate_embedding(query)

        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        # Processa resultados
        search_results = []

        for i in range(len(results['ids'][0])):
            table_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]

            # Calcula score de relevância (1 - distance normalizado)
            relevance_score = 1.0 - min(distance / 2.0, 1.0)

            # Reconstrói TableMetadata do cache ou dos metadados
            if table_id in self.metadata_cache:
                table = self.metadata_cache[table_id]
            else:
                table = TableMetadata(
                    name=metadata.get('name', ''),
                    database=metadata.get('database', ''),
                    schema=metadata.get('schema', ''),
                    description=metadata.get('description', ''),
                    owner=metadata.get('owner', ''),
                    format=metadata.get('format', ''),
                    location=metadata.get('location', ''),
                    row_count=metadata.get('row_count', 0),
                    tags=metadata.get('tags', '').split(',') if metadata.get('tags') else []
                )

            # Cria snippet do documento
            snippet = document[:300] + "..." if len(document) > 300 else document

            search_result = SearchResult(
                table=table,
                relevance_score=relevance_score,
                matching_reason=f"Relevância: {relevance_score:.2%}",
                snippet=snippet
            )

            search_results.append(search_result)

        print(f"✅ Encontrados {len(search_results)} resultados")

        return search_results

    def ask(
        self,
        question: str,
        n_context: int = 3,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Responde uma pergunta sobre o data lake usando RAG

        Args:
            question: Pergunta em linguagem natural
            n_context: Número de tabelas a usar como contexto
            include_reasoning: Se True, inclui raciocínio do LLM

        Returns:
            Dicionário com resposta e metadados
        """
        print(f"❓ Pergunta: '{question}'")

        # Busca contexto relevante
        search_results = self.search(query=question, n_results=n_context)

        if not search_results:
            return {
                'question': question,
                'answer': 'Não encontrei tabelas relevantes no data lake para responder sua pergunta.',
                'relevant_tables': [],
                'confidence': 0.0
            }

        # Constrói contexto para o LLM
        context_parts = []
        context_parts.append("# Contexto: Metadados de Tabelas Relevantes\n")

        for i, result in enumerate(search_results, 1):
            context_parts.append(f"\n## Tabela {i}: {result.table.name}")
            context_parts.append(f"Relevância: {result.relevance_score:.2%}\n")
            context_parts.append(result.snippet)
            context_parts.append("\n---")

        context = "\n".join(context_parts)

        # Gera resposta com LLM
        try:
            client = self._get_openai_client()

            # Monta o prompt
            full_prompt = f"""Você é um especialista em governança de dados e data lakes.
Responda a pergunta do usuário com base nos metadados das tabelas fornecidos como contexto.

IMPORTANTE:
- Seja específico e cite as tabelas relevantes
- Se houver múltiplas opções, liste-as
- Inclua informações sobre colunas, formatos e localizações quando relevante
- Se não houver informação suficiente, seja honesto sobre isso

{context}

Pergunta do usuário: {question}

Responda em português de forma clara e estruturada:"""

            # Chama o LLM
            reasoning_effort = "low" if include_reasoning else "none"

            response = client.responses.create(
                model=self.llm_model,
                input=full_prompt,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": "medium"}
            )

            # Extrai resposta
            answer = response.output_text or ""
            if not answer and hasattr(response, "output"):
                text_chunks = []
                for item in response.output:
                    for piece in item.get("content", []):
                        if piece.get("type") == "output_text":
                            text_chunks.append(piece.get("text", ""))
                answer = "".join(text_chunks)

            # Calcula confiança média baseada na relevância dos resultados
            avg_confidence = sum(r.relevance_score for r in search_results) / len(search_results)

            return {
                'question': question,
                'answer': answer,
                'relevant_tables': [
                    {
                        'name': r.table.name,
                        'database': r.table.database,
                        'relevance': r.relevance_score,
                        'description': r.table.description
                    }
                    for r in search_results
                ],
                'confidence': avg_confidence,
                'context_used': len(search_results)
            }

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return {
                'question': question,
                'answer': f"Erro ao gerar resposta: {e}",
                'relevant_tables': [
                    {'name': r.table.name, 'relevance': r.relevance_score}
                    for r in search_results
                ],
                'confidence': 0.0
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas sobre o índice"""
        count = self.collection.count()

        # Busca alguns metadados para estatísticas
        if count > 0:
            sample = self.collection.get(limit=min(100, count))

            # Conta por database
            databases = {}
            formats = {}

            for metadata in sample['metadatas']:
                db = metadata.get('database', 'unknown')
                databases[db] = databases.get(db, 0) + 1

                fmt = metadata.get('format', 'unknown')
                formats[fmt] = formats.get(fmt, 0) + 1

            return {
                'total_tables': count,
                'databases': databases,
                'formats': formats,
                'collection_name': self.collection_name
            }

        return {
            'total_tables': 0,
            'databases': {},
            'formats': {},
            'collection_name': self.collection_name
        }

    def export_metadata(self, output_file: str = "metadata_export.json"):
        """Exporta todos os metadados indexados para um arquivo JSON"""
        print(f"📤 Exportando metadados para {output_file}...")

        # Busca todos os documentos
        all_docs = self.collection.get()

        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_tables': len(all_docs['ids']),
            'tables': []
        }

        for i in range(len(all_docs['ids'])):
            table_data = {
                'id': all_docs['ids'][i],
                'metadata': all_docs['metadatas'][i],
                'document': all_docs['documents'][i]
            }
            export_data['tables'].append(table_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Exportados {len(all_docs['ids'])} metadados para {output_file}")

    def import_from_json(self, json_file: str) -> List[TableMetadata]:
        """
        Importa metadados de um arquivo JSON

        Formato esperado:
        [
            {
                "name": "customers",
                "database": "prod",
                "schema": "public",
                "description": "Customer data",
                "columns": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "name", "type": "varchar", "description": "Customer name"}
                ],
                "owner": "data-team",
                "tags": ["pii", "critical"],
                ...
            }
        ]
        """
        print(f"📥 Importando metadados de {json_file}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tables = []
        for item in data:
            table = TableMetadata(
                name=item.get('name', ''),
                database=item.get('database', ''),
                schema=item.get('schema', ''),
                description=item.get('description', ''),
                columns=item.get('columns', []),
                row_count=item.get('row_count'),
                size_bytes=item.get('size_bytes'),
                created_at=item.get('created_at'),
                updated_at=item.get('updated_at'),
                owner=item.get('owner', ''),
                tags=item.get('tags', []),
                location=item.get('location', ''),
                format=item.get('format', ''),
                partition_keys=item.get('partition_keys', []),
                sample_data=item.get('sample_data')
            )
            tables.append(table)

        print(f"✅ Importados {len(tables)} metadados de tabelas")
        return tables

    def reset_index(self):
        """Reseta completamente o índice (USE COM CUIDADO!)"""
        print("⚠️  ATENÇÃO: Resetando índice...")
        self.chroma_client.delete_collection(name=self.collection_name)
        self._initialize_chroma()
        self.metadata_cache.clear()
        print("✅ Índice resetado")


def create_sample_metadata() -> List[TableMetadata]:
    """Cria metadados de exemplo para testes"""
    return [
        TableMetadata(
            name="customers",
            database="production",
            schema="public",
            description="Tabela principal de clientes contendo informações cadastrais e de contato",
            columns=[
                {"name": "customer_id", "type": "bigint", "description": "Identificador único do cliente"},
                {"name": "name", "type": "varchar(200)", "description": "Nome completo do cliente"},
                {"name": "email", "type": "varchar(100)", "description": "Email de contato"},
                {"name": "phone", "type": "varchar(20)", "description": "Telefone"},
                {"name": "created_at", "type": "timestamp", "description": "Data de cadastro"},
                {"name": "country", "type": "varchar(50)", "description": "País do cliente"}
            ],
            row_count=1500000,
            owner="data-team",
            tags=["pii", "critical", "customer-data"],
            location="s3://data-lake/production/customers/",
            format="delta",
            partition_keys=["country", "created_at"]
        ),
        TableMetadata(
            name="orders",
            database="production",
            schema="public",
            description="Histórico de pedidos realizados pelos clientes",
            columns=[
                {"name": "order_id", "type": "bigint", "description": "ID único do pedido"},
                {"name": "customer_id", "type": "bigint", "description": "FK para customers"},
                {"name": "order_date", "type": "timestamp", "description": "Data do pedido"},
                {"name": "total_amount", "type": "decimal(10,2)", "description": "Valor total"},
                {"name": "status", "type": "varchar(50)", "description": "Status do pedido"},
                {"name": "payment_method", "type": "varchar(50)", "description": "Forma de pagamento"}
            ],
            row_count=5000000,
            owner="data-team",
            tags=["transactional", "critical"],
            location="s3://data-lake/production/orders/",
            format="parquet",
            partition_keys=["order_date"]
        ),
        TableMetadata(
            name="product_catalog",
            database="production",
            schema="public",
            description="Catálogo de produtos disponíveis para venda",
            columns=[
                {"name": "product_id", "type": "bigint", "description": "ID do produto"},
                {"name": "name", "type": "varchar(200)", "description": "Nome do produto"},
                {"name": "category", "type": "varchar(100)", "description": "Categoria"},
                {"name": "price", "type": "decimal(10,2)", "description": "Preço unitário"},
                {"name": "stock_quantity", "type": "int", "description": "Quantidade em estoque"}
            ],
            row_count=50000,
            owner="product-team",
            tags=["product", "catalog"],
            location="s3://data-lake/production/products/",
            format="delta"
        ),
        TableMetadata(
            name="user_activity_logs",
            database="analytics",
            schema="logs",
            description="Logs de atividade dos usuários no aplicativo e website",
            columns=[
                {"name": "event_id", "type": "uuid", "description": "ID do evento"},
                {"name": "user_id", "type": "bigint", "description": "ID do usuário"},
                {"name": "event_type", "type": "varchar(100)", "description": "Tipo de evento"},
                {"name": "timestamp", "type": "timestamp", "description": "Momento do evento"},
                {"name": "page_url", "type": "varchar(500)", "description": "URL acessada"},
                {"name": "user_agent", "type": "varchar(500)", "description": "Browser/app do usuário"}
            ],
            row_count=100000000,
            owner="analytics-team",
            tags=["logs", "analytics", "behavioral"],
            location="s3://data-lake/analytics/user_activity/",
            format="parquet",
            partition_keys=["timestamp"]
        ),
        TableMetadata(
            name="financial_transactions",
            database="finance",
            schema="transactions",
            description="Transações financeiras detalhadas incluindo pagamentos e reembolsos",
            columns=[
                {"name": "transaction_id", "type": "uuid", "description": "ID da transação"},
                {"name": "order_id", "type": "bigint", "description": "ID do pedido relacionado"},
                {"name": "amount", "type": "decimal(15,2)", "description": "Valor da transação"},
                {"name": "currency", "type": "char(3)", "description": "Moeda (USD, BRL, etc)"},
                {"name": "transaction_date", "type": "timestamp", "description": "Data da transação"},
                {"name": "status", "type": "varchar(50)", "description": "Status (completed, pending, failed)"}
            ],
            row_count=8000000,
            owner="finance-team",
            tags=["pii", "financial", "critical", "audit"],
            location="s3://data-lake/finance/transactions/",
            format="delta",
            partition_keys=["transaction_date", "currency"]
        )
    ]
