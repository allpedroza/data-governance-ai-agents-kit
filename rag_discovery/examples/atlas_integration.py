"""
Exemplo avançado: Integração do RAG Agent com Apache Atlas
Demonstra como extrair metadados do Atlas e indexá-los para busca semântica
"""

import sys
import os
from pathlib import Path
import json

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_discovery_rag_agent import DataDiscoveryRAGAgent, TableMetadata


def extract_metadata_from_atlas_entity(entity: dict) -> TableMetadata:
    """
    Extrai metadados de uma entidade do Apache Atlas e converte para TableMetadata

    Args:
        entity: Dicionário com dados da entidade do Atlas

    Returns:
        TableMetadata object
    """
    attributes = entity.get('attributes', {})

    # Extrai informações básicas
    name = attributes.get('name', '')
    qualified_name = attributes.get('qualifiedName', '')

    # Parse qualified name para extrair database/schema
    # Formato típico: database.schema.table@cluster
    parts = qualified_name.split('@')[0].split('.')
    database = parts[0] if len(parts) > 2 else ''
    schema = parts[1] if len(parts) > 2 else ''

    # Extrai colunas
    columns = []
    if 'columns' in entity.get('relationshipAttributes', {}):
        for col_ref in entity['relationshipAttributes']['columns']:
            col = {
                'name': col_ref.get('displayText', ''),
                'type': col_ref.get('attributes', {}).get('type', 'unknown'),
                'description': col_ref.get('attributes', {}).get('comment', '')
            }
            columns.append(col)

    # Cria TableMetadata
    return TableMetadata(
        name=name,
        database=database,
        schema=schema,
        description=attributes.get('description', '') or attributes.get('comment', ''),
        columns=columns,
        owner=attributes.get('owner', ''),
        location=attributes.get('location', ''),
        format=attributes.get('tableType', ''),
        created_at=attributes.get('createTime'),
        updated_at=attributes.get('updateTime'),
        tags=entity.get('classificationNames', [])
    )


def load_atlas_export(export_file: str) -> list:
    """
    Carrega um arquivo de export do Apache Atlas

    Args:
        export_file: Caminho para o arquivo JSON de export do Atlas

    Returns:
        Lista de entidades
    """
    print(f"📥 Carregando export do Atlas: {export_file}")

    with open(export_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities = data.get('entities', [])
    print(f"   Encontradas {len(entities)} entidades")

    return entities


def main():
    print("=" * 80)
    print("🚀 Data Discovery RAG Agent - Integração com Apache Atlas")
    print("=" * 80)
    print()

    # 1. Inicializa o agente
    print("📌 Passo 1: Inicializando o agente RAG...")
    agent = DataDiscoveryRAGAgent(
        collection_name="atlas_metadata",
        persist_directory="./chroma_db_atlas"
    )
    print()

    # 2. Opção A: Carregar de um arquivo de export do Atlas
    # Você pode exportar entidades do Atlas usando a API REST ou UI
    atlas_export_file = "atlas_export.json"

    if os.path.exists(atlas_export_file):
        print("📌 Passo 2: Carregando metadados do Atlas...")

        # Carrega entidades do Atlas
        entities = load_atlas_export(atlas_export_file)

        # Filtra apenas tabelas (hive_table, rdbms_table, etc)
        table_types = ['hive_table', 'rdbms_table', 'Table', 'DataSet']
        table_entities = [e for e in entities if e.get('typeName') in table_types]

        print(f"   Encontradas {len(table_entities)} tabelas")

        # Converte para TableMetadata
        tables = []
        for entity in table_entities:
            try:
                table = extract_metadata_from_atlas_entity(entity)
                tables.append(table)
            except Exception as e:
                print(f"   ⚠️ Erro ao processar {entity.get('attributes', {}).get('name', 'unknown')}: {e}")

        print(f"   ✅ Convertidas {len(tables)} tabelas para TableMetadata")
        print()

        # 3. Indexa as tabelas
        print("📌 Passo 3: Indexando tabelas no banco vetorizado...")
        agent.index_tables_batch(tables, force_update=True)
        print()

    else:
        print(f"⚠️  Arquivo {atlas_export_file} não encontrado")
        print()
        print("📝 Como exportar metadados do Apache Atlas:")
        print()
        print("   Opção 1: Via API REST")
        print("   ```bash")
        print("   curl -u admin:admin \\")
        print("     http://atlas-host:21000/api/atlas/v2/search/basic \\")
        print("     -d '{\"typeName\": \"hive_table\"}' \\")
        print("     -H 'Content-Type: application/json' > atlas_export.json")
        print("   ```")
        print()
        print("   Opção 2: Via Python (apache-atlas-client)")
        print("   ```python")
        print("   from apache_atlas.client.base_client import AtlasClient")
        print("   client = AtlasClient('http://atlas-host:21000', ('admin', 'admin'))")
        print("   entities = client.search_entities('hive_table')")
        print("   ```")
        print()
        print("   Criando dados de exemplo para demonstração...")
        print()

        # Cria dados de exemplo simulando Atlas
        from data_discovery_rag_agent import create_sample_metadata
        tables = create_sample_metadata()

        print("📌 Passo 3: Indexando tabelas de exemplo...")
        agent.index_tables_batch(tables, force_update=True)
        print()

    # 4. Demonstra buscas
    print("=" * 80)
    print("🔍 EXEMPLOS DE DESCOBERTA DE DADOS")
    print("=" * 80)
    print()

    queries = [
        "Quais tabelas contêm dados de usuários?",
        "Onde estão os dados financeiros?",
        "Mostre tabelas particionadas por data",
        "Quais tabelas têm mais de 1 milhão de registros?",
        "Onde encontro dados de transações?",
    ]

    for query in queries:
        print(f"\n{'─' * 80}")
        print(f"🔎 Query: {query}")
        print('─' * 80)

        results = agent.search(query, n_results=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.table.database}.{result.table.name}")
                print(f"   Relevância: {result.relevance_score:.1%}")
                print(f"   {result.table.description[:100]}...")

                if result.table.columns:
                    print(f"   Colunas: {', '.join([c['name'] for c in result.table.columns[:5]])}")

                if result.table.row_count:
                    print(f"   Registros: {result.table.row_count:,}")

    # 5. Exemplo de pergunta com contexto
    print("\n\n" + "=" * 80)
    print("💬 PERGUNTAS COM RAG COMPLETO")
    print("=" * 80)
    print()

    questions = [
        "Como posso fazer uma análise de churn de clientes? Quais tabelas devo usar?",
        "Preciso criar um dashboard financeiro. Quais fontes de dados estão disponíveis?",
        "Quais tabelas possuem dados sensíveis que requerem cuidados especiais de segurança?",
    ]

    for question in questions:
        print(f"\n{'─' * 80}")
        print(f"❓ {question}")
        print('─' * 80)

        try:
            response = agent.ask(question, n_context=4)

            print(f"\n🤖 Resposta:")
            print(response['answer'])

            print(f"\n📊 Fontes consultadas:")
            for table in response['relevant_tables']:
                print(f"   • {table.get('database', '')}.{table['name']}")

        except Exception as e:
            print(f"   ⚠️ Erro: {e}")

    # 6. Exporta metadados
    print("\n\n" + "=" * 80)
    print("📤 EXPORTANDO METADADOS INDEXADOS")
    print("=" * 80)
    print()

    agent.export_metadata("atlas_rag_metadata_export.json")

    print("\n\n" + "=" * 80)
    print("✅ Exemplo concluído!")
    print("=" * 80)


if __name__ == "__main__":
    main()
