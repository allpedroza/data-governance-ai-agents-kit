"""
Exemplo de integração entre Data Lineage Agent e Data Discovery RAG Agent
Demonstra como usar ambos os agentes para governança completa de dados
"""

import sys
import os
from pathlib import Path

# Adiciona os diretórios ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lineage'))

from data_discovery_rag_agent import DataDiscoveryRAGAgent, TableMetadata
from data_lineage_agent import DataLineageAgent, DataAsset


def convert_lineage_assets_to_metadata(
    lineage_agent: DataLineageAgent
) -> list:
    """
    Converte os assets do Data Lineage Agent para TableMetadata do RAG Agent

    Args:
        lineage_agent: Instância do DataLineageAgent com análise completa

    Returns:
        Lista de TableMetadata
    """
    tables = []

    for asset_name, asset in lineage_agent.assets.items():
        # Extrai colunas do schema se disponível
        columns = []
        if hasattr(asset, 'schema') and asset.schema:
            for col_name, col_type in asset.schema.items():
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'description': ''
                })

        # Calcula impacto downstream e upstream
        downstream = lineage_agent.get_downstream_impact(asset_name)
        upstream = lineage_agent.get_upstream_dependencies(asset_name)

        # Cria descrição enriquecida com informações de linhagem
        description_parts = []

        if asset.metadata.get('description'):
            description_parts.append(asset.metadata['description'])

        description_parts.append(
            f"Asset de tipo {asset.type} encontrado em {asset.source_file}"
        )

        if downstream:
            description_parts.append(
                f"Impacta {len(downstream)} assets downstream: {', '.join(downstream[:5])}"
            )

        if upstream:
            description_parts.append(
                f"Depende de {len(upstream)} assets upstream: {', '.join(upstream[:5])}"
            )

        description = ". ".join(description_parts)

        # Extrai tags baseadas no tipo e contexto
        tags = [asset.type]

        if len(downstream) > 5:
            tags.append('high-impact')

        if len(upstream) == 0:
            tags.append('source-table')

        if len(downstream) == 0:
            tags.append('sink-table')

        # Cria TableMetadata
        table = TableMetadata(
            name=asset_name,
            database='',
            schema='',
            description=description,
            columns=columns,
            owner='',
            tags=tags,
            location=asset.source_file,
            format=asset.type
        )

        tables.append(table)

    return tables


def main():
    print("=" * 80)
    print("🚀 Integração: Data Lineage Agent + Data Discovery RAG Agent")
    print("=" * 80)
    print()

    # Exemplo de arquivos para análise de linhagem
    # Você pode substituir pelos seus próprios arquivos
    example_files = [
        # Adicione seus arquivos SQL, Python, etc aqui
        # Por exemplo:
        # "../../examples/pipeline.sql",
        # "../../examples/etl.py",
    ]

    if not example_files or not all(os.path.exists(f) for f in example_files):
        print("⚠️  Para este exemplo, adicione arquivos de pipeline ao array example_files")
        print("   Criando exemplo simulado...")
        print()

        # Simula análise de linhagem com dados fictícios
        lineage_agent = DataLineageAgent()

        # Cria assets simulados
        lineage_agent.assets = {
            'raw_users': DataAsset(
                name='raw_users',
                type='table',
                source_file='etl/extract.sql',
                schema={'id': 'bigint', 'name': 'varchar', 'email': 'varchar'}
            ),
            'raw_orders': DataAsset(
                name='raw_orders',
                type='table',
                source_file='etl/extract.sql',
                schema={'order_id': 'bigint', 'user_id': 'bigint', 'amount': 'decimal'}
            ),
            'clean_users': DataAsset(
                name='clean_users',
                type='table',
                source_file='etl/transform.sql',
                schema={'id': 'bigint', 'name': 'varchar', 'email': 'varchar', 'domain': 'varchar'}
            ),
            'user_orders_agg': DataAsset(
                name='user_orders_agg',
                type='table',
                source_file='etl/transform.sql',
                schema={'user_id': 'bigint', 'total_orders': 'int', 'total_amount': 'decimal'}
            ),
            'analytics_user_metrics': DataAsset(
                name='analytics_user_metrics',
                type='table',
                source_file='etl/load.sql',
                schema={'user_id': 'bigint', 'name': 'varchar', 'total_orders': 'int', 'total_spent': 'decimal'}
            )
        }

        # Simula o grafo de linhagem
        import networkx as nx
        lineage_agent.graph = nx.DiGraph()
        lineage_agent.graph.add_edge('raw_users', 'clean_users')
        lineage_agent.graph.add_edge('raw_orders', 'user_orders_agg')
        lineage_agent.graph.add_edge('clean_users', 'analytics_user_metrics')
        lineage_agent.graph.add_edge('user_orders_agg', 'analytics_user_metrics')

        print("✅ Dados de linhagem simulados criados")

    else:
        # 1. Executa análise de linhagem
        print("📌 Passo 1: Executando Data Lineage Agent...")
        lineage_agent = DataLineageAgent()
        lineage_analysis = lineage_agent.analyze_pipeline(example_files)

        print(f"   ✅ Encontrados {len(lineage_agent.assets)} assets")
        print(f"   ✅ Encontradas {len(lineage_agent.transformations)} transformações")

    print()

    # 2. Converte assets de linhagem para metadados RAG
    print("📌 Passo 2: Convertendo assets de linhagem para metadados RAG...")
    tables = convert_lineage_assets_to_metadata(lineage_agent)
    print(f"   ✅ Convertidas {len(tables)} tabelas")
    print()

    # 3. Inicializa RAG Agent e indexa
    print("📌 Passo 3: Inicializando RAG Agent e indexando metadados...")
    rag_agent = DataDiscoveryRAGAgent(
        collection_name="lineage_metadata",
        persist_directory="./chroma_db_lineage"
    )

    rag_agent.index_tables_batch(tables, force_update=True)
    print()

    # 4. Demonstra descoberta de dados com contexto de linhagem
    print("=" * 80)
    print("🔍 DESCOBERTA DE DADOS COM CONTEXTO DE LINHAGEM")
    print("=" * 80)
    print()

    queries = [
        "Quais são as tabelas de origem (source)?",
        "Mostre tabelas que têm alto impacto em outras tabelas",
        "Onde posso encontrar dados agregados de usuários?",
        "Quais tabelas intermediárias processam dados de pedidos?",
    ]

    for query in queries:
        print(f"\n{'─' * 80}")
        print(f"🔎 {query}")
        print('─' * 80)

        results = rag_agent.search(query, n_results=3)

        if results:
            for i, result in enumerate(results, 1):
                table_name = result.table.name
                print(f"\n{i}. {table_name}")
                print(f"   Relevância: {result.relevance_score:.1%}")
                print(f"   Tipo: {result.table.format}")
                print(f"   Tags: {', '.join(result.table.tags)}")

                # Adiciona informações de linhagem
                if table_name in lineage_agent.assets:
                    downstream = lineage_agent.get_downstream_impact(table_name)
                    upstream = lineage_agent.get_upstream_dependencies(table_name)

                    print(f"   Upstream: {len(upstream)} assets")
                    print(f"   Downstream: {len(downstream)} assets")

                    if upstream:
                        print(f"   Depende de: {', '.join(upstream[:3])}")

                    if downstream:
                        print(f"   Impacta: {', '.join(downstream[:3])}")

    # 5. Análise de impacto com RAG
    print("\n\n" + "=" * 80)
    print("🎯 ANÁLISE DE IMPACTO COM RAG")
    print("=" * 80)
    print()

    # Seleciona uma tabela para análise de impacto
    if lineage_agent.assets:
        target_table = list(lineage_agent.assets.keys())[0]

        print(f"Analisando impacto de mudanças em: {target_table}")
        print()

        # Usa o agente de linhagem para análise de impacto
        impact = lineage_agent.analyze_change_impact([target_table])

        print(f"📊 Análise de Impacto:")
        print(f"   Nível de risco: {impact['risk_level']}")
        print(f"   Assets afetados downstream: {len(impact['downstream_affected'])}")
        print(f"   Dependências upstream: {len(impact['upstream_dependencies'])}")

        if impact['recommendations']:
            print(f"\n💡 Recomendações:")
            for rec in impact['recommendations']:
                print(f"   {rec}")

        # Usa o RAG para explicar o contexto
        print("\n🤖 Contexto adicional via RAG:")

        question = f"Explique o contexto e uso da tabela {target_table} no pipeline de dados"

        try:
            response = rag_agent.ask(question, n_context=2)
            print(response['answer'])
        except Exception as e:
            print(f"   ⚠️ Erro ao gerar contexto: {e}")

    # 6. Exporta análise combinada
    print("\n\n" + "=" * 80)
    print("📤 EXPORTANDO ANÁLISE COMBINADA")
    print("=" * 80)
    print()

    combined_report = {
        'lineage_metrics': lineage_agent._calculate_metrics(),
        'rag_statistics': rag_agent.get_statistics(),
        'assets': {
            name: {
                'type': asset.type,
                'source_file': asset.source_file,
                'downstream_count': len(lineage_agent.get_downstream_impact(name)),
                'upstream_count': len(lineage_agent.get_upstream_dependencies(name))
            }
            for name, asset in lineage_agent.assets.items()
        }
    }

    import json
    with open('combined_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)

    print("✅ Relatório exportado para: combined_analysis_report.json")

    print("\n\n" + "=" * 80)
    print("✅ Exemplo de integração concluído!")
    print("=" * 80)
    print()
    print("💡 Casos de uso:")
    print("   • Descoberta de dados com contexto de linhagem")
    print("   • Análise de impacto enriquecida com IA")
    print("   • Busca semântica considerando dependências")
    print("   • Documentação automática de pipelines")


if __name__ == "__main__":
    main()
