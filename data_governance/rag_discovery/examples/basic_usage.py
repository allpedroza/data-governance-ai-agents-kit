"""
Exemplo básico de uso do Data Discovery RAG Agent
Demonstra indexação de metadados e busca semântica
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_discovery_rag_agent import DataDiscoveryRAGAgent, create_sample_metadata


def main():
    print("=" * 80)
    print("🚀 Data Discovery RAG Agent - Exemplo Básico")
    print("=" * 80)
    print()

    # 1. Inicializa o agente
    print("📌 Passo 1: Inicializando o agente...")
    agent = DataDiscoveryRAGAgent(
        collection_name="data_lake_demo",
        persist_directory="./chroma_db_demo"
    )
    print()

    # 2. Cria metadados de exemplo
    print("📌 Passo 2: Criando metadados de exemplo...")
    tables = create_sample_metadata()
    print(f"   Criadas {len(tables)} tabelas de exemplo")
    print()

    # 3. Indexa as tabelas
    print("📌 Passo 3: Indexando tabelas no banco vetorizado...")
    agent.index_tables_batch(tables, force_update=True)
    print()

    # 4. Mostra estatísticas
    print("📌 Passo 4: Estatísticas do índice")
    stats = agent.get_statistics()
    print(f"   Total de tabelas: {stats['total_tables']}")
    print(f"   Databases: {stats['databases']}")
    print(f"   Formatos: {stats['formats']}")
    print()

    # 5. Exemplos de busca semântica
    print("=" * 80)
    print("🔍 EXEMPLOS DE BUSCA SEMÂNTICA")
    print("=" * 80)
    print()

    queries = [
        "Onde estão armazenados os dados de clientes?",
        "Quais tabelas contêm informações financeiras?",
        "Mostre tabelas com dados de comportamento do usuário",
        "Onde encontro informações sobre produtos e estoque?",
        "Quais tabelas têm dados sensíveis (PII)?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print('─' * 80)

        results = agent.search(query, n_results=2)

        if results:
            for j, result in enumerate(results, 1):
                print(f"\n📊 Resultado {j}:")
                print(f"   Tabela: {result.table.database}.{result.table.name}")
                print(f"   Relevância: {result.relevance_score:.1%}")
                print(f"   Descrição: {result.table.description}")
                print(f"   Formato: {result.table.format}")
                print(f"   Localização: {result.table.location}")
                print(f"   Colunas: {len(result.table.columns)}")
                if result.table.tags:
                    print(f"   Tags: {', '.join(result.table.tags)}")
        else:
            print("   ❌ Nenhum resultado encontrado")

    # 6. Exemplos de perguntas com RAG completo
    print("\n\n" + "=" * 80)
    print("💬 EXEMPLOS DE PERGUNTAS COM RAG (LLM + Busca Vetorizada)")
    print("=" * 80)
    print()

    questions = [
        "Quais tabelas eu devo consultar para fazer uma análise de vendas por cliente?",
        "Como posso identificar comportamentos suspeitos de usuários?",
        "Quais dados estão particionados por data e são adequados para análise temporal?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Pergunta {i}: {question}")
        print('─' * 80)

        try:
            response = agent.ask(question, n_context=3)

            print(f"\n🤖 Resposta:")
            print(response['answer'])
            print(f"\n📊 Tabelas relevantes consultadas ({len(response['relevant_tables'])}):")
            for table in response['relevant_tables']:
                print(f"   • {table['name']} (Relevância: {table['relevance']:.1%})")
            print(f"\n🎯 Confiança: {response['confidence']:.1%}")

        except Exception as e:
            print(f"   ⚠️ Erro ao processar pergunta: {e}")
            print(f"   Certifique-se de que OPENAI_API_KEY está configurada")

    print("\n\n" + "=" * 80)
    print("✅ Exemplo concluído!")
    print("=" * 80)


if __name__ == "__main__":
    main()
