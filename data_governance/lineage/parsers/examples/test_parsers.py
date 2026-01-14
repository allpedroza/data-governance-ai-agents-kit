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
#!/usr/bin/env python3
"""
Script de teste para demonstrar o uso dos novos parsers:
- synapse_parser.py (Azure Data Factory / Synapse)
- parquet_parser.py (arquivos Parquet)

Uso:
    python test_parsers.py
"""

import sys
from pathlib import Path

# Adicionar o diretório parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synapse_parser import (
    SynapseParser,
    parse_synapse_pipeline,
    analyze_synapse_project
)


def test_synapse_parser():
    """Testa o parser de Synapse/Data Factory"""
    print("="*80)
    print("TESTE: Synapse/Data Factory Parser")
    print("="*80)

    # Caminho para o arquivo de exemplo
    example_file = Path(__file__).parent / "example_pipeline.json"

    if not example_file.exists():
        print(f"❌ Arquivo de exemplo não encontrado: {example_file}")
        return

    print(f"\n📄 Parseando pipeline: {example_file.name}")
    print("-"*80)

    # Parse do pipeline
    pipeline = parse_synapse_pipeline(str(example_file))

    if not pipeline:
        print("❌ Erro ao parsear pipeline")
        return

    # Exibir informações do pipeline
    print(f"\n✅ Pipeline parseado com sucesso!")
    print(f"\nNome: {pipeline.name}")
    print(f"Descrição: {pipeline.description}")
    print(f"Atividades: {len(pipeline.activities)}")
    print(f"Parâmetros: {len(pipeline.parameters)}")
    print(f"Variáveis: {len(pipeline.variables)}")
    print(f"Annotations: {pipeline.annotations}")

    # Detalhar atividades
    print(f"\n📋 Detalhes das Atividades:")
    print("-"*80)

    for activity_name, activity in pipeline.activities.items():
        print(f"\n  🔹 {activity_name}")
        print(f"     Tipo: {activity.type}")

        if activity.dependencies:
            print(f"     Depende de: {', '.join(activity.dependencies)}")

        if activity.inputs:
            print(f"     Inputs: {', '.join(activity.inputs)}")

        if activity.outputs:
            print(f"     Outputs: {', '.join(activity.outputs)}")

        if activity.sql_query:
            print(f"     SQL Query: {activity.sql_query[:100]}...")

        if activity.script_content:
            print(f"     Script: {activity.script_content[:100]}...")

        if activity.dataflow_name:
            print(f"     DataFlow: {activity.dataflow_name}")

    # Análise de complexidade
    print(f"\n📊 Análise de Complexidade:")
    print("-"*80)

    parser = SynapseParser()
    parser.pipelines[pipeline.name] = pipeline

    # Registrar atividades no parser
    for activity_name, activity in pipeline.activities.items():
        parser.activities[activity.get_full_id()] = activity

    analysis = parser.analyze_pipeline_complexity(pipeline.name)

    print(f"\n  Total de atividades: {analysis['total_activities']}")
    print(f"  Atividades Copy: {analysis['copy_activities']}")
    print(f"  Atividades DataFlow: {analysis['dataflow_activities']}")
    print(f"  Atividades Script: {analysis['script_activities']}")
    print(f"  Profundidade máxima: {analysis['max_dependency_depth']}")
    print(f"  Pontos de entrada: {analysis['entry_points']}")
    print(f"  Pontos de saída: {analysis['exit_points']}")
    print(f"  Parâmetros: {analysis['parameter_count']}")
    print(f"  Variáveis: {analysis['variable_count']}")

    # Gerar diagrama Mermaid
    print(f"\n🎨 Diagrama Mermaid:")
    print("-"*80)
    diagram = parser.generate_mermaid_diagram(pipeline.name)
    print(diagram)

    # Grafo de lineage
    print(f"\n🔗 Grafo de Lineage:")
    print("-"*80)
    lineage = parser.get_lineage_graph()

    print(f"\n  Nós: {len(lineage['nodes'])}")
    print(f"  Arestas: {len(lineage['edges'])}")

    print(f"\n  Nós detalhados:")
    for node in lineage['nodes']:
        print(f"    - {node['name']} (tipo: {node['type']})")

    print(f"\n  Arestas detalhadas:")
    for edge in lineage['edges']:
        print(f"    - {edge['source']} --[{edge['type']}]--> {edge['target']}")


def test_parquet_parser():
    """Testa o parser de Parquet (se pyarrow estiver disponível)"""
    print("\n\n")
    print("="*80)
    print("TESTE: Parquet Parser")
    print("="*80)

    try:
        from parquet_parser import (
            ParquetParser,
            parse_parquet_file,
            PYARROW_AVAILABLE
        )

        if not PYARROW_AVAILABLE:
            print("\n⚠️  PyArrow não está instalado.")
            print("   Para usar o Parquet Parser, instale com:")
            print("   pip install pyarrow")
            return

        print("\n✅ PyArrow disponível!")
        print("\nℹ️  Para testar o Parquet Parser:")
        print("   1. Coloque um arquivo .parquet na pasta examples/")
        print("   2. Modifique este script para apontar para o arquivo")
        print("   3. Execute novamente")

        # Exemplo de uso (comentado pois requer arquivo real)
        print("\n📝 Exemplo de uso:")
        print("""
        from parquet_parser import parse_parquet_file

        # Parse arquivo individual
        parquet_file = parse_parquet_file('data.parquet')

        print(f"Arquivo: {parquet_file.name}")
        print(f"Linhas: {parquet_file.num_rows:,}")
        print(f"Colunas: {len(parquet_file.schema.columns)}")
        print(f"Compressão: {parquet_file.compression}")

        # Exibir schema
        for col in parquet_file.schema.columns:
            print(f"  {col.name}: {col.type}")
        """)

    except ImportError as e:
        print(f"\n❌ Erro ao importar parquet_parser: {e}")
        print("   Certifique-se de que parquet_parser.py está no mesmo diretório")


def test_dataset_parser():
    """Testa o parser de datasets"""
    print("\n\n")
    print("="*80)
    print("TESTE: Dataset Parser")
    print("="*80)

    example_file = Path(__file__).parent / "example_dataset.json"

    if not example_file.exists():
        print(f"❌ Arquivo de exemplo não encontrado: {example_file}")
        return

    print(f"\n📄 Parseando dataset: {example_file.name}")
    print("-"*80)

    parser = SynapseParser()
    dataset = parser.parse_dataset(str(example_file))

    if not dataset:
        print("❌ Erro ao parsear dataset")
        return

    print(f"\n✅ Dataset parseado com sucesso!")
    print(f"\nNome: {dataset.name}")
    print(f"Tipo: {dataset.type}")
    print(f"Linked Service: {dataset.linked_service}")
    print(f"Localização: {dataset.get_full_location()}")

    if dataset.schema:
        print(f"\n📋 Schema ({len(dataset.schema)} colunas):")
        print("-"*80)
        for col in dataset.schema:
            col_type = col.get('type', 'unknown')
            print(f"  - {col['name']}: {col_type}")

    if dataset.location:
        print(f"\n📍 Detalhes de Localização:")
        print("-"*80)
        for key, value in dataset.location.items():
            print(f"  {key}: {value}")

    if dataset.properties:
        print(f"\n⚙️  Propriedades:")
        print("-"*80)
        for key, value in dataset.properties.items():
            if key not in ['location']:  # Já exibimos location
                print(f"  {key}: {value}")


def main():
    """Função principal"""
    print("\n" + "🚀 "*20)
    print("TESTE DOS NOVOS PARSERS")
    print("🚀 "*20 + "\n")

    # Teste 1: Synapse Parser com pipeline
    try:
        test_synapse_parser()
    except Exception as e:
        print(f"\n❌ Erro no teste Synapse Parser: {e}")
        import traceback
        traceback.print_exc()

    # Teste 2: Dataset Parser
    try:
        test_dataset_parser()
    except Exception as e:
        print(f"\n❌ Erro no teste Dataset Parser: {e}")
        import traceback
        traceback.print_exc()

    # Teste 3: Parquet Parser
    try:
        test_parquet_parser()
    except Exception as e:
        print(f"\n❌ Erro no teste Parquet Parser: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "✨ "*20)
    print("TESTES CONCLUÍDOS")
    print("✨ "*20 + "\n")


if __name__ == "__main__":
    main()
