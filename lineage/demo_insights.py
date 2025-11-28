#!/usr/bin/env python3
"""
Demonstração da Feature de Insights Automáticos
Mostra como os resumos e explicações automáticas funcionam
"""

import tempfile
from pathlib import Path
from lineage_system import DataLineageSystem


def create_complex_pipeline():
    """Cria um pipeline complexo para demonstrar insights"""
    temp_dir = tempfile.mkdtemp(prefix="insights_demo_")

    # Pipeline com dependências complexas e potenciais problemas
    sql_pipeline = """
-- Tabela central (ponto único de falha potencial)
CREATE TABLE core_customers AS
SELECT * FROM raw_customers;

-- Múltiplas dependências downstream (alto impacto)
CREATE TABLE customer_orders AS
SELECT c.*, o.*
FROM core_customers c
JOIN orders o ON c.id = o.customer_id;

CREATE TABLE customer_analytics AS
SELECT c.*, a.*
FROM core_customers c
JOIN analytics a ON c.id = a.customer_id;

CREATE TABLE customer_segments AS
SELECT c.*, s.*
FROM core_customers c
JOIN segments s ON c.segment_id = s.id;

-- Bottleneck (muitas entradas e saídas)
CREATE TABLE unified_customer_view AS
SELECT co.*, ca.analytics_score, cs.segment_name
FROM customer_orders co
JOIN customer_analytics ca ON co.id = ca.id
JOIN customer_segments cs ON co.id = cs.id;

-- Dashboard dependencies
CREATE TABLE daily_metrics AS
SELECT date, COUNT(*) as orders
FROM unified_customer_view
GROUP BY date;

CREATE TABLE revenue_report AS
SELECT segment_name, SUM(amount) as revenue
FROM unified_customer_view
GROUP BY segment_name;
"""

    sql_file = Path(temp_dir) / "complex_pipeline.sql"
    with open(sql_file, 'w') as f:
        f.write(sql_pipeline)

    return temp_dir


def main():
    print("="*80)
    print("🤖 DEMONSTRAÇÃO: Insights Automáticos do Grafo")
    print("="*80)

    # Cria pipeline de exemplo
    print("\n📁 Criando pipeline complexo...")
    pipeline_dir = create_complex_pipeline()

    # Analisa com o sistema
    print("📊 Analisando pipeline...\n")
    system = DataLineageSystem(verbose=True)

    analysis = system.analyze_project(
        pipeline_dir,
        file_patterns=['*.sql'],
        recursive=False
    )

    print("\n" + "="*80)
    print("📋 RESULTADOS DA ANÁLISE")
    print("="*80)

    # Mostra componentes críticos
    critical = analysis.get('critical_components', {})

    print("\n🔴 PONTOS ÚNICOS DE FALHA:")
    for spof in critical.get('single_points_of_failure', []):
        print(f"  • {spof['asset']}")
        print(f"    Tipo: {spof['type']}")
        print(f"    Impacto downstream: {spof['downstream_impact']} assets\n")

    print("⚠️  BOTTLENECKS:")
    for bn in critical.get('bottleneck_assets', []):
        print(f"  • {bn['asset']}")
        print(f"    {bn['in_degree']} entradas → {bn['out_degree']} saídas\n")

    print("📈 ASSETS DE ALTO IMPACTO:")
    for hi in critical.get('high_impact_assets', []):
        print(f"  • {hi['asset']}")
        print(f"    {hi['downstream_count']} dependências downstream\n")

    # Mostra insights
    insights = analysis.get('insights', {})

    print("🤖 RESUMO EXECUTIVO:")
    print(f"  {insights.get('summary', 'N/A')}\n")

    print(f"💡 AVALIAÇÃO DE RISCO:")
    print(f"  {insights.get('risk_assessment', 'N/A')}\n")

    print("📋 RECOMENDAÇÕES:")
    for i, rec in enumerate(insights.get('recommendations', []), 1):
        print(f"  {i}. {rec}")

    # Exporta relatório
    print("\n" + "="*80)
    print("📄 Gerando relatório HTML...")
    report_file = system.generate_report()
    print(f"✅ Relatório salvo em: {report_file}")

    print("\n💡 Abra o relatório HTML para ver a visualização completa dos insights!")
    print("="*80)

    # Cleanup
    import shutil
    shutil.rmtree(pipeline_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
