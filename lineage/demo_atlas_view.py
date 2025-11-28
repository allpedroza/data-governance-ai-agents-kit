#!/usr/bin/env python3
"""
Demonstração da Nova Visualização Estilo Atlas/Neo4j
Mostra interface intuitiva e suporte a múltiplos idiomas
"""

import tempfile
from pathlib import Path
from lineage_system import DataLineageSystem
from i18n import set_language
import shutil


def create_sample_pipeline():
    """Cria pipeline com conexões reais para demonstração"""
    temp_dir = tempfile.mkdtemp(prefix="atlas_demo_")

    # SQL com transformações reais
    sql_content = """
-- Tabelas fonte
CREATE TABLE raw_customers AS SELECT * FROM source_customers;
CREATE TABLE raw_orders AS SELECT * FROM source_orders;
CREATE TABLE raw_products AS SELECT * FROM source_products;

-- Tabelas intermediárias
CREATE TABLE staging_customers AS
SELECT customer_id, name, email
FROM raw_customers
WHERE active = true;

CREATE TABLE staging_orders AS
SELECT order_id, customer_id, product_id, amount
FROM raw_orders
WHERE status = 'completed';

-- Tabelas de fatos
CREATE TABLE fact_sales AS
SELECT
    o.order_id,
    c.customer_id,
    c.name as customer_name,
    p.product_id,
    p.name as product_name,
    o.amount
FROM staging_orders o
JOIN staging_customers c ON o.customer_id = c.customer_id
JOIN raw_products p ON o.product_id = p.product_id;

-- Views analíticas
CREATE VIEW analytics_revenue AS
SELECT
    customer_name,
    SUM(amount) as total_revenue
FROM fact_sales
GROUP BY customer_name;

CREATE VIEW analytics_products AS
SELECT
    product_name,
    COUNT(*) as total_sales
FROM fact_sales
GROUP BY product_name;
"""

    sql_file = Path(temp_dir) / "pipeline.sql"
    with open(sql_file, 'w') as f:
        f.write(sql_content)

    return temp_dir


def demo_atlas_view():
    """Demonstra a visualização estilo Atlas"""
    print("="*80)
    print("🎨 DEMONSTRAÇÃO: Visualização Estilo Apache Atlas/Neo4j")
    print("="*80)

    # Cria pipeline
    print("\n📁 Criando pipeline de exemplo...")
    pipeline_dir = create_sample_pipeline()

    try:
        # Teste em Português
        print("\n" + "="*80)
        print("🇧🇷 DEMONSTRAÇÃO EM PORTUGUÊS")
        print("="*80)

        system_pt = DataLineageSystem(verbose=True, language='pt')
        analysis_pt = system_pt.analyze_project(
            pipeline_dir,
            file_patterns=['*.sql'],
            recursive=False
        )

        # Gera visualização Atlas
        print("\n🎨 Gerando visualização estilo Atlas (PT)...")
        atlas_file_pt = system_pt.visualize('atlas', 'lineage_atlas_pt.html')
        print(f"✅ Visualização salva em: {atlas_file_pt}")

        # Teste em Inglês
        print("\n" + "="*80)
        print("🇺🇸 DEMONSTRATION IN ENGLISH")
        print("="*80)

        system_en = DataLineageSystem(verbose=True, language='en')
        analysis_en = system_en.analyze_project(
            pipeline_dir,
            file_patterns=['*.sql'],
            recursive=False
        )

        # Gera visualização Atlas
        print("\n🎨 Generating Atlas-style visualization (EN)...")
        atlas_file_en = system_en.visualize('atlas', 'lineage_atlas_en.html')
        print(f"✅ Visualization saved at: {atlas_file_en}")

        # Compara todas as visualizações
        print("\n" + "="*80)
        print("📊 COMPARANDO TODAS AS VISUALIZAÇÕES")
        print("="*80)

        viz_types = [
            ('atlas', 'Estilo Apache Atlas/Neo4j'),
            ('force', 'Force-Directed (padrão anterior)'),
            ('hierarchical', 'Hierárquica'),
            ('sankey', 'Sankey Diagram')
        ]

        for viz_type, description in viz_types:
            filename = f"compare_{viz_type}.html"
            print(f"\n📊 Gerando: {description}...")
            system_pt.visualize(viz_type, filename)
            print(f"   ✅ {filename}")

        print("\n" + "="*80)
        print("✨ DEMONSTRAÇÃO COMPLETA!")
        print("="*80)

        print("\n📁 Arquivos gerados:")
        print("  • lineage_atlas_pt.html  - Visualização Atlas em Português")
        print("  • lineage_atlas_en.html  - Visualização Atlas em Inglês")
        print("  • compare_atlas.html     - Estilo Atlas")
        print("  • compare_force.html     - Force-Directed")
        print("  • compare_hierarchical.html - Hierárquica")
        print("  • compare_sankey.html    - Sankey")

        print("\n💡 Características da Visualização Atlas:")
        print("  ✓ Nodes grandes e coloridos por tipo")
        print("  ✓ Labels sempre visíveis")
        print("  ✓ Layout hierárquico limpo")
        print("  ✓ Setas direcionais claras")
        print("  ✓ Legenda por tipo de asset")
        print("  ✓ Tooltips ricos com informações")
        print("  ✓ Zoom e pan interativos")

        print("\n🌍 Suporte a Idiomas:")
        print("  • Português (pt) - padrão")
        print("  • Inglês (en)")
        print("  • Configure via: DATA_LINEAGE_LANGUAGE=en")

    finally:
        # Limpa
        print(f"\n🧹 Limpando arquivos temporários...")
        shutil.rmtree(pipeline_dir, ignore_errors=True)

    print("\n" + "="*80)


if __name__ == "__main__":
    demo_atlas_view()
