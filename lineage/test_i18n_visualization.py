#!/usr/bin/env python3
"""
Script de teste para verificar i18n e visualização Atlas interativa
"""

import sys
import os

# Adiciona o diretório ao path
sys.path.insert(0, os.path.dirname(__file__))

from lineage_system import DataLineageSystem
import tempfile
import shutil

def test_portuguese():
    """Testa visualização em português"""
    print("\n" + "="*60)
    print("TESTANDO VISUALIZAÇÃO EM PORTUGUÊS")
    print("="*60 + "\n")

    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp(prefix='lineage_test_pt_')
    print(f"📁 Diretório temporário: {temp_dir}\n")

    # Cria arquivos de teste
    sql_file = os.path.join(temp_dir, 'test.sql')
    with open(sql_file, 'w') as f:
        f.write("""
-- Tabelas de origem
CREATE TABLE raw.customers (
    customer_id INT,
    name VARCHAR(100)
);

CREATE TABLE raw.orders (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2)
);

-- Transformações
CREATE TABLE staging.enriched_orders AS
SELECT
    o.order_id,
    c.name as customer_name,
    o.amount
FROM raw.orders o
JOIN raw.customers c ON o.customer_id = c.customer_id;

-- Agregação
CREATE TABLE analytics.customer_summary AS
SELECT
    customer_name,
    COUNT(*) as order_count,
    SUM(amount) as total_amount
FROM staging.enriched_orders
GROUP BY customer_name;
""")

    # Inicializa sistema em português
    system = DataLineageSystem(verbose=True, language='pt')

    # Analisa projeto
    analysis = system.analyze_project(
        temp_dir,
        file_patterns=['*.sql'],
        recursive=False
    )

    if analysis:
        print("\n✅ Análise concluída!")

        # Testa visualização Atlas interativa
        print("\n🎨 Gerando visualização Atlas interativa em português...")
        atlas_file = system.visualize('atlas-interactive', 'test_atlas_pt.html')
        if atlas_file:
            print(f"✅ Visualização Atlas gerada: {atlas_file}")

        # Testa relatório
        print("\n📝 Gerando relatório em português...")
        report = system.generate_report()
        if report:
            print(f"✅ Relatório gerado: {report}")

    # Limpa
    shutil.rmtree(temp_dir)
    print("\n🧹 Diretório temporário removido")

def test_english():
    """Testa visualização em inglês"""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION IN ENGLISH")
    print("="*60 + "\n")

    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp(prefix='lineage_test_en_')
    print(f"📁 Temporary directory: {temp_dir}\n")

    # Cria arquivos de teste
    sql_file = os.path.join(temp_dir, 'test.sql')
    with open(sql_file, 'w') as f:
        f.write("""
CREATE TABLE raw.products (id INT, name VARCHAR(100));
CREATE TABLE analytics.product_stats AS SELECT name, COUNT(*) FROM raw.products GROUP BY name;
""")

    # Inicializa sistema em inglês
    system = DataLineageSystem(verbose=True, language='en')

    # Analisa projeto
    analysis = system.analyze_project(
        temp_dir,
        file_patterns=['*.sql'],
        recursive=False
    )

    if analysis:
        print("\n✅ Analysis completed!")

        # Testa visualização Atlas interativa
        print("\n🎨 Generating interactive Atlas visualization in English...")
        atlas_file = system.visualize('atlas-interactive', 'test_atlas_en.html')
        if atlas_file:
            print(f"✅ Atlas visualization generated: {atlas_file}")

    # Limpa
    shutil.rmtree(temp_dir)
    print("\n🧹 Temporary directory removed")

if __name__ == '__main__':
    try:
        test_portuguese()
        test_english()
        print("\n" + "="*60)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
