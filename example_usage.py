#!/usr/bin/env python3
"""
Exemplo de Uso - Data Lineage AI Agent
Demonstração com pipeline de dados de exemplo
"""

import os
import tempfile
from pathlib import Path
import shutil
from lineage_system import DataLineageSystem


def create_sample_pipeline():
    """
    Cria um pipeline de exemplo para demonstração
    """
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp(prefix="lineage_demo_")
    print(f"📁 Criando pipeline de exemplo em: {temp_dir}")
    
    # 1. Script Python - Extração de dados
    python_extract = """
import pandas as pd
from pyspark.sql import SparkSession

# Extração de dados de vendas
def extract_sales_data():
    # Leitura de arquivo CSV
    sales_df = pd.read_csv("raw_sales.csv")
    
    # Leitura de banco de dados
    customers_df = pd.read_sql("SELECT * FROM customers", connection)
    
    # Salva dados processados
    sales_df.to_parquet("processed_sales.parquet")
    
    return sales_df

# Spark ETL
spark = SparkSession.builder.appName("SalesETL").getOrCreate()

# Leitura de tabela Delta
orders_df = spark.read.format("delta").load("delta_lake/orders")

# Join com dimensão de produtos
products_df = spark.table("dim_products")
enriched_orders = orders_df.join(products_df, "product_id")

# Salva resultado
enriched_orders.write.mode("overwrite").saveAsTable("fact_enriched_orders")
"""
    
    # 2. Script SQL - Transformações
    sql_transform = """
-- Criação de tabela agregada de vendas
CREATE TABLE sales_summary AS
SELECT 
    d.date,
    p.product_category,
    c.customer_segment,
    SUM(s.amount) as total_sales,
    COUNT(DISTINCT s.customer_id) as unique_customers
FROM fact_sales s
JOIN dim_date d ON s.date_id = d.date_id
JOIN dim_product p ON s.product_id = p.product_id
JOIN dim_customer c ON s.customer_id = c.customer_id
GROUP BY d.date, p.product_category, c.customer_segment;

-- View materializada para dashboard
CREATE MATERIALIZED VIEW mv_daily_kpis AS
SELECT 
    date,
    SUM(total_sales) as daily_revenue,
    SUM(unique_customers) as daily_customers
FROM sales_summary
GROUP BY date;

-- Atualização de tabela de métricas
INSERT INTO metrics_history
SELECT 
    CURRENT_DATE as snapshot_date,
    COUNT(*) as record_count,
    SUM(total_sales) as total_revenue
FROM sales_summary;

-- Procedure para atualizar dimensões
UPDATE dim_customer
SET last_purchase_date = (
    SELECT MAX(date) 
    FROM fact_sales s 
    WHERE s.customer_id = dim_customer.customer_id
);
"""
    
    # 3. Script Python - ML Pipeline
    ml_pipeline = """
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Carrega dados para treinamento
training_data = pd.read_parquet("processed_sales.parquet")
customer_features = pd.read_csv("customer_features.csv")

# Merge datasets
model_data = training_data.merge(customer_features, on='customer_id')

# Feature engineering
model_data['recency_score'] = calculate_recency(model_data)
model_data['frequency_score'] = calculate_frequency(model_data)

# Salva dataset preparado
model_data.to_parquet("ml_prepared_data.parquet")

# Treina modelo
X_train, X_test = train_test_split(model_data)
model = train_churn_model(X_train)

# Salva modelo
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Gera previsões
predictions_df = pd.DataFrame({
    'customer_id': X_test['customer_id'],
    'churn_probability': model.predict_proba(X_test)[:, 1]
})

# Salva previsões
predictions_df.to_csv("churn_predictions.csv")
"""
    
    # 4. Databricks Notebook (Python)
    databricks_notebook = """
# COMMAND ----------
# Databricks notebook for real-time processing

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# COMMAND ----------
# Leitura de stream Kafka
events_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "user_events")
    .load()
)

# COMMAND ----------
# Processamento do stream
processed_stream = (
    events_stream
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), event_schema).alias("data"))
    .select("data.*")
)

# COMMAND ----------
# Agregações em tempo real
aggregated_stream = (
    processed_stream
    .groupBy(window("event_timestamp", "1 hour"), "event_type")
    .agg(count("*").alias("event_count"))
)

# COMMAND ----------
# Salva em Delta Lake
query = (
    aggregated_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/delta/checkpoints/events")
    .start("/delta/tables/event_aggregations")
)
"""
    
    # 5. Terraform - Infraestrutura
    terraform_config = """
resource "aws_s3_bucket" "data_lake" {
  bucket = "company-data-lake"
  
  tags = {
    Environment = "Production"
    Purpose     = "DataLake"
  }
}

resource "aws_glue_catalog_database" "analytics" {
  name = "analytics_db"
  
  description = "Analytics database for processed data"
}

resource "aws_glue_catalog_table" "sales_fact" {
  database_name = aws_glue_catalog_database.analytics.name
  name          = "fact_sales"
  
  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/fact_sales/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    
    columns {
      name = "date_id"
      type = "bigint"
    }
    
    columns {
      name = "customer_id"
      type = "string"
    }
    
    columns {
      name = "amount"
      type = "double"
    }
  }
}

resource "databricks_table" "ml_features" {
  name = "ml_feature_store"
  catalog_name = "hive_metastore"
  schema_name = "ml"
  table_type = "MANAGED"
  
  column {
    name = "customer_id"
    type = "STRING"
  }
  
  column {
    name = "feature_vector"
    type = "ARRAY<DOUBLE>"
  }
}
"""
    
    # 6. SQL - Queries de análise
    analysis_queries = """
-- Query 1: Análise de tendências
SELECT 
    m.metric_date,
    m.total_revenue,
    LAG(m.total_revenue, 1) OVER (ORDER BY m.metric_date) as prev_revenue,
    (m.total_revenue - LAG(m.total_revenue, 1) OVER (ORDER BY m.metric_date)) / 
        LAG(m.total_revenue, 1) OVER (ORDER BY m.metric_date) * 100 as growth_rate
FROM metrics_history m
WHERE m.metric_date >= DATE_SUB(CURRENT_DATE, 30);

-- Query 2: Segmentação de clientes
CREATE VIEW customer_segments AS
SELECT 
    c.customer_id,
    c.customer_segment,
    s.total_purchases,
    s.avg_purchase_value,
    CASE 
        WHEN s.total_purchases > 100 THEN 'VIP'
        WHEN s.total_purchases > 50 THEN 'Regular'
        ELSE 'Occasional'
    END as customer_tier
FROM dim_customer c
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(*) as total_purchases,
        AVG(amount) as avg_purchase_value
    FROM fact_sales
    GROUP BY customer_id
) s ON c.customer_id = s.customer_id;

-- Query 3: Product performance
CREATE TABLE product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.product_category,
    SUM(s.quantity) as units_sold,
    SUM(s.amount) as revenue,
    COUNT(DISTINCT s.customer_id) as unique_buyers
FROM dim_product p
JOIN fact_sales s ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name, p.product_category;
"""
    
    # Salva arquivos
    files = {
        'extract_sales.py': python_extract,
        'transform_data.sql': sql_transform,
        'ml_pipeline.py': ml_pipeline,
        'streaming_notebook.py': databricks_notebook,
        'infrastructure.tf': terraform_config,
        'analysis_queries.sql': analysis_queries
    }
    
    for filename, content in files.items():
        filepath = Path(temp_dir) / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Criado: {filename}")
    
    return temp_dir


def demonstrate_lineage_analysis():
    """
    Demonstra as capacidades do sistema
    """
    print("\n" + "="*80)
    print("🚀 DEMONSTRAÇÃO DO DATA LINEAGE AI AGENT")
    print("="*80 + "\n")
    
    # Cria pipeline de exemplo
    sample_dir = create_sample_pipeline()
    
    try:
        # Inicializa o sistema
        print("\n📊 Inicializando sistema de análise...")
        system = DataLineageSystem(verbose=True)
        
        # 1. ANÁLISE DO PIPELINE
        print("\n" + "-"*60)
        print("1️⃣ ANÁLISE DO PIPELINE")
        print("-"*60)
        
        analysis = system.analyze_project(
            sample_dir,
            file_patterns=['*.py', '*.sql', '*.tf'],
            recursive=True
        )
        
        # 2. ANÁLISE DE IMPACTO
        print("\n" + "-"*60)
        print("2️⃣ ANÁLISE DE IMPACTO")
        print("-"*60)
        
        # Simula mudança em tabelas críticas
        changed_assets = ['fact_sales', 'dim_customer', 'processed_sales.parquet']
        print(f"\n🎯 Simulando mudanças em: {', '.join(changed_assets)}")
        
        impact = system.analyze_impact(changed_assets)
        
        # 3. VISUALIZAÇÕES
        print("\n" + "-"*60)
        print("3️⃣ GERANDO VISUALIZAÇÕES")
        print("-"*60)
        
        # Dashboard principal
        print("\n📊 Gerando dashboard...")
        dashboard_file = system.visualize('dashboard', 'demo_dashboard.html')
        
        # Grafo force-directed
        print("🌐 Gerando grafo interativo...")
        force_file = system.visualize('force', 'demo_force_graph.html')
        
        # Diagrama Sankey
        print("🔄 Gerando diagrama de fluxo...")
        sankey_file = system.visualize('sankey', 'demo_sankey.html')
        
        # Análise de impacto visual
        if changed_assets:
            print("💥 Gerando visualização de impacto...")
            impact_file = system.visualize(
                'impact', 
                'demo_impact.html',
                changed_nodes=changed_assets
            )
        
        # 4. EXPORTAÇÃO
        print("\n" + "-"*60)
        print("4️⃣ EXPORTANDO RESULTADOS")
        print("-"*60)
        
        # Exporta JSON
        print("\n💾 Exportando análise em JSON...")
        json_file = system.export_analysis('json', 'demo_analysis.json')
        
        # Exporta documentação
        print("📝 Exportando documentação...")
        doc_file = system.export_analysis('md', 'demo_documentation.md')
        
        # 5. RELATÓRIO COMPLETO
        print("\n" + "-"*60)
        print("5️⃣ GERANDO RELATÓRIO COMPLETO")
        print("-"*60)
        
        report_file = system.generate_report()
        
        # RESUMO FINAL
        print("\n" + "="*80)
        print("✨ DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*80)
        
        print("\n📁 Arquivos gerados:")
        generated_files = [
            dashboard_file, force_file, sankey_file,
            json_file, doc_file, report_file
        ]
        
        if 'impact_file' in locals():
            generated_files.append(impact_file)
        
        for f in generated_files:
            if f and os.path.exists(f):
                size = os.path.getsize(f) / 1024  # KB
                print(f"  • {f} ({size:.1f} KB)")
        
        print("\n💡 Dicas:")
        print("  • Abra os arquivos HTML no navegador para visualização interativa")
        print("  • Use demo_analysis.json para integração com outras ferramentas")
        print("  • Consulte demo_documentation.md para documentação detalhada")
        print("  • O relatório principal (lineage_report_*.html) contém links para todas as visualizações")
        
    finally:
        # Limpa diretório temporário
        print(f"\n🧹 Limpando arquivos temporários em: {sample_dir}")
        shutil.rmtree(sample_dir, ignore_errors=True)


def example_programmatic_usage():
    """
    Exemplo de uso programático do sistema
    """
    print("\n" + "="*80)
    print("💻 EXEMPLO DE USO PROGRAMÁTICO")
    print("="*80 + "\n")
    
    # Cria sistema
    system = DataLineageSystem(verbose=False)
    
    # Cria um pipeline simples para teste
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Cria arquivo SQL de exemplo
        sql_content = """
        CREATE TABLE users AS SELECT * FROM raw_users;
        CREATE TABLE orders AS SELECT * FROM raw_orders WHERE user_id IN (SELECT id FROM users);
        CREATE TABLE analytics AS SELECT u.*, o.total FROM users u JOIN orders o ON u.id = o.user_id;
        """
        
        sql_file = Path(temp_dir) / "pipeline.sql"
        with open(sql_file, 'w') as f:
            f.write(sql_content)
        
        # Analisa
        print("📊 Analisando pipeline...")
        analysis = system.analyze_project(temp_dir)
        
        # Acessa resultados programaticamente
        print("\n📈 Resultados da análise:")
        print(f"  • Assets encontrados: {len(system.agent.assets)}")
        print(f"  • Transformações: {len(system.agent.transformations)}")
        
        # Lista assets
        print("\n📦 Assets detectados:")
        for asset_name, asset in system.agent.assets.items():
            print(f"  • {asset_name} ({asset.type})")
        
        # Verifica dependências
        print("\n🔗 Dependências:")
        for asset_name in system.agent.assets:
            upstream = system.agent.get_upstream_dependencies(asset_name)
            downstream = system.agent.get_downstream_impact(asset_name)
            
            if upstream or downstream:
                print(f"\n  {asset_name}:")
                if upstream:
                    print(f"    ← Depende de: {', '.join(upstream)}")
                if downstream:
                    print(f"    → Impacta: {', '.join(downstream)}")
        
        # Análise de impacto programática
        print("\n💥 Simulando mudança em 'users'...")
        impact = system.agent.analyze_change_impact(['users'])
        
        print(f"  • Assets afetados downstream: {len(impact['downstream_affected'])}")
        print(f"  • Nível de risco: {impact['risk_level']}")
        
        # Acessa o grafo NetworkX diretamente
        print("\n🔧 Acessando grafo NetworkX:")
        graph = system.agent.graph
        print(f"  • Nós: {graph.number_of_nodes()}")
        print(f"  • Arestas: {graph.number_of_edges()}")
        print(f"  • É DAG? {nx.is_directed_acyclic_graph(graph)}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Executa demonstração completa
    demonstrate_lineage_analysis()
    
    # Mostra uso programático
    example_programmatic_usage()
    
    print("\n✅ Demonstração completa finalizada!")
    print("📚 Para mais informações, consulte a documentação ou use --help")
