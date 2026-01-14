#!/usr/bin/env python3
"""
Data Lineage AI Agent - Main CLI Interface
Command-line interface for data lineage analysis
"""

import click
import json
import sys
from pathlib import Path
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime

# Import core modules
from data_lineage_agent import DataLineageAgent
from visualization_engine import DataLineageVisualizer
from lineage_system import DataLineageSystem
from parsers.terraform_parser import parse_terraform_directory
from parsers.databricks_parser import parse_databricks_workspace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='2.0.0')
def cli():
    """
    Data Lineage AI Agent - Análise inteligente de linhagem de dados
    
    Use 'lineage COMMAND --help' para mais informações sobre cada comando.
    """
    pass


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), help='Arquivo de saída para resultados')
@click.option('--format', '-f', type=click.Choice(['json', 'html', 'markdown', 'csv']), 
              default='json', help='Formato de saída')
@click.option('--include-terraform', '-t', is_flag=True, help='Incluir análise Terraform')
@click.option('--include-databricks', '-d', is_flag=True, help='Incluir análise Databricks')
@click.option('--verbose', '-v', is_flag=True, help='Saída detalhada')
def analyze(files, output, format, include_terraform, include_databricks, verbose):
    """
    Analisa arquivos de pipeline e extrai linhagem de dados
    
    Exemplo:
        lineage analyze *.py *.sql -o results.json
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    click.echo(f"🔍 Analisando {len(files)} arquivo(s)...")
    
    # Initialize agent
    agent = DataLineageAgent()
    
    # Analyze files
    try:
        results = agent.analyze_pipeline(list(files))
        
        # Additional analyses
        if include_terraform:
            click.echo("🏗️ Analisando infraestrutura Terraform...")
            tf_files = [f for f in files if f.endswith('.tf') or f.endswith('.tf.json')]
            if tf_files:
                tf_dir = Path(tf_files[0]).parent
                tf_results = parse_terraform_directory(str(tf_dir))
                results['terraform'] = tf_results
        
        if include_databricks:
            click.echo("📊 Analisando notebooks Databricks...")
            db_dir = Path(files[0]).parent
            db_results = parse_databricks_workspace(str(db_dir))
            if db_results['assets']:
                results['databricks'] = db_results
        
        # Format output
        if format == 'json':
            output_data = json.dumps(results, default=str, indent=2)
        elif format == 'markdown':
            output_data = agent.generate_documentation()
        elif format == 'html':
            output_data = generate_html_report(results)
        elif format == 'csv':
            output_data = generate_csv_report(results)
        else:
            output_data = json.dumps(results, default=str, indent=2)
        
        # Save or display results
        if output:
            Path(output).write_text(output_data)
            click.echo(f"✅ Resultados salvos em: {output}")
        else:
            click.echo(output_data)
        
        # Display summary
        display_summary(results)
        
    except Exception as e:
        click.echo(f"❌ Erro na análise: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('assets', nargs=-1, required=True)
@click.option('--pipeline-dir', '-p', type=click.Path(exists=True), 
              help='Diretório do pipeline', required=True)
@click.option('--output', '-o', type=click.Path(), help='Arquivo de saída')
@click.option('--visualize', '-v', is_flag=True, help='Gerar visualização')
def impact(assets, pipeline_dir, output, visualize):
    """
    Analisa o impacto de mudanças em assets específicos
    
    Exemplo:
        lineage impact bronze.orders silver.customers -p ./pipeline
    """
    click.echo(f"💥 Analisando impacto de mudanças em {len(assets)} asset(s)...")
    
    # Analyze pipeline first
    agent = DataLineageAgent()
    pipeline_files = list(Path(pipeline_dir).glob('**/*'))
    pipeline_files = [str(f) for f in pipeline_files if f.is_file()]
    
    results = agent.analyze_pipeline(pipeline_files)
    
    # Analyze impact
    impact_results = agent.analyze_change_impact(list(assets))
    
    # Display results
    click.echo(f"\n📊 Resultados da Análise de Impacto:")
    click.echo(f"  • Nível de Risco: {impact_results['risk_level']}")
    click.echo(f"  • Assets Afetados Diretamente: {len(impact_results['directly_affected'])}")
    click.echo(f"  • Assets Downstream: {len(impact_results['downstream_affected'])}")
    click.echo(f"  • Dependências Upstream: {len(impact_results['upstream_dependencies'])}")
    
    if impact_results['affected_pipelines']:
        click.echo(f"\n⚠️ Pipelines Críticos Afetados:")
        for pipeline in impact_results['affected_pipelines'][:5]:
            click.echo(f"  • {pipeline}")
    
    if impact_results['recommendations']:
        click.echo(f"\n💡 Recomendações:")
        for rec in impact_results['recommendations']:
            click.echo(f"  {rec}")
    
    # Save results
    if output:
        Path(output).write_text(json.dumps(impact_results, indent=2))
        click.echo(f"\n✅ Resultados salvos em: {output}")
    
    # Generate visualization
    if visualize:
        generate_impact_visualization(results['graph'], assets, impact_results)


@cli.command()
@click.argument('old-dir', type=click.Path(exists=True))
@click.argument('new-dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Arquivo de saída')
@click.option('--detailed', '-d', is_flag=True, help='Comparação detalhada')
def compare(old_dir, new_dir, output, detailed):
    """
    Compara duas versões de pipeline
    
    Exemplo:
        lineage compare ./v1 ./v2 -o comparison.json
    """
    click.echo(f"🔄 Comparando pipelines...")
    click.echo(f"  • Versão antiga: {old_dir}")
    click.echo(f"  • Versão nova: {new_dir}")
    
    # Get files from both directories
    old_files = [str(f) for f in Path(old_dir).glob('**/*') if f.is_file()]
    new_files = [str(f) for f in Path(new_dir).glob('**/*') if f.is_file()]
    
    # Compare versions
    agent = DataLineageAgent()
    comparison = agent.compare_versions(old_files, new_files)
    
    # Display results
    click.echo(f"\n📊 Resultados da Comparação:")
    click.echo(f"  • Assets Adicionados: {len(comparison['added_assets'])}")
    click.echo(f"  • Assets Removidos: {len(comparison['removed_assets'])}")
    click.echo(f"  • Conexões Adicionadas: {len(comparison['added_connections'])}")
    click.echo(f"  • Conexões Removidas: {len(comparison['removed_connections'])}")
    
    if detailed:
        if comparison['added_assets']:
            click.echo(f"\n✅ Assets Adicionados:")
            for asset in comparison['added_assets'][:10]:
                click.echo(f"  + {asset}")
        
        if comparison['removed_assets']:
            click.echo(f"\n❌ Assets Removidos:")
            for asset in comparison['removed_assets'][:10]:
                click.echo(f"  - {asset}")
        
        if comparison.get('risk_assessment', {}).get('removed_assets_impact'):
            click.echo(f"\n⚠️ Impacto das Remoções:")
            for asset, impact in list(comparison['risk_assessment']['removed_assets_impact'].items())[:5]:
                click.echo(f"  • {asset}: afeta {len(impact)} assets")
    
    # Save results
    if output:
        Path(output).write_text(json.dumps(comparison, indent=2))
        click.echo(f"\n✅ Resultados salvos em: {output}")


@cli.command()
@click.option('--port', '-p', default=8501, help='Porta para o servidor web')
@click.option('--host', '-h', default='localhost', help='Host para o servidor')
@click.option('--debug', is_flag=True, help='Modo debug')
def web(port, host, debug):
    """
    Inicia a interface web interativa
    
    Exemplo:
        lineage web -p 8080
    """
    click.echo(f"🌐 Iniciando interface web...")
    click.echo(f"  • URL: http://{host}:{port}")
    click.echo(f"  • Pressione Ctrl+C para parar")
    
    import subprocess
    
    try:
        cmd = ['streamlit', 'run', 'app.py', 
               '--server.port', str(port),
               '--server.address', host]
        
        if not debug:
            cmd.extend(['--server.headless', 'true'])
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        click.echo("\n👋 Servidor encerrado.")
    except Exception as e:
        click.echo(f"❌ Erro ao iniciar servidor: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pipeline-dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Arquivo de documentação')
@click.option('--format', '-f', type=click.Choice(['markdown', 'html', 'pdf']), 
              default='markdown', help='Formato da documentação')
@click.option('--include-graphs', '-g', is_flag=True, help='Incluir visualizações')
def document(pipeline_dir, output, format, include_graphs):
    """
    Gera documentação automática do pipeline
    
    Exemplo:
        lineage document ./pipeline -o docs.md
    """
    click.echo(f"📝 Gerando documentação do pipeline...")
    
    # Analyze pipeline
    agent = DataLineageAgent()
    pipeline_files = [str(f) for f in Path(pipeline_dir).glob('**/*') if f.is_file()]
    results = agent.analyze_pipeline(pipeline_files)
    
    # Generate documentation
    doc = agent.generate_documentation()
    
    if include_graphs:
        # Add graph visualizations to documentation
        doc += "\n\n## 📊 Visualizações\n\n"
        doc += "### Grafo de Linhagem\n"
        doc += "![Lineage Graph](lineage_graph.png)\n\n"
        
        # Generate graph image
        visualizer = DataLineageVisualizer(results['graph'])
        fig = visualizer.visualize_force_directed()
        fig.write_image("lineage_graph.png")
    
    # Convert format if needed
    if format == 'html':
        import markdown
        doc = markdown.markdown(doc, extensions=['tables', 'fenced_code'])
    elif format == 'pdf':
        # Would require additional PDF library
        click.echo("⚠️ Conversão para PDF requer bibliotecas adicionais")
    
    # Save documentation
    if output:
        Path(output).write_text(doc)
        click.echo(f"✅ Documentação salva em: {output}")
    else:
        click.echo(doc)


@cli.command()
@click.option('--example', '-e', 
              type=click.Choice(['ecommerce', 'financial', 'streaming', 'ml']),
              help='Tipo de exemplo')
def init(example):
    """
    Inicializa um projeto de exemplo
    
    Exemplo:
        lineage init --example ecommerce
    """
    click.echo(f"🚀 Inicializando projeto de exemplo...")
    
    from example_usage import create_sample_pipeline
    import tempfile
    import shutil
    
    # Create example directory
    example_dir = Path(f"lineage_example_{example or 'default'}")
    
    if example_dir.exists():
        click.confirm(f"Diretório {example_dir} já existe. Sobrescrever?", abort=True)
        shutil.rmtree(example_dir)
    
    example_dir.mkdir(parents=True)
    
    # Create example files based on type
    if example == 'ecommerce':
        create_ecommerce_example(example_dir)
    elif example == 'financial':
        create_financial_example(example_dir)
    elif example == 'streaming':
        create_streaming_example(example_dir)
    elif example == 'ml':
        create_ml_example(example_dir)
    else:
        # Default example
        create_default_example(example_dir)
    
    click.echo(f"✅ Projeto criado em: {example_dir}/")
    click.echo(f"\n📝 Próximos passos:")
    click.echo(f"  1. cd {example_dir}")
    click.echo(f"  2. lineage analyze *.py *.sql")
    click.echo(f"  3. lineage web")


def display_summary(results):
    """Display analysis summary"""
    click.echo(f"\n📊 Resumo da Análise:")
    click.echo(f"  • Total de Assets: {len(results.get('assets', []))}")
    click.echo(f"  • Transformações: {len(results.get('transformations', []))}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        if 'graph_metrics' in metrics:
            gm = metrics['graph_metrics']
            click.echo(f"  • Nós no Grafo: {gm.get('total_nodes', 0)}")
            click.echo(f"  • Conexões: {gm.get('total_edges', 0)}")
        
        if 'complexity_metrics' in metrics:
            cm = metrics['complexity_metrics']
            if cm.get('has_cycles'):
                click.echo(f"  ⚠️ Ciclos detectados: {len(cm.get('cycles', []))}")


def generate_html_report(results):
    """Generate HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Lineage Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .asset {{ background: #e8f4f8; padding: 5px; margin: 5px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>📊 Data Lineage Analysis Report</h1>
        <div class="metric">
            <h2>Summary</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Assets: {len(results.get('assets', []))}</p>
            <p>Total Transformations: {len(results.get('transformations', []))}</p>
        </div>
        
        <h2>Assets</h2>
        <table>
            <tr><th>Name</th><th>Type</th><th>Source File</th></tr>
    """
    
    for asset in results.get('assets', [])[:20]:
        html += f"""
            <tr>
                <td>{asset.name}</td>
                <td>{asset.type}</td>
                <td>{asset.source_file}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    return html


def generate_csv_report(results):
    """Generate CSV report"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write assets
    writer.writerow(['Asset Name', 'Type', 'Source File', 'Line Number'])
    for asset in results.get('assets', []):
        writer.writerow([asset.name, asset.type, asset.source_file, asset.line_number])
    
    return output.getvalue()


def generate_impact_visualization(graph, changed_assets, impact_results):
    """Generate and save impact visualization"""
    visualizer = DataLineageVisualizer(graph)
    
    # Highlight affected nodes
    highlight_nodes = list(changed_assets) + list(impact_results['downstream_affected'])
    
    fig = visualizer.visualize_force_directed(
        highlight_nodes=highlight_nodes,
        title="Impact Analysis Visualization"
    )
    
    # Save as HTML
    output_file = f"impact_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(output_file)
    
    click.echo(f"📊 Visualização salva em: {output_file}")


def create_ecommerce_example(directory):
    """Create e-commerce pipeline example"""
    files = {
        'extract_orders.py': """
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("OrdersETL").getOrCreate()

# Extract orders from source
orders_df = spark.read.format("delta").load("bronze.orders")
customers_df = spark.table("silver.customers")
products_df = spark.read.parquet("s3://data-lake/products.parquet")

# Join and transform
enriched_orders = orders_df.join(customers_df, "customer_id")
enriched_orders = enriched_orders.join(products_df, "product_id")

# Write to gold layer
enriched_orders.write.format("delta").mode("overwrite").saveAsTable("gold.enriched_orders")
""",
        'aggregate_sales.sql': """
-- Create daily sales summary
CREATE OR REPLACE TABLE gold.daily_sales AS
SELECT 
    DATE(order_date) as sale_date,
    product_category,
    customer_segment,
    COUNT(*) as order_count,
    SUM(total_amount) as total_sales,
    AVG(total_amount) as avg_order_value
FROM gold.enriched_orders
GROUP BY 1, 2, 3;

-- Update customer lifetime value
MERGE INTO gold.customer_metrics AS target
USING (
    SELECT 
        customer_id,
        COUNT(*) as total_orders,
        SUM(total_amount) as lifetime_value,
        MAX(order_date) as last_order_date
    FROM gold.enriched_orders
    GROUP BY customer_id
) AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
""",
        'recommendations.py': """
import mlflow
from pyspark.ml.recommendation import ALS

# Load customer purchase history
purchases = spark.table("gold.enriched_orders")

# Prepare data for ALS
ratings = purchases.select("customer_id", "product_id", "rating")

# Train recommendation model
als = ALS(userCol="customer_id", itemCol="product_id", ratingCol="rating")
model = als.fit(ratings)

# Generate recommendations
recommendations = model.recommendForAllUsers(10)

# Save recommendations
recommendations.write.format("delta").mode("overwrite").saveAsTable("gold.product_recommendations")

# Log model
mlflow.spark.log_model(model, "recommendation_model")
""",
        'infrastructure.tf': """
resource "aws_s3_bucket" "data_lake" {
  bucket = "ecommerce-data-lake"
}

resource "aws_glue_catalog_database" "analytics" {
  name = "ecommerce_analytics"
}

resource "databricks_cluster" "etl_cluster" {
  cluster_name = "ecommerce-etl"
  spark_version = "11.3.x-scala2.12"
  node_type_id = "i3.xlarge"
  num_workers = 4
}

resource "databricks_job" "daily_pipeline" {
  name = "Daily Sales Pipeline"
  
  task {
    task_key = "extract_orders"
    notebook_task {
      notebook_path = "/pipelines/extract_orders"
    }
  }
  
  task {
    task_key = "aggregate_sales"
    depends_on {
      task_key = "extract_orders"
    }
    sql_task {
      warehouse_id = databricks_sql_endpoint.analytics.id
      query {
        query_text = file("aggregate_sales.sql")
      }
    }
  }
}
"""
    }
    
    for filename, content in files.items():
        (directory / filename).write_text(content)


def create_financial_example(directory):
    """Create financial pipeline example"""
    files = {
        'risk_analytics.py': """
import pandas as pd
from pyspark.sql import functions as F

# Load trading data
trades = spark.table("bronze.trades")
positions = spark.table("bronze.positions")
market_data = spark.read.format("delta").load("bronze.market_data")

# Calculate risk metrics
risk_metrics = trades.join(positions, "account_id") \
    .join(market_data, "symbol") \
    .groupBy("account_id", "symbol") \
    .agg(
        F.sum("exposure").alias("total_exposure"),
        F.stddev("returns").alias("volatility"),
        F.mean("returns").alias("expected_return")
    )

# Calculate VaR
risk_metrics = risk_metrics.withColumn(
    "var_95",
    F.col("expected_return") - 1.645 * F.col("volatility")
)

risk_metrics.write.format("delta").mode("overwrite").saveAsTable("gold.risk_metrics")
""",
        'compliance_check.sql': """
-- Check for regulatory compliance
CREATE OR REPLACE VIEW gold.compliance_violations AS
SELECT 
    account_id,
    trade_date,
    symbol,
    trade_amount,
    'LARGE_TRADE' as violation_type
FROM bronze.trades
WHERE trade_amount > 1000000
   OR trade_count > 100

UNION ALL

SELECT 
    account_id,
    trade_date,
    symbol,
    exposure_amount,
    'EXPOSURE_LIMIT' as violation_type
FROM gold.risk_metrics
WHERE total_exposure > account_limit;
"""
    }
    
    for filename, content in files.items():
        (directory / filename).write_text(content)


def create_streaming_example(directory):
    """Create streaming pipeline example"""
    files = {
        'streaming_ingestion.py': """
from pyspark.sql.functions import *

# Read streaming data from Kafka
stream_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Parse JSON events
parsed_stream = stream_df \
    .select(from_json(col("value").cast("string"), event_schema).alias("data")) \
    .select("data.*")

# Aggregate in windows
windowed_counts = parsed_stream \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("event_type")
    ) \
    .count()

# Write to Delta Lake
query = windowed_counts \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoint/events") \
    .trigger(processingTime='10 seconds') \
    .table("silver.event_aggregates")
"""
    }
    
    for filename, content in files.items():
        (directory / filename).write_text(content)


def create_ml_example(directory):
    """Create ML pipeline example"""
    files = {
        'feature_engineering.py': """
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Load raw data
raw_data = spark.table("bronze.user_activity")

# Feature engineering
features = raw_data.groupBy("user_id").agg(
    count("event").alias("event_count"),
    avg("duration").alias("avg_duration"),
    max("timestamp").alias("last_activity")
)

# Assemble features
assembler = VectorAssembler(
    inputCols=["event_count", "avg_duration"],
    outputCol="features"
)

features_df = assembler.transform(features)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(features_df)
scaled_features = scaler_model.transform(features_df)

# Save feature store
scaled_features.write.format("delta").mode("overwrite").saveAsTable("gold.ml_features")
"""
    }
    
    for filename, content in files.items():
        (directory / filename).write_text(content)


def create_default_example(directory):
    """Create default example"""
    files = {
        'etl_pipeline.py': """
import pandas as pd

# Extract
source_data = pd.read_csv("input_data.csv")

# Transform
transformed_data = source_data.groupby('category').agg({
    'amount': 'sum',
    'quantity': 'mean'
})

# Load
transformed_data.to_parquet("output_data.parquet")
""",
        'analysis.sql': """
-- Create summary table
CREATE TABLE IF NOT EXISTS summary AS
SELECT 
    category,
    SUM(amount) as total_amount,
    COUNT(*) as record_count
FROM raw_data
GROUP BY category;
"""
    }
    
    for filename, content in files.items():
        (directory / filename).write_text(content)


if __name__ == '__main__':
    cli()
