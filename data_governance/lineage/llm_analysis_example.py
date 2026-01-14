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
Example: Using LLM-Enhanced Graph Analysis for Automatic Summaries
Demonstrates natural language explanations and recommendations
"""

import networkx as nx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_lineage_agent import DataLineageAgent
from visualization_engine import DataLineageVisualizer
from llm_graph_analyzer import GraphLLMAnalyzer
import tempfile
from pathlib import Path


def create_sample_pipeline():
    """Create a sample data pipeline for demonstration"""
    
    temp_dir = tempfile.mkdtemp(prefix="llm_demo_")
    print(f"📁 Creating sample pipeline in: {temp_dir}")
    
    # 1. ETL Python Script
    etl_script = Path(temp_dir) / "etl_pipeline.py"
    etl_script.write_text("""
import pandas as pd
from sqlalchemy import create_engine

# Extract from multiple sources
customers = pd.read_csv('s3://data-lake/customers.csv')
orders = pd.read_parquet('s3://data-lake/orders.parquet')
products = pd.read_json('api://products-service/v1/products')

# Transform - Join and aggregate
order_details = orders.merge(customers, on='customer_id')
order_details = order_details.merge(products, on='product_id')

# Calculate metrics
daily_sales = order_details.groupby('date').agg({
    'amount': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'order_count'})

# Load to warehouse
engine = create_engine('postgresql://warehouse:5432/analytics')
daily_sales.to_sql('fact_daily_sales', engine, if_exists='replace')
order_details.to_sql('fact_orders', engine, if_exists='append')
""")
    
    # 2. SQL Transformations
    sql_script = Path(temp_dir) / "transformations.sql"
    sql_script.write_text("""
-- Create customer segments
CREATE TABLE dim_customer_segments AS
SELECT 
    c.customer_id,
    c.customer_name,
    CASE 
        WHEN total_spent > 10000 THEN 'Premium'
        WHEN total_spent > 5000 THEN 'Gold'
        WHEN total_spent > 1000 THEN 'Silver'
        ELSE 'Bronze'
    END as segment,
    total_spent,
    order_count
FROM customers c
JOIN (
    SELECT 
        customer_id,
        SUM(amount) as total_spent,
        COUNT(*) as order_count
    FROM fact_orders
    GROUP BY customer_id
) o ON c.customer_id = o.customer_id;

-- Create product performance view
CREATE VIEW v_product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(o.quantity) as units_sold,
    SUM(o.amount) as revenue,
    AVG(o.amount/o.quantity) as avg_price
FROM products p
JOIN fact_orders o ON p.product_id = o.product_id
GROUP BY p.product_id, p.product_name, p.category;

-- Update aggregate tables
INSERT INTO monthly_sales_summary
SELECT 
    DATE_TRUNC('month', date) as month,
    SUM(amount) as total_sales,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) as transaction_count
FROM fact_orders
WHERE date >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY 1;
""")
    
    # 3. Complex pipeline with issues
    complex_pipeline = Path(temp_dir) / "complex_pipeline.py"
    complex_pipeline.write_text("""
# Complex pipeline with various patterns
import pandas as pd

# Multiple data sources (potential bottleneck)
source1 = pd.read_csv('raw_data1.csv')
source2 = pd.read_csv('raw_data2.csv')
source3 = pd.read_csv('raw_data3.csv')
source4 = pd.read_csv('raw_data4.csv')
source5 = pd.read_csv('raw_data5.csv')

# All converge to single processing point (bottleneck)
combined = pd.concat([source1, source2, source3, source4, source5])

# Long chain of transformations (critical path)
step1 = combined.drop_duplicates()
step2 = step1.fillna(method='ffill')
step3 = step2[step2['amount'] > 0]
step4 = step3.groupby('category').sum()
step5 = step4.reset_index()
step6 = step5.sort_values('amount', ascending=False)
step7 = step6.head(100)

# Single point of failure - no error handling
critical_transform = step7.pivot_table(index='category', values='amount')

# Write to multiple outputs
critical_transform.to_csv('output1.csv')
critical_transform.to_parquet('output2.parquet')
critical_transform.to_excel('output3.xlsx')

# Orphaned code (not connected to main flow)
unused_data = pd.read_csv('archived_data.csv')
""")
    
    return temp_dir


def demonstrate_llm_analysis():
    """Demonstrate LLM-enhanced graph analysis"""
    
    print("=" * 60)
    print("🤖 LLM-Enhanced Data Lineage Analysis Demo")
    print("=" * 60)
    
    # Create sample pipeline
    pipeline_dir = create_sample_pipeline()
    
    # Analyze with Data Lineage Agent
    print("\n📊 Analyzing pipeline structure...")
    agent = DataLineageAgent()
    
    files = list(Path(pipeline_dir).glob('*.py'))
    files.extend(list(Path(pipeline_dir).glob('*.sql')))
    
    analysis = agent.analyze_pipeline([str(f) for f in files])
    
    print(f"✅ Found {len(analysis['assets'])} data assets")
    print(f"✅ Found {len(analysis['transformations'])} transformations")
    
    # Create visualizer with the graph
    print("\n🎨 Creating visualizations with LLM analysis...")
    viz = DataLineageVisualizer(analysis['graph'])
    
    # Get LLM-generated summary
    print("\n" + "=" * 60)
    print("📝 NATURAL LANGUAGE SUMMARY")
    print("=" * 60)
    summary = viz.get_llm_summary()
    print(summary)
    
    # Get insights
    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS DETECTED")
    print("=" * 60)
    insights = viz.get_insights()
    
    for i, insight in enumerate(insights[:5], 1):
        print(f"\n{i}. {insight['title']} [{insight['severity']}]")
        print(f"   {insight['description']}")
        print(f"   Affected: {len(insight.get('affected_nodes', []))} nodes")
        print(f"   Action: {insight['recommendation']}")
    
    # Get recommendations
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    recommendations = viz.get_recommendations()
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. {rec['title']} (Priority: {rec['priority']})")
        print(f"   {rec['description']}")
        print("   Actions:")
        for action in rec.get('actions', [])[:3]:
            print(f"   - {action}")
        print(f"   Impact: {rec.get('impact', 'N/A')}")
        print(f"   Effort: {rec.get('effort', 'N/A')}")
    
    # Generate detailed report
    print("\n" + "=" * 60)
    print("📄 DETAILED NATURAL LANGUAGE REPORT")
    print("=" * 60)
    report = viz.get_natural_language_report()
    
    # Print first part of report
    lines = report.split('\n')
    for line in lines[:30]:  # First 30 lines
        print(line)
    
    if len(lines) > 30:
        print("\n... (report continues)")
    
    # Create interactive visualization
    print("\n🌐 Generating interactive HTML report...")
    fig = viz.visualize_force_directed(
        title="Pipeline Analysis with AI Insights"
    )
    
    output_file = Path(pipeline_dir) / "llm_analysis_report.html"
    viz.export_to_html(fig, str(output_file))
    
    print(f"✅ Report saved to: {output_file}")
    
    # Demonstrate specific analyses
    print("\n" + "=" * 60)
    print("🔍 SPECIFIC PATTERN ANALYSIS")
    print("=" * 60)
    
    # Analyze with standalone analyzer for more details
    analyzer = GraphLLMAnalyzer()
    detailed_analysis = analyzer.analyze_graph(analysis['graph'])
    
    # Check for specific patterns
    if detailed_analysis.get('subgraph_summaries'):
        print("\n📊 Component Analysis:")
        for component in detailed_analysis['subgraph_summaries'][:3]:
            print(f"\nComponent: {component['name']}")
            print(f"  Purpose: {component['purpose']}")
            print(f"  Size: {component['node_count']} nodes, {component['edge_count']} edges")
            print(f"  Data Flow: {component['data_flow']}")
    
    # Metrics summary
    print("\n" + "=" * 60)
    print("📈 PIPELINE HEALTH METRICS")
    print("=" * 60)
    
    if viz.metrics:
        print(f"Total Nodes: {viz.metrics.get('total_nodes', 0)}")
        print(f"Total Edges: {viz.metrics.get('total_edges', 0)}")
        print(f"Density: {viz.metrics.get('density', 0):.3f}")
        print(f"Is DAG: {'Yes' if viz.metrics.get('is_dag', False) else 'No'}")
        print(f"Components: {viz.metrics.get('connected_components', 0)}")
        print(f"Critical Path: {viz.metrics.get('longest_path_length', 'N/A')} steps")
    
    return pipeline_dir, analysis, viz


def demonstrate_impact_with_llm():
    """Demonstrate impact analysis with LLM explanations"""
    
    print("\n" + "=" * 60)
    print("💥 IMPACT ANALYSIS WITH LLM EXPLANATIONS")
    print("=" * 60)
    
    # Create simple graph for impact demo
    G = nx.DiGraph()
    G.add_edges_from([
        ('raw_customers', 'clean_customers'),
        ('raw_orders', 'clean_orders'),
        ('clean_customers', 'customer_segments'),
        ('clean_orders', 'order_metrics'),
        ('customer_segments', 'customer_report'),
        ('order_metrics', 'sales_dashboard'),
        ('customer_segments', 'ml_features'),
        ('order_metrics', 'ml_features'),
        ('ml_features', 'prediction_model'),
        ('prediction_model', 'recommendations'),
        ('recommendations', 'email_campaigns')
    ])
    
    # Analyze with LLM
    analyzer = GraphLLMAnalyzer()
    
    # Simulate change in 'clean_customers'
    print("\n🔄 Simulating schema change in 'clean_customers' table...")
    
    # Get downstream impact
    downstream = list(nx.descendants(G, 'clean_customers'))
    print(f"\n📊 Direct graph analysis:")
    print(f"  - Downstream nodes affected: {len(downstream)}")
    print(f"  - Affected: {', '.join(downstream)}")
    
    # Get LLM interpretation
    analysis = analyzer.analyze_graph(G)
    
    print("\n🤖 LLM Analysis:")
    
    # Find relevant insights
    for insight in analysis['insights']:
        if 'clean_customers' in insight.get('affected_nodes', []) or \
           any(node in downstream for node in insight.get('affected_nodes', [])):
            print(f"\n  {insight['title']}:")
            print(f"  {insight['description']}")
            print(f"  Recommendation: {insight['recommendation']}")
    
    # Generate specific impact explanation
    print("\n📝 Natural Language Impact Explanation:")
    print("""
    The schema change in 'clean_customers' will cascade through the entire analytics 
    pipeline. This table serves as a foundational data source that feeds into:
    
    1. Customer Segmentation: Business logic for customer classification will need 
       validation to ensure compatibility with the new schema.
    
    2. ML Feature Engineering: The feature pipeline combines customer data with orders,
       so both schema compatibility and feature calculation logic must be verified.
    
    3. Downstream Models: The prediction model relies on specific features derived
       from customer data. Schema changes could break model scoring.
    
    4. Business Applications: Email campaigns depend on customer segments and 
       predictions. Any data quality issues will directly impact customer communications.
    
    Risk Level: HIGH - This change affects both analytical and operational systems.
    
    Recommended Actions:
    - Create a staging environment to test the schema change
    - Run data quality checks at each transformation step
    - Validate ML model performance with new schema
    - Implement gradual rollout with monitoring
    - Prepare rollback plan if issues detected
    """)


if __name__ == "__main__":
    # Run main demo
    pipeline_dir, analysis, viz = demonstrate_llm_analysis()
    
    # Run impact analysis demo
    demonstrate_impact_with_llm()
    
    print("\n" + "=" * 60)
    print("✅ LLM Analysis Demo Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved in: {pipeline_dir}")
    print("\n🎉 The system now provides:")
    print("  • Natural language summaries of graph structure")
    print("  • Automatic detection of patterns and issues")
    print("  • Actionable recommendations with priority")
    print("  • Component-level analysis and explanations")
    print("  • Enhanced HTML reports with AI insights")
