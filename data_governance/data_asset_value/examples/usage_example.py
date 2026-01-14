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
"""
Example usage of the Data Asset Value Scanner Agent

This example demonstrates how to:
1. Load query logs and configurations
2. Analyze data asset value
3. Integrate with Data Lineage Agent output
4. Generate comprehensive value reports
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_asset_value import DataAssetValueAgent, AssetValueReport


def load_json_file(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Get the examples directory
    examples_dir = os.path.dirname(os.path.abspath(__file__))

    # Load sample data
    print("Loading sample data...")
    query_logs = load_json_file(os.path.join(examples_dir, 'sample_query_logs.json'))
    data_products = load_json_file(os.path.join(examples_dir, 'data_products_config.json'))
    asset_metadata = load_json_file(os.path.join(examples_dir, 'asset_metadata.json'))

    print(f"  - Query logs: {len(query_logs)} entries")
    print(f"  - Data products: {len(data_products)} products")
    print(f"  - Asset metadata: {len(asset_metadata)} assets")

    # Initialize the agent with custom weights
    print("\nInitializing Data Asset Value Agent...")
    agent = DataAssetValueAgent(
        weights={
            'usage': 0.30,        # 30% weight on usage frequency
            'joins': 0.25,        # 25% weight on join relationships
            'lineage': 0.20,      # 20% weight on lineage impact
            'data_product': 0.25  # 25% weight on data product importance
        },
        time_range_days=30,
        persist_dir=os.path.join(examples_dir, 'output')
    )

    # Run the analysis
    print("\nAnalyzing data asset value...")
    report = agent.analyze_from_query_logs(
        query_logs=query_logs,
        lineage_data=None,  # Would integrate with Lineage Agent output
        data_product_config=data_products,
        asset_metadata=asset_metadata,
        time_range_days=30
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Assets analyzed: {report.assets_analyzed}")
    print(f"Query logs processed: {report.query_logs_processed}")
    print(f"Time range: {report.time_range_days} days")

    # Print top value assets
    print("\n" + "-" * 40)
    print("TOP VALUE ASSETS")
    print("-" * 40)
    for i, score in enumerate(report.asset_scores[:10], 1):
        print(f"{i:2}. {score.asset_name:25} | Score: {score.overall_value_score:5.1f} | Category: {score.value_category}")

    # Print critical assets
    if report.critical_assets:
        print("\n" + "-" * 40)
        print("CRITICAL ASSETS")
        print("-" * 40)
        for asset in report.critical_assets:
            print(f"  - {asset}")

    # Print hub assets
    if report.hub_assets:
        print("\n" + "-" * 40)
        print("HUB ASSETS (High Connectivity)")
        print("-" * 40)
        for asset in report.hub_assets:
            print(f"  - {asset}")

    # Print orphan assets
    if report.orphan_assets:
        print("\n" + "-" * 40)
        print("ORPHAN ASSETS (No Usage Detected)")
        print("-" * 40)
        for asset in report.orphan_assets[:5]:
            print(f"  - {asset}")

    # Print recommendations
    if report.recommendations:
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

    # Compare specific assets
    print("\n" + "-" * 40)
    print("ASSET COMPARISON: customers vs orders vs products")
    print("-" * 40)
    comparison = agent.compare_assets(['customers', 'orders', 'products'], report)
    for item in comparison:
        print(f"\n{item['asset']}:")
        print(f"  Overall Score: {item['overall_score']:.1f} ({item['category']})")
        print(f"  Usage: {item['usage']:.1f} | Joins: {item['joins']:.1f}")
        print(f"  Lineage: {item['lineage']:.1f} | Data Products: {item['data_products']:.1f}")
        print(f"  Business Impact: {item['business_impact']:.1f}")

    # Save reports
    print("\n" + "-" * 40)
    print("SAVING REPORTS")
    print("-" * 40)

    # Save JSON report
    json_path = os.path.join(examples_dir, 'output', 'value_report.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(report.to_json())
    print(f"  JSON report saved to: {json_path}")

    # Save Markdown report
    md_path = os.path.join(examples_dir, 'output', 'value_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report.to_markdown())
    print(f"  Markdown report saved to: {md_path}")

    print("\nAnalysis complete!")


def example_with_lineage_agent():
    """
    Example showing integration with Data Lineage Agent

    This function demonstrates how to combine query log analysis
    with pipeline lineage data for comprehensive value assessment.
    """
    from lineage.data_lineage_agent import DataLineageAgent

    # Initialize agents
    lineage_agent = DataLineageAgent()
    value_agent = DataAssetValueAgent()

    # Analyze pipeline files with lineage agent
    pipeline_files = [
        'path/to/etl_pipeline.py',
        'path/to/transform.sql',
    ]
    lineage_data = lineage_agent.analyze_pipeline(pipeline_files)

    # Load query logs
    query_logs = [
        {
            'query': 'SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id',
            'timestamp': '2024-12-15T10:00:00Z',
            'user': 'analyst',
            'data_product': 'customer_analytics'
        }
    ]

    # Analyze with both sources
    report = value_agent.analyze_from_query_logs(
        query_logs=query_logs,
        lineage_data=lineage_data,  # Pass lineage data for impact analysis
        data_product_config=[
            {
                'name': 'customer_analytics',
                'assets': ['customers', 'orders'],
                'critical_assets': ['customers'],
                'consumers': 10,
                'revenue_impact': 'high'
            }
        ]
    )

    return report


if __name__ == '__main__':
    main()
