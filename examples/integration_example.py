"""
Example demonstrating integration of all five Data Governance AI Agents.

This example shows how to use all agents together for a complete
data governance workflow:
1. Lineage Agent - Analyze pipeline code and map data flow
2. Discovery Agent - Catalog and search data assets
3. Enrichment Agent - Generate metadata descriptions and classifications
4. Classification Agent - Classify data by sensitivity (PII/PHI/PCI)
5. Quality Agent - Monitor data quality metrics

Usage:
    python examples/integration_example.py

Requirements:
    - Set OPENAI_API_KEY environment variable
    - Install all dependencies: pip install -r requirements.txt
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "lineage"))
sys.path.insert(0, str(BASE_DIR / "rag_discovery"))
sys.path.insert(0, str(BASE_DIR / "metadata_enrichment"))
sys.path.insert(0, str(BASE_DIR / "data_classification"))
sys.path.insert(0, str(BASE_DIR / "data_quality"))

# Import agents
from lineage.data_lineage_agent import DataLineageAgent
from rag_discovery.data_discovery_rag_agent import DataDiscoveryRAGAgent, TableMetadata

# Optional imports - may require additional setup
try:
    from metadata_enrichment.agent import MetadataEnrichmentAgent, EnrichmentResult
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False
    print("Note: Metadata Enrichment Agent not available. Install dependencies.")

try:
    from data_classification.agent import DataClassificationAgent, ClassificationReport
    CLASSIFICATION_AVAILABLE = True
except ImportError:
    CLASSIFICATION_AVAILABLE = False
    print("Note: Data Classification Agent not available. Install dependencies.")

try:
    from data_quality.agent import DataQualityAgent, QualityReport
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False
    print("Note: Data Quality Agent not available. Install dependencies.")


def create_sample_data():
    """Create sample CSV data for demonstration."""
    import csv

    # Create sample customer data
    sample_data = [
        {"customer_id": "C001", "name": "John Doe", "email": "john@example.com",
         "cpf": "123.456.789-00", "balance": 1500.00, "updated_at": datetime.now().isoformat()},
        {"customer_id": "C002", "name": "Jane Smith", "email": "jane@example.com",
         "cpf": "987.654.321-00", "balance": 2300.50, "updated_at": datetime.now().isoformat()},
        {"customer_id": "C003", "name": "Bob Wilson", "email": "bob@example.com",
         "cpf": "456.789.123-00", "balance": 890.25, "updated_at": datetime.now().isoformat()},
        {"customer_id": "", "name": "Anonymous", "email": "",  # Incomplete record
         "cpf": "", "balance": None, "updated_at": (datetime.now() - timedelta(days=30)).isoformat()},
    ]

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.DictWriter(temp_file, fieldnames=sample_data[0].keys())
    writer.writeheader()
    writer.writerows(sample_data)
    temp_file.close()

    return temp_file.name


def create_sample_pipeline():
    """Create sample pipeline code for lineage analysis."""
    pipeline_code = '''
-- Sample ETL Pipeline
-- Extract customer data from source
CREATE TABLE staging.customers AS
SELECT
    customer_id,
    name,
    email,
    cpf,
    balance,
    updated_at
FROM raw.customer_source
WHERE customer_id IS NOT NULL;

-- Transform and load to analytics
INSERT INTO analytics.customer_metrics
SELECT
    c.customer_id,
    c.name,
    c.balance,
    t.total_transactions,
    t.last_transaction_date
FROM staging.customers c
JOIN staging.transactions t ON c.customer_id = t.customer_id;
'''

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False)
    temp_file.write(pipeline_code)
    temp_file.close()

    return temp_file.name


def demo_lineage_agent(pipeline_path: str):
    """Demonstrate the Data Lineage Agent."""
    print("\n" + "="*60)
    print("1. DATA LINEAGE AGENT")
    print("="*60)

    agent = DataLineageAgent()
    results = agent.analyze_pipeline([pipeline_path])

    print(f"\nAnalyzed pipeline: {pipeline_path}")
    print(f"Total assets found: {results.get('metrics', {}).get('total_assets', 0)}")
    print(f"Transformations: {results.get('metrics', {}).get('total_transformations', 0)}")

    insights = results.get("insights", {})
    print(f"\nSummary: {insights.get('summary', 'N/A')}")

    # Extract table names for discovery agent
    tables_found = []
    for asset in results.get("assets", []):
        if asset.get("type") == "table":
            tables_found.append(asset.get("name", ""))

    print(f"Tables identified: {tables_found}")
    return tables_found, results


def demo_discovery_agent(tables: list):
    """Demonstrate the Data Discovery RAG Agent."""
    print("\n" + "="*60)
    print("2. DATA DISCOVERY RAG AGENT")
    print("="*60)

    # Create temp directory for chroma
    persist_dir = tempfile.mkdtemp()

    try:
        agent = DataDiscoveryRAGAgent(
            persist_directory=persist_dir,
            collection_name="demo_catalog"
        )

        # Index tables from lineage
        indexed_tables = []
        for table_name in tables[:3]:  # Limit for demo
            if table_name:
                metadata = TableMetadata(
                    name=table_name,
                    database="demo_db",
                    schema="public",
                    description=f"Table {table_name} identified from pipeline analysis",
                    tags=["auto-discovered", "lineage"]
                )
                agent.index_table(metadata)
                indexed_tables.append(metadata)
                print(f"Indexed: {table_name}")

        # Search example
        if indexed_tables:
            print("\nSearching for 'customer' tables...")
            results = agent.search("customer", n_results=3)
            for res in results:
                print(f"  Found: {res.table.name} (score: {res.relevance_score:.2%})")

        return indexed_tables

    except Exception as e:
        print(f"Discovery agent error: {e}")
        return []


def demo_enrichment_agent(data_path: str):
    """Demonstrate the Metadata Enrichment Agent."""
    print("\n" + "="*60)
    print("3. METADATA ENRICHMENT AGENT")
    print("="*60)

    if not ENRICHMENT_AVAILABLE:
        print("Skipped: Enrichment Agent not available")
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return None

    try:
        # Initialize with providers
        embedding_provider = SentenceTransformerEmbeddings()
        llm_provider = OpenAILLM(model="gpt-4o-mini")

        persist_dir = tempfile.mkdtemp()
        vector_store = ChromaStore(
            collection_name="demo_standards",
            persist_directory=persist_dir
        )

        agent = MetadataEnrichmentAgent(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_store=vector_store,
            standards_persist_dir=persist_dir
        )

        # Enrich the sample data
        print(f"\nAnalyzing: {data_path}")
        result = agent.enrich_from_csv(
            data_path,
            sample_size=100,
            additional_context="Customer financial data from banking system"
        )

        print(f"\nTable: {result.business_name}")
        print(f"Description: {result.description[:100]}...")
        print(f"Classification: {result.classification}")
        print(f"Domain: {result.domain}")
        print(f"Tags: {', '.join(result.tags)}")
        print(f"PII Detected: {'Yes' if result.has_pii else 'No'}")

        if result.pii_columns:
            print(f"PII Columns: {', '.join(result.pii_columns)}")

        print(f"\nColumns analyzed: {len(result.columns)}")
        for col in result.columns[:3]:
            pii_marker = " [PII]" if col.is_pii else ""
            print(f"  - {col.name}: {col.description[:50]}...{pii_marker}")

        return result

    except Exception as e:
        print(f"Enrichment agent error: {e}")
        return None


def demo_classification_agent(data_path: str):
    """Demonstrate the Data Classification Agent."""
    print("\n" + "="*60)
    print("4. DATA CLASSIFICATION AGENT")
    print("="*60)

    if not CLASSIFICATION_AVAILABLE:
        print("Skipped: Classification Agent not available")
        return None

    try:
        agent = DataClassificationAgent()

        print(f"\nAnalyzing: {data_path}")
        report = agent.classify_from_csv(data_path, sample_size=1000)

        print(f"\nOverall Sensitivity: {report.overall_sensitivity.upper()}")
        print(f"Columns analyzed: {report.columns_analyzed}")
        print(f"High-risk columns: {report.high_risk_count}")

        if report.pii_columns:
            print(f"\nPII Detected: {report.pii_columns}")
        if report.phi_columns:
            print(f"PHI Detected: {report.phi_columns}")
        if report.pci_columns:
            print(f"PCI Detected: {report.pci_columns}")
        if report.financial_columns:
            print(f"Financial: {report.financial_columns}")

        if report.compliance_flags:
            print(f"\nCompliance Flags:")
            for flag in report.compliance_flags:
                print(f"  - {flag}")

        return report

    except Exception as e:
        print(f"Classification agent error: {e}")
        return None


def demo_quality_agent(data_path: str):
    """Demonstrate the Data Quality Agent."""
    print("\n" + "="*60)
    print("5. DATA QUALITY AGENT")
    print("="*60)

    if not QUALITY_AVAILABLE:
        print("Skipped: Quality Agent not available")
        return None

    try:
        persist_dir = tempfile.mkdtemp()
        agent = DataQualityAgent(
            persist_dir=persist_dir,
            enable_schema_tracking=True
        )

        print(f"\nAnalyzing: {data_path}")

        # Evaluate with freshness check
        report = agent.evaluate_file(
            data_path,
            sample_size=1000,
            freshness_config={
                "timestamp_column": "updated_at",
                "sla_hours": 24,
                "max_age_hours": 48
            }
        )

        print(f"\nOverall Score: {report.overall_score:.0%}")
        print(f"Status: {report.overall_status.upper()}")
        print(f"Rows analyzed: {report.row_count}")
        print(f"Processing time: {report.processing_time_ms}ms")

        print("\nQuality Dimensions:")
        for dim_name, dim_data in report.dimensions.items():
            score = dim_data.get("score", 0)
            status = dim_data.get("status", "unknown")
            icon = "✓" if status == "passed" else "!" if status == "warning" else "✗"
            print(f"  [{icon}] {dim_name.capitalize()}: {score:.0%}")

        if report.alerts:
            print(f"\nAlerts ({len(report.alerts)}):")
            for alert in report.alerts[:3]:
                level = alert.get("level", "info")
                print(f"  [{level.upper()}] {alert.get('rule_name')}: {alert.get('message')}")

        return report

    except Exception as e:
        print(f"Quality agent error: {e}")
        return None


def demo_integrated_workflow():
    """
    Demonstrate complete integrated workflow:
    Lineage -> Discovery -> Enrichment -> Classification -> Quality
    """
    print("\n" + "="*60)
    print("INTEGRATED DATA GOVERNANCE WORKFLOW")
    print("="*60)
    print("This demo shows all 5 agents working together.\n")

    # Create sample data
    data_path = create_sample_data()
    pipeline_path = create_sample_pipeline()

    try:
        # Step 1: Analyze pipeline for lineage
        tables, lineage_results = demo_lineage_agent(pipeline_path)

        # Step 2: Catalog discovered tables
        indexed_tables = demo_discovery_agent(tables)

        # Step 3: Enrich metadata (if available)
        enrichment_result = demo_enrichment_agent(data_path)

        # Step 4: Classify data by sensitivity
        classification_report = demo_classification_agent(data_path)

        # Step 5: Evaluate quality
        quality_report = demo_quality_agent(data_path)

        # Summary
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        print(f"✓ Lineage: Analyzed {len(tables)} tables from pipeline")
        print(f"✓ Discovery: Indexed {len(indexed_tables)} tables in catalog")

        if enrichment_result:
            pii_status = "PII detected" if enrichment_result.has_pii else "No PII"
            print(f"✓ Enrichment: {enrichment_result.classification} ({pii_status})")
        else:
            print("- Enrichment: Skipped (check OPENAI_API_KEY)")

        if classification_report:
            compliance = ", ".join(classification_report.compliance_flags[:2]) if classification_report.compliance_flags else "None"
            print(f"✓ Classification: {classification_report.overall_sensitivity} (Compliance: {compliance})")
        else:
            print("- Classification: Skipped")

        if quality_report:
            print(f"✓ Quality: {quality_report.overall_score:.0%} ({quality_report.overall_status})")
        else:
            print("- Quality: Skipped")

        print("\nWorkflow complete!")

    finally:
        # Cleanup temp files
        os.unlink(data_path)
        os.unlink(pipeline_path)


if __name__ == "__main__":
    demo_integrated_workflow()
