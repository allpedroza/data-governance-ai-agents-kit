"""
Basic usage example for Data Quality Agent

Demonstrates:
1. Quality evaluation on a CSV file
2. Freshness/SLA monitoring
3. Schema drift detection
4. Quality rules and alerts
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from data_quality.agent import DataQualityAgent
from data_quality.rules import QualityRule, RuleSet, AlertLevel


def create_sample_data():
    """Create sample CSV for testing"""
    import csv
    from datetime import datetime, timedelta

    now = datetime.now()

    rows = [
        ["id", "name", "email", "status", "amount", "updated_at"],
    ]

    for i in range(100):
        # Some nulls
        name = f"Customer {i}" if i % 10 != 0 else ""
        email = f"customer{i}@email.com" if i % 15 != 0 else ""

        # Some duplicates
        customer_id = i if i % 20 != 0 else i - 1

        # Valid/invalid emails
        if i % 25 == 0:
            email = "invalid-email"

        # Timestamps (some stale)
        if i < 80:
            ts = now - timedelta(hours=i % 5)
        else:
            ts = now - timedelta(days=3)  # Stale data

        rows.append([
            customer_id,
            name,
            email,
            "active" if i % 3 != 0 else "inactive",
            round(100 + i * 10.5, 2),
            ts.isoformat()
        ])

    sample_file = Path("./sample_quality_data.csv")
    with open(sample_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return str(sample_file)


def main():
    print("=" * 60)
    print("Data Quality Agent - Basic Usage Example")
    print("=" * 60)

    # Initialize agent
    agent = DataQualityAgent(
        persist_dir="./quality_example_data",
        enable_schema_tracking=True
    )

    # Create sample data
    print("\n1. Creating sample data...")
    sample_file = create_sample_data()
    print(f"   Created: {sample_file}")

    # Basic quality evaluation
    print("\n2. Running quality evaluation...")
    report = agent.evaluate_file(
        sample_file,
        sample_size=100,
        completeness_config={"threshold": 0.95},
        uniqueness_config={"column": "id", "threshold": 1.0},
        validity_configs=[{
            "column": "email",
            "pattern_name": "email",
            "threshold": 0.9
        }],
        freshness_config={
            "timestamp_column": "updated_at",
            "sla_hours": 24,
            "max_age_hours": 48
        }
    )

    # Print results
    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)
    print(f"\nSource: {report.source_name}")
    print(f"Overall Score: {report.overall_score:.2%}")
    print(f"Status: {report.overall_status.upper()}")
    print(f"Rows: {report.row_count:,}")
    print(f"Processing Time: {report.processing_time_ms}ms")

    print("\nDimensions:")
    for dim_name, dim_data in report.dimensions.items():
        score = dim_data.get("score", 0)
        status = dim_data.get("status", "unknown")
        icon = "✓" if status == "passed" else "⚠" if status == "warning" else "✗"
        print(f"  {icon} {dim_name.capitalize()}: {score:.2%} ({status})")

    if report.alerts:
        print("\nAlerts:")
        for alert in report.alerts:
            level = alert.get("level", "info")
            icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🔵"
            print(f"  {icon} {alert.get('rule_name')}: {alert.get('message')}")

    if report.schema_drift and report.schema_drift.get("has_drift"):
        print("\nSchema Changes:")
        for change in report.schema_drift.get("changes", []):
            print(f"  - {change.get('message')}")

    # Add custom rules
    print("\n3. Adding custom quality rules...")
    agent.add_rule(QualityRule(
        name="email_completeness_critical",
        dimension="completeness",
        table_name=Path(sample_file).stem,
        column="email",
        threshold=0.98,
        alert_level=AlertLevel.CRITICAL,
        description="Email must be 98% complete"
    ))

    # Re-evaluate with rules
    print("\n4. Re-evaluating with custom rules...")
    report2 = agent.evaluate_file(sample_file)

    print(f"\nNew alerts after adding rules:")
    for alert in report2.alerts:
        level = alert.get("level", "info")
        icon = "🔴" if level == "critical" else "🟡"
        print(f"  {icon} {alert.get('rule_name')}: {alert.get('message')}")

    # Schema drift detection
    print("\n5. Checking schema drift...")
    from data_quality.connectors import CSVConnector
    connector = CSVConnector(sample_file)
    drift_report = agent.check_schema_drift(connector)

    if drift_report.has_drift:
        print(f"   Schema changes detected: {drift_report.summary}")
    else:
        print("   No schema changes detected")

    # Export report
    print("\n6. Exporting reports...")
    agent.export_report(report, "./quality_report.json", format="json")
    agent.export_report(report, "./quality_report.md", format="markdown")
    print("   Exported: quality_report.json, quality_report.md")

    # Statistics
    print("\n7. Agent statistics:")
    stats = agent.get_statistics()
    print(f"   Rules: {stats.get('rules', {}).get('total_rules', 0)}")
    print(f"   Reports: {stats.get('reports_generated', 0)}")

    # Cleanup
    Path(sample_file).unlink()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
