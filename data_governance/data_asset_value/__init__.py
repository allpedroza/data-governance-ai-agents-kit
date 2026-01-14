"""
Data Asset Value Scanner Agent

This module provides tools for analyzing and scoring data asset value
based on usage patterns, join relationships, lineage impact, and
data product integration.

Main Components:
- DataAssetValueAgent: Main agent for value analysis
- AssetValueReport: Comprehensive analysis report
- AssetValueScore: Individual asset value scores
- ValueCalculator: Scoring logic
- QueryLogParser: SQL query log parser

Example Usage:
    from data_asset_value import DataAssetValueAgent, AssetValueReport

    agent = DataAssetValueAgent()

    # Analyze from query logs
    report = agent.analyze_from_query_logs(
        query_logs=logs,
        lineage_data=lineage_output,
        data_product_config=dp_config
    )

    # Get markdown report
    print(report.to_markdown())

    # Get specific asset value
    customer_value = agent.get_asset_value('customers', report)
"""

from .agent import (
    # Main Agent
    DataAssetValueAgent,

    # Report Classes
    AssetValueReport,
    AssetValueScore,
    AssetUsage,
    JoinRelationship,
    DataProductImpact,

    # Calculator
    ValueCalculator,
    QueryLogParser,

    # Enums
    ValueCategory,
    UsageType,
)

__all__ = [
    # Main Agent
    'DataAssetValueAgent',

    # Report Classes
    'AssetValueReport',
    'AssetValueScore',
    'AssetUsage',
    'JoinRelationship',
    'DataProductImpact',

    # Calculator
    'ValueCalculator',
    'QueryLogParser',

    # Enums
    'ValueCategory',
    'UsageType',
]

__version__ = '1.0.0'
