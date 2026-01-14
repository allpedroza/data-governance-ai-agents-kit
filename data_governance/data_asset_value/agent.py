"""
Data Asset Value Scanner Agent

This agent analyzes data lake assets to understand their business value based on:
- Direct query usage patterns
- JOIN relationships (data products)
- Lineage impact (downstream dependencies)
- Business criticality, cost, and risk factors

The agent integrates with the Data Lineage Agent output to provide
comprehensive value scoring for data governance decisions.
"""

import json
import re
import os
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML

try:
    from rag_discovery.providers.base import LLMProvider
except ImportError:
    from data_governance.rag_discovery.providers.base import LLMProvider


class ValueCategory(Enum):
    """Categories for asset value classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class UsageType(Enum):
    """Types of data asset usage"""
    DIRECT_QUERY = "direct_query"
    JOIN_SOURCE = "join_source"
    JOIN_TARGET = "join_target"
    AGGREGATION = "aggregation"
    FILTER = "filter"
    DATA_PRODUCT = "data_product"
    ETL_SOURCE = "etl_source"
    ETL_TARGET = "etl_target"
    REPORT = "report"
    DASHBOARD = "dashboard"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"


@dataclass
class AssetUsage:
    """Represents a single usage instance of a data asset"""
    asset_name: str
    usage_type: str
    query_hash: str
    timestamp: datetime
    user: Optional[str] = None
    query_duration_ms: Optional[int] = None
    rows_scanned: Optional[int] = None
    data_product: Optional[str] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset_name': self.asset_name,
            'usage_type': self.usage_type,
            'query_hash': self.query_hash,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user': self.user,
            'query_duration_ms': self.query_duration_ms,
            'rows_scanned': self.rows_scanned,
            'data_product': self.data_product,
            'context': self.context
        }


@dataclass
class JoinRelationship:
    """Represents a JOIN relationship between assets"""
    left_asset: str
    right_asset: str
    join_columns: List[Tuple[str, str]]
    join_type: str  # INNER, LEFT, RIGHT, FULL, CROSS
    frequency: int = 0
    data_products: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'left_asset': self.left_asset,
            'right_asset': self.right_asset,
            'join_columns': self.join_columns,
            'join_type': self.join_type,
            'frequency': self.frequency,
            'data_products': self.data_products
        }


@dataclass
class DataProductImpact:
    """Represents impact of an asset on a data product"""
    data_product_name: str
    asset_name: str
    usage_types: List[str]
    is_critical_path: bool = False
    downstream_consumers: int = 0
    business_domain: Optional[str] = None
    sla_hours: Optional[int] = None
    revenue_impact: Optional[str] = None  # high/medium/low
    user_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_product_name': self.data_product_name,
            'asset_name': self.asset_name,
            'usage_types': self.usage_types,
            'is_critical_path': self.is_critical_path,
            'downstream_consumers': self.downstream_consumers,
            'business_domain': self.business_domain,
            'sla_hours': self.sla_hours,
            'revenue_impact': self.revenue_impact,
            'user_count': self.user_count
        }


@dataclass
class AssetValueScore:
    """Value scores for a single data asset"""
    asset_name: str

    # Core value dimensions (0-100)
    usage_score: float = 0.0
    join_score: float = 0.0
    lineage_score: float = 0.0
    data_product_score: float = 0.0

    # Weighted composite scores
    overall_value_score: float = 0.0
    business_impact_score: float = 0.0

    # Classification
    value_category: str = "unknown"

    # Metrics
    total_queries: int = 0
    unique_users: int = 0
    join_count: int = 0
    downstream_assets: int = 0
    upstream_assets: int = 0
    data_products_count: int = 0
    critical_path_count: int = 0

    # Risk & Cost factors
    criticality: str = "medium"  # high/medium/low
    estimated_cost_impact: Optional[str] = None
    risk_level: str = "medium"

    # Additional metadata
    last_accessed: Optional[datetime] = None
    access_trend: str = "stable"  # increasing/stable/decreasing
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset_name': self.asset_name,
            'usage_score': round(self.usage_score, 2),
            'join_score': round(self.join_score, 2),
            'lineage_score': round(self.lineage_score, 2),
            'data_product_score': round(self.data_product_score, 2),
            'overall_value_score': round(self.overall_value_score, 2),
            'business_impact_score': round(self.business_impact_score, 2),
            'value_category': self.value_category,
            'total_queries': self.total_queries,
            'unique_users': self.unique_users,
            'join_count': self.join_count,
            'downstream_assets': self.downstream_assets,
            'upstream_assets': self.upstream_assets,
            'data_products_count': self.data_products_count,
            'critical_path_count': self.critical_path_count,
            'criticality': self.criticality,
            'estimated_cost_impact': self.estimated_cost_impact,
            'risk_level': self.risk_level,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_trend': self.access_trend,
            'recommendations': self.recommendations
        }


@dataclass
class AssetValueReport:
    """Complete value analysis report for data assets"""
    analysis_timestamp: datetime
    assets_analyzed: int
    query_logs_processed: int
    time_range_days: int

    # Asset scores
    asset_scores: List[AssetValueScore] = field(default_factory=list)

    # Aggregated insights
    top_value_assets: List[str] = field(default_factory=list)
    critical_assets: List[str] = field(default_factory=list)
    orphan_assets: List[str] = field(default_factory=list)
    declining_assets: List[str] = field(default_factory=list)

    # Relationship insights
    join_relationships: List[JoinRelationship] = field(default_factory=list)
    hub_assets: List[str] = field(default_factory=list)  # Assets with many connections

    # Data product insights
    data_product_impacts: List[DataProductImpact] = field(default_factory=list)

    # Summary metrics
    summary: Dict[str, Any] = field(default_factory=dict)

    # Optional GenAI review
    llm_review: Optional[Dict[str, Any]] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'assets_analyzed': self.assets_analyzed,
            'query_logs_processed': self.query_logs_processed,
            'time_range_days': self.time_range_days,
            'asset_scores': [a.to_dict() for a in self.asset_scores],
            'top_value_assets': self.top_value_assets,
            'critical_assets': self.critical_assets,
            'orphan_assets': self.orphan_assets,
            'declining_assets': self.declining_assets,
            'join_relationships': [j.to_dict() for j in self.join_relationships],
            'hub_assets': self.hub_assets,
            'data_product_impacts': [d.to_dict() for d in self.data_product_impacts],
            'summary': self.summary,
            'llm_review': self.llm_review,
            'recommendations': self.recommendations
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Generate markdown report"""
        md = []
        md.append("# Data Asset Value Report")
        md.append(f"\n**Analysis Date:** {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Assets Analyzed:** {self.assets_analyzed}")
        md.append(f"**Query Logs Processed:** {self.query_logs_processed}")
        md.append(f"**Time Range:** {self.time_range_days} days\n")

        # Summary section
        md.append("## Summary\n")
        if self.summary:
            for key, value in self.summary.items():
                md.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Top Value Assets
        if self.top_value_assets:
            md.append("\n## Top Value Assets\n")
            for i, asset in enumerate(self.top_value_assets[:10], 1):
                md.append(f"{i}. `{asset}`")

        # Critical Assets
        if self.critical_assets:
            md.append("\n## Critical Assets\n")
            md.append("These assets are on critical paths for data products:\n")
            for asset in self.critical_assets:
                md.append(f"- `{asset}`")

        # Hub Assets
        if self.hub_assets:
            md.append("\n## Hub Assets (High Connectivity)\n")
            md.append("Assets with many JOIN relationships:\n")
            for asset in self.hub_assets:
                md.append(f"- `{asset}`")

        # Asset Value Scores Table
        md.append("\n## Asset Value Scores\n")
        md.append("| Asset | Value Score | Usage | Joins | Lineage | Data Products | Category |")
        md.append("|-------|-------------|-------|-------|---------|---------------|----------|")
        for score in sorted(self.asset_scores, key=lambda x: x.overall_value_score, reverse=True)[:20]:
            md.append(
                f"| `{score.asset_name}` | {score.overall_value_score:.1f} | "
                f"{score.usage_score:.1f} | {score.join_score:.1f} | "
                f"{score.lineage_score:.1f} | {score.data_product_score:.1f} | "
                f"{score.value_category} |"
            )

        # Orphan Assets
        if self.orphan_assets:
            md.append("\n## Orphan Assets (Low/No Usage)\n")
            md.append("Consider reviewing these assets for potential deprecation:\n")
            for asset in self.orphan_assets[:10]:
                md.append(f"- `{asset}`")

        # Declining Assets
        if self.declining_assets:
            md.append("\n## Declining Usage Assets\n")
            for asset in self.declining_assets[:10]:
                md.append(f"- `{asset}`")

        # Data Product Impact
        if self.data_product_impacts:
            md.append("\n## Data Product Dependencies\n")
            md.append("| Data Product | Asset | Critical Path | Consumers | Revenue Impact |")
            md.append("|--------------|-------|---------------|-----------|----------------|")
            for impact in self.data_product_impacts[:20]:
                md.append(
                    f"| {impact.data_product_name} | `{impact.asset_name}` | "
                    f"{'Yes' if impact.is_critical_path else 'No'} | "
                    f"{impact.downstream_consumers} | {impact.revenue_impact or 'N/A'} |"
                )

        # Recommendations
        if self.recommendations:
            md.append("\n## Recommendations\n")
            for i, rec in enumerate(self.recommendations, 1):
                md.append(f"{i}. {rec}")

        if self.llm_review and isinstance(self.llm_review, dict):
            md.append("\n## LLM Review (Post-Processing)\n")
            insights = self.llm_review.get('insights') or self.llm_review.get('parsed', {}).get('insights', [])
            if insights:
                md.append("**Insights:**")
                for insight in insights:
                    md.append(f"- {insight}")

            next_steps = self.llm_review.get('next_steps') or self.llm_review.get('parsed', {}).get('next_steps', [])
            if next_steps:
                md.append("\n**Next steps suggested by LLM:**")
                for step in next_steps:
                    md.append(f"- {step}")

            re_ranked = self.llm_review.get('re_ranked_top_assets') or self.llm_review.get('parsed', {}).get('re_ranked_top_assets', [])
            if re_ranked:
                md.append("\n**LLM-prioritized assets:**")
                for asset in re_ranked:
                    md.append(f"- `{asset}`")

        return "\n".join(md)


class QueryLogParser:
    """Parses SQL query logs to extract asset usage information"""

    # Common table patterns
    TABLE_PATTERNS = [
        r'(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([`"\[]?[\w\.]+[`"\]]?)',
        r'(?:FROM|JOIN)\s+\(?\s*SELECT.*?FROM\s+([`"\[]?[\w\.]+[`"\]]?)',
    ]

    # Join patterns
    JOIN_PATTERN = r'(LEFT|RIGHT|INNER|FULL|CROSS)?\s*(?:OUTER\s+)?JOIN\s+([`"\[]?[\w\.]+[`"\]]?)\s+(?:AS\s+\w+\s+)?ON\s+(.+?)(?=\s+(?:LEFT|RIGHT|INNER|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|LIMIT|$))'

    def __init__(self):
        self.usage_cache: Dict[str, List[AssetUsage]] = defaultdict(list)
        self.join_cache: Dict[str, JoinRelationship] = {}

    def parse_query_log(
        self,
        log_entries: List[Dict[str, Any]],
        data_product: Optional[str] = None
    ) -> Tuple[List[AssetUsage], List[JoinRelationship]]:
        """
        Parse query log entries to extract asset usage.

        Expected log entry format:
        {
            'query': str,
            'timestamp': str (ISO format) or datetime,
            'user': str (optional),
            'duration_ms': int (optional),
            'rows_scanned': int (optional),
            'data_product': str (optional),
            'context': str (optional)
        }
        """
        usages = []
        joins = []

        for entry in log_entries:
            query = entry.get('query', '')
            if not query:
                continue

            timestamp = entry.get('timestamp')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()

            query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

            # Extract tables and usage types
            parsed_usages, parsed_joins = self._parse_single_query(
                query=query,
                query_hash=query_hash,
                timestamp=timestamp,
                user=entry.get('user'),
                duration_ms=entry.get('duration_ms'),
                rows_scanned=entry.get('rows_scanned'),
                data_product=data_product or entry.get('data_product'),
                context=entry.get('context')
            )

            usages.extend(parsed_usages)
            joins.extend(parsed_joins)

        return usages, joins

    def _parse_single_query(
        self,
        query: str,
        query_hash: str,
        timestamp: datetime,
        user: Optional[str] = None,
        duration_ms: Optional[int] = None,
        rows_scanned: Optional[int] = None,
        data_product: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[List[AssetUsage], List[JoinRelationship]]:
        """Parse a single SQL query"""
        usages = []
        joins = []

        try:
            # Use sqlparse for better parsing
            parsed = sqlparse.parse(query)
            if not parsed:
                return usages, joins

            statement = parsed[0]

            # Determine query type
            query_type = statement.get_type()

            # Extract tables
            tables = self._extract_tables(statement)

            # Determine usage types based on query structure
            has_join = 'JOIN' in query.upper()
            has_aggregation = any(agg in query.upper() for agg in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(', 'GROUP BY'])

            for i, table in enumerate(tables):
                table_name = self._normalize_table_name(table)

                # Determine usage type
                if query_type == 'SELECT':
                    if i == 0:
                        usage_type = UsageType.DIRECT_QUERY.value
                    elif has_join:
                        usage_type = UsageType.JOIN_SOURCE.value
                    else:
                        usage_type = UsageType.DIRECT_QUERY.value

                    if has_aggregation:
                        usage_type = UsageType.AGGREGATION.value
                elif query_type == 'INSERT':
                    usage_type = UsageType.ETL_TARGET.value if i == 0 else UsageType.ETL_SOURCE.value
                elif query_type == 'UPDATE':
                    usage_type = UsageType.ETL_TARGET.value
                elif query_type == 'DELETE':
                    usage_type = UsageType.ETL_TARGET.value
                else:
                    usage_type = UsageType.DIRECT_QUERY.value

                usages.append(AssetUsage(
                    asset_name=table_name,
                    usage_type=usage_type,
                    query_hash=query_hash,
                    timestamp=timestamp,
                    user=user,
                    query_duration_ms=duration_ms,
                    rows_scanned=rows_scanned,
                    data_product=data_product,
                    context=context
                ))

            # Extract JOIN relationships
            if has_join:
                joins.extend(self._extract_joins(query, tables, data_product))

        except Exception as e:
            # Fallback to regex-based extraction
            tables = self._extract_tables_regex(query)
            for table in tables:
                table_name = self._normalize_table_name(table)
                usages.append(AssetUsage(
                    asset_name=table_name,
                    usage_type=UsageType.DIRECT_QUERY.value,
                    query_hash=query_hash,
                    timestamp=timestamp,
                    user=user,
                    query_duration_ms=duration_ms,
                    rows_scanned=rows_scanned,
                    data_product=data_product,
                    context=context
                ))

        return usages, joins

    def _extract_tables(self, statement) -> List[str]:
        """Extract table names from parsed SQL statement"""
        tables = []

        def extract_from_token(token):
            if isinstance(token, Identifier):
                name = token.get_real_name()
                if name:
                    tables.append(name)
            elif isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    extract_from_token(identifier)
            elif hasattr(token, 'tokens'):
                # Check for FROM/JOIN keywords
                from_seen = False
                for sub_token in token.tokens:
                    if sub_token.ttype is Keyword and sub_token.value.upper() in ('FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE'):
                        from_seen = True
                    elif from_seen and isinstance(sub_token, (Identifier, IdentifierList)):
                        extract_from_token(sub_token)
                        from_seen = False
                    elif from_seen and sub_token.ttype not in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                        if sub_token.ttype is None and sub_token.value.strip():
                            # Might be a simple table name
                            name = sub_token.value.strip()
                            if name and not name.upper() in ('SELECT', 'WHERE', 'AND', 'OR', 'ON'):
                                tables.append(name.split()[0])
                        from_seen = False

        extract_from_token(statement)

        # Fallback to regex if no tables found
        if not tables:
            tables = self._extract_tables_regex(str(statement))

        return tables

    def _extract_tables_regex(self, query: str) -> List[str]:
        """Fallback regex-based table extraction"""
        tables = []
        query_upper = query.upper()

        # Pattern for FROM and JOIN clauses
        patterns = [
            r'\bFROM\s+([`"\[\]]?[\w\.]+[`"\]\]]?)',
            r'\bJOIN\s+([`"\[\]]?[\w\.]+[`"\]\]]?)',
            r'\bINTO\s+([`"\[\]]?[\w\.]+[`"\]\]]?)',
            r'\bUPDATE\s+([`"\[\]]?[\w\.]+[`"\]\]]?)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _normalize_table_name(self, table: str) -> str:
        """Normalize table name by removing quotes and aliases"""
        # Remove quotes/brackets
        table = re.sub(r'[`"\[\]]', '', table)
        # Remove alias
        table = table.split()[0] if ' ' in table else table
        # Lowercase for consistency
        return table.lower().strip()

    def _extract_joins(
        self,
        query: str,
        tables: List[str],
        data_product: Optional[str]
    ) -> List[JoinRelationship]:
        """Extract JOIN relationships from query"""
        joins = []

        # Simple pattern to extract JOIN type and table
        join_pattern = r'(LEFT|RIGHT|INNER|FULL|CROSS)?\s*(?:OUTER\s+)?JOIN\s+([`"\[\]]?[\w\.]+[`"\]\]]?)'

        matches = re.findall(join_pattern, query, re.IGNORECASE)

        if tables and matches:
            left_table = self._normalize_table_name(tables[0])

            for join_type, right_table in matches:
                right_table = self._normalize_table_name(right_table)
                join_type = (join_type or 'INNER').upper()

                # Try to extract join columns
                join_cols = self._extract_join_columns(query, left_table, right_table)

                join_key = f"{left_table}_{right_table}_{join_type}"

                if join_key not in self.join_cache:
                    self.join_cache[join_key] = JoinRelationship(
                        left_asset=left_table,
                        right_asset=right_table,
                        join_columns=join_cols,
                        join_type=join_type,
                        frequency=1,
                        data_products=[data_product] if data_product else []
                    )
                else:
                    self.join_cache[join_key].frequency += 1
                    if data_product and data_product not in self.join_cache[join_key].data_products:
                        self.join_cache[join_key].data_products.append(data_product)

                joins.append(self.join_cache[join_key])

        return joins

    def _extract_join_columns(
        self,
        query: str,
        left_table: str,
        right_table: str
    ) -> List[Tuple[str, str]]:
        """Try to extract columns used in JOIN condition"""
        columns = []

        # Pattern to match ON conditions
        on_pattern = rf'JOIN\s+[`"\[\]]?{right_table}[`"\]\]]?\s+(?:AS\s+\w+\s+)?ON\s+(.+?)(?=\s+(?:LEFT|RIGHT|INNER|JOIN|WHERE|GROUP|ORDER|LIMIT|$))'

        match = re.search(on_pattern, query, re.IGNORECASE)
        if match:
            condition = match.group(1)
            # Extract column comparisons
            col_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
            col_matches = re.findall(col_pattern, condition)

            for t1, c1, t2, c2 in col_matches:
                columns.append((f"{t1}.{c1}", f"{t2}.{c2}"))

        return columns


class ValueCalculator:
    """Calculates value scores for data assets"""

    # Default weights for score calculation
    DEFAULT_WEIGHTS = {
        'usage': 0.30,       # Direct usage frequency
        'joins': 0.25,       # Join relationships
        'lineage': 0.25,     # Lineage impact
        'data_product': 0.20 # Data product importance
    }

    # Thresholds for categorization
    CATEGORY_THRESHOLDS = {
        'critical': 80,
        'high': 60,
        'medium': 40,
        'low': 20
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v/total for k, v in self.weights.items()}

    def calculate_usage_score(
        self,
        usages: List[AssetUsage],
        time_range_days: int = 30,
        max_queries_benchmark: int = 1000
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate usage score based on query frequency and patterns.

        Returns:
            Tuple of (score 0-100, metrics dict)
        """
        if not usages:
            return 0.0, {'total_queries': 0, 'unique_users': 0}

        now = datetime.now()
        recent_cutoff = now - timedelta(days=time_range_days)

        # Filter to recent usages
        recent_usages = [u for u in usages if u.timestamp and u.timestamp >= recent_cutoff]

        total_queries = len(recent_usages)
        unique_users = len(set(u.user for u in recent_usages if u.user))
        unique_data_products = len(set(u.data_product for u in recent_usages if u.data_product))

        # Usage type diversity
        usage_types = set(u.usage_type for u in recent_usages)
        type_diversity = len(usage_types) / len(UsageType)

        # Calculate base score (normalized by benchmark)
        frequency_score = min(100, (total_queries / max_queries_benchmark) * 100)

        # Adjust for user diversity (more users = more valuable)
        user_factor = min(1.5, 1.0 + (unique_users / 10))

        # Adjust for data product integration
        dp_factor = min(1.3, 1.0 + (unique_data_products / 5))

        # Calculate final score
        score = min(100, frequency_score * user_factor * dp_factor * (0.7 + 0.3 * type_diversity))

        metrics = {
            'total_queries': total_queries,
            'unique_users': unique_users,
            'unique_data_products': unique_data_products,
            'usage_types': list(usage_types),
            'type_diversity': round(type_diversity, 2)
        }

        return round(score, 2), metrics

    def calculate_join_score(
        self,
        asset_name: str,
        join_relationships: List[JoinRelationship],
        max_joins_benchmark: int = 50
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate JOIN score based on relationship patterns.

        Assets that are frequently joined with are more valuable as they
        serve as integration points in the data ecosystem.
        """
        # Find all joins involving this asset
        relevant_joins = [
            j for j in join_relationships
            if j.left_asset == asset_name or j.right_asset == asset_name
        ]

        if not relevant_joins:
            return 0.0, {'join_count': 0, 'connected_assets': []}

        total_join_frequency = sum(j.frequency for j in relevant_joins)
        connected_assets = set()
        data_products_involved = set()

        for j in relevant_joins:
            other_asset = j.right_asset if j.left_asset == asset_name else j.left_asset
            connected_assets.add(other_asset)
            data_products_involved.update(j.data_products)

        # Base score from join frequency
        frequency_score = min(100, (total_join_frequency / max_joins_benchmark) * 100)

        # Connectivity bonus (more connected assets = more valuable)
        connectivity_factor = min(1.5, 1.0 + (len(connected_assets) / 20))

        # Data product integration bonus
        dp_factor = min(1.3, 1.0 + (len(data_products_involved) / 5))

        score = min(100, frequency_score * connectivity_factor * dp_factor)

        metrics = {
            'join_count': len(relevant_joins),
            'total_frequency': total_join_frequency,
            'connected_assets': list(connected_assets),
            'data_products_involved': list(data_products_involved)
        }

        return round(score, 2), metrics

    def calculate_lineage_score(
        self,
        asset_name: str,
        lineage_data: Optional[Dict[str, Any]] = None,
        max_downstream_benchmark: int = 20
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate lineage score based on upstream/downstream dependencies.

        Uses output from Data Lineage Agent if available.
        """
        if not lineage_data:
            return 0.0, {'downstream_assets': 0, 'upstream_assets': 0}

        # Extract asset info from lineage data
        assets = lineage_data.get('assets', [])
        transformations = lineage_data.get('transformations', [])
        graph = lineage_data.get('graph')

        downstream_count = 0
        upstream_count = 0
        is_critical = False

        # If we have a networkx graph, use it for accurate counts
        if graph is not None:
            try:
                import networkx as nx

                if asset_name in graph:
                    # Count downstream (successors in DAG)
                    downstream_count = len(list(nx.descendants(graph, asset_name)))
                    # Count upstream (predecessors in DAG)
                    upstream_count = len(list(nx.ancestors(graph, asset_name)))

                    # Check if on critical path
                    critical_components = lineage_data.get('critical_components', {})
                    critical_assets = critical_components.get('high_impact_assets', [])
                    is_critical = asset_name in [a.get('name', '') for a in critical_assets]
            except ImportError:
                pass
        else:
            # Fallback: count from transformations
            for t in transformations:
                source = t.get('source', '') if isinstance(t, dict) else getattr(t, 'source', '')
                target = t.get('target', '') if isinstance(t, dict) else getattr(t, 'target', '')

                if source == asset_name:
                    downstream_count += 1
                if target == asset_name:
                    upstream_count += 1

        # Calculate score based on downstream impact
        downstream_score = min(100, (downstream_count / max_downstream_benchmark) * 100)

        # Being a source (many downstreams) is more valuable than being a sink
        source_factor = 1.0 + (0.5 if downstream_count > upstream_count else 0)

        # Critical path bonus
        critical_factor = 1.5 if is_critical else 1.0

        score = min(100, downstream_score * source_factor * critical_factor)

        metrics = {
            'downstream_assets': downstream_count,
            'upstream_assets': upstream_count,
            'is_critical_path': is_critical
        }

        return round(score, 2), metrics

    def calculate_data_product_score(
        self,
        asset_name: str,
        data_product_impacts: List[DataProductImpact],
        max_products_benchmark: int = 10
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate score based on data product integration and impact.
        """
        relevant_impacts = [
            dp for dp in data_product_impacts
            if dp.asset_name == asset_name
        ]

        if not relevant_impacts:
            return 0.0, {'data_products_count': 0}

        products_count = len(relevant_impacts)
        critical_path_count = sum(1 for dp in relevant_impacts if dp.is_critical_path)
        total_consumers = sum(dp.downstream_consumers for dp in relevant_impacts)
        total_users = sum(dp.user_count for dp in relevant_impacts)

        # Revenue impact weights
        revenue_weights = {'high': 3, 'medium': 2, 'low': 1, None: 0}
        revenue_score = sum(revenue_weights.get(dp.revenue_impact, 0) for dp in relevant_impacts)

        # Base score from product count
        product_score = min(100, (products_count / max_products_benchmark) * 100)

        # Critical path bonus
        critical_factor = 1.0 + (critical_path_count * 0.2)

        # Consumer reach bonus
        consumer_factor = min(1.5, 1.0 + (total_consumers / 50))

        # Revenue impact bonus
        revenue_factor = min(1.5, 1.0 + (revenue_score / 10))

        score = min(100, product_score * critical_factor * consumer_factor * revenue_factor)

        metrics = {
            'data_products_count': products_count,
            'critical_path_count': critical_path_count,
            'total_downstream_consumers': total_consumers,
            'total_users': total_users,
            'revenue_impact_score': revenue_score
        }

        return round(score, 2), metrics

    def calculate_overall_score(
        self,
        usage_score: float,
        join_score: float,
        lineage_score: float,
        data_product_score: float
    ) -> float:
        """Calculate weighted overall value score"""
        score = (
            usage_score * self.weights['usage'] +
            join_score * self.weights['joins'] +
            lineage_score * self.weights['lineage'] +
            data_product_score * self.weights['data_product']
        )
        return round(score, 2)

    def calculate_business_impact_score(
        self,
        overall_score: float,
        criticality: str = 'medium',
        risk_level: str = 'medium',
        has_pii: bool = False,
        sla_count: int = 0
    ) -> float:
        """
        Calculate business impact considering criticality and risk.
        """
        # Criticality multipliers
        criticality_mult = {'high': 1.3, 'medium': 1.0, 'low': 0.7}
        crit_factor = criticality_mult.get(criticality, 1.0)

        # Risk adds to impact (high risk = high impact if compromised)
        risk_mult = {'high': 1.2, 'medium': 1.0, 'low': 0.9}
        risk_factor = risk_mult.get(risk_level, 1.0)

        # PII/sensitive data factor
        pii_factor = 1.2 if has_pii else 1.0

        # SLA factor (more SLAs = more business critical)
        sla_factor = min(1.3, 1.0 + (sla_count * 0.1))

        score = overall_score * crit_factor * risk_factor * pii_factor * sla_factor
        return min(100, round(score, 2))

    def categorize_value(self, score: float) -> str:
        """Categorize asset value based on score"""
        if score >= self.CATEGORY_THRESHOLDS['critical']:
            return ValueCategory.CRITICAL.value
        elif score >= self.CATEGORY_THRESHOLDS['high']:
            return ValueCategory.HIGH.value
        elif score >= self.CATEGORY_THRESHOLDS['medium']:
            return ValueCategory.MEDIUM.value
        elif score >= self.CATEGORY_THRESHOLDS['low']:
            return ValueCategory.LOW.value
        else:
            return ValueCategory.UNKNOWN.value


class DataAssetValueAgent:
    """
    Agent for scanning and analyzing data asset value in a data lake.

    This agent integrates with:
    - Query logs to understand usage patterns
    - Data Lineage Agent output for dependency analysis
    - Data Product metadata for business impact assessment

    Example usage:
        agent = DataAssetValueAgent()

        # Analyze from query logs
        report = agent.analyze_from_query_logs(
            query_logs=logs,
            lineage_data=lineage_agent_output,
            data_product_config=dp_config
        )

        print(report.to_markdown())
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        time_range_days: int = 30,
        persist_dir: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize the Data Asset Value Agent.

        Args:
            weights: Custom weights for value calculation
            time_range_days: Default time range for analysis
            persist_dir: Directory for persisting analysis results
        """
        self.query_parser = QueryLogParser()
        self.value_calculator = ValueCalculator(weights)
        self.time_range_days = time_range_days
        self.persist_dir = persist_dir
        self.llm_provider = llm_provider

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    def analyze_from_query_logs(
        self,
        query_logs: List[Dict[str, Any]],
        lineage_data: Optional[Dict[str, Any]] = None,
        data_product_config: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        time_range_days: Optional[int] = None,
        llm_review: bool = False,
        llm_additional_context: Optional[str] = None
    ) -> AssetValueReport:
        """
        Analyze data asset value from query logs.

        Args:
            query_logs: List of query log entries
            lineage_data: Output from Data Lineage Agent
            data_product_config: Data product configuration with impact info
            asset_metadata: Additional asset metadata (criticality, cost, risk)
            time_range_days: Override default time range

        Returns:
            AssetValueReport with comprehensive value analysis
        """
        time_range = time_range_days or self.time_range_days

        # Parse query logs
        all_usages, all_joins = self.query_parser.parse_query_log(query_logs)

        # Group usages by asset
        usages_by_asset: Dict[str, List[AssetUsage]] = defaultdict(list)
        for usage in all_usages:
            usages_by_asset[usage.asset_name].append(usage)

        # Deduplicate joins
        unique_joins = list(self.query_parser.join_cache.values())

        # Build data product impacts if config provided
        data_product_impacts = []
        if data_product_config:
            data_product_impacts = self._build_data_product_impacts(
                data_product_config, usages_by_asset
            )

        # Calculate scores for each asset
        asset_scores = []
        all_assets = set(usages_by_asset.keys())

        # Also include assets from lineage that might not have direct usage
        if lineage_data:
            for asset in lineage_data.get('assets', []):
                name = asset.get('name', '') if isinstance(asset, dict) else getattr(asset, 'name', '')
                if name:
                    all_assets.add(name.lower())

        for asset_name in all_assets:
            score = self._calculate_asset_score(
                asset_name=asset_name,
                usages=usages_by_asset.get(asset_name, []),
                join_relationships=unique_joins,
                lineage_data=lineage_data,
                data_product_impacts=data_product_impacts,
                asset_metadata=asset_metadata,
                time_range_days=time_range
            )
            asset_scores.append(score)

        # Sort by overall value score
        asset_scores.sort(key=lambda x: x.overall_value_score, reverse=True)

        # Generate insights
        report = self._generate_report(
            asset_scores=asset_scores,
            join_relationships=unique_joins,
            data_product_impacts=data_product_impacts,
            query_logs_count=len(query_logs),
            time_range_days=time_range
        )

        # Persist if configured
        if self.persist_dir:
            self._persist_report(report)

        if llm_review and self.llm_provider:
            report.llm_review = self._run_llm_review(
                report=report,
                data_product_config=data_product_config,
                asset_metadata=asset_metadata,
                additional_context=llm_additional_context
            )

        return report

    def analyze_from_files(
        self,
        query_log_path: str,
        lineage_output_path: Optional[str] = None,
        data_product_config_path: Optional[str] = None,
        asset_metadata_path: Optional[str] = None,
        time_range_days: Optional[int] = None
    ) -> AssetValueReport:
        """
        Analyze data asset value from file inputs.

        Args:
            query_log_path: Path to JSON file with query logs
            lineage_output_path: Path to Data Lineage Agent output
            data_product_config_path: Path to data product config
            asset_metadata_path: Path to asset metadata file

        Returns:
            AssetValueReport
        """
        # Load query logs
        with open(query_log_path, 'r', encoding='utf-8') as f:
            query_logs = json.load(f)

        # Load lineage data if provided
        lineage_data = None
        if lineage_output_path and os.path.exists(lineage_output_path):
            with open(lineage_output_path, 'r', encoding='utf-8') as f:
                lineage_data = json.load(f)

        # Load data product config if provided
        data_product_config = None
        if data_product_config_path and os.path.exists(data_product_config_path):
            with open(data_product_config_path, 'r', encoding='utf-8') as f:
                data_product_config = json.load(f)

        # Load asset metadata if provided
        asset_metadata = None
        if asset_metadata_path and os.path.exists(asset_metadata_path):
            with open(asset_metadata_path, 'r', encoding='utf-8') as f:
                asset_metadata = json.load(f)

        return self.analyze_from_query_logs(
            query_logs=query_logs,
            lineage_data=lineage_data,
            data_product_config=data_product_config,
            asset_metadata=asset_metadata,
            time_range_days=time_range_days
        )

    def analyze_with_lineage_agent(
        self,
        query_logs: List[Dict[str, Any]],
        pipeline_files: List[str],
        data_product_config: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        time_range_days: Optional[int] = None
    ) -> AssetValueReport:
        """
        Analyze with integrated Data Lineage Agent execution.

        Args:
            query_logs: Query log entries
            pipeline_files: Files to analyze with Lineage Agent
            data_product_config: Data product configuration
            asset_metadata: Asset metadata

        Returns:
            AssetValueReport
        """
        # Import and run lineage agent
        try:
            from lineage.data_lineage_agent import DataLineageAgent

            lineage_agent = DataLineageAgent()
            lineage_data = lineage_agent.analyze_pipeline(pipeline_files)
        except ImportError:
            lineage_data = None
        except Exception as e:
            print(f"Warning: Could not run lineage analysis: {e}")
            lineage_data = None

        return self.analyze_from_query_logs(
            query_logs=query_logs,
            lineage_data=lineage_data,
            data_product_config=data_product_config,
            asset_metadata=asset_metadata,
            time_range_days=time_range_days
        )

    def _calculate_asset_score(
        self,
        asset_name: str,
        usages: List[AssetUsage],
        join_relationships: List[JoinRelationship],
        lineage_data: Optional[Dict[str, Any]],
        data_product_impacts: List[DataProductImpact],
        asset_metadata: Optional[Dict[str, Dict[str, Any]]],
        time_range_days: int
    ) -> AssetValueScore:
        """Calculate comprehensive value score for a single asset"""

        # Calculate individual dimension scores
        usage_score, usage_metrics = self.value_calculator.calculate_usage_score(
            usages, time_range_days
        )

        join_score, join_metrics = self.value_calculator.calculate_join_score(
            asset_name, join_relationships
        )

        lineage_score, lineage_metrics = self.value_calculator.calculate_lineage_score(
            asset_name, lineage_data
        )

        dp_score, dp_metrics = self.value_calculator.calculate_data_product_score(
            asset_name, data_product_impacts
        )

        # Calculate overall score
        overall_score = self.value_calculator.calculate_overall_score(
            usage_score, join_score, lineage_score, dp_score
        )

        # Get asset metadata if available
        metadata = {}
        if asset_metadata and asset_name in asset_metadata:
            metadata = asset_metadata[asset_name]

        criticality = metadata.get('criticality', 'medium')
        risk_level = metadata.get('risk_level', 'medium')
        has_pii = metadata.get('has_pii', False)
        sla_count = metadata.get('sla_count', 0)

        # Calculate business impact score
        business_impact = self.value_calculator.calculate_business_impact_score(
            overall_score, criticality, risk_level, has_pii, sla_count
        )

        # Categorize
        value_category = self.value_calculator.categorize_value(overall_score)

        # Determine access trend
        access_trend = self._calculate_access_trend(usages, time_range_days)

        # Get last access time
        last_accessed = None
        if usages:
            last_accessed = max(u.timestamp for u in usages if u.timestamp)

        # Generate recommendations
        recommendations = self._generate_asset_recommendations(
            asset_name=asset_name,
            overall_score=overall_score,
            usage_score=usage_score,
            join_score=join_score,
            access_trend=access_trend,
            criticality=criticality
        )

        return AssetValueScore(
            asset_name=asset_name,
            usage_score=usage_score,
            join_score=join_score,
            lineage_score=lineage_score,
            data_product_score=dp_score,
            overall_value_score=overall_score,
            business_impact_score=business_impact,
            value_category=value_category,
            total_queries=usage_metrics.get('total_queries', 0),
            unique_users=usage_metrics.get('unique_users', 0),
            join_count=join_metrics.get('join_count', 0),
            downstream_assets=lineage_metrics.get('downstream_assets', 0),
            upstream_assets=lineage_metrics.get('upstream_assets', 0),
            data_products_count=dp_metrics.get('data_products_count', 0),
            critical_path_count=dp_metrics.get('critical_path_count', 0),
            criticality=criticality,
            estimated_cost_impact=metadata.get('estimated_cost_impact'),
            risk_level=risk_level,
            last_accessed=last_accessed,
            access_trend=access_trend,
            recommendations=recommendations
        )

    def _calculate_access_trend(
        self,
        usages: List[AssetUsage],
        time_range_days: int
    ) -> str:
        """Calculate if usage is increasing, stable, or decreasing"""
        if not usages or len(usages) < 2:
            return "stable"

        now = datetime.now()
        half_range = time_range_days // 2

        recent_cutoff = now - timedelta(days=half_range)
        older_cutoff = now - timedelta(days=time_range_days)

        recent_count = sum(
            1 for u in usages
            if u.timestamp and u.timestamp >= recent_cutoff
        )
        older_count = sum(
            1 for u in usages
            if u.timestamp and older_cutoff <= u.timestamp < recent_cutoff
        )

        if older_count == 0:
            return "increasing" if recent_count > 0 else "stable"

        ratio = recent_count / older_count

        if ratio > 1.2:
            return "increasing"
        elif ratio < 0.8:
            return "decreasing"
        else:
            return "stable"

    def _generate_asset_recommendations(
        self,
        asset_name: str,
        overall_score: float,
        usage_score: float,
        join_score: float,
        access_trend: str,
        criticality: str
    ) -> List[str]:
        """Generate recommendations for a specific asset"""
        recommendations = []

        if overall_score >= 80:
            recommendations.append(
                f"High-value asset - ensure robust backup and monitoring"
            )

        if usage_score < 10 and access_trend == "decreasing":
            recommendations.append(
                f"Consider deprecation review - low usage with declining trend"
            )

        if join_score > 70:
            recommendations.append(
                f"Hub asset - document integration patterns and dependencies"
            )

        if criticality == "high" and overall_score < 40:
            recommendations.append(
                f"Marked as critical but low usage - verify criticality classification"
            )

        if access_trend == "increasing" and overall_score >= 50:
            recommendations.append(
                f"Growing importance - consider scaling and optimization"
            )

        return recommendations

    def _build_data_product_impacts(
        self,
        data_product_config: List[Dict[str, Any]],
        usages_by_asset: Dict[str, List[AssetUsage]]
    ) -> List[DataProductImpact]:
        """Build DataProductImpact objects from configuration"""
        impacts = []

        for dp_config in data_product_config:
            dp_name = dp_config.get('name', '')
            assets = dp_config.get('assets', [])

            for asset_name in assets:
                asset_name_lower = asset_name.lower()

                # Get usage types for this asset in this data product
                asset_usages = usages_by_asset.get(asset_name_lower, [])
                dp_usages = [u for u in asset_usages if u.data_product == dp_name]
                usage_types = list(set(u.usage_type for u in dp_usages))

                impacts.append(DataProductImpact(
                    data_product_name=dp_name,
                    asset_name=asset_name_lower,
                    usage_types=usage_types or ['unknown'],
                    is_critical_path=dp_config.get('critical_assets', []).count(asset_name) > 0,
                    downstream_consumers=dp_config.get('consumers', 0),
                    business_domain=dp_config.get('domain'),
                    sla_hours=dp_config.get('sla_hours'),
                    revenue_impact=dp_config.get('revenue_impact'),
                    user_count=dp_config.get('user_count', 0)
                ))

        return impacts

    def _generate_report(
        self,
        asset_scores: List[AssetValueScore],
        join_relationships: List[JoinRelationship],
        data_product_impacts: List[DataProductImpact],
        query_logs_count: int,
        time_range_days: int
    ) -> AssetValueReport:
        """Generate comprehensive value report"""

        # Identify key asset categories
        top_value_assets = [
            s.asset_name for s in asset_scores[:10]
            if s.overall_value_score >= 50
        ]

        critical_assets = [
            s.asset_name for s in asset_scores
            if s.value_category == 'critical' or s.critical_path_count > 0
        ]

        orphan_assets = [
            s.asset_name for s in asset_scores
            if s.total_queries == 0 and s.join_count == 0
        ]

        declining_assets = [
            s.asset_name for s in asset_scores
            if s.access_trend == 'decreasing'
        ]

        # Identify hub assets (high connectivity)
        hub_assets = [
            s.asset_name for s in asset_scores
            if s.join_count >= 5
        ]

        # Generate summary
        summary = {
            'total_assets': len(asset_scores),
            'critical_assets_count': len(critical_assets),
            'high_value_assets_count': len([s for s in asset_scores if s.value_category in ('critical', 'high')]),
            'orphan_assets_count': len(orphan_assets),
            'declining_assets_count': len(declining_assets),
            'average_value_score': round(
                sum(s.overall_value_score for s in asset_scores) / len(asset_scores), 2
            ) if asset_scores else 0,
            'total_join_relationships': len(join_relationships),
            'data_products_analyzed': len(set(dp.data_product_name for dp in data_product_impacts))
        }

        # Generate recommendations
        recommendations = self._generate_global_recommendations(
            asset_scores, orphan_assets, declining_assets, hub_assets
        )

        return AssetValueReport(
            analysis_timestamp=datetime.now(),
            assets_analyzed=len(asset_scores),
            query_logs_processed=query_logs_count,
            time_range_days=time_range_days,
            asset_scores=asset_scores,
            top_value_assets=top_value_assets,
            critical_assets=critical_assets,
            orphan_assets=orphan_assets,
            declining_assets=declining_assets,
            join_relationships=join_relationships,
            hub_assets=hub_assets,
            data_product_impacts=data_product_impacts,
            summary=summary,
            recommendations=recommendations
        )

    def _run_llm_review(
        self,
        report: AssetValueReport,
        data_product_config: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask an LLM to re-run the analysis using the deterministic summary as context."""

        summary_payload = {
            'summary': report.summary,
            'top_value_assets': report.top_value_assets,
            'critical_assets': report.critical_assets,
            'orphan_assets': report.orphan_assets,
            'declining_assets': report.declining_assets,
            'recommendations': report.recommendations,
            'data_products': data_product_config or [],
            'asset_metadata': asset_metadata or {},
            'additional_context': additional_context,
        }

        prompt = (
            "Você é um arquiteto de dados e governança. Recebeu um resumo determinístico "
            "sobre valor de ativos (uso, joins, linhagem e data products) e precisa revisar o ranking, "
            "apontar riscos e sugerir próximos passos. Responda em JSON com as chaves: "
            "insights (lista), re_ranked_top_assets (lista de strings), risk_flags (lista), next_steps (lista).\n\n"
            f"Contexto:\n{json.dumps(summary_payload, ensure_ascii=False)}"
        )

        system_prompt = (
            "Analise o contexto de forma crítica, mantenha respostas concisas e priorize ativos com impacto "
            "de negócio ou risco (PII/SLAs)."
        )

        llm_response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=600
        )

        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(llm_response.content)
        except Exception:
            parsed = {'insights': [llm_response.content]}

        return {
            'model': getattr(self.llm_provider, 'model_name', ''),
            'prompt': prompt,
            'parsed': parsed,
            'raw_response': llm_response.content,
            'insights': parsed.get('insights', []),
            're_ranked_top_assets': parsed.get('re_ranked_top_assets', []),
            'risk_flags': parsed.get('risk_flags', []),
            'next_steps': parsed.get('next_steps', []),
        }

    def _generate_global_recommendations(
        self,
        asset_scores: List[AssetValueScore],
        orphan_assets: List[str],
        declining_assets: List[str],
        hub_assets: List[str]
    ) -> List[str]:
        """Generate global recommendations based on analysis"""
        recommendations = []

        if orphan_assets:
            pct = (len(orphan_assets) / len(asset_scores)) * 100
            recommendations.append(
                f"{len(orphan_assets)} assets ({pct:.1f}%) have no recorded usage. "
                f"Consider reviewing for potential deprecation or verify if monitoring is complete."
            )

        if declining_assets:
            recommendations.append(
                f"{len(declining_assets)} assets show declining usage trends. "
                f"Review for potential consolidation or phase-out."
            )

        if hub_assets:
            recommendations.append(
                f"{len(hub_assets)} hub assets identified with high connectivity. "
                f"Prioritize documentation and change management for these integration points."
            )

        critical_low_usage = [
            s for s in asset_scores
            if s.criticality == 'high' and s.usage_score < 20
        ]
        if critical_low_usage:
            recommendations.append(
                f"{len(critical_low_usage)} assets marked as critical have low usage. "
                f"Verify criticality classifications are accurate."
            )

        high_value_no_docs = [
            s for s in asset_scores
            if s.overall_value_score >= 70 and 'document' not in str(s.recommendations).lower()
        ]
        if high_value_no_docs:
            recommendations.append(
                f"Ensure high-value assets have comprehensive documentation and ownership."
            )

        return recommendations

    def _persist_report(self, report: AssetValueReport) -> str:
        """Persist report to disk"""
        if not self.persist_dir:
            return ""

        timestamp = report.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"asset_value_report_{timestamp}.json"
        filepath = os.path.join(self.persist_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_json())

        return filepath

    def get_asset_value(self, asset_name: str, report: AssetValueReport) -> Optional[AssetValueScore]:
        """Get value score for a specific asset from a report"""
        for score in report.asset_scores:
            if score.asset_name.lower() == asset_name.lower():
                return score
        return None

    def compare_assets(
        self,
        asset_names: List[str],
        report: AssetValueReport
    ) -> List[Dict[str, Any]]:
        """Compare multiple assets side by side"""
        comparison = []
        for name in asset_names:
            score = self.get_asset_value(name, report)
            if score:
                comparison.append({
                    'asset': name,
                    'overall_score': score.overall_value_score,
                    'category': score.value_category,
                    'usage': score.usage_score,
                    'joins': score.join_score,
                    'lineage': score.lineage_score,
                    'data_products': score.data_product_score,
                    'business_impact': score.business_impact_score
                })

        return sorted(comparison, key=lambda x: x['overall_score'], reverse=True)
