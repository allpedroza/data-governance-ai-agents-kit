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
Data Product Scoring Layer

Scores data products using a mix of contract completeness, governance,
quality, delivery readiness, and business value. Integrates directly
with Data Quality Agent and Data Asset Value Agent outputs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from data_governance.data_asset_value import AssetValueReport, DataAssetValueAgent
from data_governance.data_quality import DataQualityAgent, QualityReport


@dataclass
class DataProductContract:
    """Data contract definition for a data product."""
    schema: Optional[List[Dict[str, Any]]] = None
    granularity: Optional[str] = None
    sla_hours: Optional[int] = None
    freshness_sla_hours: Optional[int] = None
    semantics: Optional[Dict[str, str]] = None
    quality_rules: Optional[List[Dict[str, Any]]] = None
    version: Optional[str] = None


@dataclass
class DataProductGovernance:
    """Governance metadata embedded in the data product."""
    owner: Optional[str] = None
    steward: Optional[str] = None
    classification: Optional[str] = None
    lineage: Optional[str] = None
    access_policies: Optional[List[str]] = None
    pii: Optional[bool] = None
    retention_policy: Optional[str] = None


@dataclass
class DataProductDefinition:
    """Core definition of a data product."""
    name: str
    domain: Optional[str] = None
    purpose: Optional[str] = None
    business_outcomes: Optional[List[str]] = None
    consumers: Optional[List[str]] = None
    contract: Optional[DataProductContract] = None
    governance: Optional[DataProductGovernance] = None
    delivery_format: Optional[str] = None
    delivery_channel: Optional[str] = None
    assets: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProductScore:
    """Scoring output for a data product."""
    product_name: str
    timestamp: str
    overall_score: float
    status: str
    purpose_score: float
    consumers_score: float
    contract_score: float
    quality_score: float
    governance_score: float
    delivery_score: float
    value_score: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_name": self.product_name,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 2),
            "status": self.status,
            "purpose_score": round(self.purpose_score, 2),
            "consumers_score": round(self.consumers_score, 2),
            "contract_score": round(self.contract_score, 2),
            "quality_score": round(self.quality_score, 2),
            "governance_score": round(self.governance_score, 2),
            "delivery_score": round(self.delivery_score, 2),
            "value_score": round(self.value_score, 2),
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Data Product Score: {self.product_name}",
            f"*Generated: {self.timestamp}*",
            "",
            "## Summary",
            "",
            "| Dimension | Score |",
            "|---|---|",
            f"| Overall | **{self.overall_score:.1f}** |",
            f"| Purpose | {self.purpose_score:.1f} |",
            f"| Consumers | {self.consumers_score:.1f} |",
            f"| Contract | {self.contract_score:.1f} |",
            f"| Quality | {self.quality_score:.1f} |",
            f"| Governance | {self.governance_score:.1f} |",
            f"| Delivery | {self.delivery_score:.1f} |",
            f"| Value | {self.value_score:.1f} |",
            "",
            f"**Status:** {self.status}",
        ]

        if self.notes:
            lines.append("")
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")

        return "\n".join(lines)


@dataclass
class DataProductScoringReport:
    """Aggregate scoring report for multiple data products."""
    generated_at: str
    scores: List[DataProductScore]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "scores": [score.to_dict() for score in self.scores],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class DataProductScoringWeights:
    """Weighting configuration for data product scoring."""
    purpose: float = 0.15
    consumers: float = 0.10
    contract: float = 0.20
    quality: float = 0.20
    governance: float = 0.15
    delivery: float = 0.10
    value: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "purpose": self.purpose,
            "consumers": self.consumers,
            "contract": self.contract,
            "quality": self.quality,
            "governance": self.governance,
            "delivery": self.delivery,
            "value": self.value,
        }

    def normalize(self) -> None:
        total = sum(self.as_dict().values())
        if total == 0:
            return
        self.purpose /= total
        self.consumers /= total
        self.contract /= total
        self.quality /= total
        self.governance /= total
        self.delivery /= total
        self.value /= total


class DataProductScoringAgent:
    """Scores data products using internal heuristics and agent outputs."""

    def __init__(
        self,
        weights: Optional[DataProductScoringWeights] = None,
        quality_agent: Optional[DataQualityAgent] = None,
        value_agent: Optional[DataAssetValueAgent] = None,
    ) -> None:
        self.weights = weights or DataProductScoringWeights()
        self.weights.normalize()
        self.quality_agent = quality_agent or DataQualityAgent()
        self.value_agent = value_agent or DataAssetValueAgent()

    def score_product(
        self,
        product: DataProductDefinition,
        quality_report: Optional[QualityReport] = None,
        value_report: Optional[AssetValueReport] = None,
    ) -> DataProductScore:
        notes: List[str] = []
        purpose_score = self._score_purpose(product)
        consumers_score = self._score_consumers(product)
        contract_score = self._score_contract(product)
        governance_score = self._score_governance(product)
        delivery_score = self._score_delivery(product)
        quality_score = self._score_quality(quality_report, notes)
        value_score = self._score_value(product, value_report, notes)

        overall = (
            purpose_score * self.weights.purpose
            + consumers_score * self.weights.consumers
            + contract_score * self.weights.contract
            + quality_score * self.weights.quality
            + governance_score * self.weights.governance
            + delivery_score * self.weights.delivery
            + value_score * self.weights.value
        )
        status = self._status_from_score(overall)

        return DataProductScore(
            product_name=product.name,
            timestamp=datetime.utcnow().isoformat(),
            overall_score=round(overall, 2),
            status=status,
            purpose_score=purpose_score,
            consumers_score=consumers_score,
            contract_score=contract_score,
            quality_score=quality_score,
            governance_score=governance_score,
            delivery_score=delivery_score,
            value_score=value_score,
            notes=notes,
        )

    def score_product_with_agents(
        self,
        product: DataProductDefinition,
        quality_source: Optional[str] = None,
        quality_kwargs: Optional[Dict[str, Any]] = None,
        query_logs: Optional[List[Dict[str, Any]]] = None,
        data_product_config: Optional[List[Dict[str, Any]]] = None,
        lineage_data: Optional[Dict[str, Any]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[DataProductScore, Optional[QualityReport], Optional[AssetValueReport]]:
        quality_report = None
        if quality_source:
            quality_report = self.quality_agent.evaluate_file(
                quality_source, **(quality_kwargs or {})
            )

        value_report = None
        if query_logs is not None:
            value_report = self.value_agent.analyze_from_query_logs(
                query_logs=query_logs,
                lineage_data=lineage_data,
                data_product_config=data_product_config or [],
                asset_metadata=asset_metadata or [],
            )

        score = self.score_product(
            product=product,
            quality_report=quality_report,
            value_report=value_report,
        )
        return score, quality_report, value_report

    def score_portfolio(
        self,
        products: List[DataProductDefinition],
        quality_reports: Optional[Dict[str, QualityReport]] = None,
        value_reports: Optional[Dict[str, AssetValueReport]] = None,
    ) -> DataProductScoringReport:
        quality_reports = quality_reports or {}
        value_reports = value_reports or {}
        scores = []
        for product in products:
            scores.append(
                self.score_product(
                    product,
                    quality_reports.get(product.name),
                    value_reports.get(product.name),
                )
            )
        summary = self._summarize(scores)
        return DataProductScoringReport(
            generated_at=datetime.utcnow().isoformat(),
            scores=scores,
            summary=summary,
        )

    def _score_purpose(self, product: DataProductDefinition) -> float:
        points = 0
        if product.purpose:
            points += 1
        if product.business_outcomes:
            points += 1
        return self._score_ratio(points, 2)

    def _score_consumers(self, product: DataProductDefinition) -> float:
        consumers = product.consumers or []
        if not consumers:
            return 0.0
        if len(consumers) == 1:
            return 60.0
        if len(consumers) <= 3:
            return 80.0
        return 100.0

    def _score_contract(self, product: DataProductDefinition) -> float:
        contract = product.contract
        if not contract:
            return 0.0
        fields = [
            bool(contract.schema),
            bool(contract.granularity),
            bool(contract.sla_hours),
            bool(contract.freshness_sla_hours),
            bool(contract.semantics),
            bool(contract.quality_rules),
            bool(contract.version),
        ]
        return self._score_ratio(sum(fields), len(fields))

    def _score_quality(
        self,
        report: Optional[QualityReport],
        notes: List[str],
    ) -> float:
        if report is None:
            notes.append("Quality report ausente: execute Data Quality Agent para pontuar.")
            return 0.0
        return max(0.0, min(100.0, report.overall_score * 100))

    def _score_governance(self, product: DataProductDefinition) -> float:
        governance = product.governance
        if not governance:
            return 0.0
        fields = [
            bool(governance.owner),
            bool(governance.steward),
            bool(governance.classification),
            bool(governance.lineage),
            bool(governance.access_policies),
            governance.pii is not None,
            bool(governance.retention_policy),
        ]
        return self._score_ratio(sum(fields), len(fields))

    def _score_delivery(self, product: DataProductDefinition) -> float:
        fields = [
            bool(product.delivery_format),
            bool(product.delivery_channel),
        ]
        return self._score_ratio(sum(fields), len(fields))

    def _score_value(
        self,
        product: DataProductDefinition,
        report: Optional[AssetValueReport],
        notes: List[str],
    ) -> float:
        if report is None:
            notes.append("Value report ausente: execute Data Asset Value Agent para pontuar.")
            return 0.0
        assets = {asset.lower() for asset in (product.assets or [])}
        if not assets:
            notes.append("Data product sem lista de assets para calcular valor.")
            return 0.0
        scores = [
            s.overall_value_score
            for s in report.asset_scores
            if s.asset_name.lower() in assets
        ]
        if not scores:
            notes.append("Nenhum asset do data product encontrado no relatório de valor.")
            return 0.0
        return round(sum(scores) / len(scores), 2)

    def _score_ratio(self, points: int, total: int) -> float:
        if total == 0:
            return 0.0
        return round((points / total) * 100, 2)

    def _status_from_score(self, score: float) -> str:
        if score >= 80:
            return "excellent"
        if score >= 60:
            return "good"
        if score >= 40:
            return "warning"
        return "critical"

    def _summarize(self, scores: List[DataProductScore]) -> Dict[str, Any]:
        if not scores:
            return {}
        avg_score = sum(score.overall_score for score in scores) / len(scores)
        status_breakdown: Dict[str, int] = {}
        for score in scores:
            status_breakdown[score.status] = status_breakdown.get(score.status, 0) + 1
        return {
            "products_scored": len(scores),
            "average_score": round(avg_score, 2),
            "status_breakdown": status_breakdown,
        }
