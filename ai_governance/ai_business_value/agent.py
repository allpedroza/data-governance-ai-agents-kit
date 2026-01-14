"""
AI Business Value Agent

This agent analyzes AI initiatives to understand their business value based on:
- Cost analysis (development, infrastructure, maintenance)
- Benefit projections (revenue, efficiency, cost savings)
- ROI calculations and payback period
- Risk assessment (technical, organizational, market)
- Strategic alignment with business objectives
- Maturity level and readiness assessment

The agent provides comprehensive scoring and recommendations
for AI governance and investment decisions.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class ValueCategory(Enum):
    """Categories for AI initiative value classification"""
    TRANSFORMATIONAL = "transformational"  # High strategic value, game-changer
    HIGH = "high"                          # Significant business impact
    MEDIUM = "medium"                      # Moderate value, incremental improvement
    LOW = "low"                            # Limited value or experimental
    EXPERIMENTAL = "experimental"          # R&D, exploratory initiatives


class InitiativeType(Enum):
    """Types of AI initiatives"""
    AUTOMATION = "automation"              # Process automation
    ANALYTICS = "analytics"                # Predictive/prescriptive analytics
    CUSTOMER_EXPERIENCE = "customer_experience"  # Chatbots, personalization
    DECISION_SUPPORT = "decision_support"  # AI-assisted decisions
    CONTENT_GENERATION = "content_generation"  # GenAI for content
    OPERATIONAL_AI = "operational_ai"      # MLOps, AIOps
    PRODUCT_AI = "product_ai"              # AI as product feature
    RESEARCH = "research"                  # Exploratory R&D


class RiskLevel(Enum):
    """Risk levels for AI initiatives"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class MaturityLevel(Enum):
    """AI initiative maturity levels"""
    IDEATION = "ideation"          # Concept phase
    POC = "poc"                    # Proof of Concept
    PILOT = "pilot"                # Limited deployment
    SCALING = "scaling"            # Expanding deployment
    PRODUCTION = "production"      # Full production
    OPTIMIZATION = "optimization"  # Mature, optimizing


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an AI initiative"""
    initiative_id: str

    # Development costs
    development_internal: float = 0.0      # Internal team costs
    development_external: float = 0.0      # Consultants, vendors
    development_tools: float = 0.0         # IDE, MLOps tools

    # Infrastructure costs
    compute_training: float = 0.0          # GPU/TPU for training
    compute_inference: float = 0.0         # Production inference
    storage: float = 0.0                   # Data storage
    networking: float = 0.0                # API calls, bandwidth

    # Operational costs (annual)
    maintenance_annual: float = 0.0        # Ongoing maintenance
    monitoring_annual: float = 0.0         # MLOps, monitoring
    support_annual: float = 0.0            # User support
    licensing_annual: float = 0.0          # Third-party licenses

    # One-time costs
    data_acquisition: float = 0.0          # Data purchase/licensing
    integration: float = 0.0               # System integration
    training_users: float = 0.0            # User training
    change_management: float = 0.0         # Organizational change

    # Hidden costs
    opportunity_cost: float = 0.0          # Alternative investments
    technical_debt: float = 0.0            # Estimated tech debt cost

    currency: str = "BRL"
    period_months: int = 12

    @property
    def total_development(self) -> float:
        return (self.development_internal + self.development_external +
                self.development_tools)

    @property
    def total_infrastructure(self) -> float:
        return (self.compute_training + self.compute_inference +
                self.storage + self.networking)

    @property
    def total_operational_annual(self) -> float:
        return (self.maintenance_annual + self.monitoring_annual +
                self.support_annual + self.licensing_annual)

    @property
    def total_one_time(self) -> float:
        return (self.data_acquisition + self.integration +
                self.training_users + self.change_management)

    @property
    def total_initial_investment(self) -> float:
        return self.total_development + self.total_one_time

    @property
    def total_cost_year1(self) -> float:
        return (self.total_initial_investment + self.total_infrastructure +
                self.total_operational_annual)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initiative_id': self.initiative_id,
            'development': {
                'internal': self.development_internal,
                'external': self.development_external,
                'tools': self.development_tools,
                'total': self.total_development
            },
            'infrastructure': {
                'compute_training': self.compute_training,
                'compute_inference': self.compute_inference,
                'storage': self.storage,
                'networking': self.networking,
                'total': self.total_infrastructure
            },
            'operational_annual': {
                'maintenance': self.maintenance_annual,
                'monitoring': self.monitoring_annual,
                'support': self.support_annual,
                'licensing': self.licensing_annual,
                'total': self.total_operational_annual
            },
            'one_time': {
                'data_acquisition': self.data_acquisition,
                'integration': self.integration,
                'training_users': self.training_users,
                'change_management': self.change_management,
                'total': self.total_one_time
            },
            'hidden_costs': {
                'opportunity_cost': self.opportunity_cost,
                'technical_debt': self.technical_debt
            },
            'totals': {
                'initial_investment': self.total_initial_investment,
                'year1_total': self.total_cost_year1
            },
            'currency': self.currency,
            'period_months': self.period_months
        }


@dataclass
class BenefitProjection:
    """Benefit projections for an AI initiative"""
    initiative_id: str

    # Quantifiable benefits (annual)
    revenue_increase: float = 0.0          # Direct revenue impact
    cost_reduction: float = 0.0            # Operational cost savings
    efficiency_gain_hours: float = 0.0     # Hours saved annually
    efficiency_gain_value: float = 0.0     # Monetary value of hours saved
    error_reduction_value: float = 0.0     # Cost of errors avoided

    # Customer impact
    customer_satisfaction_delta: float = 0.0  # NPS/CSAT improvement
    customer_retention_delta: float = 0.0     # Retention rate improvement
    customer_acquisition_value: float = 0.0   # New customer value

    # Strategic benefits (scored 0-100)
    competitive_advantage: float = 0.0     # Market differentiation
    innovation_score: float = 0.0          # Innovation capability
    scalability_potential: float = 0.0     # Future growth enablement
    data_asset_value: float = 0.0          # Data/model assets created

    # Risk mitigation
    compliance_risk_reduction: float = 0.0  # Regulatory risk reduction
    operational_risk_reduction: float = 0.0 # Operational risk reduction

    # Confidence levels (0-100)
    revenue_confidence: float = 50.0
    cost_reduction_confidence: float = 70.0
    efficiency_confidence: float = 80.0

    currency: str = "BRL"
    projection_years: int = 3

    @property
    def total_quantifiable_annual(self) -> float:
        return (self.revenue_increase + self.cost_reduction +
                self.efficiency_gain_value + self.error_reduction_value +
                self.customer_acquisition_value)

    @property
    def weighted_annual_benefit(self) -> float:
        """Benefit weighted by confidence levels"""
        return (
            self.revenue_increase * (self.revenue_confidence / 100) +
            self.cost_reduction * (self.cost_reduction_confidence / 100) +
            self.efficiency_gain_value * (self.efficiency_confidence / 100) +
            self.error_reduction_value * (self.efficiency_confidence / 100) +
            self.customer_acquisition_value * (self.revenue_confidence / 100)
        )

    @property
    def strategic_score(self) -> float:
        """Average strategic benefit score"""
        scores = [self.competitive_advantage, self.innovation_score,
                  self.scalability_potential, self.data_asset_value]
        return sum(scores) / len(scores) if scores else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initiative_id': self.initiative_id,
            'quantifiable_annual': {
                'revenue_increase': self.revenue_increase,
                'cost_reduction': self.cost_reduction,
                'efficiency_gain_hours': self.efficiency_gain_hours,
                'efficiency_gain_value': self.efficiency_gain_value,
                'error_reduction_value': self.error_reduction_value,
                'total': self.total_quantifiable_annual,
                'weighted_total': self.weighted_annual_benefit
            },
            'customer_impact': {
                'satisfaction_delta': self.customer_satisfaction_delta,
                'retention_delta': self.customer_retention_delta,
                'acquisition_value': self.customer_acquisition_value
            },
            'strategic_benefits': {
                'competitive_advantage': self.competitive_advantage,
                'innovation_score': self.innovation_score,
                'scalability_potential': self.scalability_potential,
                'data_asset_value': self.data_asset_value,
                'average_score': self.strategic_score
            },
            'risk_mitigation': {
                'compliance_risk_reduction': self.compliance_risk_reduction,
                'operational_risk_reduction': self.operational_risk_reduction
            },
            'confidence_levels': {
                'revenue': self.revenue_confidence,
                'cost_reduction': self.cost_reduction_confidence,
                'efficiency': self.efficiency_confidence
            },
            'currency': self.currency,
            'projection_years': self.projection_years
        }


@dataclass
class RiskAssessment:
    """Risk assessment for an AI initiative"""
    initiative_id: str

    # Technical risks (0-100, higher = more risk)
    model_performance_risk: float = 50.0   # Model may not meet targets
    data_quality_risk: float = 50.0        # Data issues may impact results
    integration_risk: float = 50.0         # System integration challenges
    scalability_risk: float = 50.0         # May not scale as expected
    technology_obsolescence: float = 30.0  # Tech may become outdated

    # Organizational risks
    adoption_risk: float = 50.0            # Users may not adopt
    skill_gap_risk: float = 50.0           # Team capability gaps
    change_resistance: float = 50.0        # Organizational resistance
    sponsor_dependency: float = 30.0       # Key sponsor dependency

    # External risks
    regulatory_risk: float = 30.0          # Regulatory changes
    vendor_dependency: float = 30.0        # Vendor lock-in/failure
    market_risk: float = 30.0              # Market changes
    competitive_risk: float = 30.0         # Competitors may leapfrog

    # AI-specific risks
    bias_fairness_risk: float = 50.0       # Model bias issues
    explainability_risk: float = 50.0      # Black-box concerns
    security_risk: float = 50.0            # AI-specific security
    ethical_risk: float = 30.0             # Ethical concerns

    # Mitigation status
    mitigations_planned: List[str] = field(default_factory=list)
    mitigations_implemented: List[str] = field(default_factory=list)

    @property
    def technical_risk_score(self) -> float:
        risks = [self.model_performance_risk, self.data_quality_risk,
                 self.integration_risk, self.scalability_risk,
                 self.technology_obsolescence]
        return sum(risks) / len(risks)

    @property
    def organizational_risk_score(self) -> float:
        risks = [self.adoption_risk, self.skill_gap_risk,
                 self.change_resistance, self.sponsor_dependency]
        return sum(risks) / len(risks)

    @property
    def external_risk_score(self) -> float:
        risks = [self.regulatory_risk, self.vendor_dependency,
                 self.market_risk, self.competitive_risk]
        return sum(risks) / len(risks)

    @property
    def ai_specific_risk_score(self) -> float:
        risks = [self.bias_fairness_risk, self.explainability_risk,
                 self.security_risk, self.ethical_risk]
        return sum(risks) / len(risks)

    @property
    def overall_risk_score(self) -> float:
        """Weighted overall risk score"""
        return (
            self.technical_risk_score * 0.30 +
            self.organizational_risk_score * 0.25 +
            self.external_risk_score * 0.20 +
            self.ai_specific_risk_score * 0.25
        )

    @property
    def risk_level(self) -> str:
        score = self.overall_risk_score
        if score >= 75:
            return RiskLevel.CRITICAL.value
        elif score >= 60:
            return RiskLevel.HIGH.value
        elif score >= 40:
            return RiskLevel.MEDIUM.value
        elif score >= 20:
            return RiskLevel.LOW.value
        else:
            return RiskLevel.MINIMAL.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initiative_id': self.initiative_id,
            'technical_risks': {
                'model_performance': self.model_performance_risk,
                'data_quality': self.data_quality_risk,
                'integration': self.integration_risk,
                'scalability': self.scalability_risk,
                'technology_obsolescence': self.technology_obsolescence,
                'average': self.technical_risk_score
            },
            'organizational_risks': {
                'adoption': self.adoption_risk,
                'skill_gap': self.skill_gap_risk,
                'change_resistance': self.change_resistance,
                'sponsor_dependency': self.sponsor_dependency,
                'average': self.organizational_risk_score
            },
            'external_risks': {
                'regulatory': self.regulatory_risk,
                'vendor_dependency': self.vendor_dependency,
                'market': self.market_risk,
                'competitive': self.competitive_risk,
                'average': self.external_risk_score
            },
            'ai_specific_risks': {
                'bias_fairness': self.bias_fairness_risk,
                'explainability': self.explainability_risk,
                'security': self.security_risk,
                'ethical': self.ethical_risk,
                'average': self.ai_specific_risk_score
            },
            'overall': {
                'score': self.overall_risk_score,
                'level': self.risk_level
            },
            'mitigations': {
                'planned': self.mitigations_planned,
                'implemented': self.mitigations_implemented
            }
        }


@dataclass
class AIInitiativeScore:
    """Complete value score for an AI initiative"""
    initiative_id: str
    name: str
    description: str
    initiative_type: str
    maturity_level: str

    # Component scores (0-100)
    financial_score: float = 0.0           # ROI, payback period
    strategic_score: float = 0.0           # Strategic alignment
    operational_score: float = 0.0         # Operational impact
    risk_adjusted_score: float = 0.0       # Risk-adjusted value

    # Financial metrics
    roi_percent: float = 0.0               # Return on Investment %
    npv: float = 0.0                       # Net Present Value
    payback_months: int = 0                # Payback period
    total_investment: float = 0.0
    total_benefit_3yr: float = 0.0

    # Composite scores
    overall_value_score: float = 0.0
    value_category: str = "medium"

    # Ranking
    priority_rank: int = 0
    investment_recommendation: str = ""

    # Supporting data
    costs: Optional[CostBreakdown] = None
    benefits: Optional[BenefitProjection] = None
    risks: Optional[RiskAssessment] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initiative_id': self.initiative_id,
            'name': self.name,
            'description': self.description,
            'initiative_type': self.initiative_type,
            'maturity_level': self.maturity_level,
            'scores': {
                'financial': round(self.financial_score, 2),
                'strategic': round(self.strategic_score, 2),
                'operational': round(self.operational_score, 2),
                'risk_adjusted': round(self.risk_adjusted_score, 2),
                'overall': round(self.overall_value_score, 2)
            },
            'financial_metrics': {
                'roi_percent': round(self.roi_percent, 2),
                'npv': round(self.npv, 2),
                'payback_months': self.payback_months,
                'total_investment': self.total_investment,
                'total_benefit_3yr': self.total_benefit_3yr
            },
            'classification': {
                'value_category': self.value_category,
                'priority_rank': self.priority_rank,
                'investment_recommendation': self.investment_recommendation
            },
            'costs': self.costs.to_dict() if self.costs else None,
            'benefits': self.benefits.to_dict() if self.benefits else None,
            'risks': self.risks.to_dict() if self.risks else None,
            'recommendations': self.recommendations,
            'next_steps': self.next_steps
        }


@dataclass
class AIInitiativeReport:
    """Complete AI business value analysis report"""
    analysis_timestamp: datetime
    initiatives_analyzed: int
    currency: str
    projection_years: int

    # Initiative scores
    initiative_scores: List[AIInitiativeScore] = field(default_factory=list)

    # Portfolio insights
    total_portfolio_investment: float = 0.0
    total_portfolio_benefit_3yr: float = 0.0
    portfolio_roi_percent: float = 0.0

    # Categorization
    transformational_initiatives: List[str] = field(default_factory=list)
    high_value_initiatives: List[str] = field(default_factory=list)
    quick_wins: List[str] = field(default_factory=list)  # Low cost, high value
    high_risk_initiatives: List[str] = field(default_factory=list)

    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)

    # Strategic recommendations
    portfolio_recommendations: List[str] = field(default_factory=list)
    investment_priorities: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'initiatives_analyzed': self.initiatives_analyzed,
            'currency': self.currency,
            'projection_years': self.projection_years,
            'initiative_scores': [s.to_dict() for s in self.initiative_scores],
            'portfolio_metrics': {
                'total_investment': self.total_portfolio_investment,
                'total_benefit_3yr': self.total_portfolio_benefit_3yr,
                'portfolio_roi_percent': round(self.portfolio_roi_percent, 2)
            },
            'categorization': {
                'transformational': self.transformational_initiatives,
                'high_value': self.high_value_initiatives,
                'quick_wins': self.quick_wins,
                'high_risk': self.high_risk_initiatives
            },
            'summary': self.summary,
            'portfolio_recommendations': self.portfolio_recommendations,
            'investment_priorities': self.investment_priorities
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Generate markdown report"""
        md = []
        md.append("# AI Business Value Report")
        md.append(f"\n**Analysis Date:** {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Initiatives Analyzed:** {self.initiatives_analyzed}")
        md.append(f"**Currency:** {self.currency}")
        md.append(f"**Projection Period:** {self.projection_years} years\n")

        # Portfolio Summary
        md.append("## Portfolio Summary\n")
        md.append(f"- **Total Investment:** {self.currency} {self.total_portfolio_investment:,.2f}")
        md.append(f"- **Total 3-Year Benefit:** {self.currency} {self.total_portfolio_benefit_3yr:,.2f}")
        md.append(f"- **Portfolio ROI:** {self.portfolio_roi_percent:.1f}%\n")

        # Summary Metrics
        if self.summary:
            md.append("### Key Metrics\n")
            for key, value in self.summary.items():
                display_key = key.replace('_', ' ').title()
                md.append(f"- **{display_key}:** {value}")

        # Transformational Initiatives
        if self.transformational_initiatives:
            md.append("\n## Transformational Initiatives\n")
            md.append("High strategic value initiatives that can transform the business:\n")
            for init in self.transformational_initiatives:
                md.append(f"- `{init}`")

        # Quick Wins
        if self.quick_wins:
            md.append("\n## Quick Wins\n")
            md.append("Low investment, high value initiatives for immediate action:\n")
            for init in self.quick_wins:
                md.append(f"- `{init}`")

        # High Risk Initiatives
        if self.high_risk_initiatives:
            md.append("\n## High Risk Initiatives\n")
            md.append("Initiatives requiring additional risk mitigation:\n")
            for init in self.high_risk_initiatives:
                md.append(f"- `{init}`")

        # Initiative Scores Table
        md.append("\n## Initiative Value Scores\n")
        md.append("| Initiative | Type | Value Score | ROI % | Payback | Risk | Category |")
        md.append("|------------|------|-------------|-------|---------|------|----------|")

        for score in sorted(self.initiative_scores,
                           key=lambda x: x.overall_value_score, reverse=True):
            risk_level = score.risks.risk_level if score.risks else "N/A"
            md.append(
                f"| {score.name} | {score.initiative_type} | "
                f"{score.overall_value_score:.1f} | {score.roi_percent:.1f}% | "
                f"{score.payback_months}mo | {risk_level} | {score.value_category} |"
            )

        # Investment Priorities
        if self.investment_priorities:
            md.append("\n## Investment Priorities\n")
            for i, priority in enumerate(self.investment_priorities, 1):
                md.append(f"\n### {i}. {priority.get('initiative', 'Unknown')}")
                md.append(f"- **Investment:** {self.currency} {priority.get('investment', 0):,.2f}")
                md.append(f"- **Expected ROI:** {priority.get('roi', 0):.1f}%")
                md.append(f"- **Rationale:** {priority.get('rationale', 'N/A')}")

        # Portfolio Recommendations
        if self.portfolio_recommendations:
            md.append("\n## Portfolio Recommendations\n")
            for i, rec in enumerate(self.portfolio_recommendations, 1):
                md.append(f"{i}. {rec}")

        # Individual Initiative Details
        md.append("\n## Initiative Details\n")
        for score in self.initiative_scores:
            md.append(f"\n### {score.name}")
            md.append(f"**Type:** {score.initiative_type} | **Maturity:** {score.maturity_level}")
            md.append(f"\n{score.description}\n")

            md.append("**Scores:**")
            md.append(f"- Financial: {score.financial_score:.1f}")
            md.append(f"- Strategic: {score.strategic_score:.1f}")
            md.append(f"- Operational: {score.operational_score:.1f}")
            md.append(f"- Risk-Adjusted: {score.risk_adjusted_score:.1f}")

            if score.recommendations:
                md.append("\n**Recommendations:**")
                for rec in score.recommendations:
                    md.append(f"- {rec}")

            if score.next_steps:
                md.append("\n**Next Steps:**")
                for step in score.next_steps:
                    md.append(f"- {step}")

        return "\n".join(md)


class ROICalculator:
    """Calculator for ROI and financial metrics"""

    def __init__(self, discount_rate: float = 0.10):
        """
        Initialize ROI Calculator.

        Args:
            discount_rate: Annual discount rate for NPV calculation (default 10%)
        """
        self.discount_rate = discount_rate

    def calculate_roi(
        self,
        total_investment: float,
        total_benefit: float,
        years: int = 3
    ) -> float:
        """
        Calculate Return on Investment percentage.

        ROI = ((Total Benefit - Total Investment) / Total Investment) * 100
        """
        if total_investment == 0:
            return 0.0

        net_benefit = total_benefit - total_investment
        roi = (net_benefit / total_investment) * 100
        return round(roi, 2)

    def calculate_npv(
        self,
        initial_investment: float,
        annual_benefits: List[float],
        annual_costs: List[float]
    ) -> float:
        """
        Calculate Net Present Value.

        NPV = -Initial Investment + Sum of (Net Cash Flow / (1 + r)^t)
        """
        npv = -initial_investment

        for t, (benefit, cost) in enumerate(zip(annual_benefits, annual_costs), 1):
            net_cash_flow = benefit - cost
            discounted = net_cash_flow / ((1 + self.discount_rate) ** t)
            npv += discounted

        return round(npv, 2)

    def calculate_payback_period(
        self,
        initial_investment: float,
        annual_benefit: float,
        annual_cost: float = 0
    ) -> int:
        """
        Calculate payback period in months.

        Returns the number of months to recover the initial investment.
        """
        if annual_benefit <= annual_cost:
            return 999  # No payback

        net_annual = annual_benefit - annual_cost
        years = initial_investment / net_annual
        months = int(years * 12)

        return min(months, 120)  # Cap at 10 years

    def calculate_irr(
        self,
        initial_investment: float,
        annual_benefits: List[float],
        annual_costs: List[float],
        max_iterations: int = 100
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.
        """
        cash_flows = [-initial_investment]
        for benefit, cost in zip(annual_benefits, annual_costs):
            cash_flows.append(benefit - cost)

        # Initial guess
        rate = 0.1

        for _ in range(max_iterations):
            npv = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

            # Derivative
            d_npv = sum(-t * cf / ((1 + rate) ** (t + 1))
                       for t, cf in enumerate(cash_flows) if t > 0)

            if abs(d_npv) < 1e-10:
                break

            new_rate = rate - npv / d_npv

            if abs(new_rate - rate) < 1e-6:
                return round(new_rate * 100, 2)

            rate = new_rate

        return round(rate * 100, 2)


class ValueCalculator:
    """Calculator for AI initiative business value scores"""

    # Weights for overall score calculation
    DEFAULT_WEIGHTS = {
        'financial': 0.35,
        'strategic': 0.25,
        'operational': 0.20,
        'risk_adjusted': 0.20
    }

    # Category thresholds
    CATEGORY_THRESHOLDS = {
        'transformational': 85,
        'high': 70,
        'medium': 50,
        'low': 30
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        roi_calculator: Optional[ROICalculator] = None
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.roi_calculator = roi_calculator or ROICalculator()

        # Normalize weights
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v/total for k, v in self.weights.items()}

    def calculate_financial_score(
        self,
        costs: CostBreakdown,
        benefits: BenefitProjection,
        projection_years: int = 3
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate financial score based on ROI, NPV, and payback.

        Returns:
            Tuple of (score 0-100, financial metrics dict)
        """
        total_investment = costs.total_cost_year1
        annual_benefit = benefits.weighted_annual_benefit
        annual_cost = costs.total_operational_annual

        # Calculate total benefit over projection period
        total_benefit = annual_benefit * projection_years

        # Calculate financial metrics
        roi = self.roi_calculator.calculate_roi(
            total_investment, total_benefit, projection_years
        )

        # Create annual projections for NPV
        annual_benefits = [annual_benefit * (1 + 0.05 * i)
                         for i in range(projection_years)]
        annual_costs = [annual_cost] * projection_years

        npv = self.roi_calculator.calculate_npv(
            total_investment, annual_benefits, annual_costs
        )

        payback_months = self.roi_calculator.calculate_payback_period(
            total_investment, annual_benefit, annual_cost
        )

        # Score calculation
        # ROI score (higher is better, cap at 200%)
        roi_score = min(100, max(0, (roi / 200) * 100))

        # Payback score (faster is better, ideal < 12 months)
        if payback_months <= 12:
            payback_score = 100
        elif payback_months <= 24:
            payback_score = 80
        elif payback_months <= 36:
            payback_score = 60
        elif payback_months <= 48:
            payback_score = 40
        else:
            payback_score = max(0, 40 - (payback_months - 48) / 2)

        # NPV score (positive is good)
        if npv > 0:
            npv_score = min(100, 50 + (npv / total_investment) * 50)
        else:
            npv_score = max(0, 50 + (npv / total_investment) * 50)

        # Weighted financial score
        score = roi_score * 0.4 + payback_score * 0.3 + npv_score * 0.3

        metrics = {
            'roi_percent': roi,
            'npv': npv,
            'payback_months': payback_months,
            'total_investment': total_investment,
            'total_benefit_3yr': total_benefit,
            'annual_benefit': annual_benefit
        }

        return round(score, 2), metrics

    def calculate_strategic_score(
        self,
        benefits: BenefitProjection,
        strategic_alignment: float = 50.0,
        executive_sponsorship: float = 50.0,
        market_timing: float = 50.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate strategic value score.

        Args:
            benefits: Benefit projections with strategic scores
            strategic_alignment: Alignment with business strategy (0-100)
            executive_sponsorship: Level of executive support (0-100)
            market_timing: Market opportunity timing (0-100)

        Returns:
            Tuple of (score 0-100, strategic metrics dict)
        """
        # Base strategic score from benefits
        base_score = benefits.strategic_score

        # Weighted factors
        score = (
            base_score * 0.30 +
            strategic_alignment * 0.30 +
            executive_sponsorship * 0.20 +
            market_timing * 0.20
        )

        metrics = {
            'competitive_advantage': benefits.competitive_advantage,
            'innovation_score': benefits.innovation_score,
            'scalability_potential': benefits.scalability_potential,
            'strategic_alignment': strategic_alignment,
            'executive_sponsorship': executive_sponsorship,
            'market_timing': market_timing
        }

        return round(score, 2), metrics

    def calculate_operational_score(
        self,
        benefits: BenefitProjection,
        implementation_readiness: float = 50.0,
        team_capability: float = 50.0,
        infrastructure_readiness: float = 50.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate operational impact score.
        """
        # Efficiency impact
        if benefits.efficiency_gain_hours > 0:
            efficiency_score = min(100, benefits.efficiency_gain_hours / 100)
        else:
            efficiency_score = 0

        # Customer impact
        customer_score = (
            benefits.customer_satisfaction_delta * 2 +
            benefits.customer_retention_delta * 3
        )
        customer_score = min(100, max(0, customer_score))

        # Risk reduction impact
        risk_reduction_score = (
            benefits.compliance_risk_reduction +
            benefits.operational_risk_reduction
        ) / 2

        # Readiness factors
        readiness_score = (
            implementation_readiness * 0.4 +
            team_capability * 0.3 +
            infrastructure_readiness * 0.3
        )

        # Combined score
        score = (
            efficiency_score * 0.30 +
            customer_score * 0.25 +
            risk_reduction_score * 0.20 +
            readiness_score * 0.25
        )

        metrics = {
            'efficiency_score': efficiency_score,
            'customer_impact_score': customer_score,
            'risk_reduction_score': risk_reduction_score,
            'readiness_score': readiness_score,
            'implementation_readiness': implementation_readiness,
            'team_capability': team_capability,
            'infrastructure_readiness': infrastructure_readiness
        }

        return round(score, 2), metrics

    def calculate_risk_adjusted_score(
        self,
        base_score: float,
        risks: RiskAssessment
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Adjust score based on risk assessment.
        """
        risk_score = risks.overall_risk_score

        # Risk multiplier (lower risk = higher multiplier)
        if risk_score < 20:
            risk_multiplier = 1.1
        elif risk_score < 40:
            risk_multiplier = 1.0
        elif risk_score < 60:
            risk_multiplier = 0.85
        elif risk_score < 80:
            risk_multiplier = 0.70
        else:
            risk_multiplier = 0.50

        # Mitigation bonus
        mitigation_count = len(risks.mitigations_implemented)
        mitigation_bonus = min(0.1, mitigation_count * 0.02)

        adjusted_score = base_score * (risk_multiplier + mitigation_bonus)
        adjusted_score = min(100, max(0, adjusted_score))

        metrics = {
            'base_score': base_score,
            'risk_score': risk_score,
            'risk_level': risks.risk_level,
            'risk_multiplier': risk_multiplier,
            'mitigation_bonus': mitigation_bonus,
            'mitigations_count': mitigation_count
        }

        return round(adjusted_score, 2), metrics

    def calculate_overall_score(
        self,
        financial_score: float,
        strategic_score: float,
        operational_score: float,
        risk_adjusted_score: float
    ) -> float:
        """Calculate weighted overall value score"""
        score = (
            financial_score * self.weights['financial'] +
            strategic_score * self.weights['strategic'] +
            operational_score * self.weights['operational'] +
            risk_adjusted_score * self.weights['risk_adjusted']
        )
        return round(score, 2)

    def categorize_value(self, score: float) -> str:
        """Categorize initiative value based on score"""
        if score >= self.CATEGORY_THRESHOLDS['transformational']:
            return ValueCategory.TRANSFORMATIONAL.value
        elif score >= self.CATEGORY_THRESHOLDS['high']:
            return ValueCategory.HIGH.value
        elif score >= self.CATEGORY_THRESHOLDS['medium']:
            return ValueCategory.MEDIUM.value
        elif score >= self.CATEGORY_THRESHOLDS['low']:
            return ValueCategory.LOW.value
        else:
            return ValueCategory.EXPERIMENTAL.value

    def generate_investment_recommendation(
        self,
        score: float,
        roi: float,
        payback_months: int,
        risk_level: str
    ) -> str:
        """Generate investment recommendation"""
        if score >= 85 and roi > 100 and risk_level in ('low', 'minimal'):
            return "INVEST - High priority strategic initiative"
        elif score >= 70 and roi > 50:
            return "INVEST - Strong business case"
        elif score >= 50 and roi > 0:
            return "CONSIDER - Moderate value, review priorities"
        elif score >= 30:
            return "DEFER - Limited value, consider alternatives"
        else:
            return "DECLINE - Insufficient business case"


class AIBusinessValueAgent:
    """
    Agent for analyzing AI initiative business value.

    This agent provides comprehensive analysis of AI initiatives including:
    - Cost-benefit analysis
    - ROI and NPV calculations
    - Risk assessment
    - Strategic alignment scoring
    - Portfolio optimization recommendations

    Example usage:
        agent = AIBusinessValueAgent()

        report = agent.analyze_initiatives(
            initiatives=initiatives_data,
            cost_data=costs,
            benefit_projections=benefits
        )

        print(report.to_markdown())
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        discount_rate: float = 0.10,
        currency: str = "BRL",
        projection_years: int = 3,
        persist_dir: Optional[str] = None
    ):
        """
        Initialize the AI Business Value Agent.

        Args:
            weights: Custom weights for value calculation
            discount_rate: Discount rate for NPV calculation
            currency: Default currency for reports
            projection_years: Default projection period
            persist_dir: Directory for persisting reports
        """
        self.roi_calculator = ROICalculator(discount_rate)
        self.value_calculator = ValueCalculator(weights, self.roi_calculator)
        self.currency = currency
        self.projection_years = projection_years
        self.persist_dir = persist_dir

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    def analyze_initiatives(
        self,
        initiatives: List[Dict[str, Any]],
        cost_data: Optional[Dict[str, CostBreakdown]] = None,
        benefit_projections: Optional[Dict[str, BenefitProjection]] = None,
        risk_assessments: Optional[Dict[str, RiskAssessment]] = None,
        strategic_context: Optional[Dict[str, Any]] = None
    ) -> AIInitiativeReport:
        """
        Analyze multiple AI initiatives and generate comprehensive report.

        Args:
            initiatives: List of initiative definitions
            cost_data: Cost breakdowns by initiative_id
            benefit_projections: Benefit projections by initiative_id
            risk_assessments: Risk assessments by initiative_id
            strategic_context: Strategic context (alignment scores, etc.)

        Returns:
            AIInitiativeReport with complete analysis
        """
        cost_data = cost_data or {}
        benefit_projections = benefit_projections or {}
        risk_assessments = risk_assessments or {}
        strategic_context = strategic_context or {}

        initiative_scores = []
        total_investment = 0.0
        total_benefit = 0.0

        for init_data in initiatives:
            init_id = init_data.get('id', init_data.get('name', 'unknown'))

            # Get or create component data
            costs = cost_data.get(init_id) or self._create_default_costs(init_id, init_data)
            benefits = benefit_projections.get(init_id) or self._create_default_benefits(init_id, init_data)
            risks = risk_assessments.get(init_id) or self._create_default_risks(init_id, init_data)

            # Calculate scores
            score = self._calculate_initiative_score(
                init_data=init_data,
                costs=costs,
                benefits=benefits,
                risks=risks,
                strategic_context=strategic_context.get(init_id, {})
            )

            initiative_scores.append(score)
            total_investment += score.total_investment
            total_benefit += score.total_benefit_3yr

        # Sort by overall score
        initiative_scores.sort(key=lambda x: x.overall_value_score, reverse=True)

        # Assign priority ranks
        for i, score in enumerate(initiative_scores, 1):
            score.priority_rank = i

        # Generate report
        report = self._generate_report(
            initiative_scores=initiative_scores,
            total_investment=total_investment,
            total_benefit=total_benefit
        )

        # Persist if configured
        if self.persist_dir:
            self._persist_report(report)

        return report

    def analyze_single_initiative(
        self,
        initiative: Dict[str, Any],
        costs: Optional[CostBreakdown] = None,
        benefits: Optional[BenefitProjection] = None,
        risks: Optional[RiskAssessment] = None,
        strategic_context: Optional[Dict[str, Any]] = None
    ) -> AIInitiativeScore:
        """
        Analyze a single AI initiative.

        Args:
            initiative: Initiative definition
            costs: Cost breakdown
            benefits: Benefit projection
            risks: Risk assessment
            strategic_context: Strategic alignment info

        Returns:
            AIInitiativeScore with complete analysis
        """
        init_id = initiative.get('id', initiative.get('name', 'unknown'))

        costs = costs or self._create_default_costs(init_id, initiative)
        benefits = benefits or self._create_default_benefits(init_id, initiative)
        risks = risks or self._create_default_risks(init_id, initiative)
        strategic_context = strategic_context or {}

        return self._calculate_initiative_score(
            init_data=initiative,
            costs=costs,
            benefits=benefits,
            risks=risks,
            strategic_context=strategic_context
        )

    def _calculate_initiative_score(
        self,
        init_data: Dict[str, Any],
        costs: CostBreakdown,
        benefits: BenefitProjection,
        risks: RiskAssessment,
        strategic_context: Dict[str, Any]
    ) -> AIInitiativeScore:
        """Calculate complete score for an initiative"""

        # Extract initiative info
        init_id = init_data.get('id', init_data.get('name', 'unknown'))
        name = init_data.get('name', init_id)
        description = init_data.get('description', '')
        init_type = init_data.get('type', InitiativeType.AUTOMATION.value)
        maturity = init_data.get('maturity', MaturityLevel.IDEATION.value)

        # Calculate component scores
        financial_score, financial_metrics = self.value_calculator.calculate_financial_score(
            costs, benefits, self.projection_years
        )

        strategic_score, _ = self.value_calculator.calculate_strategic_score(
            benefits,
            strategic_context.get('alignment', 50),
            strategic_context.get('sponsorship', 50),
            strategic_context.get('market_timing', 50)
        )

        operational_score, _ = self.value_calculator.calculate_operational_score(
            benefits,
            strategic_context.get('implementation_readiness', 50),
            strategic_context.get('team_capability', 50),
            strategic_context.get('infrastructure_readiness', 50)
        )

        # Base score before risk adjustment
        base_score = (
            financial_score * 0.4 +
            strategic_score * 0.3 +
            operational_score * 0.3
        )

        risk_adjusted_score, _ = self.value_calculator.calculate_risk_adjusted_score(
            base_score, risks
        )

        # Calculate overall score
        overall_score = self.value_calculator.calculate_overall_score(
            financial_score, strategic_score, operational_score, risk_adjusted_score
        )

        # Categorize
        value_category = self.value_calculator.categorize_value(overall_score)

        # Investment recommendation
        investment_rec = self.value_calculator.generate_investment_recommendation(
            overall_score,
            financial_metrics['roi_percent'],
            financial_metrics['payback_months'],
            risks.risk_level
        )

        # Generate recommendations
        recommendations = self._generate_initiative_recommendations(
            overall_score=overall_score,
            financial_score=financial_score,
            strategic_score=strategic_score,
            risk_level=risks.risk_level,
            maturity=maturity
        )

        # Generate next steps
        next_steps = self._generate_next_steps(
            value_category=value_category,
            maturity=maturity,
            risk_level=risks.risk_level
        )

        return AIInitiativeScore(
            initiative_id=init_id,
            name=name,
            description=description,
            initiative_type=init_type,
            maturity_level=maturity,
            financial_score=financial_score,
            strategic_score=strategic_score,
            operational_score=operational_score,
            risk_adjusted_score=risk_adjusted_score,
            roi_percent=financial_metrics['roi_percent'],
            npv=financial_metrics['npv'],
            payback_months=financial_metrics['payback_months'],
            total_investment=financial_metrics['total_investment'],
            total_benefit_3yr=financial_metrics['total_benefit_3yr'],
            overall_value_score=overall_score,
            value_category=value_category,
            investment_recommendation=investment_rec,
            costs=costs,
            benefits=benefits,
            risks=risks,
            recommendations=recommendations,
            next_steps=next_steps
        )

    def _create_default_costs(
        self,
        init_id: str,
        init_data: Dict[str, Any]
    ) -> CostBreakdown:
        """Create default cost breakdown from initiative data"""
        return CostBreakdown(
            initiative_id=init_id,
            development_internal=init_data.get('dev_cost_internal', 0),
            development_external=init_data.get('dev_cost_external', 0),
            compute_training=init_data.get('compute_cost', 0),
            compute_inference=init_data.get('inference_cost', 0),
            maintenance_annual=init_data.get('maintenance_cost', 0),
            integration=init_data.get('integration_cost', 0),
            currency=self.currency
        )

    def _create_default_benefits(
        self,
        init_id: str,
        init_data: Dict[str, Any]
    ) -> BenefitProjection:
        """Create default benefit projection from initiative data"""
        return BenefitProjection(
            initiative_id=init_id,
            revenue_increase=init_data.get('expected_revenue', 0),
            cost_reduction=init_data.get('expected_savings', 0),
            efficiency_gain_hours=init_data.get('hours_saved', 0),
            efficiency_gain_value=init_data.get('efficiency_value', 0),
            competitive_advantage=init_data.get('competitive_score', 50),
            innovation_score=init_data.get('innovation_score', 50),
            currency=self.currency,
            projection_years=self.projection_years
        )

    def _create_default_risks(
        self,
        init_id: str,
        init_data: Dict[str, Any]
    ) -> RiskAssessment:
        """Create default risk assessment from initiative data"""
        return RiskAssessment(
            initiative_id=init_id,
            model_performance_risk=init_data.get('tech_risk', 50),
            data_quality_risk=init_data.get('data_risk', 50),
            adoption_risk=init_data.get('adoption_risk', 50),
            regulatory_risk=init_data.get('regulatory_risk', 30),
            bias_fairness_risk=init_data.get('bias_risk', 50)
        )

    def _generate_initiative_recommendations(
        self,
        overall_score: float,
        financial_score: float,
        strategic_score: float,
        risk_level: str,
        maturity: str
    ) -> List[str]:
        """Generate recommendations for an initiative"""
        recommendations = []

        if overall_score >= 80:
            recommendations.append(
                "High-value initiative - prioritize resources and executive attention"
            )

        if financial_score < 50 and strategic_score > 70:
            recommendations.append(
                "Strong strategic value but weak financials - explore cost optimization"
            )

        if risk_level in ('high', 'critical'):
            recommendations.append(
                "High risk profile - develop comprehensive risk mitigation plan"
            )

        if maturity == MaturityLevel.IDEATION.value:
            recommendations.append(
                "Early stage - consider proof-of-concept before major investment"
            )

        if maturity == MaturityLevel.PILOT.value and overall_score > 60:
            recommendations.append(
                "Successful pilot with good scores - prepare scaling plan"
            )

        return recommendations

    def _generate_next_steps(
        self,
        value_category: str,
        maturity: str,
        risk_level: str
    ) -> List[str]:
        """Generate next steps for an initiative"""
        next_steps = []

        if value_category in ('transformational', 'high'):
            next_steps.append("Schedule executive steering committee review")
            next_steps.append("Allocate dedicated project team")

        if maturity == MaturityLevel.IDEATION.value:
            next_steps.append("Define success criteria and KPIs")
            next_steps.append("Develop proof-of-concept scope")
        elif maturity == MaturityLevel.POC.value:
            next_steps.append("Document POC results and learnings")
            next_steps.append("Prepare pilot deployment plan")
        elif maturity == MaturityLevel.PILOT.value:
            next_steps.append("Gather user feedback and adoption metrics")
            next_steps.append("Develop production deployment roadmap")

        if risk_level in ('high', 'critical'):
            next_steps.append("Conduct detailed risk workshop")
            next_steps.append("Implement priority risk mitigations")

        return next_steps

    def _generate_report(
        self,
        initiative_scores: List[AIInitiativeScore],
        total_investment: float,
        total_benefit: float
    ) -> AIInitiativeReport:
        """Generate comprehensive portfolio report"""

        # Calculate portfolio ROI
        portfolio_roi = 0.0
        if total_investment > 0:
            portfolio_roi = ((total_benefit - total_investment) / total_investment) * 100

        # Categorize initiatives
        transformational = [s.name for s in initiative_scores
                          if s.value_category == 'transformational']
        high_value = [s.name for s in initiative_scores
                     if s.value_category == 'high']

        # Find quick wins (high score, low investment, fast payback)
        avg_investment = total_investment / len(initiative_scores) if initiative_scores else 0
        quick_wins = [
            s.name for s in initiative_scores
            if s.overall_value_score >= 60
            and s.total_investment < avg_investment
            and s.payback_months <= 18
        ]

        # Find high risk initiatives
        high_risk = [
            s.name for s in initiative_scores
            if s.risks and s.risks.risk_level in ('high', 'critical')
        ]

        # Generate summary
        summary = {
            'total_initiatives': len(initiative_scores),
            'average_value_score': round(
                sum(s.overall_value_score for s in initiative_scores) / len(initiative_scores), 2
            ) if initiative_scores else 0,
            'average_roi': round(
                sum(s.roi_percent for s in initiative_scores) / len(initiative_scores), 2
            ) if initiative_scores else 0,
            'transformational_count': len(transformational),
            'high_value_count': len(high_value),
            'quick_wins_count': len(quick_wins),
            'high_risk_count': len(high_risk)
        }

        # Generate portfolio recommendations
        portfolio_recommendations = self._generate_portfolio_recommendations(
            initiative_scores, transformational, quick_wins, high_risk
        )

        # Generate investment priorities
        investment_priorities = self._generate_investment_priorities(initiative_scores)

        return AIInitiativeReport(
            analysis_timestamp=datetime.now(),
            initiatives_analyzed=len(initiative_scores),
            currency=self.currency,
            projection_years=self.projection_years,
            initiative_scores=initiative_scores,
            total_portfolio_investment=total_investment,
            total_portfolio_benefit_3yr=total_benefit,
            portfolio_roi_percent=portfolio_roi,
            transformational_initiatives=transformational,
            high_value_initiatives=high_value,
            quick_wins=quick_wins,
            high_risk_initiatives=high_risk,
            summary=summary,
            portfolio_recommendations=portfolio_recommendations,
            investment_priorities=investment_priorities
        )

    def _generate_portfolio_recommendations(
        self,
        scores: List[AIInitiativeScore],
        transformational: List[str],
        quick_wins: List[str],
        high_risk: List[str]
    ) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []

        if transformational:
            recommendations.append(
                f"Prioritize {len(transformational)} transformational initiative(s) "
                f"for strategic investment and executive sponsorship."
            )

        if quick_wins:
            recommendations.append(
                f"Execute {len(quick_wins)} quick win(s) to demonstrate early value "
                f"and build organizational momentum for AI adoption."
            )

        if high_risk:
            recommendations.append(
                f"Address {len(high_risk)} high-risk initiative(s) with dedicated "
                f"risk mitigation plans before proceeding."
            )

        # Portfolio balance recommendation
        types = [s.initiative_type for s in scores]
        type_counts = {t: types.count(t) for t in set(types)}

        if len(type_counts) < 3:
            recommendations.append(
                "Consider diversifying AI portfolio across more initiative types "
                "to balance risk and capture different value opportunities."
            )

        # Maturity distribution
        maturities = [s.maturity_level for s in scores]
        early_stage = sum(1 for m in maturities
                        if m in ('ideation', 'poc'))

        if early_stage > len(scores) * 0.6:
            recommendations.append(
                "High concentration of early-stage initiatives. "
                "Focus on advancing key initiatives through the maturity pipeline."
            )

        return recommendations

    def _generate_investment_priorities(
        self,
        scores: List[AIInitiativeScore]
    ) -> List[Dict[str, Any]]:
        """Generate ranked investment priorities"""
        priorities = []

        for score in scores[:5]:  # Top 5 priorities
            rationale = self._generate_priority_rationale(score)

            priorities.append({
                'initiative': score.name,
                'rank': score.priority_rank,
                'investment': score.total_investment,
                'roi': score.roi_percent,
                'value_score': score.overall_value_score,
                'category': score.value_category,
                'rationale': rationale
            })

        return priorities

    def _generate_priority_rationale(
        self,
        score: AIInitiativeScore
    ) -> str:
        """Generate rationale for investment priority"""
        reasons = []

        if score.overall_value_score >= 80:
            reasons.append("excellent overall value score")
        elif score.overall_value_score >= 60:
            reasons.append("strong value proposition")

        if score.roi_percent >= 100:
            reasons.append(f"high ROI ({score.roi_percent:.0f}%)")

        if score.payback_months <= 12:
            reasons.append("fast payback period")

        if score.strategic_score >= 70:
            reasons.append("strong strategic alignment")

        if score.risks and score.risks.risk_level in ('low', 'minimal'):
            reasons.append("favorable risk profile")

        if not reasons:
            reasons.append("balanced business case")

        return " | ".join(reasons).capitalize()

    def _persist_report(self, report: AIInitiativeReport) -> str:
        """Persist report to disk"""
        if not self.persist_dir:
            return ""

        timestamp = report.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"ai_business_value_report_{timestamp}.json"
        filepath = os.path.join(self.persist_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_json())

        return filepath

    def get_initiative_value(
        self,
        initiative_id: str,
        report: AIInitiativeReport
    ) -> Optional[AIInitiativeScore]:
        """Get value score for a specific initiative from a report"""
        for score in report.initiative_scores:
            if score.initiative_id == initiative_id or score.name == initiative_id:
                return score
        return None

    def compare_initiatives(
        self,
        initiative_ids: List[str],
        report: AIInitiativeReport
    ) -> List[Dict[str, Any]]:
        """Compare multiple initiatives side by side"""
        comparison = []

        for init_id in initiative_ids:
            score = self.get_initiative_value(init_id, report)
            if score:
                comparison.append({
                    'initiative': score.name,
                    'overall_score': score.overall_value_score,
                    'category': score.value_category,
                    'financial': score.financial_score,
                    'strategic': score.strategic_score,
                    'operational': score.operational_score,
                    'roi_percent': score.roi_percent,
                    'payback_months': score.payback_months,
                    'risk_level': score.risks.risk_level if score.risks else 'N/A'
                })

        return sorted(comparison, key=lambda x: x['overall_score'], reverse=True)
