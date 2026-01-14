"""
AI Business Value Agent

This module provides tools for analyzing and scoring AI initiatives
based on business value, ROI, risk assessment, and strategic alignment.

Main Components:
- AIBusinessValueAgent: Main agent for AI initiative value analysis
- AIInitiativeReport: Comprehensive analysis report
- AIInitiativeScore: Individual initiative value scores
- ValueCalculator: Business value scoring logic
- ROICalculator: Return on Investment calculations

Example Usage:
    from ai_business_value import AIBusinessValueAgent, AIInitiativeReport

    agent = AIBusinessValueAgent()

    # Analyze AI initiatives
    report = agent.analyze_initiatives(
        initiatives=initiatives_data,
        cost_data=cost_info,
        benefit_projections=benefits
    )

    # Get markdown report
    print(report.to_markdown())

    # Get specific initiative value
    chatbot_value = agent.get_initiative_value('customer_chatbot', report)
"""

from .agent import (
    # Main Agent
    AIBusinessValueAgent,

    # Report Classes
    AIInitiativeReport,
    AIInitiativeScore,
    CostBreakdown,
    BenefitProjection,
    RiskAssessment,

    # Calculator
    ValueCalculator,
    ROICalculator,

    # Enums
    ValueCategory,
    InitiativeType,
    RiskLevel,
    MaturityLevel,
)

__all__ = [
    # Main Agent
    'AIBusinessValueAgent',

    # Report Classes
    'AIInitiativeReport',
    'AIInitiativeScore',
    'CostBreakdown',
    'BenefitProjection',
    'RiskAssessment',

    # Calculator
    'ValueCalculator',
    'ROICalculator',

    # Enums
    'ValueCategory',
    'InitiativeType',
    'RiskLevel',
    'MaturityLevel',
]

__version__ = '1.0.0'
