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
Example usage of the AI Business Value Agent

This example demonstrates how to:
1. Define AI initiatives with costs and benefits
2. Analyze individual initiatives
3. Analyze a portfolio of initiatives
4. Generate reports and recommendations
"""

import sys
sys.path.insert(0, '..')

from agent import (
    AIBusinessValueAgent,
    CostBreakdown,
    BenefitProjection,
    RiskAssessment,
    InitiativeType,
    MaturityLevel
)


def example_single_initiative():
    """Analyze a single AI initiative"""
    print("=" * 60)
    print("Example: Single Initiative Analysis")
    print("=" * 60)

    # Initialize agent
    agent = AIBusinessValueAgent(
        currency="BRL",
        projection_years=3,
        discount_rate=0.10
    )

    # Define initiative
    initiative = {
        "id": "chatbot_atendimento",
        "name": "Chatbot de Atendimento ao Cliente",
        "description": "Implementacao de chatbot com IA generativa para atendimento "
                       "automatizado de clientes, reduzindo tempo de espera e custos operacionais.",
        "type": InitiativeType.CUSTOMER_EXPERIENCE.value,
        "maturity": MaturityLevel.PILOT.value
    }

    # Define costs
    costs = CostBreakdown(
        initiative_id="chatbot_atendimento",
        # Development
        development_internal=150000,    # 3 devs x 50k each
        development_external=80000,     # Consultoria especializada
        development_tools=15000,        # Ferramentas ML/LLM

        # Infrastructure
        compute_training=25000,         # Fine-tuning do modelo
        compute_inference=48000,        # API calls (anual)
        storage=6000,                   # Logs e historico

        # Operational (annual)
        maintenance_annual=36000,       # 1 dev meio periodo
        monitoring_annual=12000,        # Observabilidade
        licensing_annual=24000,         # Licencas LLM

        # One-time
        integration=45000,              # Integracao com CRM/sistemas
        training_users=20000,           # Treinamento da equipe
        change_management=15000,        # Gestao de mudanca

        currency="BRL"
    )

    # Define benefits
    benefits = BenefitProjection(
        initiative_id="chatbot_atendimento",

        # Quantifiable (annual)
        cost_reduction=240000,          # Reducao de 4 atendentes
        efficiency_gain_hours=6000,     # Horas de atendimento automatizado
        efficiency_gain_value=180000,   # Valor das horas economizadas
        error_reduction_value=30000,    # Menos erros de atendimento

        # Customer impact
        customer_satisfaction_delta=12, # NPS melhoria
        customer_retention_delta=3,     # Retencao +3%
        customer_acquisition_value=50000,  # Novos clientes

        # Strategic (0-100)
        competitive_advantage=70,       # Diferenciacao no mercado
        innovation_score=65,            # Capacidade de inovacao
        scalability_potential=85,       # Pode escalar facilmente
        data_asset_value=60,            # Dados de conversas

        # Confidence levels
        revenue_confidence=50,
        cost_reduction_confidence=80,
        efficiency_confidence=85,

        currency="BRL",
        projection_years=3
    )

    # Define risks
    risks = RiskAssessment(
        initiative_id="chatbot_atendimento",

        # Technical risks
        model_performance_risk=35,      # Modelo bem testado
        data_quality_risk=30,           # Dados de treinamento bons
        integration_risk=45,            # Integracao com sistemas legados
        scalability_risk=25,            # Arquitetura escalavel

        # Organizational risks
        adoption_risk=40,               # Alguma resistencia esperada
        skill_gap_risk=35,              # Equipe precisa capacitacao
        change_resistance=45,           # Atendentes podem resistir

        # AI-specific risks
        bias_fairness_risk=30,          # Baixo risco de vies
        explainability_risk=50,         # LLM tem explicabilidade limitada
        security_risk=35,               # Dados de clientes
        ethical_risk=25,                # Uso etico do chatbot

        # Mitigations
        mitigations_planned=[
            "Plano de treinamento para equipe",
            "Auditoria trimestral de vieses",
            "Fallback para atendente humano"
        ],
        mitigations_implemented=[
            "Monitoramento de satisfacao em tempo real"
        ]
    )

    # Strategic context
    strategic_context = {
        "alignment": 80,            # Alta prioridade estrategica
        "sponsorship": 75,          # Bom suporte executivo
        "market_timing": 85,        # Momento ideal para chatbots
        "implementation_readiness": 70,
        "team_capability": 65,
        "infrastructure_readiness": 80
    }

    # Run analysis
    score = agent.analyze_single_initiative(
        initiative=initiative,
        costs=costs,
        benefits=benefits,
        risks=risks,
        strategic_context=strategic_context
    )

    # Print results
    print(f"\nIniciativa: {score.name}")
    print(f"Tipo: {score.initiative_type}")
    print(f"Maturidade: {score.maturity_level}")
    print()
    print("SCORES:")
    print(f"  - Financeiro: {score.financial_score:.1f}")
    print(f"  - Estrategico: {score.strategic_score:.1f}")
    print(f"  - Operacional: {score.operational_score:.1f}")
    print(f"  - Ajustado Risco: {score.risk_adjusted_score:.1f}")
    print(f"  - SCORE GERAL: {score.overall_value_score:.1f}")
    print()
    print("METRICAS FINANCEIRAS:")
    print(f"  - Investimento Total: R$ {score.total_investment:,.2f}")
    print(f"  - Beneficio 3 Anos: R$ {score.total_benefit_3yr:,.2f}")
    print(f"  - ROI: {score.roi_percent:.1f}%")
    print(f"  - VPL: R$ {score.npv:,.2f}")
    print(f"  - Payback: {score.payback_months} meses")
    print()
    print(f"CATEGORIA: {score.value_category.upper()}")
    print(f"RECOMENDACAO: {score.investment_recommendation}")
    print()

    if score.recommendations:
        print("RECOMENDACOES:")
        for rec in score.recommendations:
            print(f"  - {rec}")
        print()

    if score.next_steps:
        print("PROXIMOS PASSOS:")
        for step in score.next_steps:
            print(f"  - {step}")

    return score


def example_portfolio_analysis():
    """Analyze a portfolio of AI initiatives"""
    print("\n" + "=" * 60)
    print("Example: Portfolio Analysis")
    print("=" * 60)

    # Initialize agent
    agent = AIBusinessValueAgent(
        currency="BRL",
        projection_years=3
    )

    # Define multiple initiatives (simplified format)
    initiatives = [
        {
            "id": "chatbot_atendimento",
            "name": "Chatbot de Atendimento",
            "description": "Chatbot IA para atendimento ao cliente",
            "type": "customer_experience",
            "maturity": "pilot",
            "dev_cost_internal": 150000,
            "dev_cost_external": 80000,
            "compute_cost": 25000,
            "inference_cost": 48000,
            "maintenance_cost": 36000,
            "integration_cost": 45000,
            "expected_savings": 240000,
            "efficiency_value": 180000,
            "competitive_score": 70,
            "innovation_score": 65,
            "tech_risk": 35,
            "adoption_risk": 40
        },
        {
            "id": "previsao_demanda",
            "name": "Previsao de Demanda ML",
            "description": "Modelo ML para previsao de demanda de produtos",
            "type": "analytics",
            "maturity": "poc",
            "dev_cost_internal": 200000,
            "dev_cost_external": 50000,
            "compute_cost": 40000,
            "inference_cost": 24000,
            "maintenance_cost": 48000,
            "integration_cost": 60000,
            "expected_savings": 500000,
            "efficiency_value": 100000,
            "competitive_score": 80,
            "innovation_score": 75,
            "tech_risk": 50,
            "data_risk": 45,
            "adoption_risk": 30
        },
        {
            "id": "deteccao_fraude",
            "name": "Deteccao de Fraude",
            "description": "Sistema ML para deteccao de fraudes em transacoes",
            "type": "operational_ai",
            "maturity": "production",
            "dev_cost_internal": 300000,
            "dev_cost_external": 100000,
            "compute_cost": 60000,
            "inference_cost": 120000,
            "maintenance_cost": 72000,
            "integration_cost": 80000,
            "expected_savings": 800000,
            "efficiency_value": 200000,
            "competitive_score": 85,
            "innovation_score": 70,
            "tech_risk": 30,
            "data_risk": 35,
            "adoption_risk": 20,
            "regulatory_risk": 40
        },
        {
            "id": "recomendacao_produtos",
            "name": "Sistema de Recomendacao",
            "description": "Recomendacao personalizada de produtos",
            "type": "customer_experience",
            "maturity": "scaling",
            "dev_cost_internal": 180000,
            "dev_cost_external": 60000,
            "compute_cost": 35000,
            "inference_cost": 60000,
            "maintenance_cost": 42000,
            "integration_cost": 50000,
            "expected_revenue": 400000,
            "expected_savings": 100000,
            "competitive_score": 75,
            "innovation_score": 60,
            "tech_risk": 25,
            "adoption_risk": 25
        },
        {
            "id": "analise_sentimento",
            "name": "Analise de Sentimento",
            "description": "Analise de sentimento em redes sociais e reviews",
            "type": "analytics",
            "maturity": "ideation",
            "dev_cost_internal": 80000,
            "dev_cost_external": 20000,
            "compute_cost": 10000,
            "inference_cost": 12000,
            "maintenance_cost": 18000,
            "integration_cost": 25000,
            "expected_savings": 50000,
            "efficiency_value": 40000,
            "competitive_score": 50,
            "innovation_score": 55,
            "tech_risk": 40,
            "adoption_risk": 50
        }
    ]

    # Run portfolio analysis
    report = agent.analyze_initiatives(initiatives=initiatives)

    # Print portfolio summary
    print(f"\nINICIATIVAS ANALISADAS: {report.initiatives_analyzed}")
    print()
    print("RESUMO DO PORTFOLIO:")
    print(f"  - Investimento Total: R$ {report.total_portfolio_investment:,.2f}")
    print(f"  - Beneficio 3 Anos: R$ {report.total_portfolio_benefit_3yr:,.2f}")
    print(f"  - ROI do Portfolio: {report.portfolio_roi_percent:.1f}%")
    print()

    if report.transformational_initiatives:
        print("INICIATIVAS TRANSFORMACIONAIS:")
        for init in report.transformational_initiatives:
            print(f"  - {init}")
        print()

    if report.quick_wins:
        print("QUICK WINS:")
        for init in report.quick_wins:
            print(f"  - {init}")
        print()

    if report.high_risk_initiatives:
        print("ALTO RISCO:")
        for init in report.high_risk_initiatives:
            print(f"  - {init}")
        print()

    print("RANKING DE INICIATIVAS:")
    print("-" * 80)
    print(f"{'Rank':<6}{'Iniciativa':<30}{'Score':<10}{'ROI':<12}{'Categoria':<15}")
    print("-" * 80)

    for score in report.initiative_scores:
        print(f"{score.priority_rank:<6}{score.name:<30}{score.overall_value_score:<10.1f}"
              f"{score.roi_percent:<12.1f}%{score.value_category:<15}")

    print()

    if report.portfolio_recommendations:
        print("RECOMENDACOES DO PORTFOLIO:")
        for rec in report.portfolio_recommendations:
            print(f"  - {rec}")
        print()

    if report.investment_priorities:
        print("PRIORIDADES DE INVESTIMENTO:")
        for priority in report.investment_priorities[:3]:
            print(f"\n  {priority['rank']}. {priority['initiative']}")
            print(f"     Investimento: R$ {priority['investment']:,.2f}")
            print(f"     ROI: {priority['roi']:.1f}%")
            print(f"     Racional: {priority['rationale']}")

    return report


def example_comparison():
    """Compare specific initiatives"""
    print("\n" + "=" * 60)
    print("Example: Initiative Comparison")
    print("=" * 60)

    # First run portfolio analysis
    agent = AIBusinessValueAgent(currency="BRL")

    initiatives = [
        {
            "id": "opcao_a",
            "name": "Opcao A - Build Interno",
            "description": "Desenvolvimento interno da solucao",
            "type": "automation",
            "maturity": "ideation",
            "dev_cost_internal": 400000,
            "expected_savings": 300000,
            "competitive_score": 80,
            "tech_risk": 50
        },
        {
            "id": "opcao_b",
            "name": "Opcao B - Comprar Solucao",
            "description": "Compra de solucao de mercado",
            "type": "automation",
            "maturity": "poc",
            "dev_cost_external": 150000,
            "maintenance_cost": 60000,
            "expected_savings": 250000,
            "competitive_score": 50,
            "tech_risk": 25
        },
        {
            "id": "opcao_c",
            "name": "Opcao C - Hibrido",
            "description": "Solucao hibrida build + buy",
            "type": "automation",
            "maturity": "poc",
            "dev_cost_internal": 200000,
            "dev_cost_external": 100000,
            "maintenance_cost": 40000,
            "expected_savings": 280000,
            "competitive_score": 70,
            "tech_risk": 35
        }
    ]

    report = agent.analyze_initiatives(initiatives=initiatives)

    # Compare
    comparison = agent.compare_initiatives(
        ["opcao_a", "opcao_b", "opcao_c"],
        report
    )

    print("\nCOMPARATIVO DE OPCOES:")
    print("-" * 100)
    print(f"{'Opcao':<25}{'Score':<10}{'Financeiro':<12}{'Estrategico':<12}"
          f"{'ROI':<10}{'Payback':<10}{'Risco':<10}")
    print("-" * 100)

    for item in comparison:
        print(f"{item['initiative']:<25}{item['overall_score']:<10.1f}"
              f"{item['financial']:<12.1f}{item['strategic']:<12.1f}"
              f"{item['roi_percent']:<10.1f}%{item['payback_months']:<10}mo"
              f"{item['risk_level']:<10}")

    print()
    print(f"RECOMENDACAO: {comparison[0]['initiative']} (maior score)")


if __name__ == "__main__":
    # Run examples
    example_single_initiative()
    example_portfolio_analysis()
    example_comparison()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
