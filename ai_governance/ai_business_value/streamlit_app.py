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
Streamlit interface for AI Business Value Agent
"""

import streamlit as st
import json
from datetime import datetime

from agent import (
    AIBusinessValueAgent,
    CostBreakdown,
    BenefitProjection,
    RiskAssessment,
    InitiativeType,
    MaturityLevel
)


def main():
    st.set_page_config(
        page_title="AI Business Value Agent",
        page_icon="💼",
        layout="wide"
    )

    st.title("AI Business Value Agent")
    st.markdown("Avaliacao de valor de negocios para iniciativas de IA")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuracoes")

        currency = st.selectbox(
            "Moeda",
            ["BRL", "USD", "EUR"],
            index=0
        )

        projection_years = st.slider(
            "Anos de Projecao",
            min_value=1,
            max_value=5,
            value=3
        )

        discount_rate = st.slider(
            "Taxa de Desconto (%)",
            min_value=5,
            max_value=20,
            value=10
        ) / 100

    # Initialize agent
    agent = AIBusinessValueAgent(
        currency=currency,
        projection_years=projection_years,
        discount_rate=discount_rate
    )

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Nova Iniciativa",
        "Analise de Portfolio",
        "Comparativo",
        "Documentacao"
    ])

    with tab1:
        render_single_initiative_tab(agent, currency)

    with tab2:
        render_portfolio_tab(agent, currency)

    with tab3:
        render_comparison_tab()

    with tab4:
        render_documentation_tab()


def render_single_initiative_tab(agent: AIBusinessValueAgent, currency: str):
    """Render single initiative analysis tab"""

    st.header("Analise de Iniciativa Individual")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informacoes Basicas")

        init_name = st.text_input("Nome da Iniciativa", "Chatbot de Atendimento")
        init_description = st.text_area(
            "Descricao",
            "Implementacao de chatbot com IA para atendimento ao cliente"
        )

        init_type = st.selectbox(
            "Tipo de Iniciativa",
            [t.value for t in InitiativeType],
            index=2  # customer_experience
        )

        maturity = st.selectbox(
            "Nivel de Maturidade",
            [m.value for m in MaturityLevel],
            index=1  # poc
        )

    with col2:
        st.subheader("Contexto Estrategico")

        strategic_alignment = st.slider(
            "Alinhamento Estrategico",
            0, 100, 70,
            help="Quao alinhada esta a iniciativa com a estrategia de negocios"
        )

        executive_sponsorship = st.slider(
            "Patrocinio Executivo",
            0, 100, 60,
            help="Nivel de suporte executivo"
        )

        implementation_readiness = st.slider(
            "Prontidao para Implementacao",
            0, 100, 50,
            help="Quao pronta esta a organizacao para implementar"
        )

        team_capability = st.slider(
            "Capacidade da Equipe",
            0, 100, 55,
            help="Habilidades e experiencia da equipe"
        )

    st.divider()

    # Cost inputs
    st.subheader("Custos")
    cost_col1, cost_col2, cost_col3 = st.columns(3)

    with cost_col1:
        st.markdown("**Desenvolvimento**")
        dev_internal = st.number_input(
            f"Equipe Interna ({currency})",
            min_value=0,
            value=150000,
            step=10000
        )
        dev_external = st.number_input(
            f"Consultores ({currency})",
            min_value=0,
            value=50000,
            step=10000
        )

    with cost_col2:
        st.markdown("**Infraestrutura**")
        compute_training = st.number_input(
            f"Computacao Treinamento ({currency})",
            min_value=0,
            value=20000,
            step=5000
        )
        compute_inference = st.number_input(
            f"Computacao Inferencia ({currency}/ano)",
            min_value=0,
            value=36000,
            step=5000
        )

    with cost_col3:
        st.markdown("**Outros**")
        integration = st.number_input(
            f"Integracao ({currency})",
            min_value=0,
            value=30000,
            step=5000
        )
        maintenance = st.number_input(
            f"Manutencao ({currency}/ano)",
            min_value=0,
            value=24000,
            step=5000
        )

    st.divider()

    # Benefit inputs
    st.subheader("Beneficios Esperados")
    ben_col1, ben_col2, ben_col3 = st.columns(3)

    with ben_col1:
        st.markdown("**Financeiros (anual)**")
        revenue_increase = st.number_input(
            f"Aumento de Receita ({currency})",
            min_value=0,
            value=0,
            step=10000
        )
        cost_reduction = st.number_input(
            f"Reducao de Custos ({currency})",
            min_value=0,
            value=180000,
            step=10000
        )

    with ben_col2:
        st.markdown("**Eficiencia**")
        efficiency_hours = st.number_input(
            "Horas Economizadas (ano)",
            min_value=0,
            value=3000,
            step=100
        )
        hour_value = st.number_input(
            f"Valor da Hora ({currency})",
            min_value=0,
            value=50,
            step=5
        )

    with ben_col3:
        st.markdown("**Cliente**")
        csat_delta = st.slider(
            "Melhoria CSAT/NPS (pontos)",
            -20, 50, 10
        )
        retention_delta = st.slider(
            "Melhoria Retencao (%)",
            -10, 20, 3
        )

    st.divider()

    # Risk inputs
    st.subheader("Avaliacao de Riscos")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.markdown("**Riscos Tecnicos**")
        model_risk = st.slider("Risco de Performance do Modelo", 0, 100, 40)
        data_risk = st.slider("Risco de Qualidade de Dados", 0, 100, 35)
        integration_risk = st.slider("Risco de Integracao", 0, 100, 45)

    with risk_col2:
        st.markdown("**Riscos de IA**")
        bias_risk = st.slider("Risco de Vies/Fairness", 0, 100, 40)
        explainability_risk = st.slider("Risco de Explicabilidade", 0, 100, 50)
        adoption_risk = st.slider("Risco de Adocao", 0, 100, 35)

    # Analyze button
    if st.button("Analisar Iniciativa", type="primary"):
        # Create data structures
        initiative = {
            "id": init_name.lower().replace(" ", "_"),
            "name": init_name,
            "description": init_description,
            "type": init_type,
            "maturity": maturity
        }

        costs = CostBreakdown(
            initiative_id=initiative["id"],
            development_internal=dev_internal,
            development_external=dev_external,
            compute_training=compute_training,
            compute_inference=compute_inference,
            integration=integration,
            maintenance_annual=maintenance,
            currency=currency
        )

        benefits = BenefitProjection(
            initiative_id=initiative["id"],
            revenue_increase=revenue_increase,
            cost_reduction=cost_reduction,
            efficiency_gain_hours=efficiency_hours,
            efficiency_gain_value=efficiency_hours * hour_value,
            customer_satisfaction_delta=csat_delta,
            customer_retention_delta=retention_delta,
            competitive_advantage=strategic_alignment,
            currency=currency
        )

        risks = RiskAssessment(
            initiative_id=initiative["id"],
            model_performance_risk=model_risk,
            data_quality_risk=data_risk,
            integration_risk=integration_risk,
            bias_fairness_risk=bias_risk,
            explainability_risk=explainability_risk,
            adoption_risk=adoption_risk
        )

        strategic_context = {
            "alignment": strategic_alignment,
            "sponsorship": executive_sponsorship,
            "implementation_readiness": implementation_readiness,
            "team_capability": team_capability
        }

        # Run analysis
        score = agent.analyze_single_initiative(
            initiative=initiative,
            costs=costs,
            benefits=benefits,
            risks=risks,
            strategic_context=strategic_context
        )

        # Display results
        st.divider()
        st.header("Resultados da Analise")

        # Key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Score Geral",
                f"{score.overall_value_score:.1f}",
                help="Score ponderado de valor"
            )

        with metric_col2:
            st.metric(
                "ROI",
                f"{score.roi_percent:.1f}%",
                help="Retorno sobre investimento"
            )

        with metric_col3:
            st.metric(
                "Payback",
                f"{score.payback_months} meses",
                help="Periodo de retorno"
            )

        with metric_col4:
            st.metric(
                "Categoria",
                score.value_category.upper(),
                help="Classificacao de valor"
            )

        # Score breakdown
        st.subheader("Breakdown de Scores")
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)

        with score_col1:
            st.progress(score.financial_score / 100)
            st.caption(f"Financeiro: {score.financial_score:.1f}")

        with score_col2:
            st.progress(score.strategic_score / 100)
            st.caption(f"Estrategico: {score.strategic_score:.1f}")

        with score_col3:
            st.progress(score.operational_score / 100)
            st.caption(f"Operacional: {score.operational_score:.1f}")

        with score_col4:
            st.progress(score.risk_adjusted_score / 100)
            st.caption(f"Ajustado Risco: {score.risk_adjusted_score:.1f}")

        # Investment recommendation
        st.subheader("Recomendacao")

        if "INVEST" in score.investment_recommendation:
            st.success(score.investment_recommendation)
        elif "CONSIDER" in score.investment_recommendation:
            st.warning(score.investment_recommendation)
        else:
            st.error(score.investment_recommendation)

        # Recommendations
        if score.recommendations:
            st.subheader("Recomendacoes")
            for rec in score.recommendations:
                st.info(rec)

        # Next steps
        if score.next_steps:
            st.subheader("Proximos Passos")
            for step in score.next_steps:
                st.write(f"- {step}")

        # Financial details
        with st.expander("Detalhes Financeiros"):
            fin_col1, fin_col2 = st.columns(2)

            with fin_col1:
                st.markdown("**Custos**")
                st.write(f"- Investimento Inicial: {currency} {score.total_investment:,.2f}")
                st.write(f"- Desenvolvimento: {currency} {costs.total_development:,.2f}")
                st.write(f"- Infraestrutura: {currency} {costs.total_infrastructure:,.2f}")
                st.write(f"- Operacional Anual: {currency} {costs.total_operational_annual:,.2f}")

            with fin_col2:
                st.markdown("**Beneficios (3 anos)**")
                st.write(f"- Beneficio Total: {currency} {score.total_benefit_3yr:,.2f}")
                st.write(f"- VPL: {currency} {score.npv:,.2f}")

        # Risk details
        with st.expander("Detalhes de Risco"):
            st.write(f"**Nivel de Risco:** {risks.risk_level}")
            st.write(f"- Risco Tecnico: {risks.technical_risk_score:.1f}")
            st.write(f"- Risco Organizacional: {risks.organizational_risk_score:.1f}")
            st.write(f"- Risco de IA: {risks.ai_specific_risk_score:.1f}")
            st.write(f"- Score Geral de Risco: {risks.overall_risk_score:.1f}")

        # JSON export
        with st.expander("Exportar JSON"):
            st.json(score.to_dict())


def render_portfolio_tab(agent: AIBusinessValueAgent, currency: str):
    """Render portfolio analysis tab"""

    st.header("Analise de Portfolio")

    st.info(
        "Cole um JSON com as iniciativas do seu portfolio para analise completa. "
        "Veja a aba Documentacao para o formato esperado."
    )

    json_input = st.text_area(
        "JSON das Iniciativas",
        height=300,
        placeholder='[\n  {\n    "id": "iniciativa_1",\n    "name": "Nome",\n    "description": "Descricao",\n    "type": "automation",\n    "maturity": "poc",\n    "dev_cost_internal": 100000,\n    "expected_savings": 150000\n  }\n]'
    )

    if st.button("Analisar Portfolio", type="primary"):
        if json_input:
            try:
                initiatives = json.loads(json_input)

                report = agent.analyze_initiatives(initiatives=initiatives)

                st.divider()
                st.header("Resultados do Portfolio")

                # Portfolio summary
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric(
                        "Investimento Total",
                        f"{currency} {report.total_portfolio_investment:,.0f}"
                    )

                with metric_col2:
                    st.metric(
                        "Beneficio 3 Anos",
                        f"{currency} {report.total_portfolio_benefit_3yr:,.0f}"
                    )

                with metric_col3:
                    st.metric(
                        "ROI Portfolio",
                        f"{report.portfolio_roi_percent:.1f}%"
                    )

                with metric_col4:
                    st.metric(
                        "Iniciativas",
                        report.initiatives_analyzed
                    )

                # Categories
                cat_col1, cat_col2, cat_col3 = st.columns(3)

                with cat_col1:
                    if report.transformational_initiatives:
                        st.success(f"Transformacionais: {len(report.transformational_initiatives)}")
                        for init in report.transformational_initiatives:
                            st.write(f"  - {init}")

                with cat_col2:
                    if report.quick_wins:
                        st.info(f"Quick Wins: {len(report.quick_wins)}")
                        for init in report.quick_wins:
                            st.write(f"  - {init}")

                with cat_col3:
                    if report.high_risk_initiatives:
                        st.warning(f"Alto Risco: {len(report.high_risk_initiatives)}")
                        for init in report.high_risk_initiatives:
                            st.write(f"  - {init}")

                # Initiative table
                st.subheader("Ranking de Iniciativas")

                table_data = []
                for score in report.initiative_scores:
                    table_data.append({
                        "Rank": score.priority_rank,
                        "Iniciativa": score.name,
                        "Tipo": score.initiative_type,
                        "Score": f"{score.overall_value_score:.1f}",
                        "ROI": f"{score.roi_percent:.1f}%",
                        "Payback": f"{score.payback_months}mo",
                        "Categoria": score.value_category
                    })

                st.table(table_data)

                # Portfolio recommendations
                if report.portfolio_recommendations:
                    st.subheader("Recomendacoes do Portfolio")
                    for rec in report.portfolio_recommendations:
                        st.write(f"- {rec}")

                # Markdown report
                with st.expander("Relatorio Completo (Markdown)"):
                    st.markdown(report.to_markdown())

                # JSON export
                with st.expander("Exportar JSON"):
                    st.json(report.to_dict())

            except json.JSONDecodeError as e:
                st.error(f"Erro ao parsear JSON: {e}")
        else:
            st.warning("Por favor, insira o JSON das iniciativas.")


def render_comparison_tab():
    """Render comparison tab"""

    st.header("Comparativo de Iniciativas")

    st.info("Use esta aba para comparar visualmente diferentes cenarios de investimento.")

    # Placeholder for comparison feature
    st.write("Em desenvolvimento...")


def render_documentation_tab():
    """Render documentation tab"""

    st.header("Documentacao")

    st.markdown("""
    ## Formato JSON para Portfolio

    ```json
    [
      {
        "id": "chatbot_atendimento",
        "name": "Chatbot de Atendimento",
        "description": "Chatbot com IA para atendimento",
        "type": "customer_experience",
        "maturity": "pilot",
        "dev_cost_internal": 150000,
        "dev_cost_external": 50000,
        "compute_cost": 20000,
        "inference_cost": 36000,
        "maintenance_cost": 24000,
        "integration_cost": 30000,
        "expected_revenue": 0,
        "expected_savings": 180000,
        "hours_saved": 3000,
        "efficiency_value": 150000,
        "competitive_score": 70,
        "innovation_score": 60,
        "tech_risk": 40,
        "data_risk": 35,
        "adoption_risk": 30,
        "bias_risk": 40
      }
    ]
    ```

    ## Tipos de Iniciativa

    - `automation`: Automacao de processos
    - `analytics`: Analytics preditivo/prescritivo
    - `customer_experience`: Chatbots, personalizacao
    - `decision_support`: Suporte a decisao
    - `content_generation`: GenAI para conteudo
    - `operational_ai`: MLOps, AIOps
    - `product_ai`: IA como feature de produto
    - `research`: Pesquisa e desenvolvimento

    ## Niveis de Maturidade

    - `ideation`: Fase de conceito
    - `poc`: Prova de conceito
    - `pilot`: Deploy limitado
    - `scaling`: Expandindo deploy
    - `production`: Producao completa
    - `optimization`: Maduro, otimizando

    ## Categorias de Valor

    | Categoria | Score | Descricao |
    |-----------|-------|-----------|
    | Transformational | >= 85 | Game-changer |
    | High | >= 70 | Impacto significativo |
    | Medium | >= 50 | Melhoria incremental |
    | Low | >= 30 | Valor limitado |
    | Experimental | < 30 | R&D/exploratorio |
    """)


if __name__ == "__main__":
    main()
