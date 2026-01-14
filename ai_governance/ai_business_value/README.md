# AI Business Value Agent

Agente para avaliacao e analise de valor de negocios de iniciativas de IA.

## Visao Geral

O AI Business Value Agent analisa iniciativas de IA para determinar seu valor de negocio com base em:

- **Analise de Custos**: Desenvolvimento, infraestrutura, operacao, custos ocultos
- **Projecoes de Beneficios**: Receita, economia, eficiencia, impacto ao cliente
- **Calculo de ROI**: Retorno sobre investimento, VPL, payback
- **Avaliacao de Riscos**: Tecnico, organizacional, externo, especifico de IA
- **Alinhamento Estrategico**: Vantagem competitiva, inovacao, escalabilidade

## Uso Rapido

```python
from ai_governance.ai_business_value import AIBusinessValueAgent, CostBreakdown, BenefitProjection

# Inicializar agente
agent = AIBusinessValueAgent(currency="BRL", projection_years=3)

# Definir iniciativas
initiatives = [
    {
        "id": "chatbot_atendimento",
        "name": "Chatbot de Atendimento",
        "description": "Chatbot com IA para atendimento ao cliente",
        "type": "customer_experience",
        "maturity": "pilot"
    },
    {
        "id": "previsao_demanda",
        "name": "Previsao de Demanda",
        "description": "ML para previsao de demanda de produtos",
        "type": "analytics",
        "maturity": "poc"
    }
]

# Definir custos
costs = {
    "chatbot_atendimento": CostBreakdown(
        initiative_id="chatbot_atendimento",
        development_internal=150000,
        development_external=100000,
        compute_inference=24000,
        maintenance_annual=36000,
        integration=50000
    )
}

# Definir beneficios
benefits = {
    "chatbot_atendimento": BenefitProjection(
        initiative_id="chatbot_atendimento",
        cost_reduction=200000,
        efficiency_gain_hours=5000,
        efficiency_gain_value=150000,
        customer_satisfaction_delta=15,
        competitive_advantage=70
    )
}

# Analisar iniciativas
report = agent.analyze_initiatives(
    initiatives=initiatives,
    cost_data=costs,
    benefit_projections=benefits
)

# Gerar relatorio
print(report.to_markdown())
```

## Componentes Principais

### CostBreakdown

Estrutura detalhada de custos:

```python
CostBreakdown(
    initiative_id="minha_iniciativa",

    # Custos de desenvolvimento
    development_internal=100000,    # Equipe interna
    development_external=50000,     # Consultores
    development_tools=10000,        # Ferramentas

    # Custos de infraestrutura
    compute_training=20000,         # GPU para treinamento
    compute_inference=30000,        # Inferencia em producao
    storage=5000,                   # Armazenamento

    # Custos operacionais (anuais)
    maintenance_annual=24000,       # Manutencao
    monitoring_annual=12000,        # Monitoramento
    licensing_annual=36000,         # Licencas

    # Custos unicos
    data_acquisition=15000,         # Aquisicao de dados
    integration=40000,              # Integracao
    training_users=10000            # Treinamento
)
```

### BenefitProjection

Projecao de beneficios:

```python
BenefitProjection(
    initiative_id="minha_iniciativa",

    # Beneficios quantificaveis (anuais)
    revenue_increase=500000,        # Aumento de receita
    cost_reduction=200000,          # Reducao de custos
    efficiency_gain_hours=2000,     # Horas economizadas
    efficiency_gain_value=100000,   # Valor das horas

    # Impacto ao cliente
    customer_satisfaction_delta=10, # Melhoria NPS/CSAT
    customer_retention_delta=5,     # Melhoria retencao %

    # Beneficios estrategicos (0-100)
    competitive_advantage=80,       # Diferenciacao
    innovation_score=70,            # Capacidade de inovacao
    scalability_potential=85,       # Potencial de crescimento

    # Niveis de confianca (0-100)
    revenue_confidence=60,
    cost_reduction_confidence=80
)
```

### RiskAssessment

Avaliacao de riscos:

```python
RiskAssessment(
    initiative_id="minha_iniciativa",

    # Riscos tecnicos (0-100)
    model_performance_risk=40,
    data_quality_risk=30,
    integration_risk=50,
    scalability_risk=35,

    # Riscos organizacionais
    adoption_risk=45,
    skill_gap_risk=50,
    change_resistance=40,

    # Riscos de IA
    bias_fairness_risk=35,
    explainability_risk=45,
    security_risk=30,
    ethical_risk=25,

    # Mitigacoes
    mitigations_planned=["Plano de treinamento", "Auditoria de vieses"],
    mitigations_implemented=["Monitoramento de drift"]
)
```

## Metricas Calculadas

### Financeiras

- **ROI (%)**: Retorno sobre investimento
- **VPL**: Valor presente liquido
- **Payback**: Periodo de retorno em meses
- **TIR**: Taxa interna de retorno

### Scores (0-100)

- **Financial Score**: Baseado em ROI, VPL e payback
- **Strategic Score**: Alinhamento estrategico e diferenciacao
- **Operational Score**: Impacto operacional e prontidao
- **Risk-Adjusted Score**: Score ajustado pelo nivel de risco

### Categorias de Valor

| Categoria | Score | Descricao |
|-----------|-------|-----------|
| Transformational | >= 85 | Iniciativa game-changer |
| High | >= 70 | Impacto significativo |
| Medium | >= 50 | Melhoria incremental |
| Low | >= 30 | Valor limitado |
| Experimental | < 30 | R&D/exploratorio |

## Recomendacoes de Investimento

O agente gera recomendacoes automaticas:

- **INVEST - High priority**: Score >= 85, ROI > 100%, risco baixo
- **INVEST - Strong case**: Score >= 70, ROI > 50%
- **CONSIDER**: Score >= 50, ROI > 0%
- **DEFER**: Score >= 30, valor limitado
- **DECLINE**: Score < 30, caso de negocio insuficiente

## Relatorio de Portfolio

O relatorio inclui:

1. **Resumo do Portfolio**: Investimento total, beneficio, ROI
2. **Iniciativas Transformacionais**: Alto valor estrategico
3. **Quick Wins**: Baixo custo, alto valor, payback rapido
4. **Alto Risco**: Iniciativas que precisam de mitigacao
5. **Tabela de Scores**: Comparativo de todas iniciativas
6. **Prioridades de Investimento**: Ranking com racionais
7. **Recomendacoes**: Sugestoes para o portfolio

## Integracao com Outros Agentes

O AI Business Value Agent pode integrar com:

- **Sensitive Data NER**: Avaliar riscos de PII/dados sensiveis
- **Data Quality Agent**: Avaliar qualidade dos dados de IA
- **Data Lineage Agent**: Entender dependencias de dados
