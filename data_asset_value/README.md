# Data Asset Value Agent

## Resumo
Agente de IA para avaliar o valor de ativos de dados combinando uso, relacionamentos de JOIN, impacto de data products e dependências de linhagem.

## Summary
AI agent to assess the value of data assets by combining usage, join relationships, data product impact, and lineage dependencies.

*English details available at the end of the file.*

## Visão Geral
Processa logs de consulta, configurações de data products e (opcionalmente) a saída do Data Lineage Agent para priorizar ativos críticos.

## Características
- **Uso real**: frequência, diversidade de usuários e tipos de consulta.
- **JOINs e conectividade** para identificar hubs.
- **Impacto de linhagem** em caminhos upstream/downstream.
- **Data products** com criticidade e impacto de receita.
- **Recomendações de governança** e categorização (crítico, hub, órfão).
- **Valor de modelos de IA** com custos (API/infra/pessoas), benefícios e tracking pós-deploy.

## Uso Rápido
```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent(
    weights={"usage": 0.30, "joins": 0.25, "lineage": 0.20, "data_product": 0.25},
    time_range_days=30,
)

report = agent.analyze_from_query_logs(
    query_logs=[{"query": "SELECT * FROM customers c JOIN orders o ON c.id=o.customer_id", "timestamp": "2024-12-15T10:00:00Z", "user": "analyst"}],
    lineage_data=None,
    data_product_config=[],
    asset_metadata=[],
)

print(report.to_markdown())
```

Ative `llm_review=True` e configure um `llm_provider` para revisar ranking e ações sugeridas.

## Valor de Modelos/Use Cases de IA
```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent()
model_report = agent.analyze_model_value([
    {
        "model_name": "fraud_detector_v2",
        "use_case": "detecção de fraude",
        "costs": {"api_cost": 1200, "infra_cost": 800, "people_cost": 600},
        "benefits": {"revenue_gain": 5000, "cost_savings": 2000},
        "post_deploy_metrics": {"adoption_rate": 0.7, "requests_per_day": 1500, "quality_score": 0.88}
    }
])

print(model_report.to_dict())
```

## Exemplos
Execute `python data_asset_value/examples/usage_example.py` para ver um fluxo completo com logs de exemplo.

## Integrações
- Consuma a saída do **Data Lineage Agent** para refletir impacto real nas dependências.
- Combine com **Data Classification** para aplicar multiplicadores de risco/PII.
- Publique resultados em catálogos via **Metadata Enrichment** ou **Discovery**.

---

## Summary
AI agent to assess the value of data assets by combining usage, join relationships, data product impact, and lineage dependencies.

## Overview
Processes query logs, data product configurations, and optionally Data Lineage Agent output to prioritize critical assets.

## Features
- **Real usage**: frequency, user diversity, and query types.
- **Joins and connectivity** to spot hubs.
- **Lineage impact** across upstream/downstream paths.
- **Data products** with criticality and revenue impact.
- **Governance recommendations** and categorization (critical, hub, orphan).
- **AI model value** with costs (API/infra/people), benefits, and post-deploy tracking.

## Quickstart
```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent(
    weights={"usage": 0.30, "joins": 0.25, "lineage": 0.20, "data_product": 0.25},
    time_range_days=30,
)

report = agent.analyze_from_query_logs(
    query_logs=[{"query": "SELECT * FROM customers c JOIN orders o ON c.id=o.customer_id", "timestamp": "2024-12-15T10:00:00Z", "user": "analyst"}],
    lineage_data=None,
    data_product_config=[],
    asset_metadata=[],
)

print(report.to_markdown())
```

Enable `llm_review=True` and set an `llm_provider` to review rankings and suggested actions.

## AI Model/Use Case Value
```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent()
model_report = agent.analyze_model_value([
    {
        "model_name": "fraud_detector_v2",
        "use_case": "fraud detection",
        "costs": {"api_cost": 1200, "infra_cost": 800, "people_cost": 600},
        "benefits": {"revenue_gain": 5000, "cost_savings": 2000},
        "post_deploy_metrics": {"adoption_rate": 0.7, "requests_per_day": 1500, "quality_score": 0.88}
    }
])

print(model_report.to_dict())
```

## Examples
Run `python data_asset_value/examples/usage_example.py` to see a complete flow with sample logs.

## Integrations
- Consume **Data Lineage Agent** output to reflect real dependency impact.
- Combine with **Data Classification** to apply risk/PII multipliers.
- Publish results to catalogs via **Metadata Enrichment** or **Discovery**.
