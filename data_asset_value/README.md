# Data Asset Value Agent

Agente de IA para avaliar o **valor de ativos de dados** combinando uso, relacionamentos de JOIN, impacto de data products e dependências de linhagem.
*AI agent to assess the **value of data assets** by combining usage, join relationships, data product impact, and lineage dependencies.*

## Visão Geral
*Overview.*

Processa logs de consulta, configurações de data products e (opcionalmente) saída do Data Lineage Agent para priorizar ativos críticos.
*Processes query logs, data product configurations, and optionally Data Lineage Agent output to prioritize critical assets.*

## Características
*Features.*

- **Uso real**: frequência, diversidade de usuários e tipos de consulta.
  *Real usage: frequency, user diversity, and query types.*
- **JOINs e conectividade** para identificar hubs.
  *Joins and connectivity to spot hubs.*
- **Impacto de linhagem** em caminhos upstream/downstream.
  *Lineage impact across upstream/downstream paths.*
- **Data products** com criticidade e impacto de receita.
  *Data products with criticality and revenue impact.*
- **Recomendações de governança** e categorização (crítico, hub, órfão).
  *Governance recommendations and categorization (critical, hub, orphan).* 

## Uso Rápido
*Quickstart.*

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
*Enable `llm_review=True` and set an `llm_provider` to review rankings and suggested actions.*

## Exemplos
*Examples.*

Execute `python data_asset_value/examples/usage_example.py` para ver um fluxo completo com logs de exemplo.
*Run `python data_asset_value/examples/usage_example.py` to see a complete flow with sample logs.*

## Integrações
*Integrations.*

- Consuma a saída do **Data Lineage Agent** para refletir impacto real nas dependências.
  *Consume **Data Lineage Agent** output to reflect real dependency impact.*
- Combine com **Data Classification** para aplicar multiplicadores de risco/PII.
  *Combine with **Data Classification** to apply risk/PII multipliers.*
- Publique resultados em catálogos via **Metadata Enrichment** ou **Discovery**.
  *Publish results to catalogs via **Metadata Enrichment** or **Discovery**.*
