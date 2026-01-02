# Data Quality Agent

Agente de IA para monitorar qualidade de dados com métricas multi-dimensionais, SLA de freshness, detecção de schema drift e alertas configuráveis.
*AI agent for data quality monitoring with multi-dimensional metrics, freshness SLAs, schema drift detection, and configurable alerts.*

## Visão Geral
*Overview.*

O agente mede completude, unicidade, validade, consistência, freshness e mudanças de schema em arquivos ou conectores suportados.
*The agent measures completeness, uniqueness, validity, consistency, freshness, and schema changes across supported files/connectors.*

## Características
*Features.*

- **Métricas multi-dimensionais** com score consolidado.
  *Multi-dimensional metrics with a consolidated score.*
- **Freshness vs SLA** para monitorar atualidade.
  *Freshness versus SLA to monitor data recency.*
- **Schema drift automático** com histórico de versões.
  *Automatic schema drift tracking with version history.*
- **Regras e alertas declarativos** com níveis de severidade.
  *Declarative rules and alerts with severity levels.*
- **Suporte a CSV, Parquet, SQL, Delta Lake**.
  *Support for CSV, Parquet, SQL, and Delta Lake.*

## Instalação
*Installation.*

```bash
pip install -r data_quality/requirements.txt
```

## Uso Rápido
*Quickstart.*

```python
from data_quality.agent import DataQualityAgent

agent = DataQualityAgent(enable_schema_tracking=True)
report = agent.evaluate_file(
    "orders.parquet",
    freshness_config={"timestamp_column": "updated_at", "sla_hours": 4}
)

print(report.overall_score)
print(report.overall_status)
print(report.schema_drift)
```

Relatórios podem ser exportados com `to_markdown()` ou `to_json()`.
*Reports can be exported via `to_markdown()` or `to_json()`.*

## Regras e Alertas
*Rules and alerts.*

```python
from data_quality.rules import QualityRule, AlertLevel

agent.add_rule(QualityRule(
    name="orders_freshness_sla",
    dimension="freshness",
    table_name="orders",
    column="updated_at",
    threshold=0.95,
    alert_level=AlertLevel.CRITICAL,
    params={"sla_hours": 4},
))
```

As regras permitem disparar notificações ou ações de governança quando métricas caem abaixo do limite definido.
*Rules let you trigger notifications or governance actions when metrics fall below defined thresholds.*

## Interface
*Interface.*

Execute `streamlit run data_quality/streamlit_app.py` ou use a interface unificada em `streamlit run app.py`.
*Run `streamlit run data_quality/streamlit_app.py` or the unified interface via `streamlit run app.py`.*
