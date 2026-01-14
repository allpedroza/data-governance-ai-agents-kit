# Data Quality Agent

## Resumo
Agente de IA para monitorar qualidade de dados com métricas multidimensionais, SLA de frescor, detecção de schema drift e alertas configuráveis.

## Summary
AI agent for data quality monitoring with multi-dimensional metrics, freshness SLAs, schema drift detection, and configurable alerts.

*English details available at the end of the file.*

## Visão Geral
O agente mede completude, unicidade, validade, consistência, frescor e mudanças de schema em arquivos ou conectores suportados.

## Características
- **Métricas multidimensionais** com score consolidado.
- **Freshness vs SLA** para monitorar atualidade.
- **Schema drift automático** com histórico de versões.
- **Regras e alertas declarativos** com níveis de severidade.
- **Suporte a CSV, Parquet, SQL, Delta Lake**.

## Instalação
```bash
pip install -r data_quality/requirements.txt
```

## Uso Rápido
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

## Regras e Alertas
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

## Interface
Execute `streamlit run data_quality/streamlit_app.py` ou use a interface unificada em `streamlit run app.py`.

---

## Summary
AI agent for data quality monitoring with multi-dimensional metrics, freshness SLAs, schema drift detection, and configurable alerts.

## Overview
The agent measures completeness, uniqueness, validity, consistency, freshness, and schema changes across supported files/connectors.

## Features
- **Multi-dimensional metrics** with a consolidated score.
- **Freshness versus SLA** to monitor data recency.
- **Automatic schema drift tracking** with version history.
- **Declarative rules and alerts** with severity levels.
- **Support for CSV, Parquet, SQL, and Delta Lake.**

## Installation
```bash
pip install -r data_quality/requirements.txt
```

## Quickstart
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

Reports can be exported via `to_markdown()` or `to_json()`.

## Rules and Alerts
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

Rules let you trigger notifications or governance actions when metrics fall below defined thresholds.

## Interface
Run `streamlit run data_quality/streamlit_app.py` or the unified interface via `streamlit run app.py`.
