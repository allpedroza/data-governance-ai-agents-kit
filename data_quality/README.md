# Data Quality Agent

Agente de IA para monitoramento de qualidade de dados com métricas automatizadas, detecção de schema drift e alertas baseados em SLA.

## Visão Geral

O **Data Quality Agent** monitora a qualidade dos dados em múltiplas dimensões:

| Dimensão | Descrição |
|----------|-----------|
| **Completeness** | Presença de dados (nulos, vazios) |
| **Uniqueness** | Ausência de duplicatas |
| **Validity** | Conformidade com formatos esperados |
| **Consistency** | Coerência entre campos |
| **Freshness** | Atualidade dos dados vs SLA |
| **Schema** | Detecção de mudanças de schema |

### Características

- 📊 **Métricas Multi-dimensionais**: Score consolidado de qualidade
- ⏱️ **Freshness com SLA**: Monitoramento de atualidade vs SLA definido
- 🔄 **Schema Drift**: Detecção automática de mudanças de schema
- 🔔 **Alertas Configuráveis**: Regras declarativas com níveis de severidade
- 📁 **Multi-fonte**: CSV, Parquet, SQL, Delta Lake

## Instalação

```bash
pip install -r data_quality/requirements.txt
```

## Uso Rápido

### Via Python

```python
from data_quality.agent import DataQualityAgent

# Inicializar
agent = DataQualityAgent(
    persist_dir="./quality_data",
    enable_schema_tracking=True
)

# Avaliar qualidade de um arquivo
report = agent.evaluate_file("data.csv")

print(f"Score: {report.overall_score:.2%}")
print(f"Status: {report.overall_status}")

# Com configuração de Freshness/SLA
report = agent.evaluate_file(
    "orders.parquet",
    freshness_config={
        "timestamp_column": "updated_at",
        "sla_hours": 4,
        "max_age_hours": 8
    }
)

# Exportar relatório
print(report.to_markdown())
```

### Via Streamlit

```bash
# Interface standalone
streamlit run data_quality/streamlit_app.py

# Ou interface unificada
streamlit run app.py
```

## Dimensões de Qualidade

### Completeness (Completude)

Mede a presença de dados obrigatórios.

```python
# Verificar completude geral
report = agent.evaluate_file(
    "data.csv",
    completeness_config={
        "threshold": 0.95,
        "treat_empty_as_null": True,
        "required_columns": ["id", "name", "email"]
    }
)
```

### Uniqueness (Unicidade)

Detecta duplicatas em colunas-chave.

```python
report = agent.evaluate_file(
    "data.csv",
    uniqueness_config={
        "column": "customer_id",  # Coluna única
        "threshold": 1.0,
        "key_columns": ["order_id", "product_id"]  # Chave composta
    }
)
```

### Validity (Validade)

Valida formatos e ranges.

```python
report = agent.evaluate_file(
    "data.csv",
    validity_configs=[
        {
            "column": "email",
            "pattern_name": "email",
            "threshold": 0.95
        },
        {
            "column": "cpf",
            "pattern_name": "cpf",
            "threshold": 0.99
        },
        {
            "column": "amount",
            "min_value": 0,
            "max_value": 1000000,
            "threshold": 0.99
        }
    ]
)
```

**Padrões suportados:**
- `email`, `cpf`, `cnpj`, `phone_br`
- `date_iso`, `date_br`, `uuid`, `url`
- `integer`, `decimal`, `boolean`

### Freshness (Atualidade/SLA)

Monitora se os dados estão atualizados conforme o SLA.

```python
report = agent.evaluate_file(
    "orders.parquet",
    freshness_config={
        "timestamp_column": "updated_at",
        "sla_hours": 4,          # SLA: 4 horas
        "max_age_hours": 8       # Máximo: 8 horas
    }
)

# Detalhes de freshness
print(f"Última atualização: {report.metrics[0]['details']['latest_timestamp']}")
print(f"Idade (horas): {report.metrics[0]['details']['age_hours']}")
print(f"SLA Cumprido: {report.metrics[0]['details']['sla_compliant']}")
```

## Schema Drift Detection

Detecta mudanças no schema entre execuções.

```python
from data_quality.connectors import CSVConnector

connector = CSVConnector("data.csv")
drift_report = agent.check_schema_drift(connector)

if drift_report.has_drift:
    print(f"Mudanças detectadas: {drift_report.summary}")
    for change in drift_report.changes:
        print(f"  - {change.change_type}: {change.message}")
```

**Tipos de mudança detectados:**
- `COLUMN_ADDED`: Nova coluna
- `COLUMN_REMOVED`: Coluna removida (breaking)
- `TYPE_CHANGED`: Tipo alterado
- `NULLABLE_CHANGED`: Nullable alterado
- `COLUMN_RENAMED`: Renomeação detectada

## Regras e Alertas

### Definindo Regras

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
    description="Orders must be updated within 4 hours"
))
```

### Carregando Regras de Arquivo

```python
# JSON com regras
agent.load_rules_from_file("rules.json")
```

**Formato do arquivo:**
```json
{
  "name": "production_rules",
  "rules": [
    {
      "name": "email_validity",
      "dimension": "validity",
      "table_name": "customers",
      "column": "email",
      "threshold": 0.95,
      "alert_level": "warning",
      "params": {"pattern_name": "email"}
    }
  ]
}
```

### Gerenciando Alertas

```python
# Obter alertas ativos
alerts = agent.get_active_alerts(level=AlertLevel.CRITICAL)

for alert in alerts:
    print(f"[{alert.level}] {alert.message}")
    # Reconhecer alerta
    agent.acknowledge_alert(alert.alert_id)
```

## Conectores de Dados

### CSV

```python
from data_quality.connectors import CSVConnector

connector = CSVConnector(
    "data.csv",
    encoding="utf-8",
    separator=","
)
report = agent.evaluate(connector)
```

### Parquet

```python
from data_quality.connectors import ParquetConnector

connector = ParquetConnector("data.parquet")
report = agent.evaluate(connector)
```

### SQL

```python
report = agent.evaluate_sql(
    connection_string="postgresql://user:pass@host:5432/db",
    table_name="orders",
    schema="public"
)
```

### Delta Lake

```python
report = agent.evaluate_delta(
    table_path="/path/to/delta/table",
    version=5  # Versão específica (opcional)
)
```

## Relatórios

### QualityReport

```python
report = agent.evaluate_file("data.csv")

# Propriedades
report.overall_score      # 0.0 a 1.0
report.overall_status     # "passed", "warning", "failed"
report.dimensions         # Scores por dimensão
report.metrics            # Detalhes de cada métrica
report.alerts            # Alertas gerados
report.schema_drift      # Info de schema drift

# Exportação
report.to_json()          # JSON
report.to_markdown()      # Markdown
report.to_dict()          # Dict
```

### Exportar para Arquivo

```python
agent.export_report(report, "report.json", format="json")
agent.export_report(report, "report.md", format="markdown")
```

## Arquitetura

```
data_quality/
├── agent.py                    # Agente principal
├── metrics/
│   ├── quality_metrics.py      # Métricas de qualidade
│   └── schema_drift.py         # Detector de schema drift
├── rules/
│   └── quality_rules.py        # Sistema de regras e alertas
├── connectors/
│   └── data_connector.py       # Conectores de dados
├── examples/
│   ├── basic_usage.py          # Exemplo básico
│   └── sample_rules.json       # Regras de exemplo
├── streamlit_app.py            # UI Streamlit
├── requirements.txt
└── README.md
```

## Integração com Outros Agentes

```python
# Usar com Metadata Enrichment
from metadata_enrichment.agent import MetadataEnrichmentAgent

# Avaliar qualidade
quality_report = quality_agent.evaluate_file("data.csv")

# Enriquecer metadados
enrichment_result = enrichment_agent.enrich_from_csv("data.csv")

# Combinar informações
print(f"Qualidade: {quality_report.overall_score:.0%}")
print(f"PII detectado: {enrichment_result.has_pii}")
```

## Exemplos

Veja o diretório `examples/`:

- `basic_usage.py`: Uso básico do agente
- `sample_rules.json`: Regras de qualidade de exemplo

## Boas Práticas

1. **Defina SLAs claros** para freshness por tipo de dado
2. **Configure regras por ambiente** (dev, staging, prod)
3. **Monitore schema drift** em pipelines de CI/CD
4. **Revise alertas regularmente** e ajuste thresholds
5. **Integre com observability** (logs, métricas)
