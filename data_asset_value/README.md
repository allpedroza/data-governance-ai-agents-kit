# Data Asset Value Agent

Agente de IA para **avaliar o valor de ativos de dados** combinando padrões de uso, relacionamentos de JOIN, impacto de data products e dependências de linhagem. Gera escores comparáveis por ativo, insights de conectividade e recomendações de governança.

## Visão Geral

O **Data Asset Value Agent** processa logs de consulta, metadados de produtos de dados e (opcionalmente) a saída do Data Lineage Agent para priorizar ativos em um data lake.

- 📈 **Uso real**: frequência de consultas, diversidade de usuários e tipos de uso.
- 🔗 **JOINs e conectividade**: frequência e variedade de relacionamentos entre ativos.
- 🧭 **Impacto de linhagem**: ativos críticos em caminhos upstream/downstream.
- 🧩 **Data products**: alcance, criticidade e impacto de receita dos produtos que dependem do ativo.
- 🛡️ **Risco e PII**: multiplicadores por criticidade, risco e presença de dados sensíveis.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso Rápido

### Analisar logs de consulta

```python
from data_asset_value import DataAssetValueAgent

agent = DataAssetValueAgent(
    weights={
        "usage": 0.30,
        "joins": 0.25,
        "lineage": 0.20,
        "data_product": 0.25,
    },
    time_range_days=30,
    persist_dir="./output"
)

report = agent.analyze_from_query_logs(
    query_logs=[
        {
            "query": "SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id",
            "timestamp": "2024-12-15T10:00:00Z",
            "user": "analyst",
            "data_product": "customer_analytics"
        }
    ],
    lineage_data=None,              # Integração opcional com Data Lineage Agent
    data_product_config=[           # Impacto de data products
        {
            "name": "customer_analytics",
            "assets": ["customers", "orders"],
            "critical_assets": ["customers"],
            "consumers": 10,
            "revenue_impact": "high"
        }
    ],
    asset_metadata=[               # Metadados com criticidade/risco/PII
        {"name": "customers", "criticality": "high", "has_pii": True},
        {"name": "orders", "criticality": "medium", "has_pii": False}
    ],
)

print(report.to_markdown())
```

### Comparar ativos e exportar relatórios

```python
# Comparar ativos específicos
comparison = agent.compare_assets(["customers", "orders", "products"], report)

# Exportar
with open("value_report.json", "w", encoding="utf-8") as f:
    f.write(report.to_json())

with open("value_report.md", "w", encoding="utf-8") as f:
    f.write(report.to_markdown())
```

## Exemplos

Um fluxo completo está em `data_asset_value/examples/usage_example.py` com dados de exemplo (`sample_query_logs.json`, `data_products_config.json`, `asset_metadata.json`).

```bash
python data_asset_value/examples/usage_example.py
```

## Principais Classes

- **DataAssetValueAgent**: orquestra parsing de logs, cálculo de escores e geração de recomendações.
- **QueryLogParser**: extrai tabelas usadas, JOINs, tipos de uso e metadados básicos de logs SQL.
- **ValueCalculator**: combina escores de uso, JOINs, linhagem e data products, aplicando multiplicadores de criticidade, risco e PII para o `business_impact_score`.
- **AssetValueReport**: consolida resultados, converte para JSON/Markdown e lista ativos críticos, órfãos e hubs de conectividade.

## Integrações

- **Data Lineage Agent**: passe `lineage_data` para considerar impacto upstream/downstream e caminhos críticos.
- **Data Products**: informe `data_product_config` para refletir impacto de receita, criticidade e consumidores dos produtos em cada ativo.

## Métricas e Saídas

- **Escores por ativo**: `usage_score`, `join_score`, `lineage_score`, `data_product_score`, `overall_value_score` e `business_impact_score` com categoria (`critical/high/medium/low`).
- **Insights**: ativos mais valiosos, críticos, órfãos, com uso em declínio e hubs de JOIN.
- **Relatórios**: exportação em JSON ou Markdown com tabelas e recomendações.
