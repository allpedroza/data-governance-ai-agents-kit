# Data Product Scoring Layer

## Resumo
Camada de pontuação para **Data Products** que combina:
- **Outputs do Data Quality Agent** (qualidade, frescor e confiabilidade).
- **Outputs do Data Asset Value Agent** (valor e impacto dos ativos).
- **Completude do Data Contract**, governança e forma de entrega.

A pontuação ajuda a metrificar data products de forma objetiva, alinhada à definição:
"conjunto reutilizável de dados + lógica + metadados + governança, entregue de forma confiável para resolver um problema de negócio específico".

## Como funciona
O scoring agrega sete dimensões:
1. **Propósito de negócio** (propósito + outcomes)
2. **Consumidores definidos**
3. **Contrato de dados** (schema, granularidade, SLA, semântica, regras de qualidade)
4. **Qualidade e confiabilidade** (Data Quality Agent)
5. **Governança embutida** (owner, steward, classificação, lineage, políticas)
6. **Forma de entrega** (tabela/API/feature store etc.)
7. **Valor de ativos** (Data Asset Value Agent)

## Uso rápido
```python
from data_governance.data_product_scoring import (
    DataProductScoringAgent,
    DataProductDefinition,
    DataProductContract,
    DataProductGovernance,
)

product = DataProductDefinition(
    name="customer_churn",
    purpose="Reduzir churn em 10%",
    business_outcomes=["redução de churn", "priorização de retenção"],
    consumers=["time_crm", "dashboard_churn"],
    assets=["customers", "orders"],
    delivery_format="gold_table",
    delivery_channel="warehouse",
    contract=DataProductContract(
        schema=[{"name": "customer_id", "type": "string"}],
        granularity="customer",
        sla_hours=12,
        freshness_sla_hours=6,
        semantics={"churn_flag": "1 se o cliente cancelou"},
        quality_rules=[{"rule": "not_null", "column": "customer_id"}],
        version="v1",
    ),
    governance=DataProductGovernance(
        owner="data.product@empresa.com",
        steward="data.governance@empresa.com",
        classification="PII",
        lineage="lineage://dp/customer_churn",
        access_policies=["role:analytics"],
        pii=True,
        retention_policy="2 anos",
    ),
)

agent = DataProductScoringAgent()
score = agent.score_product(product)
print(score.to_markdown())
```

## Integração direta com os agentes
```python
from data_governance.data_product_scoring import DataProductScoringAgent, DataProductDefinition

agent = DataProductScoringAgent()
score, quality_report, value_report = agent.score_product_with_agents(
    product=DataProductDefinition(name="customer_churn", assets=["customers"]) ,
    quality_source="data/customers.parquet",
    query_logs=[{"query": "SELECT * FROM customers", "timestamp": "2024-12-15T10:00:00Z"}],
    data_product_config=[{"name": "customer_churn", "assets": ["customers"]}],
)
```

## Observações
- Se o relatório de qualidade ou valor não for fornecido, a dimensão recebe nota zero.
- Use `score_portfolio` para gerar um relatório consolidado.
