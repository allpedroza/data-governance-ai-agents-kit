# Data Contracts Module

Este módulo define contratos de dados (data contracts) para pipelines de ingestão, incluindo:

- **Schema**: lista de campos com tipo, obrigatoriedade e classificação.
- **Regras de qualidade**: validações como *not null*, *unique*, *range*, *regex* e valores permitidos.
- **SLAs**: freshness, latência e disponibilidade.

## Uso básico

```python
from data_governance.data_contracts import DataContractAgent, ContractField, DataQualityRule, DataContractSLA

agent = DataContractAgent()
contract = agent.create_contract(
    name="orders",
    version="1.0",
    owner="data-platform",
    domain="sales",
    fields=[
        ContractField(name="order_id", data_type="string", required=True),
        ContractField(name="amount", data_type="float", required=True),
    ],
    quality_rules=[
        DataQualityRule(
            name="order_id_not_null",
            rule_type="not_null",
            column="order_id",
            severity="critical",
            parameters={"max_null_pct": 0.0},
        )
    ],
    sla=DataContractSLA(freshness_hours=2, latency_minutes=15, availability_pct=99.5),
)

agent.save_contract(contract)
```

## Validação de dados

```python
import pandas as pd
from data_governance.data_contracts import DataContractAgent

agent = DataContractAgent()
contract = agent.load_contract("./data_contracts/orders_v1.0.json")

df = pd.read_csv("orders.csv")
report = agent.validate_dataframe(contract, df)
print(report.to_json())
```
