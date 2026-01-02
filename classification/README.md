# 🛡️ Data Classification Agent

Agente de IA para **classificar automaticamente dados sensíveis (PII, PHI, financeiros)** usando apenas schemas e metadados.
*AI agent to automatically classify sensitive data (PII, PHI, financial) using only schemas and metadata.*

## Visão Geral
*Overview.*

O agente avalia nomes, tipos, descrições e tags para detectar categorias sensíveis alinhadas a LGPD/GDPR, sem ler valores das tabelas.
*The agent checks column names, types, descriptions, and tags to detect sensitive categories aligned with LGPD/GDPR without reading raw values.*

## Características
*Features.*

- **Classificação sem acessar dados brutos** (apenas metadados).
  *Classification without touching raw data (metadata only).* 
- **Detecção de PII/PHI/financeiro** por regras ponderadas.
  *PII/PHI/financial detection using weighted rules.*
- **Validação opcional com LLM** para revisão contextual.
  *Optional LLM validation for contextual review.*
- **Níveis de sensibilidade**: LOW, MEDIUM, HIGH, CRITICAL.
  *Sensitivity levels: LOW, MEDIUM, HIGH, CRITICAL.*
- **Extensível** com regras e controles adicionais.
  *Extensible with additional rules and controls.*

## Guia Rápido
*Quickstart.*

```python
from classification import ColumnMetadata, DataClassificationAgent, TableSchema

table = TableSchema(
    name="customers",
    schema="public",
    description="Cadastro de clientes com CPF, email e telefone",
    columns=[
        ColumnMetadata(name="customer_id", type="bigint", description="Identificador"),
        ColumnMetadata(name="cpf", type="varchar", description="Documento nacional", tags=["pii"]),
        ColumnMetadata(name="email", type="varchar"),
        ColumnMetadata(name="phone_number", type="varchar"),
    ],
    tags=["gold", "pii"],
)

agent = DataClassificationAgent()
result = agent.classify_table(table)

print(result.sensitivity_level)
print(result.detected_categories)
for column in result.columns:
    print(column.column.name, column.categories, column.suggested_controls)
```

Para validação generativa, inicialize com um `LLMProvider` e chame `classify_table_with_llm`.
*For generative validation, initialize with an `LLMProvider` and call `classify_table_with_llm`.*

## Arquitetura Lógica
*Logical architecture.*

1. Entrada de metadados (`TableSchema`/`ColumnMetadata`).
   *Metadata input (`TableSchema`/`ColumnMetadata`).*
2. Regras de sensibilidade (`SensitiveDataRule`).
   *Sensitivity rules (`SensitiveDataRule`).*
3. Pontuação por coluna e consolidação por tabela.
   *Column scoring and table-level aggregation.*
4. Recomendações de controles LGPD/GDPR.
   *LGPD/GDPR control recommendations.*
5. Saída auditável (`TableClassification`).
   *Auditable output (`TableClassification`).*

## Como Estender
*How to extend.*

- Adicione regras customizadas ao inicializar o agente.
  *Add custom rules when initializing the agent.*
- Inclua controles adicionais via `lgpd_requirements` e `gdpr_requirements`.
  *Include extra controls via `lgpd_requirements` and `gdpr_requirements`.*
- Integre com Lineage e Discovery para workflows completos.
  *Integrate with Lineage and Discovery for end-to-end workflows.*
