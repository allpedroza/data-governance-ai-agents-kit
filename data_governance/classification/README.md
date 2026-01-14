# 🛡️ Data Classification Agent

## Resumo
Agente de IA para classificar automaticamente dados sensíveis (PII, PHI, financeiros) usando apenas schemas e metadados.

## Summary
AI agent to automatically classify sensitive data (PII, PHI, financial) using only schemas and metadata.

*English details available at the end of the file.*

## Visão Geral
O agente avalia nomes, tipos, descrições e tags para detectar categorias sensíveis alinhadas a LGPD/GDPR, sem ler valores das tabelas.

## Características
- **Classificação sem acessar dados brutos** (apenas metadados).
- **Detecção de PII/PHI/financeiro** por regras ponderadas.
- **Validação opcional com LLM** para revisão contextual.
- **Níveis de sensibilidade**: LOW, MEDIUM, HIGH, CRITICAL.
- **Extensível** com regras e controles adicionais.

## Guia Rápido
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

## Arquitetura Lógica
1. Entrada de metadados (`TableSchema`/`ColumnMetadata`).
2. Regras de sensibilidade (`SensitiveDataRule`).
3. Pontuação por coluna e consolidação por tabela.
4. Recomendações de controles LGPD/GDPR.
5. Saída auditável (`TableClassification`).

## Como Estender
- Adicione regras customizadas ao inicializar o agente.
- Inclua controles adicionais via `lgpd_requirements` e `gdpr_requirements`.
- Integre com Lineage e Discovery para workflows completos.

---

## Summary
AI agent to automatically classify sensitive data (PII, PHI, financial) using only schemas and metadata.

## Overview
The agent checks column names, types, descriptions, and tags to detect sensitive categories aligned with LGPD/GDPR without reading raw values.

## Features
- **Classification without touching raw data** (metadata only).
- **PII/PHI/financial detection** using weighted rules.
- **Optional LLM validation** for contextual review.
- **Sensitivity levels**: LOW, MEDIUM, HIGH, CRITICAL.
- **Extensible** with additional rules and controls.

## Quickstart
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

For generative validation, initialize with an `LLMProvider` and call `classify_table_with_llm`.

## Logical Architecture
1. Metadata input (`TableSchema`/`ColumnMetadata`).
2. Sensitivity rules (`SensitiveDataRule`).
3. Column scoring and table-level aggregation.
4. LGPD/GDPR control recommendations.
5. Auditable output (`TableClassification`).

## How to Extend
- Add custom rules when initializing the agent.
- Include extra controls via `lgpd_requirements` and `gdpr_requirements`.
- Integrate with Lineage and Discovery for end-to-end workflows.
