# Data Classification Agent

## Resumo
Sistema de IA para classificar automaticamente dados em níveis de sensibilidade (PII, PHI, PCI, financeiro e termos estratégicos) usando schemas, amostras opcionais e validação com LLM.

## Summary
AI system to automatically classify data sensitivity (PII, PHI, PCI, financial, and strategic terms) using schemas, optional samples, and LLM validation.

*English details available at the end of the file.*

## Características
- **Detecção de PII/PHI/PCI/financeiro** com regras de palavras-chave, padrões e tipos.
- **Vocabulário proprietário customizável** para termos estratégicos do negócio.
- **Compliance automático** (LGPD/GDPR, HIPAA, PCI-DSS, SOX).
- **Validação generativa opcional** sem ler dados brutos (schema-only com LLM).
- **Exportação de relatórios** em JSON ou Markdown.

## Instalação
```bash
pip install -r requirements.txt
```

## Uso Rápido
```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()
report = agent.classify_from_csv("customers.csv")

print(report.overall_sensitivity)
print(report.pii_columns)
print(report.compliance_flags)
```

Para revisão apenas com schema, forneça colunas e, opcionalmente, um `LLMProvider` via `classify_schema_with_llm`.

## Vocabulário Estratégico
```python
agent.add_business_terms([
    "Projeto Arara Azul",
    "Mercado LATAM",
    "Aquisição Orion",
])
```

O vocabulário é aplicado tanto em metadados quanto em amostras para marcar colunas como proprietárias/confidenciais.

## Níveis de Sensibilidade
| Nível | Descrição | Exemplos |
|-------|-----------|----------|
| **Public** | Dados públicos | Códigos de produto, categorias |
| **Internal** | Uso interno | IDs internos, timestamps |
| **Confidential** | Dados sensíveis | Dados financeiros, PII básico |
| **Restricted** | Altamente restrito | PHI, PCI, PII crítico |

## Integração
- Combine com **Data Lineage Agent** para rastrear onde dados sensíveis fluem.
- Envie resultados para **Metadata Enrichment** e **Discovery** para catalogação e busca semântica.

---

## Summary
AI system to automatically classify data sensitivity (PII, PHI, PCI, financial, and strategic terms) using schemas, optional samples, and LLM validation.

## Features
- **Detection of PII/PHI/PCI/financial data** using keyword, pattern, and type rules.
- **Customizable proprietary vocabulary** for business-sensitive terms.
- **Automatic compliance flags** (LGPD/GDPR, HIPAA, PCI-DSS, SOX).
- **Optional generative validation** without raw data (schema-only LLM review).
- **Report export** in JSON or Markdown.

## Installation
```bash
pip install -r requirements.txt
```

## Quickstart
```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()
report = agent.classify_from_csv("customers.csv")

print(report.overall_sensitivity)
print(report.pii_columns)
print(report.compliance_flags)
```

For schema-only review, supply columns and optionally an `LLMProvider` through `classify_schema_with_llm`.

## Strategic Vocabulary
```python
agent.add_business_terms([
    "Projeto Arara Azul",
    "Mercado LATAM",
    "Aquisição Orion",
])
```

The vocabulary applies to metadata and samples to flag proprietary/confidential columns.

## Sensitivity Levels
| Level | Description | Examples |
|-------|-------------|----------|
| **Public** | Public data | Product codes, categories |
| **Internal** | Internal use | Internal IDs, timestamps |
| **Confidential** | Sensitive data | Financial data, basic PII |
| **Restricted** | Highly restricted | PHI, PCI, critical PII |

## Integration
- Combine with the **Data Lineage Agent** to trace where sensitive data flows.
- Send results to **Metadata Enrichment** and **Discovery** for cataloging and semantic search.
