# Data Classification Agent

Sistema de IA para **classificar automaticamente dados** em níveis de sensibilidade (PII, PHI, PCI, financeiro e termos estratégicos) usando schemas, amostras opcionais e validação com LLM.
*AI system to automatically classify data sensitivity (PII, PHI, PCI, financial, and strategic terms) using schemas, optional samples, and LLM validation.*

## Características
*Features.*

- **Detecção de PII/PHI/PCI/financeiro** com regras de palavras-chave, padrões e tipos.
  *Detection of PII/PHI/PCI/financial data using keyword, pattern, and type rules.*
- **Vocabulário proprietário customizável** para termos estratégicos do negócio.
  *Customizable proprietary vocabulary for business-sensitive terms.*
- **Compliance automático** (LGPD/GDPR, HIPAA, PCI-DSS, SOX).
  *Automatic compliance flags (LGPD/GDPR, HIPAA, PCI-DSS, SOX).* 
- **Validação generativa opcional** sem ler dados brutos (schema-only com LLM).
  *Optional generative validation without raw data (schema-only LLM review).* 
- **Exportação de relatórios** em JSON ou Markdown.
  *Report export in JSON or Markdown.*

## Instalação
*Installation.*

```bash
pip install -r requirements.txt
```

## Uso Rápido
*Quickstart.*

```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()
report = agent.classify_from_csv("customers.csv")

print(report.overall_sensitivity)
print(report.pii_columns)
print(report.compliance_flags)
```

Para revisão apenas com schema, forneça colunas e, opcionalmente, um `LLMProvider` via `classify_schema_with_llm`.
*For schema-only review, supply columns and optionally an `LLMProvider` through `classify_schema_with_llm`.*

## Vocabulário Estratégico
*Strategic vocabulary.*

```python
agent.add_business_terms([
    "Projeto Arara Azul",
    "Mercado LATAM",
    "Aquisição Orion",
])
```

O vocabulário é aplicado tanto em metadados quanto em amostras para marcar colunas como proprietárias/confidenciais.
*The vocabulary applies to metadata and samples to flag proprietary/confidential columns.*

## Níveis de Sensibilidade
*Sensitivity levels.*

| Nível | Descrição | Exemplos |
|-------|-----------|----------|
| **Public** | Dados públicos | Códigos de produto, categorias |
| **Internal** | Uso interno | IDs internos, timestamps |
| **Confidential** | Dados sensíveis | Dados financeiros, PII básico |
| **Restricted** | Altamente restrito | PHI, PCI, PII crítico |

## Integração
*Integration.*

- Combine com **Data Lineage Agent** para rastrear onde dados sensíveis fluem.
  *Combine with the Data Lineage Agent to trace where sensitive data flows.*
- Envie resultados para **Metadata Enrichment** e **Discovery** para catalogação e busca semântica.
  *Send results to Metadata Enrichment and Discovery for cataloging and semantic search.*
