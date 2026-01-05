# Sensitive Data NER Agent

## Resumo
Agente de NER para detectar, anonimizar e bloquear dados sensíveis em texto livre antes de enviar a LLMs.

## Summary
NER agent to detect, anonymize, and block sensitive data in free text before sending to LLMs.

*English details available at the end of the file.*

## Proposta de Valor
Funciona como um gateway de proteção: detecta PII/PHI/PCI/financeiro/termos estratégicos, anonimiza automaticamente e registra tentativas de vazamento.

## Categorias Suportadas
- **PII** (CPF, CNPJ, email, telefone, endereço).
- **PHI** (CID-10, CNS, prontuário).
- **PCI** (número de cartão, CVV, validade).
- **Financeiro** (IBAN, contas, PIX, criptomoedas).
- **Business** (projetos confidenciais, aquisições, termos proprietários).

## Instalação
```bash
pip install -r requirements.txt
```

## Uso Básico
```python
from sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()
text = "O cliente João Silva, CPF 123.456.789-09, email joao@email.com"
result = agent.analyze(text)

print(result.statistics)
print(result.filter_action)
print(result.anonymized_text)
```

Combina regex determinístico (padrões e checksums) com heurísticas de contexto para ajustar confiança e reduzir falsos positivos.

---

## Summary
NER agent to detect, anonymize, and block sensitive data in free text before sending to LLMs.

## Value Proposition
Acts as a protection gateway: detects PII/PHI/PCI/financial/strategic terms, anonymizes automatically, and logs leakage attempts.

## Supported Categories
- **PII** (CPF, CNPJ, email, phone, address).
- **PHI** (ICD-10 codes, CNS, medical records).
- **PCI** (card number, CVV, expiry).
- **Financial** (IBAN, accounts, PIX, crypto).
- **Business** (confidential projects, acquisitions, proprietary terms).

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage
```python
from sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()
text = "O cliente João Silva, CPF 123.456.789-09, email joao@email.com"
result = agent.analyze(text)

print(result.statistics)
print(result.filter_action)
print(result.anonymized_text)
```

Combines deterministic regex (patterns and checksums) with contextual heuristics to adjust confidence and reduce false positives.
