# Sensitive Data NER Agent

Agente de NER para detectar, anonimizar e bloquear dados sensíveis em texto livre antes de enviar a LLMs.
*NER agent to detect, anonymize, and block sensitive data in free text before sending to LLMs.*

## Proposta de Valor
*Value proposition.*

Funciona como um gateway de proteção: detecta PII/PHI/PCI/financeiro/termos estratégicos, anonimiza automaticamente e registra tentativas de vazamento.
*Acts as a protection gateway: detects PII/PHI/PCI/financial/strategic terms, anonymizes automatically, and logs leakage attempts.*

## Categorias Suportadas
*Supported categories.*

- **PII** (CPF, CNPJ, email, telefone, endereço).
  *PII (CPF, CNPJ, email, phone, address).* 
- **PHI** (CID-10, CNS, prontuário).
  *PHI (ICD-10 codes, CNS, medical records).* 
- **PCI** (número de cartão, CVV, validade).
  *PCI (card number, CVV, expiry).* 
- **Financeiro** (IBAN, contas, PIX, criptomoedas).
  *Financial (IBAN, accounts, PIX, crypto).* 
- **Business** (projetos confidenciais, aquisições, termos proprietários).
  *Business (confidential projects, acquisitions, proprietary terms).* 

## Instalação
*Installation.*

```bash
pip install -r requirements.txt
```

## Uso Básico
*Basic usage.*

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
*Combines deterministic regex (patterns and checksums) with contextual heuristics to adjust confidence and reduce false positives.*
