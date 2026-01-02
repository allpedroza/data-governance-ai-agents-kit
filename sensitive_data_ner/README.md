# Sensitive Data NER Agent

Agente de Named Entity Recognition (NER) para detecção e anonimização de dados sensíveis em texto livre. Projetado para ser usado como filtro de proteção para requisições a LLMs, prevenindo vazamento de informações confidenciais.

## Proposta de Valor

Este agente permite que empresas disponibilizem IA Generativa (LLMs) para seus colaboradores de forma segura, atuando como um **gateway de proteção** que:

- **Detecta** dados sensíveis em prompts antes do envio a LLMs terceiras
- **Anonimiza** automaticamente informações confidenciais
- **Bloqueia** requisições que violam políticas de segurança
- **Audita** tentativas de vazamento de dados

## Categorias Suportadas

| Categoria | Descrição | Exemplos |
|-----------|-----------|----------|
| **PII** | Dados Pessoais Identificáveis | CPF, CNPJ, Email, Telefone, Nome, Endereço |
| **PHI** | Informações de Saúde Protegidas | CID-10, CNS, Prontuário, Diagnósticos |
| **PCI** | Dados de Cartão de Pagamento | Número do cartão, CVV, Data de validade |
| **FINANCIAL** | Informações Financeiras | IBAN, Contas bancárias, PIX, Criptomoedas |
| **BUSINESS** | Dados Estratégicos de Negócio | Projetos confidenciais, Aquisições, Termos proprietários |

## Detecção Preditiva vs Determinística

O agente combina duas abordagens para maximizar precisão:

### Detecção Determinística (Regex)
- Padrões bem definidos (CPF, cartão de crédito, email)
- Alta precisão para formatos conhecidos
- 50+ padrões pré-configurados

### Detecção Preditiva (Heurísticas)
- **Validação de Checksum**: CPF, CNPJ, Cartão de crédito, IBAN, CNS
- **Análise de Contexto**: Palavras-chave próximas aumentam confiança
- **Qualidade de Formato**: Formatação correta indica dado real
- **Detecção de Negação**: "exemplo", "teste", "inválido" reduzem confiança

```python
# Exemplo: CPF com checksum válido + contexto "cliente" = alta confiança
"O CPF do cliente é 123.456.789-09"  # Confiança: 0.95

# Exemplo: Mesmo padrão sem contexto = confiança menor
"123.456.789-09"  # Confiança: 0.65

# Exemplo: Padrão em contexto de teste = baixa confiança
"CPF de teste: 000.000.000-00"  # Confiança: 0.25 (ignorado)
```

## Instalação

```bash
# Já incluído no projeto principal
cd data-governance-ai-agents-kit
pip install -r requirements.txt
```

## Uso Básico

### Análise de Texto

```python
from sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()

text = """
O cliente João Silva, CPF 123.456.789-09, solicitou atualização.
Email: joao@email.com, telefone (11) 98765-4321.
Pagamento via cartão 4532 1234 5678 9010.
"""

result = agent.analyze(text)

print(f"Entidades encontradas: {result.statistics['total']}")
print(f"Score de risco: {result.risk_score:.1%}")
print(f"Ação recomendada: {result.filter_action.value}")
print(f"\nTexto anonimizado:\n{result.anonymized_text}")
```

**Saída:**
```
Entidades encontradas: 5
Score de risco: 72.3%
Ação recomendada: anonymize

Texto anonimizado:
O cliente [NOME], [CPF], solicitou atualização.
Email: [EMAIL], telefone [TELEFONE].
Pagamento via cartão [CARTÃO].
```

### Filtro para LLM

```python
from sensitive_data_ner import SensitiveDataNERAgent, FilterPolicy, FilterAction

# Configurar política de filtro
policy = FilterPolicy(
    pii_action=FilterAction.ANONYMIZE,
    phi_action=FilterAction.BLOCK,  # Bloquear dados de saúde
    pci_action=FilterAction.BLOCK,  # Bloquear dados de cartão
    business_action=FilterAction.BLOCK,  # Bloquear dados estratégicos
)

agent = SensitiveDataNERAgent(filter_policy=policy)

# Usar como gateway para LLM
try:
    safe_prompt, result = agent.filter_llm_request(user_prompt)
    llm_response = send_to_llm(safe_prompt)  # Sua função de LLM
except ValueError as e:
    print(f"Requisição bloqueada: {e}")
```

### Termos de Negócio

```python
agent = SensitiveDataNERAgent(
    business_terms=[
        "Projeto Arara Azul",
        "Aquisição Orion",
        "Parceria LATAM 2025",
        "Operação Fênix",
    ]
)

text = "Atualização sobre o Projeto Arara Azul para a reunião de amanhã."
result = agent.analyze(text)

# Detecta "Projeto Arara Azul" como dado estratégico
print(result.entities[0].category)  # EntityCategory.BUSINESS
```

### Integração com Classification Agent

```python
from data_classification import DataClassificationAgent
from sensitive_data_ner import SensitiveDataNERAgent

# Classificar dados estruturados primeiro
classifier = DataClassificationAgent(
    business_sensitive_terms=["código interno", "margem"]
)
report = classifier.classify_from_csv("vendas.csv")

# Importar termos detectados para o NER
ner_agent = SensitiveDataNERAgent()
ner_agent.import_from_classification_agent(report.to_dict())

# Agora o NER conhece os termos do Classification Agent
result = ner_agent.analyze(user_text)
```

## Estratégias de Anonimização

| Estratégia | Descrição | Exemplo |
|------------|-----------|---------|
| `REDACT` | Substitui por rótulo | `123.456.789-09` → `[CPF]` |
| `MASK` | Mascara com asteriscos | `123.456.789-09` → `***.***.**9-**` |
| `PARTIAL` | Mantém início/fim | `123.456.789-09` → `12*.***.***-09` |
| `HASH` | Hash determinístico | `123.456.789-09` → `CPF_a1b2c3d4` |
| `PSEUDONYMIZE` | Dados falsos consistentes | `João Silva` → `Carlos Souza` |
| `ENCRYPT` | Criptografia reversível | (requer chave) |

```python
from sensitive_data_ner import AnonymizationStrategy, FilterPolicy

policy = FilterPolicy(
    anonymization_strategy=AnonymizationStrategy.PARTIAL
)

agent = SensitiveDataNERAgent(filter_policy=policy)
```

## Configuração de Política

```python
from sensitive_data_ner import FilterPolicy, FilterAction

policy = FilterPolicy(
    # Ações por categoria
    pii_action=FilterAction.ANONYMIZE,
    phi_action=FilterAction.BLOCK,
    pci_action=FilterAction.BLOCK,
    financial_action=FilterAction.ANONYMIZE,
    business_action=FilterAction.BLOCK,

    # Limites
    min_confidence=0.5,  # Ignorar entidades com confiança < 50%
    max_entities_before_block=10,  # Bloquear se > 10 entidades
    risk_score_block_threshold=0.8,  # Bloquear se risco > 80%
)
```

## Localização

O agente suporta padrões específicos por país:

```python
# Apenas padrões brasileiros e universais
agent = SensitiveDataNERAgent(locales=["br"])

# Brasil + EUA
agent = SensitiveDataNERAgent(locales=["br", "us"])

# Todos os padrões (padrão)
agent = SensitiveDataNERAgent(locales=None)
```

### Padrões por Locale

| Locale | Padrões Específicos |
|--------|---------------------|
| `br` | CPF, CNPJ, RG, CNS, CEP, Telefone BR, CRM, PIX |
| `us` | SSN, EIN, ZIP Code, NPI, DEA |
| `es` | NIF/NIE |
| `pt` | NIF Portugal |
| `universal` | Email, IP, Cartão de crédito, IBAN |

## Interface Streamlit

```bash
cd sensitive_data_ner
streamlit run streamlit_app.py
```

Acesse `http://localhost:8501` para interface visual.

## Arquitetura

```
sensitive_data_ner/
├── __init__.py              # Exports principais
├── agent.py                 # SensitiveDataNERAgent (principal)
├── anonymizers.py           # Estratégias de anonimização
├── streamlit_app.py         # Interface visual
├── patterns/
│   ├── __init__.py
│   └── entity_patterns.py   # Padrões regex por categoria
├── predictive/
│   ├── __init__.py
│   ├── validators.py        # Validação de checksum
│   └── heuristics.py        # Análise de contexto e confiança
└── examples/
    └── usage_example.py     # Exemplos de uso
```

## Casos de Uso

### 1. Gateway de LLM Corporativo

```python
from fastapi import FastAPI, HTTPException
from sensitive_data_ner import SensitiveDataNERAgent, FilterPolicy

app = FastAPI()
agent = SensitiveDataNERAgent(
    business_terms=load_corporate_terms(),
    filter_policy=FilterPolicy(...)
)

@app.post("/llm/complete")
async def llm_complete(prompt: str):
    try:
        safe_prompt, audit = agent.filter_llm_request(prompt)
        response = await call_external_llm(safe_prompt)
        log_audit(audit)  # Auditoria
        return {"response": response}
    except ValueError as e:
        raise HTTPException(403, str(e))
```

### 2. Validação de Documentos

```python
def validate_document_for_sharing(document_text: str) -> bool:
    result = agent.analyze(document_text)

    if result.filter_action == FilterAction.BLOCK:
        raise SecurityException(
            f"Documento contém dados sensíveis: {result.blocked_reason}"
        )

    return result.is_safe
```

### 3. Pré-processamento de Dados

```python
def anonymize_dataset(texts: List[str]) -> List[str]:
    agent = SensitiveDataNERAgent()
    anonymized = []

    for result in agent.batch_analyze(texts):
        anonymized.append(result.anonymized_text or result.original_text)

    return anonymized
```

## Métricas de Qualidade

O agente fornece métricas detalhadas:

```python
result = agent.analyze(text)

print(result.statistics)
# {
#     'total': 5,
#     'pii': 3,
#     'phi': 0,
#     'pci': 1,
#     'financial': 1,
#     'business': 0,
#     'validated': 4,  # Entidades com checksum válido
#     'high_confidence': 3,  # Confiança >= 80%
# }

print(f"Tempo de processamento: {result.processing_time_ms:.2f}ms")
```

## Compliance

O agente ajuda no cumprimento de:

- **LGPD** (Lei Geral de Proteção de Dados)
- **GDPR** (General Data Protection Regulation)
- **HIPAA** (Health Insurance Portability and Accountability Act)
- **PCI-DSS** (Payment Card Industry Data Security Standard)
- **SOX** (Sarbanes-Oxley Act)

## Limitações

1. **Falsos Negativos**: Dados mal formatados podem não ser detectados
2. **Contexto Limitado**: Análise de contexto usa janela fixa de 100 caracteres
3. **Nomes Próprios**: Detecção de nomes requer contexto adicional
4. **Performance**: Textos muito longos podem ter latência maior

## Contribuindo

Para adicionar novos padrões:

1. Adicione em `patterns/entity_patterns.py`
2. Implemente validador em `predictive/validators.py` (se aplicável)
3. Adicione testes
4. Atualize documentação
