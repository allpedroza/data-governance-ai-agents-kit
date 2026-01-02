# Data Classification Agent

Sistema de IA para **classificação automática de dados** por níveis de sensibilidade, detectando PII, PHI, PCI, dados financeiros e termos estratégicos do negócio via dicionário proprietário.

## Características

- **Detecção de PII** (Personally Identifiable Information)
  - CPF, CNPJ, RG, NIS/PIS
  - Email, telefone, endereço
  - Passaporte, CNH
  - IP, MAC address

- **Detecção de PHI** (Protected Health Information)
  - CID-10 (códigos de doenças)
  - CNS (Cartão Nacional de Saúde)
  - CRM, prontuário médico
  - Tipo sanguíneo

- **Detecção de PCI** (Payment Card Industry)
  - Números de cartão de crédito
  - CVV/CVC
  - IBAN, SWIFT

- **Detecção de dados financeiros**
  - Contas bancárias, agências
  - Valores, transações
  - Boletos, PIX
- **Termos estratégicos do negócio**
  - Dicionário customizável com nomes de projetos e iniciativas
  - Classificação como dado proprietário/confidencial
  - Indicadores de cobertura em relatórios (colunas e contagens)

- **Compliance automático**
  - LGPD / GDPR
  - HIPAA
  - PCI-DSS
  - SOX

## Instalação

```bash
pip install -r requirements.txt
```

## Uso Rápido

```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()

# Classificar arquivo CSV
report = agent.classify_from_csv("customers.csv")

print(f"Sensibilidade: {report.overall_sensitivity}")
print(f"Colunas PII: {report.pii_columns}")
print(f"Colunas PHI: {report.phi_columns}")
print(f"Termos estratégicos: {report.proprietary_columns}")
print(f"Compliance: {report.compliance_flags}")
```

## Dicionário de termos estratégicos

Use o dicionário embutido para proteger informações sensíveis à estratégia do negócio (roadmaps, iniciativas e nomes de projetos). O agente usa esses termos tanto em metadados quanto em amostras de valores para classificar colunas como **proprietary/confidential**.

```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent()

# Popular o vocabulário proprietário
agent.add_business_terms([
    "Projeto Arara Azul",
    "Mercado LATAM",
    "Aquisição Orion",
])

# Classificar com termos estratégicos
report = agent.classify_from_csv("roadmap.csv")

print(report.proprietary_columns)           # Colunas que contém termos estratégicos
print(report.metrics["proprietary_count"]) # Quantidade de colunas proprietárias
```

## Níveis de Sensibilidade

| Nível | Descrição | Exemplos |
|-------|-----------|----------|
| **Public** | Dados públicos | Códigos de produto, categorias |
| **Internal** | Uso interno | IDs internos, timestamps |
| **Confidential** | Dados sensíveis | Dados financeiros, PII básico |
| **Restricted** | Altamente restrito | PHI, PCI, PII crítico |

## Exemplos

### Classificação de CSV

```python
from data_classification import DataClassificationAgent

agent = DataClassificationAgent(sample_size=5000)

report = agent.classify_from_csv(
    "dados_clientes.csv",
    encoding="utf-8",
    separator=";"
)

# Verificar PII detectado
if report.pii_columns:
    print(f"⚠️ PII detectado em: {report.pii_columns}")
    for col in report.columns:
        if "pii" in col.categories:
            print(f"  - {col.name}: {col.pii_type} ({col.confidence:.0%})")
```

### Classificação de Parquet

```python
report = agent.classify_from_parquet("data_lake/customers.parquet")

# Verificar compliance
for flag in report.compliance_flags:
    print(f"📋 {flag}")
```

### Adicionar padrões customizados

```python
# Adicionar padrão para protocolo interno
agent.add_custom_pattern(
    "protocolo_interno",
    r"PROT-\d{4}-\d{8}"
)

# Classificar com o novo padrão
report = agent.classify_from_csv("protocolos.csv")
```

### Adicionar termos estratégicos de negócio

```python
# Popular o dicionário de termos críticos da estratégia
agent.add_business_terms([
    "Projeto Arara Azul",
    "Mercado LATAM",
    "Aquisição Orion"
])

# Classificar usando o vocabulário proprietário
report = agent.classify_from_csv("roadmap.csv")
print(report.proprietary_columns)
```

### Exportar relatório

```python
# JSON
with open("classification_report.json", "w") as f:
    f.write(report.to_json())

# Markdown
with open("classification_report.md", "w") as f:
    f.write(report.to_markdown())
```

## Integração com outros agentes

### Com Metadata Enrichment Agent

```python
from data_classification import DataClassificationAgent
from metadata_enrichment import MetadataEnrichmentAgent

# Classificar primeiro
classifier = DataClassificationAgent()
classification = classifier.classify_from_csv("data.csv")

# Usar classificação no enriquecimento
enricher = MetadataEnrichmentAgent(...)
enrichment = enricher.enrich_from_csv(
    "data.csv",
    additional_context=f"Sensitivity: {classification.overall_sensitivity}"
)
```

### Com Data Quality Agent

```python
from data_classification import DataClassificationAgent
from data_quality import DataQualityAgent

# Classificar
classifier = DataClassificationAgent()
classification = classifier.classify_from_csv("data.csv")

# Aplicar regras de qualidade diferentes por sensibilidade
quality_agent = DataQualityAgent()

if classification.overall_sensitivity == "restricted":
    # Regras mais rígidas para dados sensíveis
    report = quality_agent.evaluate_file(
        "data.csv",
        validity_configs=[{
            "column": col,
            "threshold": 0.99  # 99% de validade para PII
        } for col in classification.pii_columns]
    )
```

## Interface Streamlit

```bash
streamlit run streamlit_app.py
```

## API Reference

### DataClassificationAgent

```python
class DataClassificationAgent:
    def __init__(
        self,
        custom_patterns: Dict[str, str] = None,  # Padrões regex customizados
        sensitivity_rules: Dict[str, str] = None,  # Regras de sensibilidade
        sample_size: int = 1000  # Linhas para amostragem
    )

    def classify_from_csv(
        self,
        file_path: str,
        encoding: str = "utf-8",
        separator: str = ",",
        sample_size: int = None
    ) -> ClassificationReport

    def classify_from_parquet(
        self,
        file_path: str,
        sample_size: int = None
    ) -> ClassificationReport

    def classify_from_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str = "dataframe"
    ) -> ClassificationReport

    def classify_from_sql(
        self,
        connection_string: str,
        query: str,
        table_name: str,
        sample_size: int = None
    ) -> ClassificationReport

    def add_custom_pattern(
        self,
        name: str,
        pattern: str
    ) -> None
```

### ClassificationReport

```python
@dataclass
class ClassificationReport:
    source_name: str
    source_type: str
    classification_timestamp: str
    overall_sensitivity: str  # public, internal, confidential, restricted
    categories_found: List[str]
    columns: List[ColumnClassification]
    pii_columns: List[str]
    phi_columns: List[str]
    pci_columns: List[str]
    financial_columns: List[str]
    row_count: int
    columns_analyzed: int
    high_risk_count: int
    recommendations: List[str]
    compliance_flags: List[str]

    def to_json(self) -> str
    def to_markdown(self) -> str
```

## Padrões Detectados

### PII (Dados Pessoais)

| Padrão | Descrição | Exemplo |
|--------|-----------|---------|
| cpf | CPF brasileiro | 123.456.789-00 |
| cnpj | CNPJ brasileiro | 12.345.678/0001-90 |
| email | Endereço de email | user@example.com |
| phone_br | Telefone brasileiro | (11) 98765-4321 |
| ssn | Social Security Number | 123-45-6789 |
| ip_address | Endereço IP | 192.168.1.1 |

### PHI (Dados de Saúde)

| Padrão | Descrição | Exemplo |
|--------|-----------|---------|
| cid10 | Código CID-10 | J45.0 |
| cns | Cartão Nacional de Saúde | 123456789012345 |
| crm | Registro CRM | CRM-12345-SP |
| blood_type | Tipo sanguíneo | A+, O- |

### Financial (Dados Financeiros)

| Padrão | Descrição | Exemplo |
|--------|-----------|---------|
| credit_card | Cartão de crédito | 4111-1111-1111-1111 |
| iban | Código IBAN | BR12 3456 7890 1234 5678 9012 3 |
| swift | Código SWIFT | BRASBRRJXXX |
