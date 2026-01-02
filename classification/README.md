# 🛡️ Data Classification Agent

Agente de IA para **classificar automaticamente dados sensíveis (PII, PHI, financeiros)** usando **apenas schemas e metadados**. Ideal para cenários de **LGPD/GDPR** onde o acesso ao dado bruto não é permitido.

## 📋 Índice
- [Visão Geral](#visão-geral)
- [Características](#características)
- [Guia Rápido](#guia-rápido)
- [Arquitetura Lógica](#arquitetura-lógica)
- [Como Estender](#como-estender)

## 🎯 Visão Geral
O **Data Classification Agent** avalia nomes de colunas, tipos, descrições e tags para detectar **PII, PHI e dados financeiros**. As recomendações são alinhadas a **controles LGPD/GDPR** e o agente nunca lê os valores das tabelas.

## ✨ Características
- 🔒 **Classificação sem dados brutos**: funciona apenas com schemas, descrições e tags.
- 🩺 **Detecção de PII/PHI/Financeiro** com regras ponderadas por palavras-chave, tipos e tags.
- ✅ **Compliance LGPD/GDPR**: sugere ações como DPIA, minimização e mascaramento.
- 🧩 **Extensível**: adicione regras customizadas sem alterar o núcleo do agente.
- 🧠 **Níveis de sensibilidade**: LOW, MEDIUM, HIGH e CRITICAL para priorização.

## 🚀 Guia Rápido
```python
from classification import (
    ColumnMetadata,
    DataClassificationAgent,
    TableSchema,
)

# Define o schema de uma tabela (sem acessar os dados)
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

print(result.sensitivity_level)          # HIGH
print(result.detected_categories)        # ['PII']
for column in result.columns:
    print(column.column.name, column.categories, column.suggested_controls)
```

## 🧱 Arquitetura Lógica
1. **Entrada de metadados**: `TableSchema` e `ColumnMetadata` descrevem nome, tipo, descrição e tags.
2. **Regras de sensibilidade**: `SensitiveDataRule` avalia palavras-chave, tipos e tags para PII, PHI e financeiro.
3. **Scoring por coluna**: combina indícios (nome, descrição, tipo, tags) com pesos ajustados para metadados.
4. **Síntese por tabela**: consolida categorias detectadas, define o nível de sensibilidade e recomendações LGPD/GDPR.
5. **Saída estruturada**: `TableClassification` com colunas classificadas, ações sugeridas e rationale auditável.

## 🛠️ Como Estender
- **Novas regras**: passe uma lista de `SensitiveDataRule` customizada ao inicializar o agente.
- **Controles adicionais**: acrescente requisitos LGPD/GDPR via parâmetros `lgpd_requirements` e `gdpr_requirements`.
- **Pipelines existentes**: o agente é independente dos demais módulos, podendo ser usado junto ao Lineage e Discovery.
