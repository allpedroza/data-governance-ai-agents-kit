# AI Model Registry Agent

Inventário vivo de modelos/LLMs com rastreamento de versões, stages, owners e propósito. **Objetivo principal: eliminar "Shadow AI" e servir como base para todos os controles de AI Governance.**

## Valor de Negócio

- **Fim do Shadow AI**: Visibilidade completa de todos os modelos em uso na organização
- **Base para Controles**: Fundação para todos os demais agentes de AI Governance
- **Compliance**: Suporte a EU AI Act, LGPD e outras regulamentações
- **Ownership Claro**: Accountability definida para cada modelo
- **Gestão de Risco**: Classificação de risco e análise de impacto

## Funcionalidades Principais

### Registro e Catalogação
- Registro de modelos com metadados completos
- Suporte a múltiplos tipos (LLM, Classifier, NER, RAG, etc.)
- Categorização por propósito, risco e sensibilidade de dados
- Tags e busca textual

### Gerenciamento de Versões
- Múltiplas versões por modelo
- Ciclo de vida: Development → Staging → Production → Archived
- Métricas de performance por versão
- Rastreamento de artefatos e dependências

### Detecção de Shadow AI
- Registro de modelos descobertos não documentados
- Workflow de legitimização
- Alertas e relatórios de Shadow AI

### Auditoria e Compliance
- Log completo de todas as mudanças
- Relatórios de compliance para auditoria regulatória
- Análise de gaps e recomendações

### Análise de Dependências
- Grafo de dependências entre modelos
- Análise de impacto de mudanças
- Identificação de modelos críticos

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-org/data-governance-ai-agents-kit.git
cd data-governance-ai-agents-kit

# O agente usa apenas a biblioteca padrão do Python
# Não há dependências externas obrigatórias
```

## Quick Start

```python
from ai_governance.ai_model_registry import (
    AIModelRegistryAgent,
    ModelType,
    ModelStage,
    RiskLevel,
    ModelOwner,
)

# Inicializar o registro
registry = AIModelRegistryAgent(
    storage_path="./model_registry.json",  # Persistência opcional
    enable_audit=True,
)

# Definir owner
owner = ModelOwner(
    name="Maria Silva",
    email="maria.silva@empresa.com",
    team="Data Science",
    department="Technology",
    role="owner",
)

# Registrar um modelo
model = registry.register_model(
    name="fraud-detection-v1",
    model_type=ModelType.CLASSIFIER,
    purpose="Detectar transações fraudulentas",
    description="Modelo XGBoost para classificação de fraude",
    owner=owner,
    risk_level=RiskLevel.HIGH,
    initial_version="1.0.0",
    tags=["fraud", "production"],
    registered_by="maria.silva",
)

print(f"Modelo registrado: {model.name} (ID: {model.model_id})")
```

## Exemplos de Uso

### Gerenciamento de Versões

```python
from ai_governance.ai_model_registry import ModelMetrics, ModelStage

# Adicionar nova versão com métricas
metrics = ModelMetrics(
    accuracy=0.95,
    precision=0.92,
    recall=0.89,
    f1_score=0.905,
    latency_p99_ms=45.8,
)

registry.add_version(
    model_id_or_name="fraud-detection-v1",
    version="1.1.0",
    created_by="maria.silva",
    stage=ModelStage.STAGING,
    metrics=metrics,
    changelog="- Melhorada precisão em 5%",
)

# Promover para produção
registry.promote_version(
    model_id_or_name="fraud-detection-v1",
    version="1.1.0",
    new_stage=ModelStage.PRODUCTION,
    promoted_by="tech.lead",
    reason="Aprovado após testes",
)
```

### Detecção de Shadow AI

```python
# Registrar Shadow AI descoberto
shadow = registry.register_shadow_ai(
    name="unknown-recommender",
    model_type=ModelType.RECOMMENDER,
    purpose="Sistema não documentado descoberto",
    discovery_source="network_scan",
    discovered_by="security.team",
)

# Listar todos os Shadow AIs
shadow_models = registry.get_shadow_ai_models()

# Legitimizar após investigação
registry.legitimize_shadow_ai(
    model_id_or_name=shadow.name,
    new_owner=new_owner,
    legitimized_by="governance.team",
)
```

### Busca e Filtros

```python
# Busca textual
results = registry.search(query="fraud")

# Listar por tipo
llms = registry.list_models(model_type=ModelType.LLM)

# Modelos de alto risco
high_risk = registry.get_high_risk_models()

# Modelos em produção
production = registry.get_production_models()
```

### Análise de Dependências

```python
# Adicionar dependência
registry.add_dependency(
    model_id_or_name="semantic-search",
    depends_on_id_or_name="embeddings-model",
    dependency_type="uses",
)

# Analisar impacto de mudanças
impact = registry.get_impact_analysis("embeddings-model")
print(f"Modelos afetados: {impact['total_affected']}")
```

### Relatórios e Compliance

```python
# Estatísticas gerais
stats = registry.get_statistics()
print(f"Total de modelos: {stats.total_models}")
print(f"Shadow AI: {stats.shadow_ai_count}")

# Relatório de compliance
compliance = registry.generate_compliance_report()
for rec in compliance['recommendations']:
    print(f"- {rec}")

# Relatório completo em Markdown
report = registry.generate_report(format="markdown")
```

### Auditoria

```python
# Consultar log de auditoria
audit = registry.get_audit_log(
    model_id_or_name="fraud-detection-v1",
    limit=50,
)

for entry in audit:
    print(f"[{entry.timestamp}] {entry.change_type.value}: {entry.reason}")
```

## Tipos de Modelo Suportados

| Tipo | Descrição |
|------|-----------|
| `LLM` | Large Language Models (GPT, Claude, Llama, etc.) |
| `CLASSIFIER` | Modelos de classificação |
| `REGRESSOR` | Modelos de regressão |
| `NER` | Named Entity Recognition |
| `EMBEDDING` | Modelos de embeddings |
| `RAG` | Retrieval-Augmented Generation |
| `RECOMMENDER` | Sistemas de recomendação |
| `COMPUTER_VISION` | Visão computacional |
| `TIME_SERIES` | Séries temporais |
| `AGENT` | Agentes autônomos |
| `FINE_TUNED` | Modelos fine-tuned |
| `CUSTOM` | Outros tipos |

## Níveis de Risco (EU AI Act)

| Nível | Descrição |
|-------|-----------|
| `UNACCEPTABLE` | Risco inaceitável - proibido |
| `HIGH` | Alto risco - requer conformidade rigorosa |
| `LIMITED` | Risco limitado - requer transparência |
| `MINIMAL` | Risco mínimo - sem requisitos específicos |
| `UNASSESSED` | Ainda não avaliado |

## Ciclo de Vida de Versões

```
DEVELOPMENT → STAGING → PRODUCTION → DEPRECATED → ARCHIVED/RETIRED
```

## Estrutura do Projeto

```
ai_model_registry/
├── __init__.py          # Exports públicos
├── agent.py             # Implementação principal
├── README.md            # Esta documentação
├── requirements.txt     # Dependências
└── examples/
    └── usage_example.py # Exemplos de uso
```

## Integração com Outros Agentes

O AI Model Registry serve como base para:

- **AI Business Value Agent**: Análise de valor de iniciativas de AI
- **Sensitive Data NER Agent**: Identificação de modelos que processam dados sensíveis
- **Risk Assessment**: Avaliação de risco de modelos
- **Compliance Monitoring**: Monitoramento contínuo de conformidade

## Roadmap

- [ ] Integração com MLflow/Weights & Biases
- [ ] API REST para integração externa
- [ ] Dashboard Streamlit
- [ ] Notificações (Slack, Email)
- [ ] Integração com CI/CD pipelines
- [ ] Suporte a storage distribuído (Redis, PostgreSQL)

## Contribuindo

Contribuições são bem-vindas! Por favor, siga as diretrizes do projeto.

## Licença

MIT License - veja LICENSE para detalhes.
