# Data Steward Agent / Agente Data Steward

## PT-BR

Copiloto operacional para data stewards. Absorve trabalho repetitivo, operacional e documental enquanto decisao, accountability e aprovacao final continuam humanas.

**O agente propoe. O steward aprova.**

### Capacidades

1. **Intake e Triagem de Issues** -- Recebe incidentes em texto livre ("o KPI mudou", "campo vazio", "qual definicao oficial?"), classifica, sugere severidade, identifica dominio/dataset/owner e propoe proximos passos.

2. **Curadoria de Glossario de Negocio** -- Consolida definicoes de multiplas fontes, detecta conflitos semanticos, propoe definicao padrao com owner/steward/sistema de referencia sugeridos.

3. **Draft de Regras de Quality** -- Sugere regras em linguagem de negocio + expressao tecnica candidata. Ex: "CPF deve ser unico por cliente ativo" + `COUNT(*) = COUNT(DISTINCT cpf)`.

4. **Explicador de Impacto** -- Quando ha mudanca em campo/regra/definicao, responde quais KPIs podem quebrar, quais times sao impactados, quais regras dependem do atributo e excecoes regulatorias.

5. **Workflow de Aprovacoes** -- Monta pacote de evidencias, propoe aprovadores, rastreia decisoes e gera changelog legivel.

### Uso Rapido

```python
from data_governance.data_steward import DataStewardAgent

agent = DataStewardAgent(persist_dir="./steward_data")

# Triar issue
issue = agent.triage_issue("O KPI de receita caiu 20% sem razao")

# Curar termo de glossario
term = agent.curate_term("receita_liquida", [
    {"source": "SQL", "definition": "Soma vendas menos impostos"},
    {"source": "Dashboard", "definition": "Receita apos descontos"},
], domain="finance")

# Sugerir regras de quality
rules = agent.draft_rules("dim_customers", "customer", columns=[
    {"name": "cpf", "type": "string", "nullable": False},
])

# Analisar impacto
impact = agent.explain_impact("Remover coluna cpf", "dim_customers", "cpf")

# Workflow de aprovacao
agent.submit_term_for_approval(term.term_id)
agent.submit_decision(request_id, "approved", "Ana Silva", "OK")
```

### UI Streamlit

```bash
streamlit run data_governance/data_steward/streamlit_app.py
```

---

## EN

Operational copilot for data stewards. Absorbs repetitive, operational and documental work while keeping decisions, accountability and final approval human.

**The agent proposes. The steward approves.**

### Capabilities

1. **Issue Intake & Triage** -- Receives free-text incidents, classifies them, suggests severity, identifies domain/dataset/owner and proposes next steps.

2. **Business Glossary Curation** -- Consolidates definitions from multiple sources, detects semantic conflicts, proposes standard definition with suggested owner/steward/reference system.

3. **Quality Rule Drafting** -- Suggests rules in business language + candidate technical expression. E.g.: "CPF must be unique per active customer" + `COUNT(*) = COUNT(DISTINCT cpf)`.

4. **Impact Explainer** -- When a field/rule/definition changes, answers which KPIs may break, which teams are impacted, which rules depend on the attribute and regulatory exceptions.

5. **Approval Workflow** -- Assembles evidence packages, routes to approvers, tracks decisions and generates readable changelogs.

### Integration

The agent consumes outputs from other agents (DataQualityAgent, DataClassificationAgent, MetadataEnrichmentAgent, DataLineageAgent, DataContractAgent) but does not duplicate their logic. LLM is optional -- all capabilities work in rule-based fallback mode.
