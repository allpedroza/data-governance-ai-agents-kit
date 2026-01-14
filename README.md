# AI & Data Governance Agents Kit

**Automatize governança de dados e IA com agentes inteligentes** — reduza riscos, acelere compliance e maximize o valor dos seus ativos de dados.

---

## O Problema

Organizações enfrentam desafios crescentes para governar dados e sistemas de IA:

| Desafio | Impacto no Negócio |
|---------|-------------------|
| **Dados sem documentação** | Analistas gastam 80% do tempo procurando e entendendo dados |
| **Qualidade inconsistente** | Decisões baseadas em dados errados custam milhões |
| **Compliance manual** | Auditorias demoradas, multas por não-conformidade (LGPD/GDPR) |
| **IA sem governança** | Modelos em produção com viés, sem rastreabilidade |
| **Silos de informação** | Retrabalho, múltiplas "verdades", reconciliação manual |

---

## A Solução

O **AI & Data Governance Agents Kit** oferece **12 agentes de IA especializados** que automatizam tarefas de governança de ponta a ponta:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          GOVERNANÇA AUTOMATIZADA                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   1. DESCOBRIR           2. MEDIR              3. PROTEGER     4. RASTREAR    │
│   ┌──────────────┐      ┌──────────────┐      ┌────────────┐  ┌────────────┐  │
│   │ • Discovery  │      │ • Quality    │      │ • Policy   │  │ • Model    │  │
│   │ • Lineage    │  →   │ • Asset Value│  →   │   Engine   │  │   Registry │  │
│   │ • Metadata   │      │ • Business   │      │ • NER      │  │ • Contracts│  │
│   │ • Classify   │      │   Value      │      │ • Gates    │  │ • Versions │  │
│   └──────────────┘      └──────────────┘      └────────────┘  └────────────┘  │
│                                                                                │
│   Encontre e entenda    Quantifique valor    Aplique políticas  Versione e    │
│   seus dados            e riscos             automaticamente    audite modelos │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Valor Entregue

### Para o Negócio
- **Redução de 70%+ no tempo** de descoberta e documentação de dados
- **Compliance automatizado** com LGPD, GDPR, PCI-DSS
- **ROI mensurável** em iniciativas de IA com métricas claras

### Para Times Técnicos
- **Linhagem automática** de pipelines SQL/Python/Terraform
- **Classificação de sensibilidade** (PII/PHI/PCI) em segundos
- **Gates de governança** integrados ao CI/CD

### Para Governança
- **Catálogo vivo** com metadados sempre atualizados
- **Score de maturidade** por data product
- **Auditoria completa** com evidências rastreáveis

---

## Início Rápido

### 1. Clone e configure

```bash
# Clone o repositório
git clone https://github.com/allpedroza/data-governance-ai-agents-kit.git
cd data-governance-ai-agents-kit

# Crie ambiente virtual (escolha uv ou pip)
# Opção A: uv (recomendado - mais rápido)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate

# Opção B: pip tradicional
python -m venv .venv && source .venv/bin/activate
```

### 2. Instale as dependências

```bash
# Com uv
uv pip install -r requirements.txt streamlit

# Com pip
pip install -r requirements.txt streamlit
```

### 3. Configure e execute

```bash
# Configure sua API key (necessária para agentes com LLM)
export OPENAI_API_KEY="sua-chave"

# Inicie a interface unificada
streamlit run app.py
```

Acesse `http://localhost:8501` e comece a explorar os agentes.

---

## Os 12 Agentes

### Governança de Dados

| Agente | O que faz | Valor principal |
|--------|-----------|-----------------|
| **Data Lineage** | Mapeia dependências entre datasets e pipelines | Avalie impacto de mudanças antes de quebrar produção |
| **Data Discovery** | Busca semântica com RAG híbrido | Encontre dados em segundos, não em dias |
| **Metadata Enrichment** | Gera descrições, tags e glossário automaticamente | Catálogo sempre documentado sem esforço manual |
| **Data Classification** | Classifica sensibilidade (PII/PHI/PCI) | Compliance automático e masking inteligente |
| **Data Quality** | Monitora qualidade com SLAs e alertas | Dados confiáveis para decisões críticas |
| **Data Asset Value** | Quantifica valor por uso e dependências | Priorize investimentos com base em dados |
| **Data Product Scoring** | Score unificado de maturidade | Visão consolidada de governança por produto |
| **Data Contracts** | Define e valida contratos de dados (schema, SLA, qualidade) | Garanta acordos claros entre produtores e consumidores |

### Governança de IA

| Agente | O que faz | Valor principal |
|--------|-----------|-----------------|
| **Sensitive Data NER** | Detecta e anonimiza dados sensíveis em texto | Proteja dados em prompts e respostas de LLMs |
| **AI Business Value** | Calcula ROI de iniciativas de IA | Justifique investimentos com métricas claras |
| **AI Policy Engine** | Aplica políticas como código com gates | Governança automatizada no CI/CD |
| **Model Registry** | Versiona, cataloga e audita modelos de ML/IA | Rastreabilidade completa do ciclo de vida de modelos |

---

## Interface Unificada

A aplicação Streamlit oferece **11 módulos** em uma única interface:

| Módulo | Funcionalidade |
|--------|---------------|
| **Lineage** | Visualize grafos de dependência e impacto |
| **Discovery** | Busque dados com linguagem natural |
| **Enrichment** | Enriqueça metadados automaticamente |
| **Classification** | Classifique sensibilidade de datasets |
| **Quality** | Monitore métricas e alertas de qualidade |
| **Asset Value** | Analise valor e criticidade de ativos |
| **Contracts** | Defina e valide contratos de dados |
| **NER Module** | Detecte e anonimize texto sensível |
| **Model Registry** | Gerencie versões e metadados de modelos |
| **Vault** | Gerencie retenção de dados sensíveis |
| **Settings** | Configure LLMs, catálogos e warehouses |

**Apps standalone** também disponíveis:
```bash
streamlit run data_governance/lineage/app.py
streamlit run data_governance/data_quality/streamlit_app.py
streamlit run ai_governance/ai_business_value/streamlit_app.py
```

---

## Exemplos de Uso

### Mapear linhagem de pipelines

```python
from data_governance.lineage.data_lineage_agent import DataLineageAgent

agent = DataLineageAgent()
result = agent.analyze_pipeline(["etl/transform.sql", "etl/job.py"])

print(f"Assets encontrados: {result['metrics']['total_assets']}")
print(f"Transformações: {result['metrics']['total_transformations']}")
```

### Descobrir dados com linguagem natural

```python
from data_governance.rag_discovery.data_discovery_rag_agent import DataDiscoveryRAGAgent

rag = DataDiscoveryRAGAgent()
answer = rag.discover("Quais tabelas contêm dados de clientes?")
print(answer.answer)
```

### Anonimizar dados sensíveis

```python
from ai_governance.sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()
result = agent.analyze("O CPF 123.456.789-09 pertence ao cliente João Silva")

print(result.anonymized_text)
# Output: "O CPF [CPF_REDACTED] pertence ao cliente [PERSON_REDACTED]"
```

### Calcular ROI de iniciativa de IA

```python
from ai_governance.ai_business_value import AIBusinessValueAgent, CostBreakdown, BenefitProjection

agent = AIBusinessValueAgent(currency="BRL", projection_years=3)
report = agent.analyze_initiatives(
    initiatives=[{"id": "chatbot", "name": "Chatbot Atendimento", "type": "customer_experience"}],
    cost_data={"chatbot": CostBreakdown(initiative_id="chatbot", development_internal=150000)},
    benefit_projections={"chatbot": BenefitProjection(initiative_id="chatbot", revenue_increase=250000)}
)

print(f"ROI projetado: {report.initiatives[0].roi_percentage:.1f}%")
```

### Definir e validar contratos de dados

```python
from data_governance.data_contracts import DataContractsAgent, ContractDefinition

agent = DataContractsAgent()
contract = ContractDefinition(
    name="customers_gold",
    owner="data.team@empresa.com",
    schema=[
        {"name": "customer_id", "type": "string", "nullable": False},
        {"name": "email", "type": "string", "nullable": False, "pii": True},
        {"name": "segment", "type": "string", "nullable": True},
    ],
    sla={"freshness_hours": 6, "availability": 99.5},
    quality_rules=[
        {"rule": "not_null", "columns": ["customer_id", "email"]},
        {"rule": "unique", "columns": ["customer_id"]},
    ],
)

validation = agent.validate_contract(contract, source="warehouse://gold.customers")
print(f"Contrato válido: {validation.is_valid}")
print(f"Violações: {validation.violations}")
```

### Registrar e versionar modelos

```python
from ai_governance.model_registry import ModelRegistryAgent, ModelMetadata

agent = ModelRegistryAgent()
model = ModelMetadata(
    name="churn_predictor",
    version="1.2.0",
    framework="scikit-learn",
    metrics={"auc": 0.87, "precision": 0.82, "recall": 0.79},
    training_data="warehouse://gold.customer_features",
    owner="ml.team@empresa.com",
    tags=["production", "churn", "classification"],
)

registration = agent.register_model(model, artifact_path="models/churn_v1.2.pkl")
print(f"Modelo registrado: {registration.model_uri}")
print(f"Linhagem: {registration.lineage}")

# Listar versões de um modelo
versions = agent.list_versions("churn_predictor")
for v in versions:
    print(f"  {v.version}: AUC={v.metrics['auc']:.2f} ({v.status})")
```

### Aplicar políticas de governança

O **AI Policy Engine** oferece um pack inicial de políticas em `ai_governance/policy_engine/policy_packs/ai-governance-core.yaml`:

- **G1 Risk**: Bloqueia deploy se risco não aprovado
- **G2 Validation**: Exige métricas mínimas (AUC/robustez)
- **G4 Compliance**: Valida checklist LGPD e PII autorizada
- **Runtime Guardrail**: Impede envio de PII a provedores externos

---

## Integrações

### Catálogos de Dados
- OpenMetadata
- Apache Atlas
- AWS Glue

### Data Warehouses
- Snowflake
- Amazon Redshift
- Google BigQuery
- Azure Synapse

Conectores disponíveis em `data_governance/warehouse/`.

---

## Estrutura do Repositório

```
data-governance-ai-agents-kit/
│
├── app.py                              # Interface Streamlit unificada
│
├── data_governance/
│   ├── lineage/                        # Data Lineage Agent
│   ├── rag_discovery/                  # Data Discovery RAG Agent
│   ├── metadata_enrichment/            # Metadata Enrichment Agent
│   ├── data_classification/            # Data Classification Agent
│   ├── data_quality/                   # Data Quality Agent
│   ├── data_asset_value/               # Data Asset Value Agent
│   ├── data_product_scoring/           # Data Product Scoring Layer
│   ├── data_contracts/                 # Data Contracts Agent
│   └── warehouse/                      # Conectores para DWs
│
├── ai_governance/
│   ├── sensitive_data_ner/             # Sensitive Data NER + Vault
│   ├── ai_business_value/              # AI Business Value Agent
│   ├── model_registry/                 # Model Registry Agent
│   └── policy_engine/                  # AI Policy Engine (Policy-as-Code)
│
├── examples/                           # Exemplos e notebooks
└── requirements.txt                    # Dependências
```

---

## Conceitos Fundamentais

<details>
<summary><strong>O que é Governança de Dados?</strong></summary>

**Governança de Dados** é o sistema de decisões, papéis, políticas e controles que organiza "quem decide o quê, com base em quais regras", garantindo **qualidade e reusabilidade** dos dados.

**Problemas que resolve:**
- Papéis confusos (dono/curador/consumidor)
- Múltiplas versões da "verdade"
- Degradação de qualidade (acurácia, completude)
- Cópias e redundâncias em vez de reuso

**Valor gerado:**
- Qualidade mensurável → menor custo de retrabalho
- Reusabilidade → menos reconciliação
- Time-to-Insight → decisões mais rápidas

</details>

<details>
<summary><strong>O que é Governança de IA?</strong></summary>

**Governança de IA** é o conjunto de políticas, papéis, processos e métricas que orienta o ciclo de vida de sistemas de IA para que sejam **confiáveis, seguros, transparentes e justos**.

Framework baseado em **GOVERN–MAP–MEASURE–MANAGE**.

**Problemas que resolve:**
- Papéis e accountability difusos em times humanos-IA
- Ausência de gates para validação e gestão de incidentes
- Vieses amplificados sem medidas definidas
- Falta de métricas de risco e resiliência

**Valor gerado:**
- Confiança e adoção segura de produtos de IA
- Playbook que reduz retrabalho e incidentes
- Redução de risco regulatório e reputacional

</details>

<details>
<summary><strong>Por que usar agentes de IA para governança?</strong></summary>

Agentes de IA automatizam tarefas repetitivas e intensivas em conhecimento:

| Tarefa Manual | Com Agentes |
|---------------|-------------|
| Documentar 100 tabelas: 2 semanas | 2 horas |
| Classificar sensibilidade: análise por amostragem | 100% dos dados |
| Mapear linhagem: diagrams manuais | Grafo automático |
| Validar compliance: checklists manuais | Gates automatizados |

O resultado é **governança contínua** em vez de **projetos pontuais**.

</details>

---

## Documentação Detalhada

Cada agente possui documentação específica:

| Agente | Documentação |
|--------|-------------|
| Data Lineage | [`data_governance/lineage/README.md`](data_governance/lineage/README.md) |
| Data Discovery | [`data_governance/rag_discovery/README.md`](data_governance/rag_discovery/README.md) |
| Metadata Enrichment | [`data_governance/metadata_enrichment/README.md`](data_governance/metadata_enrichment/README.md) |
| Data Classification | [`data_governance/data_classification/README.md`](data_governance/data_classification/README.md) |
| Data Quality | [`data_governance/data_quality/README.md`](data_governance/data_quality/README.md) |
| Data Asset Value | [`data_governance/data_asset_value/README.md`](data_governance/data_asset_value/README.md) |
| Data Contracts | [`data_governance/data_contracts/README.md`](data_governance/data_contracts/README.md) |
| Sensitive Data NER | [`ai_governance/sensitive_data_ner/README.md`](ai_governance/sensitive_data_ner/README.md) |
| AI Business Value | [`ai_governance/ai_business_value/README.md`](ai_governance/ai_business_value/README.md) |
| Model Registry | [`ai_governance/model_registry/README.md`](ai_governance/model_registry/README.md) |
| AI Policy Engine | [`ai_governance/policy_engine/README.md`](ai_governance/policy_engine/README.md) |

---

## Contribuindo

Contribuições são bem-vindas! Siga o fluxo:

1. **Fork** este repositório
2. **Clone** seu fork: `git clone https://github.com/<seu-usuario>/data-governance-ai-agents-kit.git`
3. **Crie uma branch**: `git checkout -b feature/minha-feature`
4. **Faça suas alterações** e commit: `git commit -m "feat: adiciona nova funcionalidade"`
5. **Push** para seu fork: `git push origin feature/minha-feature`
6. **Abra um Pull Request**

---

## Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<p align="center">
  <strong>Governança de Dados e IA não precisa ser manual.</strong><br>
  Automatize com agentes inteligentes.
</p>
