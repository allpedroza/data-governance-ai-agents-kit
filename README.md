# AI & Data Governance Agents Kit

**Escale governança de dados e IA com agentes inteligentes** — reduza riscos, acelere compliance e maximize o valor dos seus ativos de dados.

<details>
<summary><strong>🇺🇸 English Overview</strong></summary>

## What is this?

The **AI & Data Governance Agents Kit** is an open-source toolkit of **10 specialized AI agents** that automate end-to-end data and AI governance tasks — from discovery and lineage mapping to compliance enforcement and ROI measurement.

Built for data teams that need governance to be **continuous**, not a one-off project.

---

## The Problem

| Challenge | Business Impact |
|-----------|----------------|
| **Undocumented data** | Analysts spend 80% of their time searching for and understanding data |
| **Inconsistent quality** | Decisions made on bad data cost millions |
| **Manual compliance** | Slow audits and regulatory fines (GDPR / LGPD) |
| **Ungoverned AI** | Models in production with bias, no traceability |
| **Information silos** | Rework, conflicting "sources of truth", manual reconciliation |

---

## The Agents

### Data Governance

| Agent | What it does | Key Value |
|-------|-------------|-----------|
| **Taxonomy** | Discovers, scores and evolves the canonical data dictionary | Single source of truth that drives every downstream agent |
| **Data Engineering** | Generates DDL, dbt models and CLI commands aligned to the taxonomy | New assets born already compliant — no drift |
| **Data Lineage** | Maps dependencies across datasets and pipelines | Assess change impact before breaking production |
| **Data Discovery** | Semantic search with hybrid RAG (optional linear-adapter boost) | Find data in seconds, not days |
| **Metadata Enrichment** | Auto-generates descriptions, tags, and glossary entries | Always-documented catalog with zero manual effort |
| **Data Classification** | Classifies sensitivity (PII/PHI/PCI) | Automated compliance and smart masking |
| **Data Quality** | Monitors quality with SLAs and alerts | Reliable data for critical decisions |
| **Data Asset Value** | Quantifies value by usage and dependencies | Prioritize investments based on real data |
| **Data Product Scoring** | Unified maturity score | Consolidated governance view per data product |

### AI Governance

| Agent | What it does | Key Value |
|-------|-------------|-----------|
| **Sensitive Data NER** | Detects and anonymizes sensitive data in text | Protect data in LLM prompts and responses |
| **AI Business Value** | Calculates ROI of AI initiatives | Justify investments with clear metrics |
| **AI Policy Engine** | Enforces policies as code with deployment gates | Automated governance in CI/CD |

---

## Quick Start

### Prerequisites
- Python 3.10+
- An OpenAI API key (required for LLM-powered agents)
- `uv` (recommended) or `pip`

### 1. Clone and set up

```bash
git clone https://github.com/allpedroza/data-governance-ai-agents-kit.git
cd data-governance-ai-agents-kit

# Option A: uv (recommended — faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate

# Option B: standard pip
python -m venv .venv && source .venv/bin/activate
```

### 2. Install dependencies

```bash
# With uv
uv pip install -r requirements.txt streamlit

# With pip
pip install -r requirements.txt streamlit
```

### 3. Configure and run

```bash
export OPENAI_API_KEY="your-key-here"

# Launch the unified UI
streamlit run app.py
```

Open `http://localhost:8501` and start exploring the agents.

---

## Usage Examples

### Map pipeline lineage

```python
from data_governance.lineage.data_lineage_agent import DataLineageAgent

agent = DataLineageAgent()
result = agent.analyze_pipeline(["etl/transform.sql", "etl/job.py"])

print(f"Assets found: {result['metrics']['total_assets']}")
print(f"Transformations: {result['metrics']['total_transformations']}")
```

### Discover data with natural language

```python
from data_governance.rag_discovery.data_discovery_rag_agent import DataDiscoveryRAGAgent

rag = DataDiscoveryRAGAgent()
answer = rag.discover("Which tables contain customer data?")
print(answer.answer)
```

### Anonymize sensitive data

```python
from ai_governance.sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()
result = agent.analyze("SSN 123-45-6789 belongs to customer John Smith")

print(result.anonymized_text)
# Output: "SSN [SSN_REDACTED] belongs to customer [PERSON_REDACTED]"
```

### Calculate AI initiative ROI

```python
from ai_governance.ai_business_value import AIBusinessValueAgent, CostBreakdown, BenefitProjection

agent = AIBusinessValueAgent(currency="USD", projection_years=3)
report = agent.analyze_initiatives(
    initiatives=[{"id": "chatbot", "name": "Support Chatbot", "type": "customer_experience"}],
    cost_data={"chatbot": CostBreakdown(initiative_id="chatbot", development_internal=150000)},
    benefit_projections={"chatbot": BenefitProjection(initiative_id="chatbot", revenue_increase=250000)}
)

print(f"Projected ROI: {report.initiatives[0].roi_percentage:.1f}%")
```

---

## Integrations

**Data Catalogs:** OpenMetadata · Apache Atlas · AWS Glue

**Data Warehouses:** Snowflake · Amazon Redshift · Google BigQuery · Azure Synapse

Connectors available in `data_governance/warehouse/`.

---

## Repository Structure

```
data-governance-ai-agents-kit/
│
├── app.py                              # Unified Streamlit UI
│
├── data_governance/
│   ├── taxonomy/                       # Taxonomy agent (scorer, discovery, governance gate)
│   ├── data_engineering_agent/         # DDL / dbt / CLI generator + copilot
│   ├── lineage/                        # Data Lineage Agent
│   ├── rag_discovery/                  # Data Discovery Agent (incl. linear-adapter wrapper)
│   ├── metadata_enrichment/            # Metadata Enrichment Agent
│   ├── data_classification/            # Data Classification Agent
│   ├── data_quality/                   # Data Quality Agent
│   ├── data_asset_value/               # Data Asset Value Agent
│   ├── data_product_scoring/           # Data Product Scoring Layer
│   └── warehouse/                      # DW connectors
│
├── ai_governance/
│   ├── sensitive_data_ner/             # Sensitive Data NER + Vault
│   ├── ai_business_value/              # AI Business Value Agent
│   └── policy_engine/                  # AI Policy Engine (Policy-as-Code)
│
├── examples/                           # Examples and notebooks
└── requirements.txt                    # Dependencies
```

---

## Contributing

Contributions are welcome! Follow this flow:

1. **Fork** this repository
2. **Clone** your fork: `git clone https://github.com/<your-username>/data-governance-ai-agents-kit.git`
3. **Create a branch**: `git checkout -b feature/my-feature`
4. **Make your changes** and commit: `git commit -m "feat: add new feature"`
5. **Push** to your fork: `git push origin feature/my-feature`
6. **Open a Pull Request**

---

## Third-Party Integrations & Acknowledgments

This project optionally integrates with two Apache 2.0 libraries published by
the [SantanderAI](https://github.com/SantanderAI) open-source program. Both
are **optional**: install only if you want the corresponding feature. We
follow open-source best practice — explicit attribution, upstream link,
license disclosure, and no vendoring of the upstream code.

### 1. `mech-gov-framework` — LLM governance gate

Wraps the taxonomy discovery LLM calls (synthesis + evaluation) with the
R1/R2/R3 mechanical-governance regimes (entropy commit-reveal, ambiguity
gate, candidate freezing) and exposes the official metric suite
(CDL, DIU, IPI, FVS, ESD, FSR) in the run result.

- **Upstream:** https://github.com/SantanderAI/mech-gov-framework
- **License:** Apache License 2.0
- **Install:** `pip install mech-gov-framework`
- **Where it plugs in:** `data_governance/taxonomy/discovery/governance.py`
  (`MechGovBackend`). A pure-Python `LocalGovernanceBackend` is provided as
  a fallback so the framework works even when the upstream library is
  absent.

```python
# Imports used to wire mech-gov into our pipeline
from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.r3_adaptive import R3Adaptive
from mech_gov.llm.registry import create_llm
from mech_gov.metrics.governance import compute_governance_metrics

# Activation in our discovery pipeline
from data_governance.taxonomy.discovery import (
    TaxonomyDiscoveryPipeline, GovernanceConfig, MechGovBackend,
)

pipeline = TaxonomyDiscoveryPipeline(
    llm=my_llm,
    governance=GovernanceConfig(regime="r2", max_retries=2),
    governance_backend=MechGovBackend(),   # uses mech_gov_framework
)
```

### 2. `linear-adapter-trainer` — RAG embedding adapter

Lifts retrieval precision in the Data Discovery RAG agent by applying a
small learned linear transformation on top of any base embedder
(Sentence-Transformer, OpenAI, …) — without retraining the embedding
model.

- **Upstream:** https://github.com/SantanderAI/linear-adapter-trainer
- **License:** Apache License 2.0
- **Install:** `pip install "linear-adapter-trainer[sentence-transformers]"`
- **Where it plugs in:**
  `data_governance/rag_discovery/providers/embeddings/linear_adapter.py`
  (`LinearAdapterEmbeddings`). A pure-numpy adapter loader is built in so
  trained `.npz` matrices work even without the upstream torch dependency.

```python
# Training (one-off, against your knowledge base)
from linear_adapter_trainer import (
    AdapterTrainer, DatasetConfig, DatasetGenerator, KnowledgeBase,
    TemplateQueryGenerator, TrainingConfig,
)
from linear_adapter_trainer.embeddings import SentenceTransformerEmbedder

# Inference (wraps any of our existing embedders)
from data_governance.rag_discovery.providers.embeddings import (
    SentenceTransformerEmbeddings, LinearAdapterEmbeddings,
)

embedder = LinearAdapterEmbeddings(
    base=SentenceTransformerEmbeddings("all-MiniLM-L6-v2"),
    adapter_path="adapter.pt",   # or "adapter.npz" for the numpy fallback
)
# Drop straight into ChromaStore / DataDiscoveryRAGAgent — same EmbeddingProvider contract.
```

### Attribution notice

Both libraries are © Santander AI Lab and distributed under Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>). When you install them, the
upstream `LICENSE` and `NOTICE` files travel with the package. We do not
redistribute or modify either source tree — our integration points consume
the published public APIs only.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

</details>

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

O **AI & Data Governance Agents Kit** oferece **10 agentes de IA especializados** que automatizam tarefas de governança de ponta a ponta:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GOVERNANÇA AUTOMATIZADA                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. DESCOBRIR           2. MEDIR              3. PROTEGER              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│   │ • Discovery  │      │ • Quality    │      │ • Policy     │          │
│   │ • Lineage    │  →   │ • Asset Value│  →   │   Engine     │          │
│   │ • Metadata   │      │ • Business   │      │ • NER        │          │
│   │ • Classify   │      │   Value      │      │ • Gates      │          │
│   └──────────────┘      └──────────────┘      └──────────────┘          │
│                                                                          │
│   Encontre e entenda    Quantifique valor    Aplique políticas          │
│   seus dados            e riscos             automaticamente            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
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

## Os Agentes

### Governança de Dados

| Agente | O que faz | Valor principal |
|--------|-----------|-----------------|
| **Taxonomy** | Descobre, pontua e evolui o dicionário canônico de dados | Fonte da verdade que alimenta todos os demais agentes |
| **Data Engineering** | Gera DDL, modelos dbt e comandos CLI alinhados à taxonomia | Novos ativos já nascem compliant — sem drift |
| **Data Lineage** | Mapeia dependências entre datasets e pipelines | Avalie impacto de mudanças antes de quebrar produção |
| **Data Discovery** | Busca semântica com RAG híbrido (boost opcional via linear adapter) | Encontre dados em segundos, não em dias |
| **Metadata Enrichment** | Gera descrições, tags e glossário automaticamente | Catálogo sempre documentado sem esforço manual |
| **Data Classification** | Classifica sensibilidade (PII/PHI/PCI) | Compliance automático e masking inteligente |
| **Data Quality** | Monitora qualidade com SLAs e alertas | Dados confiáveis para decisões críticas |
| **Data Asset Value** | Quantifica valor por uso e dependências | Priorize investimentos com base em dados |
| **Data Product Scoring** | Score unificado de maturidade | Visão consolidada de governança por produto |

### Governança de IA

| Agente | O que faz | Valor principal |
|--------|-----------|-----------------|
| **Sensitive Data NER** | Detecta e anonimiza dados sensíveis em texto | Proteja dados em prompts e respostas de LLMs |
| **AI Business Value** | Calcula ROI de iniciativas de IA | Justifique investimentos com métricas claras |
| **AI Policy Engine** | Aplica políticas como código com gates | Governança automatizada no CI/CD |

---

## Interface Unificada

A aplicação Streamlit oferece **9 módulos** em uma única interface:

| Módulo | Funcionalidade |
|--------|---------------|
| **Lineage** | Visualize grafos de dependência e impacto |
| **Discovery** | Busque dados com linguagem natural |
| **Enrichment** | Enriqueça metadados automaticamente |
| **Classification** | Classifique sensibilidade de datasets |
| **Quality** | Monitore métricas e alertas de qualidade |
| **Asset Value** | Analise valor e criticidade de ativos |
| **NER Module** | Detecte e anonimize texto sensível |
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
│   └── warehouse/                      # Conectores para DWs
│
├── ai_governance/
│   ├── sensitive_data_ner/             # Sensitive Data NER + Vault
│   ├── ai_business_value/              # AI Business Value Agent
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
| Sensitive Data NER | [`ai_governance/sensitive_data_ner/README.md`](ai_governance/sensitive_data_ner/README.md) |
| AI Business Value | [`ai_governance/ai_business_value/README.md`](ai_governance/ai_business_value/README.md) |
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

## Integrações de Terceiros e Atribuições

Este projeto integra-se opcionalmente a duas bibliotecas Apache 2.0 publicadas
pelo programa open source [SantanderAI](https://github.com/SantanderAI).
Ambas são **opcionais**: instale apenas se quiser a feature correspondente.
Seguimos as melhores práticas open source — atribuição explícita, link
upstream, divulgação de licença e zero vendoring do código.

### 1. `mech-gov-framework` — gate de governança LLM

Envolve as chamadas LLM do pipeline de descoberta da taxonomia (sintetização +
avaliação) com os regimes mecânicos R1/R2/R3 (entropy commit-reveal,
ambiguity gate, candidate freezing) e expõe o suite oficial de métricas
(CDL, DIU, IPI, FVS, ESD, FSR) no resultado da execução.

- **Upstream:** https://github.com/SantanderAI/mech-gov-framework
- **Licença:** Apache License 2.0
- **Instalação:** `pip install mech-gov-framework`
- **Onde plugar:** `data_governance/taxonomy/discovery/governance.py`
  (`MechGovBackend`). O `LocalGovernanceBackend` em puro Python serve como
  fallback quando a lib upstream não estiver instalada.

```python
# Imports usados para conectar mech-gov ao pipeline
from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.r3_adaptive import R3Adaptive
from mech_gov.llm.registry import create_llm
from mech_gov.metrics.governance import compute_governance_metrics

# Ativação no nosso pipeline de descoberta
from data_governance.taxonomy.discovery import (
    TaxonomyDiscoveryPipeline, GovernanceConfig, MechGovBackend,
)

pipeline = TaxonomyDiscoveryPipeline(
    llm=my_llm,
    governance=GovernanceConfig(regime="r2", max_retries=2),
    governance_backend=MechGovBackend(),
)
```

### 2. `linear-adapter-trainer` — adapter de embeddings para RAG

Eleva a precisão do Data Discovery RAG aplicando uma pequena transformação
linear treinada sobre qualquer embedder base (Sentence-Transformer,
OpenAI, …) — sem retreinar o modelo de embedding.

- **Upstream:** https://github.com/SantanderAI/linear-adapter-trainer
- **Licença:** Apache License 2.0
- **Instalação:** `pip install "linear-adapter-trainer[sentence-transformers]"`
- **Onde plugar:**
  `data_governance/rag_discovery/providers/embeddings/linear_adapter.py`
  (`LinearAdapterEmbeddings`). Há um loader numpy embutido para que
  matrizes treinadas em `.npz` funcionem sem a dependência torch upstream.

```python
# Treinamento (one-off, contra sua knowledge base)
from linear_adapter_trainer import (
    AdapterTrainer, DatasetConfig, DatasetGenerator, KnowledgeBase,
    TemplateQueryGenerator, TrainingConfig,
)
from linear_adapter_trainer.embeddings import SentenceTransformerEmbedder

# Inferência (envolvendo qualquer dos nossos embedders)
from data_governance.rag_discovery.providers.embeddings import (
    SentenceTransformerEmbeddings, LinearAdapterEmbeddings,
)

embedder = LinearAdapterEmbeddings(
    base=SentenceTransformerEmbeddings("all-MiniLM-L6-v2"),
    adapter_path="adapter.pt",   # ou "adapter.npz" para o fallback numpy
)
# Plug direto no ChromaStore / DataDiscoveryRAGAgent — mesmo contrato EmbeddingProvider.
```

### Aviso de atribuição

Ambas as bibliotecas são © Santander AI Lab e distribuídas sob Apache
License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>). Ao instalá-las,
os arquivos `LICENSE` e `NOTICE` upstream acompanham o pacote. Não
redistribuímos nem modificamos as árvores de código upstream — nossa
integração consome apenas as APIs públicas publicadas.

---

## Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<p align="center">
  <strong>Governança de Dados e IA não precisa ser manual.</strong><br>
  Automatize com agentes inteligentes.
</p>
