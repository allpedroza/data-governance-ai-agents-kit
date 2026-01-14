# Data Governance AI Agents Kit

Framework de **agentes de IA para governança de dados** com interface unificada (Streamlit) e módulos especializados para linhagem, descoberta, enriquecimento de metadados, classificação, qualidade, valor de ativos, proteção de dados sensíveis, **valor de negócios** e **policy-as-code** para governança de IA.

## Visão Geral

Hoje o kit reúne **10 agentes principais** (7 em `data_governance` + 3 em `ai_governance`) e conectores para catálogos e data warehouses.

Se você está chegando agora, pense no kit como um caminho simples:
1. **Descobrir** ativos (descoberta, classificação, metadados, qualidade).
2. **Medir valor e risco** (valor de ativos, business value).
3. **Aplicar governança** de forma automatizada (policy engine + gates).

| Agente | Propósito | Pacote |
| --- | --- | --- |
| **Data Lineage Agent** | Mapear dependências e impacto de mudanças | `data_governance.lineage` |
| **Data Discovery RAG Agent** | Descoberta semântica com RAG (busca híbrida) | `data_governance.rag_discovery` |
| **Metadata Enrichment Agent** | Enriquecer metadados (descrições, tags, PII) | `data_governance.metadata_enrichment` |
| **Data Classification Agent** | Classificação de sensibilidade (PII/PHI/PCI/Financeiro) | `data_governance.data_classification` |
| **Data Quality Agent** | Métricas de qualidade, SLA e schema drift | `data_governance.data_quality` |
| **Data Asset Value Agent** | Valor de ativos (uso, joins, linhagem) | `data_governance.data_asset_value` |
| **Data Product Scoring Layer** | Score de data products (contrato, qualidade, governança, valor) | `data_governance.data_product_scoring` |
| **Sensitive Data NER Agent** | Detecção/anonimização de dados sensíveis + Vault | `ai_governance.sensitive_data_ner` |
| **AI Business Value Agent** | ROI e valor de negócios de iniciativas de IA | `ai_governance.ai_business_value` |
| **AI Policy Engine (Policy-as-Code)** | Stage-gates, evidências e enforcement automatizado | `ai_governance.policy_engine` |

Além disso, há integração com:
- **Catálogos**: OpenMetadata, Apache Atlas, AWS Glue.
- **Data Warehouses**: Snowflake, Redshift, BigQuery e Synapse (conectores em `data_governance.warehouse`).

---

## Início Rápido

### Com uv (Recomendado)

```bash
# Instale o uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Crie ambiente virtual e instale dependências
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependências principais + agentes
uv pip install \
  -r data_governance/lineage/requirements.txt \
  -r data_governance/rag_discovery/requirements.txt \
  -r data_governance/metadata_enrichment/requirements.txt \
  -r data_governance/data_classification/requirements.txt \
  -r data_governance/data_quality/requirements.txt \
  -r data_governance/data_asset_value/requirements.txt \
  -r data_governance/warehouse/requirements.txt \
  -r ai_governance/sensitive_data_ner/requirements.txt \
  -r ai_governance/ai_business_value/requirements.txt \
  streamlit

# Configure a API key (necessária para alguns agentes)
export OPENAI_API_KEY="sua-chave"

# Inicie a interface unificada
streamlit run app.py
```

### Com pip (Alternativo)

```bash
# Clone o repositório
git clone <repo-url>
cd data-governance-ai-agents-kit

# Ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependências principais + agentes
pip install \
  -r data_governance/lineage/requirements.txt \
  -r data_governance/rag_discovery/requirements.txt \
  -r data_governance/metadata_enrichment/requirements.txt \
  -r data_governance/data_classification/requirements.txt \
  -r data_governance/data_quality/requirements.txt \
  -r data_governance/data_asset_value/requirements.txt \
  -r data_governance/warehouse/requirements.txt \
  -r ai_governance/sensitive_data_ner/requirements.txt \
  -r ai_governance/ai_business_value/requirements.txt \
  streamlit

# Configure a API key (necessária para alguns agentes)
export OPENAI_API_KEY="sua-chave"

# Inicie a interface unificada
streamlit run app.py
```

---

## Por que Governança de Dados e Governança de IA?

**Governança de Dados** garante que dados sejam confiáveis, rastreáveis e bem descritos para suportar decisões e produtos de dados com segurança. É por isso que você vê agentes voltados a **classificação, qualidade, linhagem, descoberta e metadados**: eles reduzem risco operacional, aceleram acesso responsável e sustentam compliance.

**Governança de IA** foca no ciclo de vida do modelo (treino, deploy, uso em runtime) e nos riscos específicos de IA (ex.: uso indevido, compliance, performance em produção). Embora muitas vezes seja conduzida pelo mesmo time, **o objetivo é diferente**: dados validam a base; IA governa a tomada de decisão automatizada sobre modelos, com gates, evidências e políticas.

## Explicação Macro de Cada Agente

### 1) Data Lineage Agent
**Objetivo**: mapear dependências ponta a ponta para entender impacto e risco de mudança.
- **Entradas**: pipelines SQL/Python/Terraform, DAGs, jobs e manifests.
- **Saídas**: grafo de linhagem, dependências críticas, métricas de impacto.
- **Uso típico**: change impact, auditoria de origem e rastreabilidade.

### 2) Data Discovery RAG Agent
**Objetivo**: acelerar descoberta de dados com busca semântica confiável.
- **Entradas**: catálogos, descrições, amostras, documentos.
- **Saídas**: respostas com ranking híbrido (semântico + lexical) e contexto.
- **Uso típico**: localizar datasets, acelerar análise e onboarding.

### 3) Metadata Enrichment Agent
**Objetivo**: melhorar a qualidade e completude dos metadados.
- **Entradas**: schemas, amostras e catálogos.
- **Saídas**: descrições, tags, sugestões de PII, glossário.
- **Uso típico**: documentação automática e padronização.

### 4) Data Classification Agent
**Objetivo**: classificar sensibilidade e risco regulatório dos dados.
- **Entradas**: amostras, schemas e padrões.
- **Saídas**: nível de sensibilidade (PII/PHI/PCI etc.), flags de compliance.
- **Uso típico**: políticas de acesso, data masking e LGPD/GDPR.

### 5) Data Quality Agent
**Objetivo**: medir e monitorar qualidade com thresholds e SLAs.
- **Entradas**: datasets, regras de validação, SLAs.
- **Saídas**: scores, alertas de drift, métricas de completude/validade.
- **Uso típico**: gates de treino/deploy e confiança em decisões.

### 6) Data Asset Value Agent
**Objetivo**: quantificar valor e criticidade dos ativos de dados.
- **Entradas**: uso, joins, dependências, linhagem.
- **Saídas**: score de valor, impacto e custo de mudança.
- **Uso típico**: priorização de governança e investimentos.

### 7) Data Product Scoring Layer
**Objetivo**: consolidar governança em um score único por data product.
- **Entradas**: contrato, qualidade, SLAs, uso e valor.
- **Saídas**: score composto, gaps de governança.
- **Uso típico**: roadmap de melhorias e compliance contínuo.

### 8) Sensitive Data NER Agent
**Objetivo**: detectar e anonimizar dados sensíveis em texto livre.
- **Entradas**: textos, logs, prompts e respostas.
- **Saídas**: entidades detectadas, texto anonimizado, políticas de retenção (Vault).
- **Uso típico**: proteção em runtime e compliance com PII.

### 9) AI Business Value Agent
**Objetivo**: medir ROI e valor de negócio de iniciativas de IA.
- **Entradas**: custos, projeções de benefício, riscos.
- **Saídas**: score de valor, ROI, relatórios executivos.
- **Uso típico**: aprovação de investimento e priorização de projetos.

### 10) AI Policy Engine
**Objetivo**: aplicar governança de IA com gates e evidências auditáveis.
- **Entradas**: evidências dos agentes (risco, qualidade, compliance, NER).
- **Saídas**: decisões allow/deny/warn, logs auditáveis, exceções.
- **Uso típico**: pré-merge, pré-deploy e runtime enforcement.

**O que existe no pack inicial (alto nível):**
- **G1 Risk**: libera promoção para PROD apenas com risco aprovado e tier aceitável.
- **G2 Validation**: exige métricas mínimas de validação (AUC/robustez) antes do deploy.
- **G4 Compliance (LGPD)**: bloqueia quando checklist não está completo ou há PII não autorizada.
- **Runtime Guardrail**: impede envio de PII a provedores externos.

### 10) AI Policy Engine
Policy-as-code para gates (risk/validation/compliance), evidências e guardrails runtime.

---

## Interface Unificada (Streamlit)

A aplicação principal oferece **9 tabs** (o Policy Engine é um pacote de políticas/artefatos e não tem UI própria):

- **Lineage**: análise de linhagem e impacto.
- **Discovery**: busca semântica com RAG.
- **Enrichment**: enriquecimento automático de metadados.
- **Classification**: classificação de dados sensíveis.
- **Quality**: avaliação de qualidade e alertas.
- **Asset Value**: valor de ativos por uso/joins.
- **NER Module**: detecção e anonimização de texto sensível.
- **Vault**: armazenamento seguro e políticas de retenção.
- **Settings**: LLMs, catálogos e data warehouses.

Há também apps standalone:

```bash
streamlit run data_governance/lineage/app.py
streamlit run data_governance/metadata_enrichment/streamlit_app.py
streamlit run data_governance/data_classification/streamlit_app.py
streamlit run data_governance/data_quality/streamlit_app.py
streamlit run ai_governance/sensitive_data_ner/streamlit_app.py
streamlit run ai_governance/ai_business_value/streamlit_app.py
```

---

## Exemplos Rápidos

### Lineage
```python
from data_governance.lineage.data_lineage_agent import DataLineageAgent

agent = DataLineageAgent()
result = agent.analyze_pipeline(["etl/transform.sql", "etl/job.py"])
print(result["metrics"])  # assets, transformações, complexidade
```

### Discovery (RAG)
```python
from data_governance.rag_discovery.data_discovery_rag_agent import DataDiscoveryRAGAgent

rag = DataDiscoveryRAGAgent()
answer = rag.discover("Onde estão os dados de clientes?")
print(answer.answer)
```

### Sensitive Data NER
```python
from ai_governance.sensitive_data_ner import SensitiveDataNERAgent

agent = SensitiveDataNERAgent()
result = agent.analyze("CPF 123.456.789-09 do cliente João")
print(result.anonymized_text)
```

### AI Business Value
```python
from ai_governance.ai_business_value import AIBusinessValueAgent, CostBreakdown, BenefitProjection

agent = AIBusinessValueAgent(currency="BRL", projection_years=3)
report = agent.analyze_initiatives(
    initiatives=[{"id": "chatbot", "name": "Chatbot", "type": "customer_experience"}],
    cost_data={"chatbot": CostBreakdown(initiative_id="chatbot", development_internal=150000)},
    benefit_projections={"chatbot": BenefitProjection(initiative_id="chatbot", revenue_increase=250000)}
)
print(report.to_markdown())
```

### AI Policy Engine (pack inicial)
Consulte o pacote de políticas em `ai_governance/policy_engine/policy_packs/ai-governance-core.yaml`.

---

## Estrutura do Repositório

```
./
├── app.py                          # Streamlit unificado
├── ai_governance/
│   ├── ai_business_value/          # AI Business Value Agent
│   ├── sensitive_data_ner/          # Sensitive Data NER + Vault
│   └── policy_engine/               # AI Policy Engine (Policy-as-Code)
├── data_governance/
│   ├── lineage/                    # Data Lineage Agent
│   ├── rag_discovery/              # Discovery RAG Agent
│   ├── metadata_enrichment/        # Metadata Enrichment Agent
│   ├── data_classification/        # Data Classification Agent
│   ├── data_quality/               # Data Quality Agent
│   ├── data_asset_value/           # Data Asset Value Agent
│   └── warehouse/                  # Conectores para DWs
├── examples/                       # Exemplos gerais
└── requirements.txt                # Dependências básicas (apps)
```

---

## Documentação por Módulo

- Lineage: `data_governance/lineage/README.md`
- Discovery: `data_governance/rag_discovery/README.md`
- Enrichment: `data_governance/metadata_enrichment/README.md`
- Classification: `data_governance/data_classification/README.md`
- Quality: `data_governance/data_quality/README.md`
- Asset Value: `data_governance/data_asset_value/README.md`
- Sensitive Data NER: `ai_governance/sensitive_data_ner/README.md`
- AI Business Value: `ai_governance/ai_business_value/README.md`
- AI Policy Engine: `ai_governance/policy_engine/README.md`

---

## Contribuindo

Contribuições são muito bem-vindas! Siga o fluxo abaixo:

1. **Fork** do repositório
2. **Clone** do seu fork
3. **Sync** com `upstream/main`
4. **Crie uma branch** (ex.: `docs/update-readme`)
5. **Commit** com mensagem descritiva
6. **Push** para o seu fork
7. **Abra um Pull Request**

```bash
# Clone SEU fork
git clone https://github.com/<seu-usuario>/data-governance-ai-agents-kit.git
cd data-governance-ai-agents-kit

# Configure o upstream
git remote add upstream https://github.com/allpedroza/data-governance-ai-agents-kit.git

# Sincronize
git fetch upstream
git checkout main
git merge upstream/main

# Crie uma branch
git checkout -b docs/update-readme

# Commit
git add .
git commit -m "docs: atualiza documentação"

# Push
git push origin docs/update-readme
```

---

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
