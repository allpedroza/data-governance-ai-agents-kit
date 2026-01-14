# AI Policy Engine

Este módulo inicia o **AI Policy Engine** como um pacote de políticas, regras e contratos de integração para transformar governança de IA em enforcement automatizado. A implementação é **policy-as-code**, focada em evidências e stage-gates (pré-merge, pré-deploy, runtime).

## Objetivos do MVP

- **Gates mínimos**: G1 (Risk), G2 (Validation), G4 (Compliance).
- **Integrações obrigatórias**: Risk Assessment, Validation & Testing, Compliance & Regulatory, Registry e CI/CD.
- **Evidence Store**: eventos imutáveis com hash e timestamp.
- **Decision Log**: trilha de auditoria com racional e evidências anexas.

## Pacote de políticas (detalhado)

O pack inicial (`ai-governance-core.yaml`) cobre o essencial para bloquear risco, validação e compliance antes de promover modelos para produção:

### 1) G1 — Risk Gate (registry.transition → prod)
**Objetivo**: só permitir promoção para PROD se o risco foi avaliado e o tier não é alto.
- Evidências esperadas:
  - `risk.assessment.status == "approved"`
  - `risk.tier in ["low","medium"]`
- Decisão: `block` se faltar evidência.

### 2) G2 — Validation Gate (cicd.deploy → prod)
**Objetivo**: garantir métricas mínimas antes do deploy em PROD.
- Evidências esperadas:
  - `validation.AUC >= 0.85`
  - `validation.robustness_score >= 0.70`
- Decisão: `block` se faltar evidência.

### 3) G4 — Compliance Gate (train.start / registry.transition)
**Objetivo**: impedir uso de dados sensíveis sem checklist e autorização.
- Evidências esperadas:
  - `compliance.LGPD == "complete"`
  - `data.contains_pii == false`
- Exceções: permitidas com aprovação de DPO/CISO por até 30 dias.
- Decisão: `block` se faltar evidência.

### 4) Runtime Guardrail — PII com LLM externo
**Objetivo**: evitar exfiltração de PII para provedores externos.
- Condição: `event == "inference.request" && provider.type == "external_llm"`
- Ação: `deny` quando `input.contains_pii == true`.

## Estrutura

```
policy_engine/
├── policy_packs/
│   └── ai-governance-core.yaml      # DSL YAML (G1/G2/G4)
├── rego/
│   └── runtime_guardrails.rego      # Guardrails runtime (OPA)
└── README.md
```

## Integração com agentes do repositório

O Policy Engine consome evidências produzidas pelos agentes de governança já existentes:

- **Data Classification Agent**: `data.classification`, `data.contains_pii`.
- **Data Quality Agent**: métricas de qualidade/robustez (`validation.AUC`, `validation.robustness_score`).
- **Sensitive Data NER Agent**: `input.contains_pii`, `masking`.
- **AI Business Value Agent**: `business.roi_estimate`, `business.owner_approval` (V2).
- **Lineage/Metadata/Discovery**: flags de completude (V2).

As evidências podem ser publicadas por evento (push) e registradas no Evidence Store via `POST /evidence`.

## API (contratos mínimos)

**POST /evaluate**
```json
{
  "event": "cicd.deploy",
  "env": "prod",
  "artifact_id": "model:fraud-123",
  "context": {
    "risk": {"tier": "medium", "assessment": {"status": "approved"}},
    "validation": {"AUC": 0.9, "robustness_score": 0.8},
    "compliance": {"LGPD": "complete"},
    "data": {"contains_pii": false}
  }
}
```

**POST /evidence**
```json
{
  "artifact_id": "model:fraud-123",
  "type": "validation.AUC",
  "value": 0.9,
  "source": "validation-agent",
  "hash": "sha256:...",
  "ts": "2026-01-14T15:32:10Z"
}
```

## Próximos passos

1. **Implementar serviço de avaliação** (OPA + adaptador DSL YAML).
2. **Adicionar storage imutável** para evidências/decisões.
3. **Criar conectores** com eventos dos agentes (classificação, qualidade, NER).
4. **Adicionar CI/CD Gate** com chamada para `POST /evaluate`.

Veja o pack inicial em `policy_packs/ai-governance-core.yaml`.
