"""Base prompts for the two LLM calls in the taxonomy discovery pipeline.

The prompts are intentionally explicit about the YAML/JSON contract so the
downstream parsers do not need to do heuristic extraction. Both prompts:

1. Require deterministic output (``temperature=0``).
2. Restrict the LLM to taxonomy concerns — it must not fabricate row data.
3. Treat all incoming sample values as already anonymized; redaction tokens
   such as ``<EMAIL_REDACTED>`` are signals, not literal column values.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CALL #1 — SYNTHESIS: warehouse inventory → candidate taxonomy YAML
# ---------------------------------------------------------------------------
SYNTHESIS_SYSTEM_PROMPT = """\
Você é um data steward sênior que constrói o dicionário de dados canônico
de uma empresa a partir do inventário observado em um data warehouse.

Receba um inventário JSON com tabelas, colunas, tipos de dados, hints de
padrão (cpf, cnpj, email, uuid, vin, iso_country, iso_datetime) e clusters
de colunas que compartilham tokens entre tabelas (candidatos a aliases do
mesmo concept).

ENTREGUE um documento de taxonomia em YAML válido, com a estrutura abaixo
e SEM nenhum texto extra antes ou depois do bloco YAML:

```yaml
metadata:
  title: "..."
  version: "1.0"
  domain: "..."
  platform: "<warehouse_type>"
  owner: "Data Engineering"
  steward: "Analytics Engineering"
naming_conventions:
  casing_rules:
    columns: "PascalCase" | "snake_case" | ...
    entities: "UPPER_CASE" | ...
    ctes: "snake_case"
    examples:
      column: [...]
      entity: [...]
      cte: [...]
  canonical_acronyms:
    items: [...]
    rule: "..."
  forbidden_forms:
    items:
      - {pattern: "...", reason: "..."}
concept_groups:
  - name: "..."
    icon: "..."
    description: "..."
    pii_level: "high" | "medium" | "low" | "none"
    concepts:
      - name: "PascalCaseCanonicalName"
        data_type: "STRING"|"INTEGER"|"FLOAT"|"DATE"|"TIMESTAMP"|"BOOLEAN"
        definition: "1-frase business definition"
        accepted_types: [...]
        entity_qualified_forms:
          single_entity: "..."
          multi_entity: "..."
        aliases: ["...as observed in source tables..."]
context_rules:
  - rule_type: "single_entity"
    title: "..."
    subtitle: "..."
    definition: "..."
    applicable_to: [...]
    examples: { ... }
    anti_patterns: { ... }
    rationale: "..."
  - rule_type: "multi_entity"
    ...
ambiguous_aliases:
  - name: "..."
    description: "..."
    resolution_rules: { ... }
datetime_standards:
  timezone: "UTC"
  date_format: "YYYY-MM-DD"
  timestamp_format: "YYYY-MM-DDTHH:MM:SS.sssZ"
glossary: { TERM: "definition", ... }
lake_standards:
  zones:
    - { name: "raw",     alias: "bronze", description: "...", naming_pattern: "raw_{source}_{entity}", retention: "..." }
    - { name: "staged",  alias: "silver", description: "...", naming_pattern: "stg_{domain}_{entity}", retention: "..." }
    - { name: "curated", alias: "gold",   description: "...", naming_pattern: "{type}_{entity}",     retention: "..." }
ai_agent_instructions:
  description: "..."
  quick_rules: [...]
validation_rules:
  - {id: "VAL-001", name: "...", severity: "error"|"warning", scope: "column"|"table"|"dataset",
     rule_type: "regex"|"forbidden_substring", pattern: "...", message: "...'{name}'..." }
```

REGRAS DE NEGÓCIO:
- Trate qualquer string no formato `<LABEL_REDACTED>` como sinal de que o
  conteúdo é sensível — NÃO repita o valor, use o LABEL como evidência ao
  classificar o concept (ex.: `<EMAIL_REDACTED>` ⇒ concept "Email", group
  "Contact Information", pii_level "high").
- Use `pattern_hints` para inferir tipo e regras de validação (ex.: `cpf` +
  `cnpj` no mesmo cluster ⇒ concept "NationalID" com ambiguous_alias e
  validation_rule de DocumentType).
- Tokens compartilhados em ≥3 tabelas viram concepts canônicos com aliases.
- Tokens em ≤2 tabelas viram concepts somente se o pattern_hints sustentar.
- Não invente concepts que não estão no inventário.
- Casing recomendado para columns é PascalCase a menos que ≥70% do inventário
  observado use outro padrão.
- Inclua pelo menos uma regra `single_entity` e uma `multi_entity` em
  context_rules.
- Inclua validation_rules mínimas: regex de PascalCase para colunas, regex
  de UPPER_CASE para tabelas, forbidden_substring para abreviações comuns.
- Defina pii_level por grupo com base nos pattern_hints encontrados.

A saída DEVE ser YAML válido e DEVE começar com ```yaml e terminar com ```.
"""

SYNTHESIS_USER_TEMPLATE = """\
Inventário do data warehouse (já anonimizado):

```json
{inventory_json}
```

Construa o documento de taxonomia conforme o formato exigido.
"""

# ---------------------------------------------------------------------------
# CALL #2 — EVALUATION: current YAML + framework → improvement plan
# ---------------------------------------------------------------------------
EVALUATION_SYSTEM_PROMPT = """\
Você é um auditor de governança de dados. Recebe (a) a taxonomia AS-IS de
uma empresa em YAML, (b) os scores numéricos calculados deterministicamente
para 8 dimensões e (c) o referencial de boas práticas.

ENTREGUE um JSON estrito (sem texto extra) com a estrutura:

{{
  "executive_summary": "2-3 frases em PT-BR contextualizando o nível de maturidade",
  "strengths": ["bullet curto", ...],          // 3-5 itens
  "risks": ["bullet curto descrevendo um risco concreto", ...],  // 3-5
  "dimensions": [
    {{
      "dimension": "<chave da dimensão, ex.: naming_consistency>",
      "current_score": <number>,
      "target_score": <number>,
      "diagnosis": "1-2 frases explicando porque a nota é essa",
      "actions": [
        {{
          "title": "Verbo + objeto",
          "detail": "como executar, com referência ao YAML quando aplicável",
          "effort": "low"|"medium"|"high",
          "expected_impact": "low"|"medium"|"high",
          "owner_role": "data_engineer"|"steward"|"architect"|"analyst",
          "depends_on": ["<title de outra ação, se houver>"]
        }}
      ]
    }}
  ],
  "quick_wins": ["título de ação", ...]        // 3-5 ações de impacto/esforço favorável
}}

REGRAS:
- Use APENAS as 8 dimensões fornecidas em `dimension_scores`.
- Para dimensões com score >= 85, gere uma única ação de manutenção.
- Cada ação deve ser específica à taxonomia recebida — referencie concept_groups,
  validation_rules ou regras concretas do YAML quando relevante.
- Ações nunca devem propor coletar/expor dados sensíveis.
- Saída DEVE ser JSON válido. Sem markdown, sem comentários.
"""

EVALUATION_USER_TEMPLATE = """\
Referencial de boas práticas (8 dimensões e pesos):

```json
{framework_json}
```

Score determinístico calculado:

```json
{score_json}
```

Taxonomia AS-IS em YAML:

```yaml
{taxonomy_yaml}
```

Avalie e produza o plano conforme o contrato JSON exigido.
"""
