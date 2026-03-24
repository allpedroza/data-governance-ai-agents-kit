"""
Data Steward Agent -- Basic Usage Example

Demonstrates the 5 core capabilities:
1. Issue intake & triage
2. Business glossary curation
3. Quality rule drafting
4. Impact analysis
5. Approval workflow

Run:
    python -m data_governance.data_steward.examples.basic_usage
"""

from data_governance.data_steward.agent import DataStewardAgent


def main():
    # Initialize agent (no LLM -- uses rule-based fallbacks)
    agent = DataStewardAgent(persist_dir="./steward_data_example")

    # ------------------------------------------------------------------
    # 1. Assign ownership first
    # ------------------------------------------------------------------
    print("=== 1. Assigning Stewards ===")
    agent.assign_steward(
        domain="finance",
        person_name="Ana Silva",
        role="data_owner",
        person_email="ana.silva@empresa.com",
        datasets=["fato_vendas", "dim_clientes"],
    )
    agent.assign_steward(
        domain="finance",
        person_name="Carlos Mendes",
        role="data_steward",
        person_email="carlos.mendes@empresa.com",
    )
    assignments = agent.list_assignments(domain="finance")
    print(f"  {len(assignments)} assignment(s) no dominio finance")

    # ------------------------------------------------------------------
    # 2. Triage a data issue
    # ------------------------------------------------------------------
    print("\n=== 2. Triaging an Issue ===")
    issue = agent.triage_issue(
        "O KPI de receita liquida caiu 20% de ontem para hoje sem nenhuma "
        "mudanca conhecida. Os dados de fato_vendas parecem ter registros "
        "duplicados no dia 2024-03-15."
    )
    print(f"  Issue: {issue.issue_id}")
    print(f"  Categoria: {issue.category}")
    print(f"  Severidade: {issue.severity}")
    print(f"  Owner sugerido: {issue.probable_owner}")
    print(f"  Proximos passos: {issue.suggested_next_steps}")

    # ------------------------------------------------------------------
    # 3. Curate a glossary term
    # ------------------------------------------------------------------
    print("\n=== 3. Curating a Glossary Term ===")
    term = agent.curate_term(
        term_name="receita_liquida",
        sources=[
            {
                "source": "SQL view v_receita",
                "definition": "Soma das vendas menos devolucoes e impostos",
            },
            {
                "source": "Dashboard Financeiro",
                "definition": "Total de receita apos descontos",
            },
            {
                "source": "Manual Contabil",
                "definition": "Receita bruta deduzida de impostos sobre vendas, devolucoes e abatimentos",
            },
        ],
        domain="finance",
        related_datasets=["fato_vendas", "dim_produtos"],
    )
    print(f"  Termo: {term.term_name} ({term.term_id})")
    print(f"  Definicao proposta: {term.proposed_definition}")
    print(f"  Conflitos: {len(term.semantic_conflicts)}")

    # Submit for approval
    req = agent.submit_term_for_approval(term.term_id)
    print(f"  Submetido para aprovacao: {req.request_id}")

    # ------------------------------------------------------------------
    # 4. Draft quality rules
    # ------------------------------------------------------------------
    print("\n=== 4. Drafting Quality Rules ===")
    rules = agent.draft_rules(
        dataset="fato_vendas",
        domain="finance",
        columns=[
            {"name": "venda_id", "type": "integer", "nullable": False},
            {"name": "cpf_cliente", "type": "string", "nullable": False},
            {"name": "valor_total", "type": "decimal", "nullable": True},
            {"name": "data_venda", "type": "date", "nullable": False},
            {"name": "email_cliente", "type": "string", "nullable": True},
        ],
    )
    print(f"  {len(rules)} regra(s) sugerida(s)")
    for r in rules:
        print(f"  - [{r.dimension}] {r.business_description}")

    # ------------------------------------------------------------------
    # 5. Impact analysis
    # ------------------------------------------------------------------
    print("\n=== 5. Impact Analysis ===")
    impact = agent.explain_impact(
        change_description="Remover coluna cpf_cliente da tabela fato_vendas",
        dataset="fato_vendas",
        attribute="cpf_cliente",
        domain="finance",
    )
    print(f"  Risco: {impact.risk_level}")
    print(f"  Resumo: {impact.human_summary}")
    if impact.regulatory_exceptions:
        print(f"  Regulatorio: {impact.regulatory_exceptions}")

    # ------------------------------------------------------------------
    # 6. Approval workflow
    # ------------------------------------------------------------------
    print("\n=== 6. Approval Workflow ===")
    pending = agent.get_pending_approvals()
    print(f"  {len(pending)} aprovacao(es) pendente(s)")
    if pending:
        # Approve the glossary term
        decision = agent.submit_decision(
            request_id=pending[0].request_id,
            status="approved",
            reviewer="Ana Silva",
            notes="Definicao alinhada com a equipe contabil.",
        )
        print(f"  Decisao: {decision.status} por {decision.reviewed_by}")
        print(f"  Changelog: {decision.changelog_entry}")

    # Check updated term status
    updated_term = agent.list_glossary(status="approved")
    print(f"\n  Termos aprovados: {len(updated_term)}")

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------
    print("\n=== Activity Log ===")
    log = agent.get_activity_log(limit=10)
    for entry in log:
        print(f"  [{entry.timestamp[:19]}] {entry.action} - {entry.actor}")

    print("\nDone!")


if __name__ == "__main__":
    main()
