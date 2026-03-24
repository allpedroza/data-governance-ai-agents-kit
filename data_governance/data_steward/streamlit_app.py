# /// script
# dependencies = [
#   "streamlit>=1.32.0",
#   "plotly>=5.0.0",
#   "pandas>=2.0.0",
# ]
# ///
"""
Data Steward Agent -- Standalone Streamlit UI

Run with:
    streamlit run data_governance/data_steward/streamlit_app.py
"""

import json
from datetime import datetime

import streamlit as st

from data_governance.data_steward.agent import DataStewardAgent
from data_governance.data_steward.models import (
    ApprovalStatus,
    GlossaryTermStatus,
    IssueCategory,
    IssueSeverity,
    IssueStatus,
    QualityRuleStatus,
)


def get_agent() -> DataStewardAgent:
    """Return a cached agent instance."""
    if "steward_agent" not in st.session_state:
        st.session_state.steward_agent = DataStewardAgent(
            persist_dir="./steward_data"
        )
    return st.session_state.steward_agent


def render_intake_tab(agent: DataStewardAgent):
    """Issue intake & triage tab."""
    st.markdown("### Triagem de Issues de Dados")
    st.markdown(
        "Descreva o problema em texto livre. O agente classifica, sugere "
        "severidade, identifica dominio e propoe proximos passos."
    )

    with st.form("issue_form"):
        description = st.text_area(
            "Descricao do problema",
            height=120,
            placeholder=(
                "Ex: O KPI de receita mudou 20% sem razao aparente, "
                "os dados de ontem parecem duplicados..."
            ),
        )
        col1, col2 = st.columns(2)
        with col1:
            context_domain = st.text_input(
                "Dominio (opcional)", placeholder="finance"
            )
        with col2:
            context_dataset = st.text_input(
                "Dataset (opcional)", placeholder="fato_vendas"
            )
        submitted = st.form_submit_button("Triar Issue")

    if submitted and description.strip():
        context = {}
        if context_domain:
            context["domain"] = context_domain
        if context_dataset:
            context["dataset"] = context_dataset

        issue = agent.triage_issue(description, context or None)
        st.success(f"Issue triada: {issue.issue_id}")
        st.markdown(issue.to_markdown())

    # List existing issues
    st.markdown("---")
    st.markdown("### Issues Abertas")
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox(
            "Status",
            ["Todos"] + [s.value for s in IssueStatus],
        )
    with col2:
        filter_severity = st.selectbox(
            "Severidade",
            ["Todos"] + [s.value for s in IssueSeverity],
        )
    with col3:
        filter_domain = st.text_input("Dominio", key="issue_filter_domain")

    issues = agent.list_issues(
        status=filter_status if filter_status != "Todos" else None,
        severity=filter_severity if filter_severity != "Todos" else None,
        domain=filter_domain or None,
    )

    if issues:
        for issue in issues:
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(issue.severity, "⚪")
            with st.expander(
                f"{severity_icon} [{issue.severity.upper()}] {issue.title} "
                f"({issue.status})"
            ):
                st.markdown(issue.to_markdown())
                if issue.status != IssueStatus.RESOLVED.value:
                    with st.form(f"resolve_{issue.issue_id}"):
                        notes = st.text_input("Notas de resolucao")
                        if st.form_submit_button("Resolver"):
                            agent.resolve_issue(issue.issue_id, notes)
                            st.success("Issue resolvida!")
                            st.rerun()
    else:
        st.info("Nenhuma issue encontrada com os filtros selecionados.")


def render_glossary_tab(agent: DataStewardAgent):
    """Business glossary curation tab."""
    st.markdown("### Glossario de Negocio")
    st.markdown(
        "Consolide definicoes de multiplas fontes. O agente propoe uma "
        "definicao padrao e detecta conflitos semanticos."
    )

    with st.form("glossary_form"):
        term_name = st.text_input(
            "Nome do termo", placeholder="receita_liquida"
        )
        domain = st.text_input("Dominio", placeholder="finance")

        st.markdown("**Fontes de definicao** (adicione pelo menos uma)")
        sources = []
        for i in range(3):
            col1, col2 = st.columns([1, 3])
            with col1:
                src = st.text_input(
                    f"Fonte {i + 1}",
                    key=f"src_{i}",
                    placeholder="SQL view / Dashboard / Doc",
                )
            with col2:
                defn = st.text_input(
                    f"Definicao {i + 1}",
                    key=f"def_{i}",
                    placeholder="Definicao do termo nesta fonte...",
                )
            if src and defn:
                sources.append({"source": src, "definition": defn})

        related_ds = st.text_input(
            "Datasets relacionados (separados por virgula)", placeholder="fato_vendas, dim_clientes"
        )
        submitted = st.form_submit_button("Curar Termo")

    if submitted and term_name and sources:
        ds_list = (
            [d.strip() for d in related_ds.split(",") if d.strip()]
            if related_ds
            else []
        )
        term = agent.curate_term(term_name, sources, domain, ds_list)
        st.success(f"Termo curado: {term.term_id}")
        st.markdown(term.to_markdown())

    # List existing terms
    st.markdown("---")
    st.markdown("### Termos do Glossario")
    filter_status = st.selectbox(
        "Status",
        ["Todos"] + [s.value for s in GlossaryTermStatus],
        key="glossary_filter",
    )
    terms = agent.list_glossary(
        status=filter_status if filter_status != "Todos" else None
    )
    if terms:
        for term in terms:
            status_icon = {
                "candidate": "📝",
                "under_review": "🔍",
                "approved": "✅",
                "deprecated": "❌",
            }.get(term.status, "⚪")
            with st.expander(
                f"{status_icon} {term.term_name} ({term.status})"
            ):
                st.markdown(term.to_markdown())
                if term.status == GlossaryTermStatus.CANDIDATE.value:
                    if st.button(
                        "Submeter para Aprovacao",
                        key=f"submit_term_{term.term_id}",
                    ):
                        req = agent.submit_term_for_approval(term.term_id)
                        st.success(
                            f"Submetido para aprovacao: {req.request_id}"
                        )
                        st.rerun()
    else:
        st.info("Nenhum termo no glossario.")


def render_rules_tab(agent: DataStewardAgent):
    """Quality rule drafting tab."""
    st.markdown("### Regras de Data Quality")
    st.markdown(
        "O agente sugere regras em linguagem de negocio + expressao tecnica. "
        "Voce revisa e aprova."
    )

    with st.form("rules_form"):
        dataset = st.text_input("Dataset", placeholder="dim_customers")
        domain = st.text_input("Dominio", placeholder="customer")
        st.markdown("**Colunas** (opcional, melhora sugestoes)")
        cols_json = st.text_area(
            "JSON de colunas",
            height=80,
            placeholder='[{"name": "cpf", "type": "string", "nullable": false}]',
        )
        submitted = st.form_submit_button("Sugerir Regras")

    if submitted and dataset and domain:
        columns = None
        if cols_json.strip():
            try:
                columns = json.loads(cols_json)
            except json.JSONDecodeError:
                st.error("JSON de colunas invalido.")
                columns = None

        rules = agent.draft_rules(dataset, domain, columns)
        st.success(f"{len(rules)} regra(s) sugerida(s)")
        for rule in rules:
            st.markdown(rule.to_markdown())

    # List existing rules
    st.markdown("---")
    st.markdown("### Regras Existentes")
    filter_status = st.selectbox(
        "Status",
        ["Todos"] + [s.value for s in QualityRuleStatus],
        key="rules_filter",
    )
    rules = agent.list_rules(
        status=filter_status if filter_status != "Todos" else None
    )
    if rules:
        for rule in rules:
            status_icon = {
                "draft": "📝",
                "pending_approval": "🔍",
                "active": "✅",
                "disabled": "❌",
            }.get(rule.status, "⚪")
            with st.expander(
                f"{status_icon} [{rule.dimension}] {rule.business_description[:60]} "
                f"({rule.status})"
            ):
                st.markdown(rule.to_markdown())
                if rule.status == QualityRuleStatus.DRAFT.value:
                    if st.button(
                        "Submeter para Aprovacao",
                        key=f"submit_rule_{rule.rule_id}",
                    ):
                        req = agent.submit_rule_for_approval(rule.rule_id)
                        st.success(
                            f"Submetido para aprovacao: {req.request_id}"
                        )
                        st.rerun()
    else:
        st.info("Nenhuma regra cadastrada.")


def render_impact_tab(agent: DataStewardAgent):
    """Impact analysis tab."""
    st.markdown("### Analise de Impacto")
    st.markdown(
        "Descreva uma mudanca proposta. O agente identifica KPIs, times, "
        "regras e implicacoes regulatorias impactadas."
    )

    with st.form("impact_form"):
        change_desc = st.text_area(
            "Descricao da mudanca",
            height=100,
            placeholder="Ex: Remover coluna cpf da tabela dim_customers",
        )
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.text_input(
                "Dataset", placeholder="dim_customers", key="impact_ds"
            )
        with col2:
            attribute = st.text_input(
                "Atributo (opcional)", placeholder="cpf", key="impact_attr"
            )
        domain = st.text_input(
            "Dominio", placeholder="customer", key="impact_domain"
        )
        submitted = st.form_submit_button("Analisar Impacto")

    if submitted and change_desc and dataset:
        report = agent.explain_impact(
            change_description=change_desc,
            dataset=dataset,
            attribute=attribute or None,
            domain=domain,
        )
        st.markdown(report.to_markdown())


def render_workflow_tab(agent: DataStewardAgent):
    """Approval workflow tab."""
    st.markdown("### Aprovacoes Pendentes")

    pending = agent.get_pending_approvals()

    if not pending:
        st.info("Nenhuma aprovacao pendente.")
    else:
        for req in pending:
            type_icon = {
                "glossary_term": "📖",
                "quality_rule": "📏",
                "change_request": "🔄",
            }.get(req.request_type, "📋")
            with st.expander(
                f"{type_icon} {req.title} (por {req.submitted_by})"
            ):
                st.markdown(req.to_markdown())
                with st.form(f"decision_{req.request_id}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        reviewer = st.text_input(
                            "Seu nome", key=f"rev_{req.request_id}"
                        )
                    with col2:
                        decision = st.selectbox(
                            "Decisao",
                            ["approved", "rejected", "revision_requested"],
                            key=f"dec_{req.request_id}",
                        )
                    notes = st.text_input(
                        "Notas", key=f"notes_{req.request_id}"
                    )
                    if st.form_submit_button("Submeter Decisao"):
                        if reviewer:
                            agent.submit_decision(
                                req.request_id, decision, reviewer, notes
                            )
                            st.success(f"Decisao registrada: {decision}")
                            st.rerun()
                        else:
                            st.error("Informe seu nome para registrar a decisao.")

    # Activity log
    st.markdown("---")
    st.markdown("### Log de Atividades")
    activities = agent.get_activity_log(limit=30)
    if activities:
        for entry in activities:
            st.text(
                f"[{entry.timestamp[:19]}] {entry.action} - {entry.actor}"
                + (f" | {entry.domain}" if entry.domain else "")
            )
    else:
        st.info("Nenhuma atividade registrada.")


def main():
    st.set_page_config(
        page_title="Data Steward Agent",
        page_icon="📋",
        layout="wide",
    )
    st.title("📋 Data Steward Agent")
    st.caption(
        "Copiloto operacional de governanca — o agente propoe, o steward aprova."
    )

    agent = get_agent()

    intake_tab, glossary_tab, rules_tab, impact_tab, workflow_tab = st.tabs(
        [
            "Triagem de Issues",
            "Glossario de Negocio",
            "Regras de Quality",
            "Analise de Impacto",
            "Aprovacoes",
        ]
    )

    with intake_tab:
        render_intake_tab(agent)
    with glossary_tab:
        render_glossary_tab(agent)
    with rules_tab:
        render_rules_tab(agent)
    with impact_tab:
        render_impact_tab(agent)
    with workflow_tab:
        render_workflow_tab(agent)


if __name__ == "__main__":
    main()
