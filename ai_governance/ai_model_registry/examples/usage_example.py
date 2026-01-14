"""
AI Model Registry Agent - Usage Examples

Este arquivo demonstra como usar o AIModelRegistryAgent para:
- Registrar modelos de AI/ML
- Gerenciar versões e ciclo de vida
- Detectar e tratar Shadow AI
- Gerar relatórios de compliance
- Analisar dependências entre modelos
"""

from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ai_governance.ai_model_registry import (
    AIModelRegistryAgent,
    ModelType,
    ModelStage,
    ModelStatus,
    RiskLevel,
    DataSensitivity,
    LicenseType,
    ModelOwner,
    ModelMetrics,
    DataSource,
)


def example_basic_usage():
    """Exemplo básico de uso do AI Model Registry."""
    print("=" * 60)
    print("EXEMPLO 1: Uso Básico do AI Model Registry")
    print("=" * 60)

    # Inicializar o agente
    registry = AIModelRegistryAgent(
        storage_path=None,  # Em memória apenas
        enable_audit=True,
    )

    # Definir o owner do modelo
    owner = ModelOwner(
        name="Maria Silva",
        email="maria.silva@empresa.com",
        team="Data Science",
        department="Technology",
        role="owner",
        slack_channel="#ds-models",
    )

    # Registrar um modelo de classificação de fraude
    fraud_model = registry.register_model(
        name="fraud-detection-v1",
        model_type=ModelType.CLASSIFIER,
        purpose="Detectar transações fraudulentas em tempo real",
        description="Modelo XGBoost treinado com dados históricos de fraude para "
                    "classificação binária de transações suspeitas.",
        owner=owner,
        risk_level=RiskLevel.HIGH,
        license_type=LicenseType.INTERNAL,
        initial_version="1.0.0",
        data_sensitivity=DataSensitivity.PII,
        use_cases=["Prevenção de fraude", "Scoring de transações"],
        business_units=["Fraud Prevention", "Risk Management"],
        applications=["payment-gateway", "mobile-app"],
        tags=["fraud", "xgboost", "real-time", "production"],
        compliance_requirements=["PCI-DSS", "LGPD"],
        registered_by="maria.silva",
    )

    print(f"\nModelo registrado: {fraud_model.name}")
    print(f"ID: {fraud_model.model_id}")
    print(f"Tipo: {fraud_model.model_type.value}")
    print(f"Risco: {fraud_model.risk_level.value}")
    print(f"Owner: {fraud_model.owner.name} ({fraud_model.owner.team})")

    return registry


def example_version_management(registry: AIModelRegistryAgent):
    """Exemplo de gerenciamento de versões."""
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Gerenciamento de Versões")
    print("=" * 60)

    # Adicionar nova versão com métricas
    metrics = ModelMetrics(
        accuracy=0.95,
        precision=0.92,
        recall=0.89,
        f1_score=0.905,
        latency_p50_ms=15.2,
        latency_p99_ms=45.8,
        throughput_rps=1200,
        daily_requests=5_000_000,
        monthly_cost_usd=2500.0,
        error_rate=0.001,
    )

    version = registry.add_version(
        model_id_or_name="fraud-detection-v1",
        version="1.1.0",
        created_by="maria.silva",
        stage=ModelStage.STAGING,
        status=ModelStatus.TESTING,
        description="Versão com feature engineering melhorado",
        changelog="- Adicionadas 15 novas features\n- Melhorada precisão em 5%",
        framework="xgboost",
        framework_version="1.7.6",
        python_version="3.10",
        dependencies=["xgboost==1.7.6", "pandas==2.0.0", "scikit-learn==1.3.0"],
        metrics=metrics,
        training_data_version="2024-01",
        hyperparameters={"max_depth": 8, "learning_rate": 0.1, "n_estimators": 500},
        tags=["improved", "new-features"],
    )

    print(f"\nVersão adicionada: {version.version}")
    print(f"Stage: {version.stage.value}")
    print(f"Métricas:")
    print(f"  - Accuracy: {version.metrics.accuracy}")
    print(f"  - F1 Score: {version.metrics.f1_score}")
    print(f"  - Latência P99: {version.metrics.latency_p99_ms}ms")

    # Promover para produção
    registry.promote_version(
        model_id_or_name="fraud-detection-v1",
        version="1.1.0",
        new_stage=ModelStage.PRODUCTION,
        promoted_by="tech.lead",
        reason="Aprovado após testes de stress e validação de métricas",
    )

    print(f"\nVersão 1.1.0 promovida para PRODUÇÃO")

    # Verificar versão em produção
    model = registry.get_model("fraud-detection-v1")
    prod_version = model.get_production_version()
    print(f"Versão atual em produção: {prod_version.version}")


def example_register_llm(registry: AIModelRegistryAgent):
    """Exemplo de registro de LLM/modelo generativo."""
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Registro de LLM")
    print("=" * 60)

    owner = ModelOwner(
        name="João Santos",
        email="joao.santos@empresa.com",
        team="AI Platform",
        department="Technology",
        role="owner",
    )

    # Registrar um modelo LLM
    llm = registry.register_model(
        name="customer-service-llm",
        model_type=ModelType.LLM,
        purpose="Atendimento automatizado ao cliente via chat",
        description="Modelo GPT fine-tuned para atendimento ao cliente, "
                    "treinado com histórico de conversas e FAQ.",
        owner=owner,
        risk_level=RiskLevel.LIMITED,
        license_type=LicenseType.API_SERVICE,
        initial_version="1.0.0",
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
        use_cases=["Chatbot", "FAQ automatizado", "Triagem de tickets"],
        business_units=["Customer Service", "Support"],
        applications=["chatbot-web", "mobile-support"],
        tags=["llm", "chatbot", "customer-service", "gpt"],
        api_endpoint="https://api.empresa.com/v1/chat",
        documentation_url="https://docs.empresa.com/customer-llm",
        compliance_requirements=["LGPD", "Política de IA Responsável"],
        metadata={
            "base_model": "gpt-4-turbo",
            "fine_tuning_dataset": "customer_conversations_2024",
            "context_window": 128000,
            "max_tokens": 4096,
        },
        registered_by="joao.santos",
    )

    print(f"\nLLM registrado: {llm.name}")
    print(f"Tipo: {llm.model_type.value}")
    print(f"API Endpoint: {llm.api_endpoint}")
    print(f"Metadados: {llm.metadata}")


def example_shadow_ai_detection(registry: AIModelRegistryAgent):
    """Exemplo de detecção e tratamento de Shadow AI."""
    print("\n" + "=" * 60)
    print("EXEMPLO 4: Detecção de Shadow AI")
    print("=" * 60)

    # Simular descoberta de Shadow AI durante scan de rede
    shadow = registry.register_shadow_ai(
        name="recommendation-model-unknown",
        model_type=ModelType.RECOMMENDER,
        purpose="Sistema de recomendação não documentado encontrado em produção",
        discovery_source="network_scan_api_endpoints",
        discovered_by="security.team",
        applications=["legacy-ecommerce-app"],
        metadata={
            "endpoint_discovered": "http://internal:8080/recommend",
            "last_request_seen": "2024-01-15T10:30:00",
            "estimated_daily_calls": 50000,
        },
    )

    print(f"\nShadow AI detectado: {shadow.name}")
    print(f"Fonte de descoberta: {shadow.discovery_source}")
    print(f"É Shadow AI: {shadow.is_shadow_ai}")
    print(f"Tags: {shadow.tags}")

    # Listar todos os Shadow AIs
    shadow_models = registry.get_shadow_ai_models()
    print(f"\nTotal de Shadow AIs no registro: {len(shadow_models)}")

    # Legitimizar o Shadow AI após investigação
    new_owner = ModelOwner(
        name="Carlos Lima",
        email="carlos.lima@empresa.com",
        team="E-commerce",
        department="Product",
        role="owner",
    )

    legitimized = registry.legitimize_shadow_ai(
        model_id_or_name=shadow.name,
        new_owner=new_owner,
        legitimized_by="governance.team",
        purpose="Sistema de recomendação de produtos baseado em collaborative filtering",
        risk_level=RiskLevel.LIMITED,
    )

    print(f"\nModelo legitimizado: {legitimized.name}")
    print(f"Novo owner: {legitimized.owner.name}")
    print(f"É Shadow AI: {legitimized.is_shadow_ai}")


def example_dependencies(registry: AIModelRegistryAgent):
    """Exemplo de gerenciamento de dependências entre modelos."""
    print("\n" + "=" * 60)
    print("EXEMPLO 5: Dependências entre Modelos")
    print("=" * 60)

    owner = ModelOwner(
        name="Ana Costa",
        email="ana.costa@empresa.com",
        team="ML Engineering",
        department="Technology",
        role="owner",
    )

    # Registrar modelo de embeddings
    embeddings = registry.register_model(
        name="product-embeddings",
        model_type=ModelType.EMBEDDING,
        purpose="Gerar embeddings de produtos para busca semântica",
        description="Modelo de embeddings treinado com descrições de produtos",
        owner=owner,
        risk_level=RiskLevel.MINIMAL,
        license_type=LicenseType.INTERNAL,
        initial_version="1.0.0",
        tags=["embeddings", "search", "products"],
        registered_by="ana.costa",
    )

    # Registrar modelo de busca que depende dos embeddings
    search_model = registry.register_model(
        name="semantic-search",
        model_type=ModelType.RAG,
        purpose="Busca semântica de produtos",
        description="Sistema RAG para busca de produtos usando embeddings",
        owner=owner,
        risk_level=RiskLevel.MINIMAL,
        license_type=LicenseType.INTERNAL,
        initial_version="1.0.0",
        tags=["search", "rag", "products"],
        registered_by="ana.costa",
    )

    # Adicionar dependência
    registry.add_dependency(
        model_id_or_name="semantic-search",
        depends_on_id_or_name="product-embeddings",
        dependency_type="uses",
        description="Usa embeddings para busca por similaridade",
        added_by="ana.costa",
    )

    print(f"\nDependência adicionada: semantic-search -> product-embeddings")

    # Analisar impacto de mudanças
    impact = registry.get_impact_analysis("product-embeddings")
    print(f"\nAnálise de impacto para '{impact['model_name']}':")
    print(f"  - Dependentes diretos: {impact['direct_dependents']}")
    print(f"  - Total afetados: {impact['total_affected']}")

    if impact['affected_models']:
        print("  - Modelos afetados:")
        for m in impact['affected_models']:
            print(f"    - {m['model_name']} (risco: {m['risk_level']})")


def example_search_and_filter(registry: AIModelRegistryAgent):
    """Exemplo de busca e filtros."""
    print("\n" + "=" * 60)
    print("EXEMPLO 6: Busca e Filtros")
    print("=" * 60)

    # Busca textual
    results = registry.search(
        query="fraud",
        search_in=['name', 'purpose', 'description', 'tags'],
    )

    print(f"\nBusca por 'fraud': {results.total_count} resultado(s)")
    for model in results.models:
        print(f"  - {model.name}: {model.purpose[:50]}...")

    # Listar por tipo
    llms = registry.list_models(model_type=ModelType.LLM)
    print(f"\nModelos do tipo LLM: {len(llms)}")
    for m in llms:
        print(f"  - {m.name}")

    # Listar modelos de alto risco
    high_risk = registry.get_high_risk_models()
    print(f"\nModelos de alto risco: {len(high_risk)}")
    for m in high_risk:
        print(f"  - {m.name} ({m.risk_level.value})")

    # Modelos em produção
    production = registry.get_production_models()
    print(f"\nModelos em produção: {len(production)}")
    for m in production:
        v = m.get_production_version()
        print(f"  - {m.name} v{v.version if v else 'N/A'}")


def example_reports(registry: AIModelRegistryAgent):
    """Exemplo de geração de relatórios."""
    print("\n" + "=" * 60)
    print("EXEMPLO 7: Relatórios e Estatísticas")
    print("=" * 60)

    # Estatísticas
    stats = registry.get_statistics()
    print("\nEstatísticas do Registro:")
    print(f"  - Total de modelos: {stats.total_models}")
    print(f"  - Modelos em produção: {stats.models_in_production}")
    print(f"  - Shadow AI: {stats.shadow_ai_count}")
    print(f"  - Total de versões: {stats.total_versions}")
    print(f"  - Média versões/modelo: {stats.avg_versions_per_model}")

    print("\n  Por tipo:")
    for t, count in stats.models_by_type.items():
        print(f"    - {t}: {count}")

    print("\n  Por risco:")
    for r, count in stats.models_by_risk_level.items():
        print(f"    - {r}: {count}")

    # Relatório de compliance
    print("\n--- Relatório de Compliance ---")
    compliance = registry.generate_compliance_report()
    print(f"\nGaps de Compliance:")
    for gap, value in compliance['compliance_gaps'].items():
        print(f"  - {gap}: {value}")

    print(f"\nRecomendações:")
    for rec in compliance['recommendations']:
        print(f"  - {rec}")


def example_audit_log(registry: AIModelRegistryAgent):
    """Exemplo de consulta ao log de auditoria."""
    print("\n" + "=" * 60)
    print("EXEMPLO 8: Log de Auditoria")
    print("=" * 60)

    # Buscar todas as entradas de auditoria
    audit_entries = registry.get_audit_log(limit=10)

    print(f"\nÚltimas {len(audit_entries)} entradas de auditoria:")
    for entry in audit_entries[:5]:
        print(f"\n  [{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"  Tipo: {entry.change_type.value}")
        print(f"  Modelo: {entry.model_name}")
        print(f"  Por: {entry.changed_by}")
        print(f"  Razão: {entry.reason}")


def example_full_report(registry: AIModelRegistryAgent):
    """Exemplo de relatório completo em Markdown."""
    print("\n" + "=" * 60)
    print("EXEMPLO 9: Relatório Completo (Markdown)")
    print("=" * 60)

    report = registry.generate_report(format="markdown")

    # Mostrar apenas as primeiras linhas
    lines = report.split('\n')
    print("\n" + "\n".join(lines[:30]))
    print("\n... (relatório continua)")


def main():
    """Executa todos os exemplos."""
    print("\n" + "=" * 60)
    print("AI MODEL REGISTRY AGENT - EXEMPLOS DE USO")
    print("=" * 60)

    # Exemplo 1: Uso básico
    registry = example_basic_usage()

    # Exemplo 2: Gerenciamento de versões
    example_version_management(registry)

    # Exemplo 3: Registro de LLM
    example_register_llm(registry)

    # Exemplo 4: Shadow AI
    example_shadow_ai_detection(registry)

    # Exemplo 5: Dependências
    example_dependencies(registry)

    # Exemplo 6: Busca e filtros
    example_search_and_filter(registry)

    # Exemplo 7: Relatórios
    example_reports(registry)

    # Exemplo 8: Auditoria
    example_audit_log(registry)

    # Exemplo 9: Relatório completo
    example_full_report(registry)

    print("\n" + "=" * 60)
    print("EXEMPLOS CONCLUÍDOS COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
