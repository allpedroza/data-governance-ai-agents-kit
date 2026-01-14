"""
Sensitive Data NER Agent - Usage Examples

This script demonstrates various use cases for the NER agent.
"""

import sys
sys.path.insert(0, "../..")

from sensitive_data_ner import (
    SensitiveDataNERAgent,
    NERResult,
    FilterPolicy,
    FilterAction,
    AnonymizationStrategy,
    AnonymizationConfig,
    EntityCategory,
)


def example_basic_analysis():
    """Basic text analysis for sensitive data."""
    print("=" * 60)
    print("EXEMPLO 1: Análise Básica de Texto")
    print("=" * 60)

    agent = SensitiveDataNERAgent()

    text = """
    Prezado cliente João da Silva,

    Confirmamos o recebimento do seu cadastro com os seguintes dados:
    - CPF: 123.456.789-09
    - Email: joao.silva@email.com
    - Telefone: (11) 98765-4321
    - Endereço: Rua das Flores, 123 - CEP 01234-567

    Seu cartão de crédito terminado em 9010 foi cadastrado com sucesso.

    Atenciosamente,
    Equipe de Atendimento
    """

    result = agent.analyze(text)

    print(f"\n📊 Resultados:")
    print(f"   Entidades encontradas: {result.statistics['total']}")
    print(f"   Score de risco: {result.risk_score:.1%}")
    print(f"   Ação recomendada: {result.filter_action.value}")

    print(f"\n📋 Entidades por categoria:")
    for category in EntityCategory:
        count = result.statistics.get(category.value, 0)
        if count > 0:
            print(f"   - {category.value.upper()}: {count}")

    print(f"\n🔒 Texto Anonimizado:")
    print(result.anonymized_text)


def example_llm_filter():
    """Using the agent as an LLM request filter."""
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Filtro para Requisições LLM")
    print("=" * 60)

    # Configure strict policy
    policy = FilterPolicy(
        pii_action=FilterAction.ANONYMIZE,
        phi_action=FilterAction.BLOCK,
        pci_action=FilterAction.BLOCK,
        business_action=FilterAction.BLOCK,
        min_confidence=0.5,
    )

    agent = SensitiveDataNERAgent(filter_policy=policy)

    # Test prompts
    prompts = [
        "Qual é a capital do Brasil?",  # Safe
        "Gere um email para joao@email.com",  # PII - will be anonymized
        "O paciente com diagnóstico CID J45.0 precisa de tratamento",  # PHI - blocked
        "Processe o pagamento do cartão 4532 1234 5678 9010",  # PCI - blocked
    ]

    for prompt in prompts:
        print(f"\n📤 Prompt: {prompt[:50]}...")
        try:
            safe_prompt, result = agent.filter_llm_request(prompt)
            print(f"   ✅ Ação: {result.filter_action.value}")
            if result.anonymized_text and result.anonymized_text != prompt:
                print(f"   📝 Prompt seguro: {safe_prompt}")
        except ValueError as e:
            print(f"   🚫 BLOQUEADO: {e}")


def example_business_terms():
    """Detecting business-sensitive terms."""
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Termos de Negócio Sensíveis")
    print("=" * 60)

    agent = SensitiveDataNERAgent(
        business_terms=[
            "Projeto Arara Azul",
            "Aquisição Orion",
            "Parceria LATAM 2025",
            "Margem bruta",
            "Operação Fênix",
        ]
    )

    text = """
    Conforme discutido na reunião de ontem, o Projeto Arara Azul
    está avançando conforme planejado. A equipe de M&A está
    finalizando os documentos da Aquisição Orion.

    A margem bruta do Q4 superou as expectativas.
    """

    result = agent.analyze(text)

    print(f"\n🔍 Termos estratégicos detectados:")
    for entity in result.entities:
        if entity.category == EntityCategory.BUSINESS:
            print(f"   - '{entity.value}' (confiança: {entity.confidence:.1%})")

    print(f"\n📊 Score de risco: {result.risk_score:.1%}")
    print(f"🔒 Ação: {result.filter_action.value}")


def example_anonymization_strategies():
    """Demonstrating different anonymization strategies."""
    print("\n" + "=" * 60)
    print("EXEMPLO 4: Estratégias de Anonimização")
    print("=" * 60)

    text = "Cliente: João Silva, CPF: 123.456.789-09, Email: joao@email.com"

    strategies = [
        AnonymizationStrategy.REDACT,
        AnonymizationStrategy.MASK,
        AnonymizationStrategy.PARTIAL,
        AnonymizationStrategy.HASH,
        AnonymizationStrategy.PSEUDONYMIZE,
    ]

    for strategy in strategies:
        policy = FilterPolicy(anonymization_strategy=strategy)
        agent = SensitiveDataNERAgent(filter_policy=policy)
        result = agent.analyze(text)

        print(f"\n📝 Estratégia: {strategy.value.upper()}")
        print(f"   {result.anonymized_text}")


def example_predictive_detection():
    """Showing predictive detection with validation."""
    print("\n" + "=" * 60)
    print("EXEMPLO 5: Detecção Preditiva com Validação")
    print("=" * 60)

    agent = SensitiveDataNERAgent(strict_mode=True)

    # CPF válido vs inválido
    texts = [
        ("CPF válido com contexto", "O CPF do cliente é 529.982.247-25"),
        ("CPF inválido (checksum errado)", "CPF: 123.456.789-00"),
        ("Padrão em contexto de teste", "CPF de exemplo: 000.000.000-00"),
        ("Cartão válido", "Cartão: 4532 0151 1280 0456"),
        ("Cartão inválido", "Cartão: 1234 5678 9012 3456"),
    ]

    for description, text in texts:
        result = agent.analyze(text, include_low_confidence=True)

        print(f"\n📋 {description}")
        print(f"   Texto: {text}")

        if result.entities:
            entity = result.entities[0]
            status = "✅ VÁLIDO" if entity.is_validated else "⚠️ Não validado"
            print(f"   {status} - Confiança: {entity.confidence:.1%}")
        else:
            print(f"   ❌ Nenhuma entidade detectada (baixa confiança)")


def example_batch_processing():
    """Processing multiple texts."""
    print("\n" + "=" * 60)
    print("EXEMPLO 6: Processamento em Lote")
    print("=" * 60)

    agent = SensitiveDataNERAgent()

    texts = [
        "Email de contato: contato@empresa.com",
        "Ligar para (21) 99999-8888",
        "Transferência para conta 12345-6",
        "Reunião sobre Projeto Secreto às 15h",
        "Previsão do tempo para amanhã",
    ]

    print("\n📊 Resultados do lote:")
    for i, result in enumerate(agent.batch_analyze(texts)):
        status = "⚠️" if result.has_sensitive_data else "✅"
        print(f"   {status} Texto {i+1}: {result.statistics['total']} entidades, risco: {result.risk_score:.1%}")


def example_export_configuration():
    """Exporting agent configuration."""
    print("\n" + "=" * 60)
    print("EXEMPLO 7: Exportar Configuração")
    print("=" * 60)

    agent = SensitiveDataNERAgent(
        business_terms=["Termo Secreto", "Projeto X"],
        locales=["br"],
    )

    config = agent.export_patterns()

    print(f"\n📋 Configuração do Agente:")
    print(f"   Padrões carregados: {len(config['patterns'])}")
    print(f"   Termos de negócio: {config['business_terms']}")
    print(f"   Confiança mínima: {config['policy']['min_confidence']}")
    print(f"   Estratégia de anonimização: {config['policy']['anonymization_strategy']}")


if __name__ == "__main__":
    print("\n🔒 SENSITIVE DATA NER AGENT - EXEMPLOS DE USO\n")

    example_basic_analysis()
    example_llm_filter()
    example_business_terms()
    example_anonymization_strategies()
    example_predictive_detection()
    example_batch_processing()
    example_export_configuration()

    print("\n" + "=" * 60)
    print("✅ Todos os exemplos executados com sucesso!")
    print("=" * 60)
