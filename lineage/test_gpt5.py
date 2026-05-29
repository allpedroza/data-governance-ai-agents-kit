import os
from openai import OpenAI

# Configura a chave
api_key = os.environ.get("DATA_LINEAGE_OPENAI_API_KEY")
if not api_key:
    print("❌ Erro: Variável DATA_LINEAGE_OPENAI_API_KEY não definida.")
    exit()

print("🚀 Testando GPT-5 com a nova Responses API...")

client = OpenAI(api_key=api_key)

try:
    # Usando a nova sintaxe baseada na documentação que você enviou
    response = client.responses.create(
        model="gpt-5-nano",  # Usando o nano para teste rápido
        input="Explain simply what a Data Lineage is in one sentence.",
        reasoning={"effort": "low"} # Novo parâmetro do GPT-5
    )
    
    # Acessando o output conforme a nova doc (output_text helper)
    print("\n✅ SUCESSO! O GPT-5 respondeu:\n")
    print(response.output_text)

except Exception as e:
    print(f"\n❌ FALHA: {e}")
