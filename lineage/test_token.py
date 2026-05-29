import os
from openai import OpenAI

# Pega a chave que você exportou
api_key = os.environ.get("DATA_LINEAGE_OPENAI_API_KEY")
model = os.environ.get("DATA_LINEAGE_LLM_MODEL", "gpt-3.5-turbo")

print(f"🔑 Testando chave: {api_key[:5]}...{api_key[-4:] if api_key else 'NENHUMA'}")
print(f"🤖 Testando modelo: {model}")

if not api_key:
    print("❌ ERRO: A variável DATA_LINEAGE_OPENAI_API_KEY não está definida.")
    exit()

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Teste de conexão. Responda apenas 'OK'."}]
    )
    print(f"✅ SUCESSO! Resposta da API: {response.choices[0].message.content}")
except Exception as e:
    print(f"\n❌ FALHA: {e}")
