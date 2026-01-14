package ai.runtime

deny[msg] {
  input.event == "inference.request"
  input.provider.type == "external_llm"
  input.input_contains_pii == true
  msg := "PII não pode sair para provedores externos"
}
