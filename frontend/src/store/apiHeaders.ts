import { useSettingsStore } from './useSettingsStore';

export function getApiHeaders(): Record<string, string> {
  const state = useSettingsStore.getState();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  
  if (state.llmProvider) headers['x-llm-provider'] = state.llmProvider;
  if (state.llmModel) headers['x-llm-model'] = state.llmModel;
  
  if (state.openaiApiKey) headers['x-openai-key'] = state.openaiApiKey;
  if (state.geminiApiKey) headers['x-gemini-key'] = state.geminiApiKey;
  if (state.anthropicApiKey) headers['x-anthropic-key'] = state.anthropicApiKey;
  if (state.deepseekApiKey) headers['x-deepseek-key'] = state.deepseekApiKey;
  
  if (state.warehouseType) headers['x-warehouse-type'] = state.warehouseType;
  if (state.warehouseHost) headers['x-warehouse-host'] = state.warehouseHost;
  if (state.warehouseUser) headers['x-warehouse-user'] = state.warehouseUser;
  if (state.warehousePassword) headers['x-warehouse-password'] = state.warehousePassword;
  
  if (state.catalogType) headers['x-catalog-type'] = state.catalogType;
  if (state.catalogHost) headers['x-catalog-host'] = state.catalogHost;
  if (state.catalogToken) headers['x-catalog-token'] = state.catalogToken;

  return headers;
}
