import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface SettingsState {
  llmProvider: string;
  llmModel: string;
  openaiApiKey: string;
  geminiApiKey: string;
  anthropicApiKey: string;
  deepseekApiKey: string;
  deepseekModel: string;
  warehouseType: string;
  warehouseHost?: string;
  warehouseUser?: string;
  warehousePassword?: string;
  catalogType: string;
  catalogHost?: string;
  catalogToken?: string;
  setLLMProvider: (provider: string) => void;
  setLLMModel: (model: string) => void;
  setOpenAIApiKey: (key: string) => void;
  setGeminiApiKey: (key: string) => void;
  setAnthropicApiKey: (key: string) => void;
  setDeepseekApiKey: (key: string) => void;
  setDeepseekModel: (model: string) => void;
  setWarehouseType: (type: string) => void;
  setWarehouseHost: (host: string) => void;
  setWarehouseUser: (user: string) => void;
  setWarehousePassword: (pwd: string) => void;
  setCatalogType: (type: string) => void;
  setCatalogHost: (host: string) => void;
  setCatalogToken: (token: string) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      llmProvider: "openai",
      llmModel: "gpt-4o",
      openaiApiKey: "",
      geminiApiKey: "",
      anthropicApiKey: "",
      deepseekApiKey: "",
      deepseekModel: "deepseek-chat",
      warehouseType: "snowflake",
      warehouseHost: "",
      warehouseUser: "",
      warehousePassword: "",
      catalogType: "openmetadata",
      catalogHost: "",
      catalogToken: "",
      setLLMProvider: (provider) => set({ llmProvider: provider }),
      setLLMModel: (model) => set({ llmModel: model }),
      setOpenAIApiKey: (key) => set({ openaiApiKey: key }),
      setGeminiApiKey: (key) => set({ geminiApiKey: key }),
      setAnthropicApiKey: (key) => set({ anthropicApiKey: key }),
      setDeepseekApiKey: (key) => set({ deepseekApiKey: key }),
      setDeepseekModel: (model) => set({ deepseekModel: model }),
      setWarehouseType: (type) => set({ warehouseType: type }),
      setWarehouseHost: (host) => set({ warehouseHost: host }),
      setWarehouseUser: (user) => set({ warehouseUser: user }),
      setWarehousePassword: (pwd) => set({ warehousePassword: pwd }),
      setCatalogType: (type) => set({ catalogType: type }),
      setCatalogHost: (host) => set({ catalogHost: host }),
      setCatalogToken: (token) => set({ catalogToken: token }),
    }),
    {
      name: 'datagov-settings',
    }
  )
)
