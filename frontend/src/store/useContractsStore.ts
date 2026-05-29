import { create } from 'zustand'

export interface ContractReport {
  is_valid?: boolean;
  errors?: string[];
  warnings?: string[];
  [key: string]: any;
}

interface ContractsState {
  report: ContractReport | null;
  isLoading: boolean;
  error: string | null;
  validate: (file: File, contractYaml: string) => Promise<void>;
}

import { getApiHeaders } from './apiHeaders';

export const useContractsStore = create<ContractsState>((set) => ({
  report: null,
  isLoading: false,
  error: null,
  
  validate: async (file: File, contractYaml: string) => {
    set({ isLoading: true, error: null, report: null });
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('contract_yaml', contractYaml);

      const headers = getApiHeaders();

      const response = await fetch('http://localhost:8000/api/v1/contracts/validate', {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de Contratos');
      }
      
      const data = await response.json();
      set({ report: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
