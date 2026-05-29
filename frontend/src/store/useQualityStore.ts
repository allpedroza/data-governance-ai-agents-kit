import { create } from 'zustand'

export interface QualityReport {
  score?: number;
  passed_rules?: number;
  failed_rules?: number;
  details?: Record<string, any>;
  [key: string]: any;
}

interface QualityState {
  report: QualityReport | null;
  isLoading: boolean;
  error: string | null;
  evaluate: (file: File) => Promise<void>;
}

import { getApiHeaders } from './apiHeaders';

export const useQualityStore = create<QualityState>((set) => ({
  report: null,
  isLoading: false,
  error: null,
  
  evaluate: async (file: File) => {
    set({ isLoading: true, error: null, report: null });
    try {
      const formData = new FormData();
      formData.append('file', file);

      const headers = getApiHeaders();
      delete headers['Content-Type'];

      const response = await fetch('http://localhost:8000/api/v1/quality/evaluate', {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de Qualidade');
      }
      
      const data = await response.json();
      set({ report: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
