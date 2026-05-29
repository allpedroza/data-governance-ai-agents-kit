import { create } from 'zustand'

export interface ClassificationReport {
  columns?: Record<string, string>;
  dataset_level_tags?: string[];
  [key: string]: any;
}

interface ClassificationState {
  report: ClassificationReport | null;
  isLoading: boolean;
  error: string | null;
  classify: (file: File) => Promise<void>;
}

import { getApiHeaders } from './apiHeaders';

export const useClassificationStore = create<ClassificationState>((set) => ({
  report: null,
  isLoading: false,
  error: null,
  
  classify: async (file: File) => {
    set({ isLoading: true, error: null, report: null });
    try {
      const formData = new FormData();
      formData.append('file', file);

      const headers = getApiHeaders();
      delete headers['Content-Type'];

      const response = await fetch('http://localhost:8000/api/v1/classification/classify', {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de Classificação');
      }
      
      const data = await response.json();
      set({ report: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
