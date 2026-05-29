import { create } from 'zustand';
import { getApiHeaders } from './apiHeaders';

interface EnrichmentState {
  result: any;
  isLoading: boolean;
  error: string | null;
  
  enrich: (file: File) => Promise<void>;
  reset: () => void;
}

export const useEnrichmentStore = create<EnrichmentState>((set) => ({
  result: null,
  isLoading: false,
  error: null,
  
  enrich: async (file: File) => {
    set({ isLoading: true, error: null, result: null });
    try {
      const headers = getApiHeaders();
      // Remove Content-Type so browser sets multipart/form-data boundary
      delete headers['Content-Type'];
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/api/v1/enrichment/enrich', {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Falha ao enriquecer metadados');
      }
      
      const data = await response.json();
      set({ result: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  },
  
  reset: () => set({ result: null, isLoading: false, error: null }),
}));
