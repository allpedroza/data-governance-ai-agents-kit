import { create } from 'zustand';
import { getApiHeaders } from './apiHeaders';

interface AssetValueState {
  report: any;
  isLoading: boolean;
  error: string | null;
  analyze: (logs: any[]) => Promise<void>;
}

export const useAssetValueStore = create<AssetValueState>((set) => ({
  report: null,
  isLoading: false,
  error: null,
  
  analyze: async (logs: any[]) => {
    if (!logs || logs.length === 0) return;
    
    set({ isLoading: true, error: null, report: null });
    try {
      const headers = getApiHeaders();

      const response = await fetch('http://localhost:8000/api/v1/value/analyze', {
        method: 'POST',
        headers,
        body: JSON.stringify({ query_logs: logs })
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de Asset Value');
      }
      
      const data = await response.json();
      set({ report: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
