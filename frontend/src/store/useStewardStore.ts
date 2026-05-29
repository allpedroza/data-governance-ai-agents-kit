import { create } from 'zustand';
import { getApiHeaders } from './apiHeaders';

interface StewardState {
  assignment: any;
  isLoading: boolean;
  error: string | null;
  assign: (assetName: string, metadata: any) => Promise<void>;
}

export const useStewardStore = create<StewardState>((set) => ({
  assignment: null,
  isLoading: false,
  error: null,
  
  assign: async (assetName: string, metadata: any) => {
    if (!assetName) return;
    
    set({ isLoading: true, error: null, assignment: null });
    try {
      const headers = getApiHeaders();

      const response = await fetch('http://localhost:8000/api/v1/steward/assign', {
        method: 'POST',
        headers,
        body: JSON.stringify({ asset_name: assetName, metadata })
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de Steward');
      }
      
      const data = await response.json();
      set({ assignment: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
