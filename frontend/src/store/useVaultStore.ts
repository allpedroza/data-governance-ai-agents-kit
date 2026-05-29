import { create } from 'zustand';
import { getApiHeaders } from './apiHeaders';

interface VaultState {
  result: any;
  isLoading: boolean;
  error: string | null;
  anonymize: (text: string) => Promise<void>;
}

export const useVaultStore = create<VaultState>((set) => ({
  result: null,
  isLoading: false,
  error: null,
  
  anonymize: async (text: string) => {
    if (!text.trim()) return;
    
    set({ isLoading: true, error: null, result: null });
    try {
      const response = await fetch('http://localhost:8000/api/v1/vault/anonymize', {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({ text, anonymize: true })
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API do Vault NER');
      }
      
      const data = await response.json();
      set({ result: data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
