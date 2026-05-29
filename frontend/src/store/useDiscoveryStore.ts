import { create } from 'zustand'

export interface DiscoveryResult {
  asset_id: string;
  name: string;
  description: string;
  type: string;
  domain: string;
  relevance_score?: number;
  tags?: string[];
  owner?: string;
  steward?: string;
}

interface DiscoveryState {
  results: DiscoveryResult[];
  isLoading: boolean;
  error: string | null;
  search: (query: string) => Promise<void>;
}

import { getApiHeaders } from './apiHeaders';

export const useDiscoveryStore = create<DiscoveryState>((set) => ({
  results: [],
  isLoading: false,
  error: null,
  
  search: async (query: string) => {
    if (!query.trim()) return;
    
    set({ isLoading: true, error: null });
    try {
      const headers = getApiHeaders();

      const response = await fetch('http://localhost:8000/api/v1/discovery', {
        method: 'POST',
        headers,
        body: JSON.stringify({ query, limit: 5 })
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API');
      }
      
      const data = await response.json();
      set({ results: data.results, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
