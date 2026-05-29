import { create } from 'zustand'

export interface LineageResult {
  asset_name: string;
  impact_level?: string;
  downstream_impacts: string[];
  upstream_sources: string[];
  business_impact?: string;
}

interface LineageState {
  analysis: any;
  cytoscapeData: any;
  isLoading: boolean;
  error: string | null;
  analyze: (files: File[]) => Promise<void>;
}

import { getApiHeaders } from './apiHeaders';

export const useLineageStore = create<LineageState>((set) => ({
  analysis: null,
  cytoscapeData: null,
  isLoading: false,
  error: null,
  
  analyze: async (files: File[]) => {
    if (!files || files.length === 0) return;
    
    set({ isLoading: true, error: null, analysis: null, cytoscapeData: null });
    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const headers = getApiHeaders();
      // Remove Content-Type since FormData automatically sets it with boundary
      delete headers['Content-Type'];

      const response = await fetch('http://localhost:8000/api/v1/lineage/analyze', {
        method: 'POST',
        headers,
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Falha ao comunicar com a API de linhagem');
      }
      
      const data = await response.json();
      set({ analysis: data.raw_results, cytoscapeData: data.cytoscape_data, isLoading: false });
    } catch (err: any) {
      set({ error: err.message, isLoading: false });
    }
  }
}));
