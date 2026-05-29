import { create } from 'zustand';
import { getApiHeaders } from './apiHeaders';

interface LakeSchema {
  name: string;
  tables: {
    name: string;
    full_name: string;
    type: string;
    columns: { name: string; type: string; nullable: boolean; comment: string }[];
  }[];
}

interface LakeMetadata {
  warehouse_type: string;
  database: string | null;
  schemas: LakeSchema[];
}

interface TaxonomyState {
  // Step tracking
  currentStep: number;
  
  // Step 1: Explore
  lakeMeta: LakeMetadata | null;
  isExploring: boolean;
  exploreError: string | null;
  
  // Step 2: Generate
  generatedYaml: string;
  isGenerating: boolean;
  generateError: string | null;
  
  // Step 3: Evaluate
  score: any;
  htmlArtifact: string | null;
  isEvaluating: boolean;
  evaluateError: string | null;
  
  // Actions
  setStep: (step: number) => void;
  explore: (warehouseType: string, database?: string, schemaName?: string) => Promise<void>;
  generate: () => Promise<void>;
  setGeneratedYaml: (yaml: string) => void;
  evaluate: () => Promise<void>;
  reset: () => void;
}

export const useTaxonomyStore = create<TaxonomyState>((set, get) => ({
  currentStep: 1,
  
  lakeMeta: null,
  isExploring: false,
  exploreError: null,
  
  generatedYaml: '',
  isGenerating: false,
  generateError: null,
  
  score: null,
  htmlArtifact: null,
  isEvaluating: false,
  evaluateError: null,
  
  setStep: (step) => set({ currentStep: step }),
  
  explore: async (warehouseType, database, schemaName) => {
    set({ isExploring: true, exploreError: null, lakeMeta: null });
    try {
      const headers = getApiHeaders();
      const body: any = { warehouse_type: warehouseType };
      if (database) body.database = database;
      if (schemaName) body.schema_name = schemaName;

      const response = await fetch('http://localhost:8000/api/v1/taxonomy/explore', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      });
      
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Falha ao explorar o warehouse');
      }
      
      const data = await response.json();
      set({ lakeMeta: data, isExploring: false, currentStep: 2 });
    } catch (err: any) {
      set({ exploreError: err.message, isExploring: false });
    }
  },
  
  generate: async () => {
    const { lakeMeta } = get();
    if (!lakeMeta) return;
    
    set({ isGenerating: true, generateError: null, generatedYaml: '' });
    try {
      const headers = getApiHeaders();
      const response = await fetch('http://localhost:8000/api/v1/taxonomy/generate', {
        method: 'POST',
        headers,
        body: JSON.stringify({ lake_metadata: lakeMeta }),
      });
      
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Falha ao gerar taxonomia via LLM');
      }
      
      const data = await response.json();
      set({ generatedYaml: data.yaml_content, isGenerating: false, currentStep: 3 });
    } catch (err: any) {
      set({ generateError: err.message, isGenerating: false });
    }
  },

  setGeneratedYaml: (yaml) => set({ generatedYaml: yaml }),
  
  evaluate: async () => {
    const { generatedYaml } = get();
    if (!generatedYaml.trim()) return;
    
    set({ isEvaluating: true, evaluateError: null, score: null, htmlArtifact: null });
    try {
      const headers = getApiHeaders();
      const response = await fetch('http://localhost:8000/api/v1/taxonomy/evaluate', {
        method: 'POST',
        headers,
        body: JSON.stringify({ yaml_content: generatedYaml }),
      });
      
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Falha ao avaliar a taxonomia');
      }
      
      const data = await response.json();
      const { html_artifact, ...scoreData } = data;
      set({ score: scoreData, htmlArtifact: html_artifact, isEvaluating: false });
    } catch (err: any) {
      set({ evaluateError: err.message, isEvaluating: false });
    }
  },
  
  reset: () => set({
    currentStep: 1,
    lakeMeta: null,
    isExploring: false,
    exploreError: null,
    generatedYaml: '',
    isGenerating: false,
    generateError: null,
    score: null,
    htmlArtifact: null,
    isEvaluating: false,
    evaluateError: null,
  }),
}));
