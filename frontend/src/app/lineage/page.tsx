"use client";

import { useState } from "react";
import { useLineageStore } from "@/store/useLineageStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import CytoscapeComponent from 'react-cytoscapejs';
import { Network, Upload, ServerCrash, ArrowRight, ArrowDown } from "lucide-react";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function LineagePage() {
  const [files, setFiles] = useState<File[]>([]);
  const { analysis, cytoscapeData, isLoading, error, analyze } = useLineageStore();

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) {
      toast.error("Envie os arquivos do pipeline");
      return;
    }
    await analyze(files);
  };

  return (
    <SettingsGuard require="llm" moduleName="Data Lineage">
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Lineage & Impact</h1>
        <p className="text-muted-foreground mt-2">
          Análise de impacto downstream e upstream usando Agentes de Inteligência Artificial.
        </p>
      </div>

      <Card className="bg-card">
        <CardContent className="pt-6">
          <form onSubmit={handleAnalyze} className="flex gap-4 items-center">
            <div className="flex-1">
              <Input 
                type="file"
                multiple
                accept=".py,.sql,.json,.yaml,.yml,.txt"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
                className="bg-background cursor-pointer"
                disabled={isLoading}
              />
              <p className="text-xs text-muted-foreground mt-2">
                Envie scripts SQL, Python, DBT ou arquivos de definição para analise interativa.
              </p>
            </div>
            <Button type="submit" disabled={isLoading || files.length === 0} className="h-10 px-8 self-start mt-1">
              {isLoading ? "Processando..." : "Gerar Grafo"}
              {!isLoading && <Network className="ml-2 w-4 h-4" />}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive bg-destructive/10">
          <CardContent className="pt-6 flex items-center gap-3 text-destructive">
            <ServerCrash className="h-5 w-5" />
            <p className="text-sm font-medium">{error}</p>
          </CardContent>
        </Card>
      )}

      {cytoscapeData && (
        <Card className="bg-card border-border shadow-sm mt-8">
          <CardHeader>
            <CardTitle className="text-lg">Grafo de Arquitetura em Camadas</CardTitle>
            <CardDescription>Vizualização de dependências upstream e downstream detectadas nos scripts submetidos.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[600px] w-full border border-border rounded-lg bg-white/5 relative overflow-hidden">
              <CytoscapeComponent 
                elements={[...cytoscapeData.nodes, ...cytoscapeData.edges]} 
                style={ { width: '100%', height: '100%' } }
                layout={ { name: 'cose' } } // Fallback simple layout
                stylesheet={[
                  {
                    selector: 'node',
                    style: {
                      'background-color': 'data(color)',
                      'label': 'data(label)',
                      'color': '#fff',
                      'text-valign': 'center',
                      'text-halign': 'center',
                      'font-size': '10px',
                      'shape': 'round-rectangle',
                      'width': '120px',
                      'height': '40px'
                    }
                  },
                  {
                    selector: 'edge',
                    style: {
                      'width': 2,
                      'line-color': '#ccc',
                      'target-arrow-color': '#ccc',
                      'target-arrow-shape': 'triangle',
                      'curve-style': 'bezier'
                    }
                  }
                ]}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {analysis && analysis.assets && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          <Card className="md:col-span-2 bg-card border-border shadow-sm">
            <CardContent className="pt-6">
              <h3 className="font-semibold text-lg mb-2">Relatório de Análise Bruta</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                O agente detectou {analysis.assets.length} ativos e {analysis.transformations.length} transformações neste pipeline.
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
    </SettingsGuard>
  );
}
