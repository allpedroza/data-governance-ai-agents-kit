"use client";

import { useState, useRef } from "react";
import { useEnrichmentStore } from "@/store/useEnrichmentStore";
import { SettingsGuard } from "@/components/SettingsGuard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import {
  Tags, Upload, Loader2, AlertCircle, ShieldAlert, CheckCircle,
  Download, RotateCcw, Database, Lock, Eye, Sparkles, FileText,
} from "lucide-react";

const DOMAIN_COLORS: Record<string, string> = {
  customer: "bg-blue-500/15 text-blue-500",
  sales: "bg-green-500/15 text-green-500",
  finance: "bg-amber-500/15 text-amber-500",
  product: "bg-purple-500/15 text-purple-500",
  marketing: "bg-pink-500/15 text-pink-500",
  hr: "bg-orange-500/15 text-orange-500",
  operations: "bg-cyan-500/15 text-cyan-500",
  analytics: "bg-indigo-500/15 text-indigo-500",
  general: "bg-gray-500/15 text-gray-400",
};

const CLASS_COLORS: Record<string, string> = {
  public: "bg-green-500/15 text-green-500",
  internal: "bg-blue-500/15 text-blue-500",
  confidential: "bg-orange-500/15 text-orange-500",
  restricted: "bg-red-500/15 text-red-500",
};

export default function EnrichmentPage() {
  const store = useEnrichmentStore();
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleEnrich = async () => {
    if (!file) {
      toast.error("Faça o upload de um arquivo CSV");
      return;
    }
    await store.enrich(file);
    if (!store.error) toast.success("Metadados enriquecidos com sucesso!");
  };

  const handleReset = () => {
    store.reset();
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const downloadJson = () => {
    if (!store.result) return;
    const blob = new Blob([JSON.stringify(store.result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${store.result.table_name}_metadata.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const r = store.result;

  return (
    <SettingsGuard require="llm" moduleName="Metadata Enrichment">
      <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
              <Tags className="w-8 h-8 text-primary" />
              Metadata Enrichment
            </h1>
            <p className="text-muted-foreground mt-2">
              Gere descrições, tags, classificações e detecção de PII automaticamente via IA.
            </p>
          </div>
          {r && (
            <Button variant="outline" size="sm" onClick={handleReset} className="gap-2">
              <RotateCcw className="w-4 h-4" /> Nova Análise
            </Button>
          )}
        </div>

        {/* Upload Card */}
        {!r && (
          <Card className="bg-card">
            <CardHeader>
              <CardTitle className="text-xl flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary" />
                Upload de Dataset
              </CardTitle>
              <CardDescription>
                Envie um arquivo CSV. O agente irá amostrar os dados, detectar padrões, e usar a LLM para gerar metadados enriquecidos.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="border-2 border-dashed border-border rounded-xl p-8 text-center hover:border-primary/40 transition-colors cursor-pointer"
                onClick={() => inputRef.current?.click()}
              >
                <Upload className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-sm text-muted-foreground">
                  {file ? (
                    <span className="text-foreground font-medium">{file.name} ({(file.size / 1024).toFixed(0)} KB)</span>
                  ) : (
                    "Clique para selecionar ou arraste um arquivo CSV"
                  )}
                </p>
                <input
                  ref={inputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
              </div>

              {store.error && (
                <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-lg flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {store.error}
                </div>
              )}

              <Button
                onClick={handleEnrich}
                disabled={store.isLoading || !file}
                className="w-full h-11 gap-2"
              >
                {store.isLoading ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Analisando e enriquecendo...</>
                ) : (
                  <><Sparkles className="w-4 h-4" /> Enriquecer Metadados</>
                )}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Results ────────────────────────────────────────── */}
        {r && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
            {/* Table-level summary */}
            <Card className="bg-card border-primary/20 shadow-[0_0_20px_rgba(79,70,229,0.08)]">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-2xl">{r.business_name}</CardTitle>
                    <CardDescription className="font-mono mt-1">{r.table_name}</CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={downloadJson} className="gap-2">
                      <Download className="w-4 h-4" /> JSON
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-foreground leading-relaxed">{r.description}</p>
                <p className="text-xs text-muted-foreground italic">{r.description_en}</p>

                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 pt-2">
                  <div className="text-center p-3 rounded-lg bg-secondary/30">
                    <Database className="w-4 h-4 mx-auto text-primary mb-1" />
                    <div className="text-lg font-bold">{r.row_count?.toLocaleString()}</div>
                    <div className="text-xs text-muted-foreground">Linhas</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-secondary/30">
                    <Tags className="w-4 h-4 mx-auto text-emerald-500 mb-1" />
                    <div className="text-lg font-bold">{r.column_count}</div>
                    <div className="text-xs text-muted-foreground">Colunas</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-secondary/30">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-bold ${DOMAIN_COLORS[r.domain] || DOMAIN_COLORS.general}`}>
                      {r.domain}
                    </span>
                    <div className="text-xs text-muted-foreground mt-1">Domínio</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-secondary/30">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-bold ${CLASS_COLORS[r.classification] || ""}`}>
                      {r.classification}
                    </span>
                    <div className="text-xs text-muted-foreground mt-1">Classificação</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-secondary/30">
                    <Eye className="w-4 h-4 mx-auto text-amber-500 mb-1" />
                    <div className="text-lg font-bold">{(r.confidence * 100).toFixed(0)}%</div>
                    <div className="text-xs text-muted-foreground">Confiança</div>
                  </div>
                </div>

                {/* Tags */}
                {r.tags?.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {r.tags.map((tag: string) => (
                      <span key={tag} className="px-2 py-1 rounded-md bg-primary/10 text-primary text-xs font-medium">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {r.owner_suggestion && (
                  <div className="text-sm text-muted-foreground">
                    <span className="font-medium text-foreground">Proprietário sugerido:</span> {r.owner_suggestion}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* PII Warning */}
            {r.has_pii && (
              <Card className="border-red-500/30 bg-red-500/5">
                <CardContent className="py-4 flex items-center gap-3">
                  <ShieldAlert className="w-6 h-6 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-red-500">⚠️ Dados Pessoais Detectados (PII)</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Colunas com PII: <span className="text-foreground font-mono">{r.pii_columns?.join(", ")}</span>
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Columns table */}
            <Card className="bg-card">
              <CardHeader>
                <CardTitle className="text-xl flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  Colunas Enriquecidas ({r.columns?.length})
                </CardTitle>
                <CardDescription>
                  Descrições, classificações e detecção de PII por coluna.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">Coluna</th>
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">Descrição</th>
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">Tipo</th>
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">Classificação</th>
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">PII</th>
                        <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase">Tags</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.columns?.map((col: any, idx: number) => (
                        <tr key={idx} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
                          <td className="px-3 py-2.5">
                            <div className="font-mono font-medium text-foreground">{col.name}</div>
                            <div className="text-xs text-muted-foreground">{col.business_name}</div>
                          </td>
                          <td className="px-3 py-2.5 max-w-[250px]">
                            <div className="text-sm text-foreground">{col.description}</div>
                          </td>
                          <td className="px-3 py-2.5">
                            <span className="px-2 py-0.5 rounded bg-secondary/50 text-xs font-mono">
                              {col.semantic_type || "—"}
                            </span>
                          </td>
                          <td className="px-3 py-2.5">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${CLASS_COLORS[col.classification] || ""}`}>
                              {col.classification}
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-center">
                            {col.is_pii ? (
                              <Lock className="w-4 h-4 text-red-500 mx-auto" />
                            ) : (
                              <CheckCircle className="w-4 h-4 text-emerald-500 mx-auto" />
                            )}
                          </td>
                          <td className="px-3 py-2.5">
                            <div className="flex flex-wrap gap-1">
                              {col.tags?.slice(0, 3).map((tag: string) => (
                                <span key={tag} className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-xs">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </SettingsGuard>
  );
}
