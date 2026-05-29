"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { useTaxonomyStore } from "@/store/useTaxonomyStore";
import { useSettingsStore } from "@/store/useSettingsStore";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import {
  Network, Database, Sparkles, ShieldCheck, Star, Activity,
  CheckCircle, AlertTriangle, AlertCircle, Download, RotateCcw,
  ArrowRight, Loader2, Table2, Columns3, Settings, ExternalLink,
  Search, Filter,
} from "lucide-react";

const STEPS = [
  { num: 1, label: "Explore Lake", icon: Database },
  { num: 2, label: "Generate YAML", icon: Sparkles },
  { num: 3, label: "Evaluate & Score", icon: ShieldCheck },
];

export default function TaxonomyPage() {
  const store = useTaxonomyStore();
  const settings = useSettingsStore();

  // Check if warehouse is configured
  const isWarehouseConfigured = !!(settings.warehouseHost && settings.warehouseUser);

  // Selection state for tables  
  const [selectedTables, setSelectedTables] = useState<Set<string>>(new Set());
  const [tableFilter, setTableFilter] = useState("");
  const [dbOverride, setDbOverride] = useState("");
  const [schemaOverride, setSchemaOverride] = useState("");

  // Auto-explore on mount if warehouse is configured and we haven't explored yet
  useEffect(() => {
    if (isWarehouseConfigured && !store.lakeMeta && !store.isExploring && store.currentStep === 1) {
      store.explore(settings.warehouseType, dbOverride || undefined, schemaOverride || undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isWarehouseConfigured]);

  // When lakeMeta arrives, pre-select all tables
  useEffect(() => {
    if (store.lakeMeta) {
      const allNames = new Set<string>();
      store.lakeMeta.schemas?.forEach((s) =>
        s.tables.forEach((t) => allNames.add(t.full_name))
      );
      setSelectedTables(allNames);
    }
  }, [store.lakeMeta]);

  const handleReExplore = () => {
    store.reset();
    store.explore(settings.warehouseType, dbOverride || undefined, schemaOverride || undefined);
  };

  // Build filtered lake metadata (only selected tables) for generation
  const filteredLakeMeta = useMemo(() => {
    if (!store.lakeMeta) return null;
    return {
      ...store.lakeMeta,
      schemas: store.lakeMeta.schemas.map((s) => ({
        ...s,
        tables: s.tables.filter((t) => selectedTables.has(t.full_name)),
      })).filter((s) => s.tables.length > 0),
    };
  }, [store.lakeMeta, selectedTables]);

  const handleGenerate = async () => {
    if (!filteredLakeMeta || filteredLakeMeta.schemas.length === 0) {
      toast.error("Selecione ao menos uma tabela para gerar a taxonomia.");
      return;
    }
    // Override the store's lakeMeta with filtered version for generation
    const originalExplore = store.lakeMeta;
    store.explore; // keep reference
    // We need to temporarily set the filtered meta — let's call generate with it directly
    const headers = (() => {
      const s = useSettingsStore.getState();
      const h: Record<string, string> = { "Content-Type": "application/json" };
      if (s.openaiApiKey) h["x-openai-key"] = s.openaiApiKey;
      if (s.geminiApiKey) h["x-gemini-key"] = s.geminiApiKey;
      if (s.anthropicApiKey) h["x-anthropic-key"] = s.anthropicApiKey;
      if (s.deepseekApiKey) h["x-deepseek-key"] = s.deepseekApiKey;
      if (s.llmProvider) h["x-llm-provider"] = s.llmProvider;
      if (s.llmModel) h["x-llm-model"] = s.llmModel;
      if (s.warehouseType) h["x-warehouse-type"] = s.warehouseType;
      if (s.warehouseHost) h["x-warehouse-host"] = s.warehouseHost;
      if (s.warehouseUser) h["x-warehouse-user"] = s.warehouseUser;
      if (s.warehousePassword) h["x-warehouse-password"] = s.warehousePassword;
      return h;
    })();

    store.setStep(2);
    useTaxonomyStore.setState({ isGenerating: true, generateError: null, generatedYaml: '' });

    try {
      const response = await fetch("http://localhost:8000/api/v1/taxonomy/generate", {
        method: "POST",
        headers,
        body: JSON.stringify({ lake_metadata: filteredLakeMeta }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Falha ao gerar taxonomia via LLM");
      }

      const data = await response.json();
      useTaxonomyStore.setState({ generatedYaml: data.yaml_content, isGenerating: false, currentStep: 3 });
      toast.success("Taxonomia YAML gerada!");
    } catch (err: any) {
      useTaxonomyStore.setState({ generateError: err.message, isGenerating: false });
    }
  };

  const handleEvaluate = async () => {
    await store.evaluate();
    if (!store.evaluateError) toast.success("Avaliação concluída!");
  };

  const downloadHtml = () => {
    if (!store.htmlArtifact) return;
    const blob = new Blob([store.htmlArtifact], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "taxonomy_report.html";
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleTable = (fullName: string) => {
    setSelectedTables((prev) => {
      const next = new Set(prev);
      if (next.has(fullName)) {
        next.delete(fullName);
      } else {
        next.add(fullName);
      }
      return next;
    });
  };

  const toggleAll = () => {
    if (!store.lakeMeta) return;
    const allNames: string[] = [];
    store.lakeMeta.schemas.forEach((s) => s.tables.forEach((t) => allNames.push(t.full_name)));
    if (selectedTables.size === allNames.length) {
      setSelectedTables(new Set());
    } else {
      setSelectedTables(new Set(allNames));
    }
  };

  // Compute stats
  const totalTables = store.lakeMeta?.schemas?.reduce(
    (acc, s) => acc + (s.tables?.length || 0), 0
  ) || 0;

  const totalColumns = store.lakeMeta?.schemas?.reduce(
    (acc, s) => acc + s.tables.reduce((t, tbl) => t + (tbl.columns?.length || 0), 0), 0
  ) || 0;

  // Filter visible tables
  const visibleSchemas = useMemo(() => {
    if (!store.lakeMeta) return [];
    if (!tableFilter.trim()) return store.lakeMeta.schemas;
    const lower = tableFilter.toLowerCase();
    return store.lakeMeta.schemas.map((s) => ({
      ...s,
      tables: s.tables.filter((t) =>
        t.full_name.toLowerCase().includes(lower) ||
        t.name.toLowerCase().includes(lower) ||
        t.columns.some((c) => c.name.toLowerCase().includes(lower))
      ),
    })).filter((s) => s.tables.length > 0);
  }, [store.lakeMeta, tableFilter]);

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Network className="w-8 h-8 text-primary" />
            Taxonomy Evaluator
          </h1>
          <p className="text-muted-foreground mt-2">
            Conecte ao Lake, gere a taxonomia via IA e avalie a maturidade em 8 dimensões.
          </p>
        </div>
        {(store.lakeMeta || store.currentStep > 1) && (
          <Button variant="outline" size="sm" onClick={() => { store.reset(); setSelectedTables(new Set()); setTableFilter(""); }} className="gap-2">
            <RotateCcw className="w-4 h-4" /> Reiniciar
          </Button>
        )}
      </div>

      {/* ─── NO WAREHOUSE: NOTIFICATION ──────────────────────── */}
      {!isWarehouseConfigured && (
        <Card className="border-orange-500/30 bg-orange-500/5">
          <CardContent className="py-8 flex flex-col items-center text-center space-y-4">
            <div className="w-14 h-14 rounded-full bg-orange-500/15 flex items-center justify-center">
              <AlertTriangle className="w-7 h-7 text-orange-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Warehouse não configurado</h3>
              <p className="text-muted-foreground text-sm mt-1 max-w-md">
                Para explorar seu Data Lake e gerar a taxonomia automaticamente, configure as credenciais do Warehouse em Settings.
              </p>
            </div>
            <Link href="/settings">
              <Button variant="default" className="gap-2">
                <Settings className="w-4 h-4" /> Ir para Settings
                <ExternalLink className="w-3 h-3" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* ─── WAREHOUSE CONFIGURED: MAIN FLOW ─────────────────── */}
      {isWarehouseConfigured && (
        <>
          {/* Step Indicator — only show after we have data or are past step 1 */}
          {(store.lakeMeta || store.currentStep > 1) && (
            <div className="flex items-center justify-center gap-2">
              {STEPS.map((step, idx) => {
                const isActive = store.currentStep === step.num;
                const isDone = store.currentStep > step.num;
                return (
                  <div key={step.num} className="flex items-center gap-2">
                    <button
                      onClick={() => isDone && store.setStep(step.num)}
                      disabled={!isDone}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        isActive
                          ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                          : isDone
                            ? "bg-primary/10 text-primary cursor-pointer hover:bg-primary/20"
                            : "bg-secondary/50 text-muted-foreground"
                      }`}
                    >
                      {isDone ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <step.icon className="w-4 h-4" />
                      )}
                      {step.label}
                    </button>
                    {idx < STEPS.length - 1 && (
                      <ArrowRight className="w-4 h-4 text-muted-foreground/50" />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* ─── LOADING STATE ─────────────────────────────── */}
          {store.isExploring && (
            <Card className="bg-card">
              <CardContent className="py-12 flex flex-col items-center space-y-4">
                <Loader2 className="w-10 h-10 text-primary animate-spin" />
                <div className="text-center">
                  <h3 className="text-lg font-semibold">Explorando o Data Lake...</h3>
                  <p className="text-muted-foreground text-sm mt-1">
                    Conectando a <span className="font-mono text-foreground">{settings.warehouseHost}</span> via <span className="capitalize text-foreground">{settings.warehouseType}</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ─── EXPLORE ERROR ─────────────────────────────── */}
          {store.exploreError && !store.isExploring && (
            <Card className="border-destructive/30 bg-destructive/5">
              <CardContent className="py-6 space-y-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold text-destructive">Falha ao conectar ao Warehouse</h3>
                    <p className="text-sm text-muted-foreground mt-1">{store.exploreError}</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Button variant="outline" size="sm" onClick={handleReExplore} className="gap-2">
                    <RotateCcw className="w-4 h-4" /> Tentar Novamente
                  </Button>
                  <Link href="/settings">
                    <Button variant="ghost" size="sm" className="gap-2">
                      <Settings className="w-4 h-4" /> Verificar Settings
                    </Button>
                  </Link>
                </div>

                {/* Optional refinement for re-explore */}
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border/50">
                  <div className="space-y-1">
                    <Label className="text-xs">Database (opcional)</Label>
                    <Input
                      placeholder="e.g. ANALYTICS_DB"
                      value={dbOverride}
                      onChange={(e) => setDbOverride(e.target.value)}
                      className="h-8 text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Schema (opcional)</Label>
                    <Input
                      placeholder="e.g. PUBLIC"
                      value={schemaOverride}
                      onChange={(e) => setSchemaOverride(e.target.value)}
                      className="h-8 text-sm"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ─── STEP 1: TABLE SELECTION ──────────────────────── */}
          {store.lakeMeta && store.currentStep === 1 && (
            <div className="space-y-4">
              {/* Summary metrics */}
              <div className="grid grid-cols-4 gap-4">
                <Card className="bg-card">
                  <CardContent className="pt-5 pb-4 text-center">
                    <div className="text-2xl font-bold">{store.lakeMeta.schemas?.length || 0}</div>
                    <div className="text-xs text-muted-foreground mt-1">Schemas</div>
                  </CardContent>
                </Card>
                <Card className="bg-card">
                  <CardContent className="pt-5 pb-4 text-center">
                    <div className="text-2xl font-bold flex items-center justify-center gap-1.5">
                      <Table2 className="w-4 h-4 text-primary" /> {totalTables}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Tabelas</div>
                  </CardContent>
                </Card>
                <Card className="bg-card">
                  <CardContent className="pt-5 pb-4 text-center">
                    <div className="text-2xl font-bold flex items-center justify-center gap-1.5">
                      <Columns3 className="w-4 h-4 text-emerald-500" /> {totalColumns}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Colunas</div>
                  </CardContent>
                </Card>
                <Card className="bg-card border-primary/20">
                  <CardContent className="pt-5 pb-4 text-center">
                    <div className="text-2xl font-bold text-primary">{selectedTables.size}</div>
                    <div className="text-xs text-muted-foreground mt-1">Selecionadas</div>
                  </CardContent>
                </Card>
              </div>

              {/* Table selection card */}
              <Card className="bg-card">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Filter className="w-5 h-5 text-primary" />
                        Selecione as Tabelas para Taxonomia
                      </CardTitle>
                      <CardDescription>
                        Refine quais tabelas a LLM deve considerar ao gerar o YAML de taxonomia.
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="sm" onClick={toggleAll} className="text-xs">
                      {selectedTables.size === totalTables ? "Desmarcar Todas" : "Selecionar Todas"}
                    </Button>
                  </div>
                  <div className="relative mt-2">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Filtrar tabelas ou colunas..."
                      value={tableFilter}
                      onChange={(e) => setTableFilter(e.target.value)}
                      className="pl-9 h-9"
                    />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="max-h-[350px] overflow-y-auto space-y-1 pr-1">
                    {visibleSchemas.map((s) => (
                      <div key={s.name}>
                        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2 py-1.5 bg-secondary/30 rounded sticky top-0 z-10">
                          {s.name}
                        </div>
                        {s.tables.map((t) => {
                          const isChecked = selectedTables.has(t.full_name);
                          return (
                            <label
                              key={t.full_name}
                              className={`flex items-center gap-3 px-3 py-2 rounded-md cursor-pointer transition-all text-sm hover:bg-secondary/40 ${
                                isChecked ? "bg-primary/5" : ""
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={isChecked}
                                onChange={() => toggleTable(t.full_name)}
                                className="rounded border-border accent-primary"
                              />
                              <span className="font-mono font-medium flex-1">{t.full_name}</span>
                              <span className="text-xs text-muted-foreground bg-secondary/50 px-2 py-0.5 rounded-full">
                                {t.columns.length} cols
                              </span>
                              <span className="text-xs text-muted-foreground">{t.type}</span>
                            </label>
                          );
                        })}
                      </div>
                    ))}

                    {visibleSchemas.length === 0 && (
                      <div className="text-center text-muted-foreground py-8 text-sm">
                        Nenhuma tabela encontrada para o filtro aplicado.
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Refine + Generate button */}
              <div className="flex gap-3">
                <Button variant="outline" onClick={handleReExplore} className="gap-2">
                  <RotateCcw className="w-4 h-4" /> Re-explorar
                </Button>
                <Button
                  onClick={handleGenerate}
                  disabled={store.isGenerating || selectedTables.size === 0}
                  className="flex-1 h-11 gap-2"
                >
                  {store.isGenerating ? (
                    <><Loader2 className="w-4 h-4 animate-spin" /> Gerando taxonomia via LLM...</>
                  ) : (
                    <><Sparkles className="w-4 h-4" /> Gerar Taxonomia YAML ({selectedTables.size} tabelas)</>
                  )}
                </Button>
              </div>
            </div>
          )}

          {/* ─── STEP 2: GENERATING (loading) ─────────────────── */}
          {store.currentStep === 2 && store.isGenerating && (
            <Card className="bg-card">
              <CardContent className="py-12 flex flex-col items-center space-y-4">
                <Loader2 className="w-10 h-10 text-primary animate-spin" />
                <div className="text-center">
                  <h3 className="text-lg font-semibold">Gerando Taxonomia via LLM...</h3>
                  <p className="text-muted-foreground text-sm mt-1">
                    Analisando {selectedTables.size} tabelas e classificando conceitos, aliases e naming conventions.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {store.currentStep === 2 && store.generateError && (
            <Card className="border-destructive/30 bg-destructive/5">
              <CardContent className="py-6 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-destructive">Falha ao gerar taxonomia</h3>
                  <p className="text-sm text-muted-foreground mt-1">{store.generateError}</p>
                  <Button variant="outline" size="sm" onClick={() => store.setStep(1)} className="mt-3 gap-2">
                    <ArrowRight className="w-4 h-4 rotate-180" /> Voltar à seleção
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ─── STEP 3: EVALUATE ────────────────────────────── */}
          {store.currentStep === 3 && (
            <div className="space-y-6">
              {/* YAML preview (editable) */}
              <Card className="bg-card">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    Taxonomia Gerada (editável)
                  </CardTitle>
                  <CardDescription>
                    Revise e ajuste o YAML antes de avaliar. Gerado a partir de {selectedTables.size} tabelas do lake.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Textarea
                    className="font-mono text-xs min-h-[200px] max-h-[300px] bg-secondary/30"
                    value={store.generatedYaml}
                    onChange={(e) => store.setGeneratedYaml(e.target.value)}
                    disabled={store.isEvaluating}
                  />
                </CardContent>
              </Card>

              {store.evaluateError && (
                <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-lg">
                  {store.evaluateError}
                </div>
              )}

              {!store.score && (
                <Button
                  onClick={handleEvaluate}
                  disabled={store.isEvaluating || !store.generatedYaml.trim()}
                  className="w-full h-11 gap-2"
                >
                  {store.isEvaluating ? (
                    <><Loader2 className="w-4 h-4 animate-spin" /> Avaliando...</>
                  ) : (
                    <><ShieldCheck className="w-4 h-4" /> Avaliar Maturidade (8 Dimensões)</>
                  )}
                </Button>
              )}

              {/* Results */}
              {store.score && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-2xl font-bold">Relatório de Avaliação</h2>
                    {store.htmlArtifact && (
                      <Button variant="outline" onClick={downloadHtml} className="gap-2">
                        <Download className="w-4 h-4" /> Download HTML Artifact
                      </Button>
                    )}
                  </div>

                  {/* Score cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-card border-primary/20 shadow-[0_0_15px_rgba(79,70,229,0.1)]">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                          <Star className="w-4 h-4 mr-2 text-primary" /> Overall Score
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-4xl font-bold">{store.score.overall_score?.toFixed(1)} / 100</div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                          <Activity className="w-4 h-4 mr-2 text-emerald-500" /> Maturity Level
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold text-emerald-500">
                          Lvl {store.score.maturity_level}: {store.score.maturity_label}
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                          <ShieldCheck className="w-4 h-4 mr-2 text-blue-400" /> Gaps
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">{store.score.gaps?.length || 0} Gaps</div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Dimension scores */}
                  {store.score.dimension_scores && (
                    <Card className="bg-card">
                      <CardHeader>
                        <CardTitle className="text-xl">Scores por Dimensão</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {Object.entries(store.score.dimension_scores).map(([dim, val]: [string, any]) => (
                            <div key={dim} className="flex flex-col space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="capitalize">{dim.replace(/_/g, ' ')}</span>
                                <span className="font-medium">{val.toFixed(1)} / 100</span>
                              </div>
                              <div className="w-full bg-secondary rounded-full h-2">
                                <div
                                  className="bg-primary h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${val}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Recommendations */}
                  {store.score.recommendations?.length > 0 && (
                    <Card className="bg-card">
                      <CardHeader>
                        <CardTitle className="text-xl">Recomendações TO-BE</CardTitle>
                        <CardDescription>Plano de ação baseado nos gaps mais críticos.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {store.score.recommendations.map((rec: any, idx: number) => (
                            <div key={idx} className="p-4 rounded-lg bg-secondary/30 border border-border/50">
                              <div className="flex items-center justify-between mb-2">
                                <h3 className="font-semibold text-lg flex items-center gap-2">
                                  {rec.effort === "high" ? <AlertCircle className="w-5 h-5 text-destructive" /> :
                                    rec.effort === "medium" ? <AlertTriangle className="w-5 h-5 text-orange-500" /> :
                                      <CheckCircle className="w-5 h-5 text-emerald-500" />}
                                  {rec.priority}. {rec.title}
                                </h3>
                                <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full font-medium">
                                  +{rec.expected_score_impact} pts
                                </span>
                              </div>
                              <p className="text-sm text-muted-foreground mb-2">{rec.description}</p>
                              <div className="flex gap-4 text-xs font-medium">
                                <span className="text-muted-foreground">
                                  Dimensão: <span className="text-foreground capitalize">{rec.dimension?.replace('_', ' ')}</span>
                                </span>
                                <span className="text-muted-foreground">
                                  Esforço: <span className="text-foreground capitalize">{rec.effort}</span>
                                </span>
                                <span className="text-muted-foreground">
                                  Prazo: <span className="text-foreground capitalize">{rec.timeline}</span>
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
