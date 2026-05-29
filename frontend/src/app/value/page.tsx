"use client";

import { useState } from "react";
import { useAssetValueStore } from "@/store/useAssetValueStore";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Calculator, DollarSign, Activity, AlertCircle, BarChart3, TrendingUp, TrendingDown } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function AssetValuePage() {
  const [logs, setLogs] = useState("");
  const { report, isLoading, error, analyze } = useAssetValueStore();

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const parsed = JSON.parse(logs);
      if (!Array.isArray(parsed)) throw new Error("Deve ser um array JSON");
      await analyze(parsed);
    } catch (e: any) {
      toast.error("JSON Inválido: " + e.message);
    }
  };

  return (
    <SettingsGuard require="both" moduleName="Data Asset Value">
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-3">
          <DollarSign className="w-8 h-8 text-primary" />
          Data Asset Value
        </h1>
        <p className="text-muted-foreground mt-2">
          Estime o valor financeiro e a importância estratégica dos seus ativos de dados baseado no histórico de consultas (Query Logs).
        </p>
      </div>

      <Card className="bg-card">
        <CardContent className="pt-6">
          <form onSubmit={handleAnalyze} className="flex flex-col gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Cole o histórico de logs de consulta (JSON)</label>
              <Textarea 
                value={logs}
                onChange={(e) => setLogs(e.target.value)}
                placeholder={`[\n  {\n    "query": "SELECT * FROM...",\n    "user": "analyst@company.com"\n  }\n]`}
                className="font-mono text-xs h-40 bg-background"
                disabled={isLoading}
              />
            </div>
            <Button type="submit" disabled={isLoading} className="h-10 px-8 self-end">
              {isLoading ? "Calculando Valor..." : "Analisar Valor do Ativo"}
              {!isLoading && <Calculator className="ml-2 w-4 h-4" />}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <div className="bg-destructive/10 border border-destructive/20 text-destructive p-4 rounded-md flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {report && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
          <Card className="md:col-span-1 bg-card border-border shadow-sm">
            <CardHeader className="bg-secondary/50 rounded-t-lg border-b border-border">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                Saúde do Ecossistema
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 flex flex-col gap-4">
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Ativos Analisados</div>
                <div className="text-3xl font-bold text-foreground">{report.assets_analyzed}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Logs Processados</div>
                <div className="text-3xl font-bold text-foreground">{report.query_logs_processed}</div>
              </div>
            </CardContent>
          </Card>

          <Card className="md:col-span-3 bg-card border-border shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="w-5 h-5 text-primary"/> Ranking de Valor de Ativos</CardTitle>
            </CardHeader>
            <CardContent>
              {report.asset_scores && report.asset_scores.length > 0 ? (
                <div className="border border-border rounded-md overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 text-muted-foreground">
                      <tr>
                        <th className="px-4 py-3 font-medium">Ativo</th>
                        <th className="px-4 py-3 font-medium">Score Total</th>
                        <th className="px-4 py-3 font-medium">Uso</th>
                        <th className="px-4 py-3 font-medium">Data Products</th>
                        <th className="px-4 py-3 font-medium">Categoria</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.asset_scores.map((score: any, i: number) => (
                        <tr key={i} className="border-t border-border hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 font-medium text-primary">{score.asset_name}</td>
                          <td className="px-4 py-3 font-bold">{score.overall_value_score}%</td>
                          <td className="px-4 py-3 text-muted-foreground">{score.usage_score}%</td>
                          <td className="px-4 py-3 text-muted-foreground">{score.data_products_count}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                              score.value_category === 'critical' ? 'bg-destructive/20 text-destructive' :
                              score.value_category === 'high' ? 'bg-orange-500/20 text-orange-500' :
                              'bg-green-500/20 text-green-500'
                            }`}>
                              {score.value_category.toUpperCase()}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">Nenhum ativo validado no log.</p>
              )}
            </CardContent>
          </Card>
          
          {report.llm_review && report.llm_review.insights && (
            <Card className="md:col-span-4 bg-primary/5 border-primary/20 shadow-sm mt-4">
              <CardHeader>
                <CardTitle className="text-lg text-primary flex items-center gap-2">
                  <TrendingUp className="w-5 h-5"/> Insights do Agente de IA
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc pl-5 space-y-2 text-sm text-foreground/80">
                  {report.llm_review.insights.map((insight: string, i: number) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

        </div>
      )}
    </div>
    </SettingsGuard>
  );
}
