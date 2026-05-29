"use client";

import { useState } from "react";
import { useQualityStore } from "@/store/useQualityStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { ShieldAlert, ServerCrash, Upload, CheckCircle, XCircle } from "lucide-react";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function QualityPage() {
  const [file, setFile] = useState<File | null>(null);
  const { report, isLoading, error, evaluate } = useQualityStore();

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Faça o upload de um arquivo CSV para avaliar");
      return;
    }
    await evaluate(file);
  };

  return (
    <SettingsGuard require="llm" moduleName="Data Quality">
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Quality</h1>
        <p className="text-muted-foreground mt-2">
          Avaliação ativa de metadados e distribuição de dados usando LLMs.
        </p>
      </div>

      <Card className="bg-card">
        <CardContent className="pt-6">
          <form onSubmit={handleAnalyze} className="flex gap-4 items-center">
            <div className="flex-1">
              <Input 
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="bg-background cursor-pointer"
                disabled={isLoading}
              />
            </div>
            <Button type="submit" disabled={isLoading || !file} className="h-10 px-8">
              {isLoading ? "Avaliando..." : "Avaliar Arquivo"}
              {!isLoading && <Upload className="ml-2 w-4 h-4" />}
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

      {report && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
          <Card className="md:col-span-1 bg-card border-border shadow-sm">
            <CardHeader className="bg-secondary/50 rounded-t-lg border-b border-border">
              <CardTitle className="text-lg flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-primary" />
                Score Geral
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 flex flex-col items-center justify-center">
              <div className="text-5xl font-bold text-primary mb-2">
                {report.overall_score !== undefined ? `${(report.overall_score * 100).toFixed(0)}%` : "N/A"}
              </div>
              <div className="flex gap-4 mt-4 text-sm font-medium">
                <span className="text-green-500 flex items-center gap-1"><CheckCircle className="w-4 h-4"/> Passou</span>
                <span className="text-destructive flex items-center gap-1"><XCircle className="w-4 h-4"/> Falhou</span>
              </div>
            </CardContent>
          </Card>

          <Card className="md:col-span-3 bg-card border-border shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Dimensões de Qualidade</CardTitle>
            </CardHeader>
            <CardContent>
              {report.dimensions ? (
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(report.dimensions).map(([dim, data]: [string, any]) => (
                    <div key={dim} className="border border-border rounded-md p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold uppercase text-xs tracking-wider">{dim}</span>
                        <span className="font-bold text-sm">{(data.score * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div className="bg-primary h-2 rounded-full" style={{ width: `${data.score * 100}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">Dimensões indisponíveis.</p>
              )}
            </CardContent>
          </Card>

          <Card className="md:col-span-4 bg-card border-border shadow-sm mt-4">
            <CardHeader>
              <CardTitle className="text-lg">Resultados das Regras</CardTitle>
            </CardHeader>
            <CardContent>
              {report.rule_results && report.rule_results.length > 0 ? (
                <div className="border border-border rounded-md overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 text-muted-foreground">
                      <tr>
                        <th className="px-4 py-3 font-medium">Status</th>
                        <th className="px-4 py-3 font-medium">Regra</th>
                        <th className="px-4 py-3 font-medium">Dimensão</th>
                        <th className="px-4 py-3 font-medium">Detalhe</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.rule_results.map((rule: any, i: number) => (
                        <tr key={i} className="border-t border-border hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3">
                            {rule.status === "passed" ? (
                              <span className="bg-green-500/20 text-green-500 px-2 py-1 rounded-full text-xs font-bold">PASS</span>
                            ) : (
                              <span className="bg-destructive/20 text-destructive px-2 py-1 rounded-full text-xs font-bold">FAIL</span>
                            )}
                          </td>
                          <td className="px-4 py-3 font-medium">{rule.rule_name}</td>
                          <td className="px-4 py-3 text-muted-foreground">{rule.dimension}</td>
                          <td className="px-4 py-3 text-muted-foreground">{rule.error_message || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <pre className="text-xs bg-secondary/30 p-4 rounded-md overflow-x-auto text-muted-foreground border border-border">
                  {JSON.stringify(report, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
    </SettingsGuard>
  );
}
