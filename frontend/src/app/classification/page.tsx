"use client";

import { useState } from "react";
import { useClassificationStore } from "@/store/useClassificationStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { ShieldCheck, ServerCrash, Upload, Tags } from "lucide-react";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function ClassificationPage() {
  const [file, setFile] = useState<File | null>(null);
  const { report, isLoading, error, classify } = useClassificationStore();

  const handleClassify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Faça o upload do dataset em CSV");
      return;
    }
    await classify(file);
  };

  const getBadgeColor = (tag: string) => {
    const lower = tag.toLowerCase();
    if (lower.includes("confidencial") || lower.includes("pii") || lower.includes("sensitive")) return "bg-destructive text-destructive-foreground";
    if (lower.includes("interno") || lower.includes("restrito")) return "bg-orange-500 text-white";
    return "bg-green-500/20 text-green-500";
  };

  return (
    <SettingsGuard require="llm" moduleName="Data Classification">
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Classification</h1>
        <p className="text-muted-foreground mt-2">
          Classificação autônoma de confidencialidade e dados sensíveis (PII) usando IA.
        </p>
      </div>

      <Card className="bg-card">
        <CardContent className="pt-6">
          <form onSubmit={handleClassify} className="flex gap-4 items-center">
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
              {isLoading ? "Classificando..." : "Analisar Dataset"}
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          <Card className="md:col-span-2 bg-card border-border shadow-sm">
            <CardHeader className="bg-secondary/50 rounded-t-lg border-b border-border">
              <CardTitle className="text-lg flex items-center gap-2">
                <Tags className="w-5 h-5 text-primary" />
                Tags do Dataset (Nível de Tabela)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {report.dataset_level_tags && report.dataset_level_tags.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {report.dataset_level_tags.map((tag, i) => (
                    <span key={i} className={`px-3 py-1 rounded-full text-xs font-bold ${getBadgeColor(tag)}`}>
                      {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Nenhuma tag a nível de dataset identificada.</p>
              )}
            </CardContent>
          </Card>

          <Card className="md:col-span-2 bg-card border-border shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-accent" />
                Classificação por Coluna
              </CardTitle>
            </CardHeader>
            <CardContent>
              {report.columns && Object.keys(report.columns).length > 0 ? (
                <div className="border border-border rounded-md overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 text-muted-foreground">
                      <tr>
                        <th className="px-4 py-3 font-medium">Nome da Coluna</th>
                        <th className="px-4 py-3 font-medium">Classificação Sugerida</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(report.columns).map(([col, tag], i) => (
                        <tr key={i} className="border-t border-border hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 font-medium">{col}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getBadgeColor(tag)}`}>
                              {tag}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Nenhuma classificação de coluna disponível.</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
    </SettingsGuard>
  );
}
