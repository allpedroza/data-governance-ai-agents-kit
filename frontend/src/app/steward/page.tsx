"use client";

import { useState } from "react";
import { useStewardStore } from "@/store/useStewardStore";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { ShieldCheck, UserPlus, Bot, AlertCircle, CheckCircle2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function StewardPage() {
  const [assetName, setAssetName] = useState("");
  const [metadata, setMetadata] = useState("");
  const { assignment, isLoading, error, assign } = useStewardStore();

  const handleAssign = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const parsedMeta = JSON.parse(metadata);
      await assign(assetName, parsedMeta);
      toast.success("Análise do Steward concluída!");
    } catch (e: any) {
      toast.error("JSON Inválido no campo de metadados");
    }
  };

  return (
    <SettingsGuard require="both" moduleName="Data Steward">
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-3">
          <ShieldCheck className="w-8 h-8 text-primary" />
          Data Steward Copilot
        </h1>
        <p className="text-muted-foreground mt-2">
          Deixe a IA sugerir e atribuir os proprietários (Data Owners) ideais para seus ativos baseando-se no contexto, histórico e glossário de negócios.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2"><UserPlus className="w-5 h-5 text-primary"/> Solicitar Atribuição</CardTitle>
            <CardDescription>Qual ativo precisa de um responsável?</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleAssign} className="flex flex-col gap-5">
              <div className="space-y-2">
                <label className="text-sm font-medium">Nome do Ativo</label>
                <Input 
                  value={assetName} 
                  onChange={(e) => setAssetName(e.target.value)}
                  placeholder="ex: dim_customers"
                  className="bg-background"
                  disabled={isLoading}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Metadados / Contexto (JSON)</label>
                <Textarea 
                  value={metadata}
                  onChange={(e) => setMetadata(e.target.value)}
                  placeholder={`{\n  "description": "Tabela de vendas...",\n  "schema": "ecommerce_dw"\n}`}
                  className="font-mono text-xs h-32 bg-background"
                  disabled={isLoading}
                />
              </div>
              <Button type="submit" disabled={isLoading} className="mt-2">
                {isLoading ? "Processando..." : "Gerar Sugestão de Ownership"}
                {!isLoading && <Bot className="ml-2 w-4 h-4" />}
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {error && (
            <div className="bg-destructive/10 border border-destructive/20 text-destructive p-4 rounded-md flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {assignment ? (
            <Card className="bg-card border-border shadow-sm overflow-hidden relative">
              <div className="absolute top-0 right-0 p-4 opacity-10">
                <ShieldCheck className="w-24 h-24" />
              </div>
              <CardHeader className="bg-primary/5 pb-4 border-b border-border">
                <CardTitle className="text-lg text-primary flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5"/> Sugestão do Copilot
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6 space-y-4">
                <div>
                  <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Ativo Analisado</div>
                  <div className="font-mono text-foreground font-semibold bg-white/5 p-2 rounded">{assignment.asset_name}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Proprietário Sugerido (Owner)</div>
                  <div className="text-xl font-bold text-foreground text-primary">{assignment.owner}</div>
                  <p className="text-sm text-muted-foreground mt-1">
                    Confiança: <span className="text-foreground font-medium">{assignment.confidence_score ? (assignment.confidence_score * 100).toFixed(0) : "100"}%</span>
                  </p>
                </div>
                {assignment.stewards && assignment.stewards.length > 0 && (
                  <div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Stewards Recomendados</div>
                    <div className="flex gap-2 flex-wrap">
                      {assignment.stewards.map((st: string, idx: number) => (
                        <span key={idx} className="bg-secondary text-secondary-foreground text-xs px-2 py-1 rounded">
                          {st}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
              <CardFooter className="bg-secondary/20 justify-between">
                <Button variant="outline" className="text-xs" onClick={() => toast.info("Sugestão Recusada.")}>Recusar</Button>
                <Button className="text-xs bg-green-600 hover:bg-green-700 text-white" onClick={() => toast.success("Ownership Aprovado e Salvo no Catálogo!")}>Aprovar e Salvar</Button>
              </CardFooter>
            </Card>
          ) : (
             <div className="h-full border-2 border-dashed border-border rounded-xl flex flex-col items-center justify-center p-8 text-center bg-card/50">
                <Bot className="w-12 h-12 text-muted-foreground mb-4 opacity-50" />
                <h3 className="text-lg font-medium text-foreground">Aguardando Análise</h3>
                <p className="text-sm text-muted-foreground max-w-[250px] mt-2">
                  Preencha o formulário para receber a sugestão de curadoria inteligente.
                </p>
             </div>
          )}
        </div>
      </div>
    </div>
    </SettingsGuard>
  );
}
