"use client";

import { useState } from "react";
import { useContractsStore } from "@/store/useContractsStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { FileSignature, ServerCrash, Upload, CheckCircle, XCircle } from "lucide-react";
import { SettingsGuard } from "@/components/SettingsGuard";

export default function ContractsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [contractYaml, setContractYaml] = useState("");
  const { report, isLoading, error, validate } = useContractsStore();

  const handleValidate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Faça o upload do dataset em CSV");
      return;
    }
    if (!contractYaml.trim()) {
      toast.error("Insira o contrato em YAML");
      return;
    }
    await validate(file, contractYaml);
  };

  return (
    <SettingsGuard require="llm" moduleName="Data Contracts">
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Contracts</h1>
        <p className="text-muted-foreground mt-2">
          Validação estrutural de schema contra um contrato YAML pré-estabelecido.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-card md:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Contrato YAML</CardTitle>
            <CardDescription>Defina as regras do contrato que o dataset deve obedecer.</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea 
              className="font-mono text-sm min-h-[300px] bg-secondary/30"
              value={contractYaml}
              onChange={(e) => setContractYaml(e.target.value)}
              disabled={isLoading}
            />
          </CardContent>
        </Card>

        <Card className="bg-card md:col-span-1 flex flex-col justify-between">
          <div>
            <CardHeader>
              <CardTitle className="text-lg">Upload do Dataset</CardTitle>
              <CardDescription>Submeta o CSV para ser validado contra o contrato.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleValidate} className="space-y-4 flex flex-col items-start">
                <div className="w-full">
                  <Label>Arquivo CSV</Label>
                  <Input 
                    type="file"
                    accept=".csv"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="bg-background cursor-pointer mt-2"
                    disabled={isLoading}
                  />
                </div>
                <Button type="submit" disabled={isLoading || !file} className="w-full h-10 mt-4">
                  {isLoading ? "Validando..." : "Validar Contrato"}
                  {!isLoading && <FileSignature className="ml-2 w-4 h-4" />}
                </Button>
              </form>
            </CardContent>
          </div>
          
          {error && (
            <CardContent>
              <div className="p-4 border border-destructive bg-destructive/10 rounded-md flex items-center gap-3 text-destructive">
                <ServerCrash className="h-5 w-5 shrink-0" />
                <p className="text-sm font-medium">{error}</p>
              </div>
            </CardContent>
          )}
        </Card>
      </div>

      {report && (
        <div className="mt-8 animate-in slide-in-from-bottom-2">
          <Card className={`border ${report.is_valid ? 'border-green-500' : 'border-destructive'} shadow-sm`}>
            <CardHeader className={`${report.is_valid ? 'bg-green-500/10' : 'bg-destructive/10'} rounded-t-lg border-b border-border`}>
              <CardTitle className="text-lg flex items-center gap-2">
                {report.is_valid ? (
                  <><CheckCircle className="w-5 h-5 text-green-500" /> Contrato Válido</>
                ) : (
                  <><XCircle className="w-5 h-5 text-destructive" /> Contrato Inválido</>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {!report.is_valid && report.errors && report.errors.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-semibold text-destructive mb-2">Erros de Validação:</h4>
                  <ul className="list-disc pl-5 text-sm text-destructive/80 space-y-1">
                    {report.errors.map((err: any, i: number) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="flex gap-4 mt-4 text-sm font-medium">
                <span className="text-green-500 flex items-center gap-1"><CheckCircle className="w-4 h-4"/> {report.passed_rules}</span>
                <span className="text-destructive flex items-center gap-1"><XCircle className="w-4 h-4"/> {report.failed_rules}</span>
              </div>
            </CardContent>
          </Card>

          <Card className="md:col-span-3 bg-card border-border shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Resultados da Validação</CardTitle>
            </CardHeader>
            <CardContent>
              {report.results && report.results.length > 0 ? (
                <div className="border border-border rounded-md overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 text-muted-foreground">
                      <tr>
                        <th className="px-4 py-3 font-medium">Status</th>
                        <th className="px-4 py-3 font-medium">Regra</th>
                        <th className="px-4 py-3 font-medium">Detalhe</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.results.map((rule: any, i: number) => (
                        <tr key={i} className="border-t border-border hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 w-24">
                            {rule.status === "passed" ? (
                              <span className="bg-green-500/20 text-green-500 px-2 py-1 rounded-full text-xs font-bold">PASS</span>
                            ) : (
                              <span className="bg-destructive/20 text-destructive px-2 py-1 rounded-full text-xs font-bold">FAIL</span>
                            )}
                          </td>
                          <td className="px-4 py-3 font-medium">{rule.rule_name}</td>
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
