"use client";

import { useState } from "react";
import { useVaultStore } from "@/store/useVaultStore";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Lock, FileText, AlertTriangle, KeySquare, Shield } from "lucide-react";

const SAMPLE_TEXT = `O cliente João da Silva, portador do CPF 123.456.789-00, realizou uma compra no valor de R$ 5.000,00 usando o cartão de crédito 4111 1111 1111 1111. 
O endereço de entrega é Rua das Flores, 123, São Paulo, SP. 
Para dúvidas, entrar em contato pelo telefone (11) 99999-9999 ou email joao.silva@email.com.`;

export default function VaultPage() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const { result, isLoading, error, anonymize } = useVaultStore();

  const handleAnonymize = async () => {
    await anonymize(text);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-3">
          <Lock className="w-8 h-8 text-primary" />
          Sensitive Data NER & Vault
        </h1>
        <p className="text-muted-foreground mt-2">
          Identificação de PII (Personally Identifiable Information) via NLP e criptografia tokenizada (Vault) para dados sensíveis em trânsito e em repouso.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-card shadow-sm border-border">
          <CardHeader className="bg-secondary/30 border-b border-border">
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="w-5 h-5 text-muted-foreground" /> Dado Bruto (Original)
            </CardTitle>
            <CardDescription>Insira o texto vazado ou log não estruturado</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="min-h-[250px] font-mono text-sm leading-relaxed"
              disabled={isLoading}
            />
            <Button 
              className="mt-4 w-full" 
              onClick={handleAnonymize} 
              disabled={isLoading || !text.trim()}
            >
              <Shield className="w-4 h-4 mr-2" />
              {isLoading ? "Varrendo PIIs..." : "Detectar PII e Criptografar"}
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-card shadow-sm border-border relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-5">
            <Lock className="w-32 h-32" />
          </div>
          <CardHeader className="bg-primary/5 border-b border-primary/20">
            <CardTitle className="text-lg text-primary flex items-center gap-2">
              <KeySquare className="w-5 h-5" /> Dado Criptografado (Vault)
            </CardTitle>
            <CardDescription>Texto com entidades protegidas (Tokenização Format-Preserving)</CardDescription>
          </CardHeader>
          <CardContent className="pt-6 h-[250px] overflow-auto">
            {error && (
              <div className="bg-destructive/10 text-destructive p-4 rounded-md flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                {error}
              </div>
            )}
            
            {result ? (
              <div className="space-y-6">
                <div className="bg-secondary/20 p-4 rounded-md font-mono text-sm whitespace-pre-wrap leading-relaxed text-foreground">
                  {result.anonymized_text}
                </div>
                
                {result.entities && result.entities.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Shield className="w-4 h-4 text-green-500" /> PIIs Interceptadas
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {result.entities.map((entity: any, i: number) => (
                        <div key={i} className="bg-background border border-border px-3 py-1 rounded-md flex flex-col gap-1 shadow-sm">
                          <span className="text-[10px] uppercase text-muted-foreground tracking-wider font-bold">
                            {entity.entity_type} ({Math.round(entity.confidence * 100)}%)
                          </span>
                          <span className="text-sm font-mono text-destructive line-through opacity-80">
                            {entity.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-60">
                <Shield className="w-12 h-12 mb-3" />
                <p>Nenhuma análise ativa</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
