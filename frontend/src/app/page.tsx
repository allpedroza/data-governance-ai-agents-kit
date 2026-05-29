import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowRight, Bot, Database, Shield } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-4xl font-extrabold tracking-tight">
          Governança de Dados com IA
        </h1>
        <p className="text-lg text-muted-foreground mt-2 max-w-2xl">
          Selecione um agente no menu lateral para começar. A plataforma oferece ferramentas avançadas de descoberta, linhagem, qualidade e segurança.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-card border-border shadow-sm">
          <CardHeader>
            <Database className="w-8 h-8 text-primary mb-2" />
            <CardTitle>Discovery (RAG)</CardTitle>
            <CardDescription>
              Descubra dados de forma inteligente usando buscas semânticas e assistentes conversacionais.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="secondary" className="w-full">
              <Link href="/discovery">
                Acessar Módulo <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-card border-border shadow-sm">
          <CardHeader>
            <Shield className="w-8 h-8 text-primary mb-2" />
            <CardTitle>Quality & Security</CardTitle>
            <CardDescription>
              Monitoramento automático de qualidade de dados e classificação de informações sensíveis (PII).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="secondary" className="w-full">
              <Link href="/quality">
                Acessar Módulo <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-card border-border shadow-sm">
          <CardHeader>
            <Bot className="w-8 h-8 text-primary mb-2" />
            <CardTitle>AI Governance</CardTitle>
            <CardDescription>
              Proteja as interações do seu modelo RAG utilizando mascaramento de PII e controles rígidos.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="secondary" className="w-full">
              <Link href="/ner-vault">
                Acessar Módulo <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
