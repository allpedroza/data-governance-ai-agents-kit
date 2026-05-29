"use client";

import { useState } from "react";
import { Search, ServerCrash, Database, Tag, ShieldCheck } from "lucide-react";
import { useDiscoveryStore } from "@/store/useDiscoveryStore";
import { SettingsGuard } from "@/components/SettingsGuard";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { toast } from "sonner";

export default function DiscoveryPage() {
  const [query, setQuery] = useState("");
  const { results, isLoading, error, search } = useDiscoveryStore();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) {
      toast.error("Digite um termo para pesquisar");
      return;
    }
    await search(query);
  };

  return (
    <SettingsGuard require="llm" moduleName="Data Discovery (RAG)">
    <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Discovery (RAG)</h1>
        <p className="text-muted-foreground mt-2">
          Busque por ativos de dados de forma semântica utilizando agentes LLM.
        </p>
      </div>

      <Card className="bg-card">
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ex: 'Encontre tabelas com informações de cartão de crédito de clientes'" 
                className="pl-10 h-10 bg-background"
                disabled={isLoading}
              />
            </div>
            <Button type="submit" disabled={isLoading} className="h-10 px-8">
              {isLoading ? "Buscando..." : "Pesquisar"}
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

      {results.length > 0 && (
        <div className="space-y-4 mt-8">
          <h2 className="text-xl font-semibold">Resultados da Busca</h2>
          <div className="grid gap-4">
            {results.map((item, idx) => (
              <Card key={idx} className="bg-card hover:bg-accent/5 transition-colors">
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <Database className="h-5 w-5 text-primary" />
                      <CardTitle className="text-lg">{item.name}</CardTitle>
                    </div>
                    {item.relevance_score && (
                      <span className="text-xs font-semibold px-2 py-1 bg-accent/10 text-accent rounded-full">
                        Score: {item.relevance_score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <CardDescription className="pt-2">{item.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Tag className="h-4 w-4" /> Tipo: {item.type}
                    </div>
                    <div className="flex items-center gap-1">
                      <ShieldCheck className="h-4 w-4" /> Domínio: {item.domain}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
    </SettingsGuard>
  );
}
