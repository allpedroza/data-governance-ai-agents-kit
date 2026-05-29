"use client";

import { useEffect, useState } from "react";
import { useSettingsStore } from "@/store/useSettingsStore";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Save } from "lucide-react";

export default function SettingsPage() {
  const settings = useSettingsStore();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch for persisted store
  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSave = () => {
    toast.success("Configurações salvas com sucesso!");
    // Later we can trigger a backend sync here if needed
  };

  if (!mounted) return null;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Configurações e Integrações</h1>
        <p className="text-muted-foreground mt-2">
          Gerencie conexões com Data Warehouses, chaves de API e provedores de LLM em um único lugar.
        </p>
      </div>

      <Tabs defaultValue="llm" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-secondary">
          <TabsTrigger value="llm">Provedores LLM</TabsTrigger>
          <TabsTrigger value="warehouse">Data Warehouses</TabsTrigger>
          <TabsTrigger value="catalog">Catálogos de Dados</TabsTrigger>
        </TabsList>

        <TabsContent value="llm" className="space-y-4 mt-6">
          <Card className="bg-card">
            <CardHeader>
              <CardTitle>Configuração de Modelos LLM</CardTitle>
              <CardDescription>
                Selecione o provedor e as chaves de acesso. Elas são salvas localmente e utilizadas nas integrações.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Provedor LLM Primário</Label>
                <Select value={settings.llmProvider} onValueChange={settings.setLLMProvider}>
                  <SelectTrigger className="w-full bg-background">
                    <SelectValue placeholder="Selecione o provedor" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="gemini">Google Gemini</SelectItem>
                    <SelectItem value="anthropic">Anthropic Claude</SelectItem>
                    <SelectItem value="deepseek">DeepSeek</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Modelo Padrão</Label>
                <Input 
                  value={settings.llmModel}
                  onChange={(e) => settings.setLLMModel(e.target.value)}
                  placeholder="ex: gpt-4o, gemini-1.5-pro, claude-3-5-sonnet" 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>OpenAI API Key</Label>
                <Input 
                  type="password" 
                  value={settings.openaiApiKey}
                  onChange={(e) => settings.setOpenAIApiKey(e.target.value)}
                  placeholder="sk-..." 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>Gemini API Key</Label>
                <Input 
                  type="password" 
                  value={settings.geminiApiKey}
                  onChange={(e) => settings.setGeminiApiKey(e.target.value)}
                  placeholder="AIza..." 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>Anthropic API Key</Label>
                <Input 
                  type="password" 
                  value={settings.anthropicApiKey || ""}
                  onChange={(e) => settings.setAnthropicApiKey(e.target.value)}
                  placeholder="sk-ant-..." 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>DeepSeek API Key</Label>
                <Input 
                  type="password" 
                  value={settings.deepseekApiKey || ""}
                  onChange={(e) => settings.setDeepseekApiKey(e.target.value)}
                  placeholder="sk-..." 
                  className="bg-background"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="warehouse" className="space-y-4 mt-6">
          <Card className="bg-card">
            <CardHeader>
              <CardTitle>Integração com Data Warehouse</CardTitle>
              <CardDescription>Conecte seus bancos de dados para validação de metadados.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Tipo de Warehouse</Label>
                <Select value={settings.warehouseType} onValueChange={settings.setWarehouseType}>
                  <SelectTrigger className="w-full bg-background">
                    <SelectValue placeholder="Selecione..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="snowflake">Snowflake</SelectItem>
                    <SelectItem value="bigquery">Google BigQuery</SelectItem>
                    <SelectItem value="redshift">AWS Redshift</SelectItem>
                    <SelectItem value="postgres">PostgreSQL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Host / Account</Label>
                <Input 
                  value={settings.warehouseHost || ""}
                  onChange={(e) => settings.setWarehouseHost(e.target.value)}
                  placeholder="ex: xy12345.us-east-1.snowflakecomputing.com" 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>Usuário</Label>
                <Input 
                  value={settings.warehouseUser || ""}
                  onChange={(e) => settings.setWarehouseUser(e.target.value)}
                  placeholder="DB_USER" 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>Senha / Token</Label>
                <Input 
                  type="password"
                  value={settings.warehousePassword || ""}
                  onChange={(e) => settings.setWarehousePassword(e.target.value)}
                  placeholder="••••••••" 
                  className="bg-background"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="catalog" className="space-y-4 mt-6">
          <Card className="bg-card">
            <CardHeader>
              <CardTitle>Integração com Catálogo</CardTitle>
              <CardDescription>Sincronize com ferramentas de linhagem e governança open source.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Catálogo Primário</Label>
                <Select value={settings.catalogType} onValueChange={settings.setCatalogType}>
                  <SelectTrigger className="w-full bg-background">
                    <SelectValue placeholder="Selecione..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openmetadata">OpenMetadata</SelectItem>
                    <SelectItem value="datahub">DataHub</SelectItem>
                    <SelectItem value="amundsen">Amundsen</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Host / URL</Label>
                <Input 
                  value={settings.catalogHost || ""}
                  onChange={(e) => settings.setCatalogHost(e.target.value)}
                  placeholder="http://openmetadata:8585" 
                  className="bg-background"
                />
              </div>
              <div className="space-y-2">
                <Label>API Token / JWT</Label>
                <Input 
                  type="password"
                  value={settings.catalogToken || ""}
                  onChange={(e) => settings.setCatalogToken(e.target.value)}
                  placeholder="ey..." 
                  className="bg-background"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end pt-4 border-t border-border mt-8">
        <Button onClick={handleSave}>
          <Save className="w-4 h-4 mr-2" />
          Salvar Configurações
        </Button>
      </div>
    </div>
  );
}
