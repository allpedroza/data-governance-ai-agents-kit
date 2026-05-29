"use client";

import Link from "next/link";
import { useSettingsStore } from "@/store/useSettingsStore";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Settings, ExternalLink, KeyRound } from "lucide-react";

type GuardType = "warehouse" | "llm" | "both";

interface SettingsGuardProps {
  /** What's required: warehouse credentials, llm keys, or both */
  require: GuardType;
  /** Module name for the message */
  moduleName: string;
  children: React.ReactNode;
}

/**
 * Shared guard component that checks if Settings are configured
 * before rendering the wrapped page. Shows a friendly notification
 * with a link to Settings if not.
 */
export function SettingsGuard({ require, moduleName, children }: SettingsGuardProps) {
  const settings = useSettingsStore();

  const hasWarehouse = !!(settings.warehouseHost && settings.warehouseUser);
  
  const hasLlm = (() => {
    const p = settings.llmProvider;
    if (p === "openai") return !!settings.openaiApiKey;
    if (p === "gemini") return !!settings.geminiApiKey;
    if (p === "anthropic") return !!settings.anthropicApiKey;
    if (p === "deepseek") return !!settings.deepseekApiKey;
    return false;
  })();

  const needsWarehouse = require === "warehouse" || require === "both";
  const needsLlm = require === "llm" || require === "both";

  const missingWarehouse = needsWarehouse && !hasWarehouse;
  const missingLlm = needsLlm && !hasLlm;

  if (!missingWarehouse && !missingLlm) {
    return <>{children}</>;
  }

  // Build message
  const missingParts: string[] = [];
  if (missingLlm) missingParts.push("chave de API da LLM");
  if (missingWarehouse) missingParts.push("credenciais do Warehouse");

  return (
    <Card className="border-orange-500/30 bg-orange-500/5 max-w-lg mx-auto mt-12">
      <CardContent className="py-8 flex flex-col items-center text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-orange-500/15 flex items-center justify-center">
          {missingWarehouse ? (
            <AlertTriangle className="w-7 h-7 text-orange-500" />
          ) : (
            <KeyRound className="w-7 h-7 text-orange-500" />
          )}
        </div>
        <div>
          <h3 className="text-lg font-semibold">Configuração necessária</h3>
          <p className="text-muted-foreground text-sm mt-1 max-w-md">
            O módulo <span className="font-semibold text-foreground">{moduleName}</span> requer{" "}
            {missingParts.join(" e ")} configurados em Settings para funcionar.
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
  );
}
