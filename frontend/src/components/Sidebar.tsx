"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  Search, ShieldAlert, ShieldCheck, Route, FileSignature, 
  Tags, Gem, UserCog, Lock, Settings, Network
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";

const navGroups = [
  {
    title: "Data Governance",
    items: [
      { name: "Discovery (RAG)", href: "/discovery", icon: Search },
      { name: "Data Quality", href: "/quality", icon: ShieldAlert },
      { name: "Classification", href: "/classification", icon: ShieldCheck },
      { name: "Taxonomy Evaluator", href: "/taxonomy", icon: Network },
      { name: "Data Lineage", href: "/lineage", icon: Route },
      { name: "Data Contracts", href: "/contracts", icon: FileSignature },
      { name: "Metadata Enrichment", href: "/enrichment", icon: Tags },
      { name: "Data Asset Value", href: "/value", icon: Gem },
      { name: "Data Steward", href: "/steward", icon: UserCog },
    ],
  },
  {
    title: "AI Governance",
    items: [
      { name: "AI NER & Vault", href: "/vault", icon: Lock },
    ],
  },
  {
    title: "Administração",
    items: [
      { name: "Settings", href: "/settings", icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-sidebar h-screen flex flex-col fixed left-0 top-0 z-30 shadow-[1px_0_8px_rgba(0,0,0,0.04)] dark:shadow-none">
      <div className="p-6">
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Data Gov AI
        </h1>
        <p className="text-xs text-muted-foreground mt-1 font-medium">
          Framework de Governança Generativa
        </p>
      </div>
      
      <div className="px-4 pb-6 flex-1 overflow-y-auto space-y-8">
        {navGroups.map((group) => (
          <div key={group.title}>
            <h2 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              {group.title}
            </h2>
            <div className="space-y-1">
              {group.items.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center w-full h-9 px-2 text-sm font-medium rounded-md transition-colors",
                      isActive 
                        ? "bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25" 
                        : "text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5"
                    )}
                  >
                    <item.icon className="mr-3 h-4 w-4 shrink-0" />
                    <span className="truncate">{item.name}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      
      <div className="p-4 border-t border-border">
        <div className="flex items-center justify-between px-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-green-500 ring-2 ring-green-500/20"></div>
            Status: Pronto
          </div>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
