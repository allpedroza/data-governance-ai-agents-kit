import { Construction } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function PlaceholderPage() {
  return (
    <div className="h-[60vh] flex items-center justify-center animate-in fade-in zoom-in duration-500">
      <Card className="max-w-md w-full bg-card border-border shadow-sm text-center">
        <CardContent className="pt-10 pb-10 flex flex-col items-center">
          <div className="h-16 w-16 rounded-full bg-accent/10 flex items-center justify-center mb-6">
            <Construction className="h-8 w-8 text-accent" />
          </div>
          <h2 className="text-2xl font-bold mb-2">Em Construção</h2>
          <p className="text-muted-foreground text-sm">
            Este módulo ainda será migrado para React. A integração da API via FastAPI ocorrerá nas próximas iterações.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
