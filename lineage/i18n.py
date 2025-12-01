"""
Internacionalização (i18n) para Data Lineage Agent
Suporta múltiplos idiomas para saídas do sistema
"""

TRANSLATIONS = {
    'pt': {
        # Análise
        'analyzing_project': '🚀 Iniciando análise do projeto',
        'files_found': '📁 {count} arquivos encontrados para análise',
        'analyzing_files': '🔍 Iniciando análise de {count} arquivos...',
        'file_processed': '✅ Processado',
        'file_error': '❌ Erro ao processar',
        'generating_insights': '🤖 Gerando insights automáticos...',

        # Resumo
        'analysis_summary': 'RESUMO DA ANÁLISE DE LINHAGEM',
        'general_stats': 'Estatísticas Gerais',
        'total_assets': 'Total de Assets',
        'total_transformations': 'Total de Transformações',
        'asset_types': 'Tipos de Assets',
        'operation_types': 'Tipos de Operações',
        'complexity_metrics': 'Métricas de Complexidade',
        'nodes': 'Nós no grafo',
        'edges': 'Arestas no grafo',
        'density': 'Densidade',
        'avg_degree': 'Grau médio',
        'cycles_detected': 'CICLOS DETECTADOS!',

        # Componentes Críticos
        'critical_components': 'Componentes Críticos',
        'single_points_failure': 'Pontos Únicos de Falha',
        'bottlenecks': 'Bottlenecks',
        'critical_paths': 'Caminhos Críticos',
        'impacts': 'impacta',
        'assets_downstream': 'assets downstream',

        # Insights
        'auto_insights': 'Insights Automáticos',
        'risk_assessment': 'Avaliação de Risco',
        'main_recommendations': 'Recomendações Principais',

        # Relatório
        'report_title': 'Relatório de Análise de Linhagem de Dados',
        'generated': 'Gerado',
        'executive_summary': 'Resumo Executivo',
        'recommendations': 'Recomendações',
        'visualizations': 'Visualizações Interativas',
        'detailed_docs': 'Documentação Detalhada',

        # Status
        'success': 'Análise concluída com sucesso!',
        'failed': 'Análise falhou',

        # Tipos de visualização
        'dashboard': 'Dashboard',
        'force_graph': 'Grafo Force-Directed',
        'sankey_diagram': 'Diagrama de Fluxo de Dados',
        'impact_analysis': 'Análise de Impacto',
        'atlas_view': 'Visualização Atlas',

        # Métricas do relatório
        'transformations': 'Transformações',
        'inputs': 'entradas',
        'outputs': 'saídas',
        'identified_bottlenecks': 'Bottlenecks Identificados',
        'critical_paths_found': 'Caminhos Críticos Encontrados',
        'length': 'comprimento',

        # Componentes críticos detalhados
        'single_points_failure_title': '<span class="error-icon">🔴</span> Pontos Únicos de Falha',
        'bottlenecks_identified_title': '<span class="warning-icon">⚠️</span> Bottlenecks Identificados',
        'critical_paths_title': '<span class="warning-icon">📊</span> Caminhos Críticos',

        # Seções do relatório
        'auto_insights_analysis': '🤖 Insights Automáticos e Análise Crítica',

        # Navegação
        'view_visualization': 'Ver Visualização',
        'expand_level': 'Expandir nível',
        'collapse_level': 'Recolher nível',
        'show_upstream': 'Mostrar upstream',
        'show_downstream': 'Mostrar downstream',
        'reset_view': 'Resetar visualização'
    },

    'en': {
        # Analysis
        'analyzing_project': '🚀 Starting project analysis',
        'files_found': '📁 {count} files found for analysis',
        'analyzing_files': '🔍 Starting analysis of {count} files...',
        'file_processed': '✅ Processed',
        'file_error': '❌ Error processing',
        'generating_insights': '🤖 Generating automatic insights...',

        # Summary
        'analysis_summary': 'LINEAGE ANALYSIS SUMMARY',
        'general_stats': 'General Statistics',
        'total_assets': 'Total Assets',
        'total_transformations': 'Total Transformations',
        'asset_types': 'Asset Types',
        'operation_types': 'Operation Types',
        'complexity_metrics': 'Complexity Metrics',
        'nodes': 'Nodes in graph',
        'edges': 'Edges in graph',
        'density': 'Density',
        'avg_degree': 'Average degree',
        'cycles_detected': 'CYCLES DETECTED!',

        # Critical Components
        'critical_components': 'Critical Components',
        'single_points_failure': 'Single Points of Failure',
        'bottlenecks': 'Bottlenecks',
        'critical_paths': 'Critical Paths',
        'impacts': 'impacts',
        'assets_downstream': 'assets downstream',

        # Insights
        'auto_insights': 'Automatic Insights',
        'risk_assessment': 'Risk Assessment',
        'main_recommendations': 'Main Recommendations',

        # Report
        'report_title': 'Data Lineage Analysis Report',
        'generated': 'Generated',
        'executive_summary': 'Executive Summary',
        'recommendations': 'Recommendations',
        'visualizations': 'Interactive Visualizations',
        'detailed_docs': 'Detailed Documentation',

        # Status
        'success': 'Analysis completed successfully!',
        'failed': 'Analysis failed',

        # Visualization types
        'dashboard': 'Dashboard',
        'force_graph': 'Force-Directed Graph',
        'sankey_diagram': 'Data Flow Diagram',
        'impact_analysis': 'Impact Analysis',
        'atlas_view': 'Atlas View',

        # Report metrics
        'transformations': 'Transformations',
        'inputs': 'inputs',
        'outputs': 'outputs',
        'identified_bottlenecks': 'Identified Bottlenecks',
        'critical_paths_found': 'Critical Paths Found',
        'length': 'length',

        # Detailed critical components
        'single_points_failure_title': '<span class="error-icon">🔴</span> Single Points of Failure',
        'bottlenecks_identified_title': '<span class="warning-icon">⚠️</span> Identified Bottlenecks',
        'critical_paths_title': '<span class="warning-icon">📊</span> Critical Paths',

        # Report sections
        'auto_insights_analysis': '🤖 Automatic Insights and Critical Analysis',

        # Navigation
        'view_visualization': 'View Visualization',
        'expand_level': 'Expand level',
        'collapse_level': 'Collapse level',
        'show_upstream': 'Show upstream',
        'show_downstream': 'Show downstream',
        'reset_view': 'Reset view'
    }
}


class I18n:
    """Gerenciador de internacionalização"""

    def __init__(self, language='pt'):
        """
        Inicializa com idioma padrão

        Args:
            language: Código do idioma ('pt', 'en')
        """
        self.language = language
        self.fallback = 'en'

    def t(self, key: str, **kwargs) -> str:
        """
        Traduz uma chave

        Args:
            key: Chave de tradução
            **kwargs: Variáveis para interpolação

        Returns:
            Texto traduzido
        """
        # Tenta obter tradução no idioma selecionado
        translation = TRANSLATIONS.get(self.language, {}).get(key)

        # Se não encontrar, tenta fallback
        if not translation:
            translation = TRANSLATIONS.get(self.fallback, {}).get(key, key)

        # Interpola variáveis se houver
        if kwargs:
            try:
                return translation.format(**kwargs)
            except:
                return translation

        return translation

    def set_language(self, language: str):
        """Altera o idioma"""
        if language in TRANSLATIONS:
            self.language = language
        else:
            print(f"⚠️ Language '{language}' not supported. Using '{self.language}'")


# Instância global
_i18n = I18n()


def get_i18n() -> I18n:
    """Retorna instância global de i18n"""
    return _i18n


def t(key: str, **kwargs) -> str:
    """Atalho para tradução"""
    return _i18n.t(key, **kwargs)


def set_language(language: str):
    """Atalho para configurar idioma"""
    _i18n.set_language(language)
