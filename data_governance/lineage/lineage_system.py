# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
#!/usr/bin/env python3
"""
Data Lineage AI System - Sistema Principal
Integração completa com análise de impacto e visualização
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import traceback

# Importa os módulos do sistema
from data_lineage_agent import DataLineageAgent, DataAsset, Transformation
from visualization_engine import DataLineageVisualizer
from i18n import set_language, get_i18n


class DataLineageSystem:
    """
    Sistema principal de análise de linhagem de dados
    """

    def __init__(self, verbose: bool = True, language: str = None):
        self.agent = DataLineageAgent()
        self.visualizer = None
        self.verbose = verbose
        self.current_analysis = None
        self.analysis_history = []

        # Configura idioma
        if language:
            set_language(language)
        else:
            # Detecta do ambiente ou usa padrão
            import os
            lang = os.getenv('DATA_LINEAGE_LANGUAGE', 'pt')
            set_language(lang)

        self.i18n = get_i18n()
        
    def log(self, message: str, level: str = "INFO"):
        """Log com timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def analyze_project(self, 
                        project_path: str,
                        file_patterns: List[str] = None,
                        recursive: bool = True) -> Dict:
        """
        Analisa um projeto completo
        """
        self.log(f"🚀 Iniciando análise do projeto: {project_path}")
        
        # Padrões default de arquivo
        if not file_patterns:
            file_patterns = ['*.py', '*.sql', '*.tf', '*.json', '*.scala']
        
        # Coleta arquivos
        files = self._collect_files(project_path, file_patterns, recursive)
        self.log(f"📁 {len(files)} arquivos encontrados para análise")
        
        if not files:
            self.log("⚠️ Nenhum arquivo encontrado com os padrões especificados", "WARNING")
            return {}
        
        # Executa análise
        try:
            analysis_result = self.agent.analyze_pipeline(files)
            self.current_analysis = analysis_result
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'project_path': project_path,
                'files_analyzed': len(files),
                'result': analysis_result
            })
            
            # Cria visualizador
            self.visualizer = DataLineageVisualizer(self.agent.graph)
            
            # Resumo
            self._print_analysis_summary(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.log(f"❌ Erro na análise: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return {}
    
    def _collect_files(self, 
                      project_path: str,
                      patterns: List[str],
                      recursive: bool) -> List[str]:
        """Coleta arquivos baseado nos padrões"""
        files = []
        path = Path(project_path)
        
        if not path.exists():
            self.log(f"❌ Caminho não encontrado: {project_path}", "ERROR")
            return files
        
        for pattern in patterns:
            if recursive:
                matched = path.rglob(pattern)
            else:
                matched = path.glob(pattern)
            
            for file_path in matched:
                if file_path.is_file():
                    files.append(str(file_path))
        
        return files
    
    def _print_analysis_summary(self, analysis: Dict):
        """Imprime resumo da análise"""
        print("\n" + "="*60)
        print("📊 RESUMO DA ANÁLISE DE LINHAGEM")
        print("="*60)
        
        metrics = analysis.get('metrics', {})
        
        # Estatísticas básicas
        print(f"\n📈 Estatísticas Gerais:")
        print(f"  • Total de Assets: {metrics.get('total_assets', 0)}")
        print(f"  • Total de Transformações: {metrics.get('total_transformations', 0)}")
        
        # Tipos de assets
        if metrics.get('asset_types'):
            print(f"\n🗂️ Tipos de Assets:")
            for asset_type, count in metrics['asset_types'].items():
                print(f"  • {asset_type}: {count}")
        
        # Operações
        if metrics.get('operation_types'):
            print(f"\n⚙️ Tipos de Operações:")
            for op_type, count in metrics['operation_types'].items():
                print(f"  • {op_type}: {count}")
        
        # Complexidade
        complexity = metrics.get('complexity_metrics', {})
        if complexity:
            print(f"\n🔧 Métricas de Complexidade:")
            print(f"  • Nós no grafo: {complexity.get('nodes', 0)}")
            print(f"  • Arestas no grafo: {complexity.get('edges', 0)}")
            print(f"  • Densidade: {complexity.get('density', 0):.3f}")
            print(f"  • Grau médio: {complexity.get('avg_degree', 0):.2f}")
            
            if complexity.get('has_cycles'):
                print(f"  • ⚠️ CICLOS DETECTADOS!")
                cycles = complexity.get('cycles', [])
                if cycles:
                    print(f"    Exemplo: {' → '.join(cycles[0] + [cycles[0][0]])}")

        # Componentes Críticos
        critical = analysis.get('critical_components', {})
        if critical:
            spof = critical.get('single_points_of_failure', [])
            if spof:
                print(f"\n🔴 Pontos Únicos de Falha:")
                for point in spof[:3]:
                    print(f"  • {point['asset']} (impacta {point['downstream_impact']} assets)")

            bottlenecks = critical.get('bottleneck_assets', [])
            if bottlenecks:
                print(f"\n⚠️ Bottlenecks:")
                for bn in bottlenecks[:3]:
                    print(f"  • {bn['asset']} ({bn['in_degree']} → {bn['out_degree']})")

            critical_paths = critical.get('critical_paths', [])
            if critical_paths:
                print(f"\n🛣️ Caminhos Críticos:")
                for cp in critical_paths[:1]:
                    print(f"  • {cp['description']}")

        # Insights e Recomendações
        insights = analysis.get('insights', {})
        if insights:
            print(f"\n🤖 Insights Automáticos:")
            summary = insights.get('summary', '')
            if summary:
                # Limita a 200 caracteres
                summary_short = summary[:200] + "..." if len(summary) > 200 else summary
                print(f"  {summary_short}")

            print(f"\n💡 Avaliação de Risco: {insights.get('risk_assessment', 'N/A')}")

            recommendations = insights.get('recommendations', [])
            if recommendations:
                print(f"\n📋 Recomendações Principais:")
                for rec in recommendations[:3]:
                    print(f"  • {rec}")

        print("\n" + "="*60 + "\n")
    
    def analyze_impact(self, changed_assets: List[str]) -> Dict:
        """
        Analisa impacto de mudanças em assets específicos
        """
        if not self.current_analysis:
            self.log("❌ Nenhuma análise disponível. Execute analyze_project primeiro.", "ERROR")
            return {}
        
        self.log(f"🎯 Analisando impacto de mudanças em: {', '.join(changed_assets)}")
        
        impact = self.agent.analyze_change_impact(changed_assets)
        
        # Imprime resultados
        print("\n" + "="*60)
        print("💥 ANÁLISE DE IMPACTO")
        print("="*60)
        
        print(f"\n📍 Assets diretamente modificados: {len(impact['directly_affected'])}")
        for asset in impact['directly_affected']:
            print(f"  • {asset}")
        
        if impact['downstream_affected']:
            print(f"\n⬇️ Impacto Downstream ({len(impact['downstream_affected'])} assets):")
            for asset in list(impact['downstream_affected'])[:10]:  # Primeiros 10
                print(f"  • {asset}")
            if len(impact['downstream_affected']) > 10:
                print(f"  ... e {len(impact['downstream_affected']) - 10} outros")
        
        if impact['upstream_dependencies']:
            print(f"\n⬆️ Dependências Upstream ({len(impact['upstream_dependencies'])} assets):")
            for asset in list(impact['upstream_dependencies'])[:10]:
                print(f"  • {asset}")
            if len(impact['upstream_dependencies']) > 10:
                print(f"  ... e {len(impact['upstream_dependencies']) - 10} outros")
        
        print(f"\n⚠️ Nível de Risco: {impact['risk_level']}")
        
        if impact['recommendations']:
            print("\n💡 Recomendações:")
            for rec in impact['recommendations']:
                print(f"  {rec}")
        
        print("\n" + "="*60 + "\n")
        
        return impact
    
    def compare_versions(self,
                        old_project_path: str,
                        new_project_path: str,
                        file_patterns: List[str] = None) -> Dict:
        """
        Compara duas versões de um pipeline
        """
        self.log("🔄 Comparando versões de pipeline...")
        
        # Analisa versão antiga
        self.log("📖 Analisando versão antiga...")
        old_files = self._collect_files(old_project_path, 
                                       file_patterns or ['*.py', '*.sql', '*.tf'],
                                       recursive=True)
        
        # Analisa versão nova
        self.log("📗 Analisando versão nova...")
        new_files = self._collect_files(new_project_path,
                                       file_patterns or ['*.py', '*.sql', '*.tf'],
                                       recursive=True)
        
        # Compara
        comparison = self.agent.compare_versions(old_files, new_files)
        
        # Imprime resultados
        print("\n" + "="*60)
        print("🔍 COMPARAÇÃO DE VERSÕES")
        print("="*60)
        
        if comparison['added_assets']:
            print(f"\n✅ Assets Adicionados ({len(comparison['added_assets'])}):")
            for asset in comparison['added_assets'][:10]:
                print(f"  + {asset}")
        
        if comparison['removed_assets']:
            print(f"\n❌ Assets Removidos ({len(comparison['removed_assets'])}):")
            for asset in comparison['removed_assets'][:10]:
                print(f"  - {asset}")
        
        if comparison['added_connections']:
            print(f"\n🔗 Conexões Adicionadas ({len(comparison['added_connections'])}):")
            for conn in comparison['added_connections'][:5]:
                print(f"  + {conn[0]} → {conn[1]}")
        
        if comparison['removed_connections']:
            print(f"\n💔 Conexões Removidas ({len(comparison['removed_connections'])}):")
            for conn in comparison['removed_connections'][:5]:
                print(f"  - {conn[0]} → {conn[1]}")
        
        # Avaliação de risco
        if comparison.get('risk_assessment'):
            print("\n⚠️ Avaliação de Risco:")
            for asset, impact in comparison['risk_assessment'].get('removed_assets_impact', {}).items():
                if impact:
                    print(f"  • Remoção de '{asset}' afeta: {', '.join(impact[:3])}")
        
        print("\n" + "="*60 + "\n")
        
        return comparison
    
    def visualize(self,
                 visualization_type: str = "force",
                 output_file: str = None,
                 **kwargs) -> str:
        """
        Gera visualização do grafo de linhagem
        """
        if not self.visualizer:
            self.log("❌ Nenhuma análise disponível para visualização", "ERROR")
            return ""
        
        self.log(f"🎨 Gerando visualização: {visualization_type}")

        # Tratamento especial para atlas-interactive
        if visualization_type == 'atlas-interactive':
            if not output_file:
                output_file = 'lineage_atlas_interactive.html'
            kwargs['language'] = self.i18n.language
            return self.visualizer.visualize_atlas_interactive(output_file, **kwargs)

        # Mapa de tipos de visualização
        viz_methods = {
            'force': self.visualizer.visualize_force_directed,
            'hierarchical': self.visualizer.visualize_hierarchical,
            'sankey': self.visualizer.visualize_sankey,
            'impact': self.visualizer.visualize_impact_analysis,
            '3d': self.visualizer.visualize_3d_graph,
            'radial': self.visualizer.visualize_radial,
            'atlas': self.visualizer.visualize_atlas_style,
            'dashboard': self.visualizer.create_dashboard
        }

        if visualization_type not in viz_methods:
            self.log(f"❌ Tipo de visualização inválido: {visualization_type}", "ERROR")
            self.log(f"   Tipos disponíveis: {', '.join(list(viz_methods.keys()) + ['atlas-interactive'])}")
            return ""
        
        try:
            # Gera visualização
            fig = viz_methods[visualization_type](**kwargs)
            
            # Salva arquivo
            if not output_file:
                output_file = f"lineage_{visualization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            output_path = self.visualizer.export_to_html(fig, output_file)
            
            self.log(f"✅ Visualização salva em: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.log(f"❌ Erro ao gerar visualização: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return ""
    
    def export_analysis(self, output_format: str = "json", output_file: str = None) -> str:
        """
        Exporta análise em diferentes formatos
        """
        if not self.current_analysis:
            self.log("❌ Nenhuma análise disponível para exportar", "ERROR")
            return ""
        
        if not output_file:
            output_file = f"lineage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        
        try:
            if output_format == "json":
                # Exporta como JSON
                with open(output_file, 'w') as f:
                    # Converte objetos para dict
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'assets': [
                            {
                                'name': a.name,
                                'type': a.type,
                                'source_file': a.source_file,
                                'schema': a.schema,
                                'metadata': a.metadata
                            }
                            for a in self.current_analysis.get('assets', [])
                        ],
                        'transformations': [
                            {
                                'source': t.source.name,
                                'target': t.target.name,
                                'operation': t.operation,
                                'source_file': t.source_file,
                                'confidence': t.confidence_score
                            }
                            for t in self.current_analysis.get('transformations', [])
                        ],
                        'metrics': self.current_analysis.get('metrics', {}),
                        'critical_components': self.current_analysis.get('critical_components', {}),
                        'insights': self.current_analysis.get('insights', {})
                    }
                    json.dump(export_data, f, indent=2)
                
            elif output_format == "md":
                # Exporta como Markdown
                doc = self.agent.generate_documentation()
                with open(output_file, 'w') as f:
                    f.write(doc)
                    
            elif output_format == "graph":
                # Exporta dados do grafo
                if self.visualizer:
                    output_file = self.visualizer.export_to_json(output_file)
                else:
                    self.log("❌ Visualizador não disponível", "ERROR")
                    return ""
            
            else:
                self.log(f"❌ Formato não suportado: {output_format}", "ERROR")
                return ""
            
            self.log(f"✅ Análise exportada para: {output_file}")
            return output_file
            
        except Exception as e:
            self.log(f"❌ Erro ao exportar: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return ""
    
    def generate_report(self) -> str:
        """
        Gera relatório completo em HTML
        """
        if not self.current_analysis:
            self.log("❌ Nenhuma análise disponível", "ERROR")
            return ""
        
        self.log("📝 Gerando relatório completo...")
        
        # Gera múltiplas visualizações
        visualizations = []
        
        # Dashboard principal
        self.visualize('dashboard', 'report_dashboard.html')
        visualizations.append((self.i18n.t('dashboard'), 'report_dashboard.html'))

        # Force-directed
        self.visualize('force', 'report_force.html')
        visualizations.append((self.i18n.t('force_graph'), 'report_force.html'))

        # Sankey
        self.visualize('sankey', 'report_sankey.html')
        visualizations.append((self.i18n.t('sankey_diagram'), 'report_sankey.html'))
        
        # Gera documentação
        doc = self.agent.generate_documentation()
        
        # Cria relatório HTML principal
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.i18n.t('report_title')}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .visualization-links {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .viz-link {{
            display: inline-block;
            margin: 10px;
            padding: 10px 20px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .viz-link:hover {{
            background: #2980b9;
        }}
        .documentation {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        pre {{
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
        }}
        .insights-section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .insight-summary {{
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .risk-low {{
            background: #d4edda;
            color: #155724;
        }}
        .risk-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        .risk-high {{
            background: #f8d7da;
            color: #721c24;
        }}
        .recommendation-list {{
            list-style: none;
            padding: 0;
        }}
        .recommendation-list li {{
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        .critical-component {{
            background: #fff3cd;
            padding: 10px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 3px solid #ffc107;
        }}
        .warning-icon {{
            color: #ff9800;
        }}
        .error-icon {{
            color: #f44336;
        }}
    </style>
</head>
<body>
    <h1>📊 {self.i18n.t('report_title')}</h1>
    <p class="timestamp">{self.i18n.t('generated')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{self.current_analysis.get('metrics', {}).get('total_assets', 0)}</div>
            <div class="metric-label">{self.i18n.t('total_assets')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{self.current_analysis.get('metrics', {}).get('total_transformations', 0)}</div>
            <div class="metric-label">{self.i18n.t('transformations')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(self.current_analysis.get('metrics', {}).get('asset_types', {}))}</div>
            <div class="metric-label">{self.i18n.t('asset_types')}</div>
        </div>
    </div>
"""

        # Adiciona seção de insights
        insights = self.current_analysis.get('insights', {})
        critical = self.current_analysis.get('critical_components', {})

        if insights or critical:
            report_html += f"""
    <h2>{self.i18n.t('auto_insights_analysis')}</h2>
    <div class="insights-section">
"""

            # Resumo e avaliação de risco
            if insights:
                summary = insights.get('summary', '')
                risk = insights.get('risk_assessment', 'N/A')

                # Garante que risk é uma string (pode vir como dict do LLM)
                if risk is None:
                    risk = 'N/A'
                    risk_level = 'N/A'
                elif isinstance(risk, dict):
                    risk_level = risk.get('level', 'N/A')
                    justification = risk.get('justification', '')
                    risk = f"{risk_level}: {justification}" if justification else risk_level
                else:
                    risk_level = str(risk)

                # Determina classe CSS do risco
                risk_class = 'risk-low'
                if 'HIGH' in risk_level.upper():
                    risk_class = 'risk-high'
                elif 'MEDIUM' in risk_level.upper():
                    risk_class = 'risk-medium'

                report_html += f"""
        <h3>📊 {self.i18n.t('executive_summary')}</h3>
        <div class="insight-summary">
            <p>{summary}</p>
        </div>

        <h3>💡 {self.i18n.t('risk_assessment')}</h3>
        <div class="risk-badge {risk_class}">{risk}</div>
"""

                # Recomendações
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    report_html += f"""
        <h3>📋 {self.i18n.t('recommendations')}</h3>
        <ul class="recommendation-list">
"""
                    for rec in recommendations:
                        report_html += f"            <li>{rec}</li>\n"

                    report_html += "        </ul>\n"

            # Componentes críticos
            if critical:
                spof = critical.get('single_points_of_failure', [])
                if spof:
                    report_html += f"""
        <h3>{self.i18n.t('single_points_failure_title')}</h3>
"""
                    for point in spof[:5]:
                        report_html += f"""        <div class="critical-component">
            <strong>{point['asset']}</strong> ({point['type']})<br>
            {self.i18n.t('impacts').capitalize()} <strong>{point['downstream_impact']}</strong> {self.i18n.t('assets_downstream')}
        </div>
"""

                bottlenecks = critical.get('bottleneck_assets', [])
                if bottlenecks:
                    report_html += f"""
        <h3>{self.i18n.t('bottlenecks_identified_title')}</h3>
"""
                    for bn in bottlenecks[:5]:
                        report_html += f"""        <div class="critical-component">
            <strong>{bn['asset']}</strong> ({bn['type']})<br>
            {bn['in_degree']} {self.i18n.t('inputs')} → {bn['out_degree']} {self.i18n.t('outputs')}
        </div>
"""

                critical_paths = critical.get('critical_paths', [])
                if critical_paths:
                    report_html += """
        <h3>🛣️ Caminhos Críticos</h3>
"""
                    for cp in critical_paths[:3]:
                        path_str = ' → '.join(cp['path'][:10])
                        if len(cp['path']) > 10:
                            path_str += ' → ...'
                        report_html += f"""        <div class="critical-component">
            <strong>{cp['description']}</strong><br>
            <code style="font-size: 0.85em;">{path_str}</code>
        </div>
"""

            report_html += "    </div>\n"

        report_html += f"""
    <h2>📈 {self.i18n.t('visualizations')}</h2>
    <div class="visualization-links">
"""

        for viz_name, viz_file in visualizations:
            report_html += f'        <a href="{viz_file}" class="viz-link" target="_blank">{viz_name}</a>\n'

        report_html += f"""
    </div>

    <h2>📝 {self.i18n.t('detailed_docs')}</h2>
    <div class="documentation">
        <pre>{doc}</pre>
    </div>
    
</body>
</html>
"""
        
        report_file = f"lineage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(report_html)
        
        self.log(f"✅ Relatório completo gerado: {report_file}")
        return report_file


def main():
    """
    Função principal - CLI
    """
    parser = argparse.ArgumentParser(
        description="Data Lineage AI Agent - Análise automática de linhagem de dados"
    )
    
    parser.add_argument(
        "project_path",
        help="Caminho para o projeto a ser analisado"
    )
    
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=['*.py', '*.sql', '*.tf', '*.json', '*.scala'],
        help="Padrões de arquivo para análise (default: *.py *.sql *.tf *.json *.scala)"
    )
    
    parser.add_argument(
        "--visualize",
        choices=['force', 'hierarchical', 'sankey', 'impact', '3d', 'radial', 'dashboard'],
        default='dashboard',
        help="Tipo de visualização a gerar (default: dashboard)"
    )
    
    parser.add_argument(
        "--impact",
        nargs="+",
        help="Analisa impacto de mudanças nos assets especificados"
    )
    
    parser.add_argument(
        "--compare",
        help="Caminho para segunda versão do projeto para comparação"
    )
    
    parser.add_argument(
        "--export",
        choices=['json', 'md', 'graph'],
        help="Formato de exportação da análise"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Gera relatório HTML completo"
    )
    
    parser.add_argument(
        "--output",
        help="Nome do arquivo de saída"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso (menos output)"
    )
    
    args = parser.parse_args()
    
    # Cria sistema
    system = DataLineageSystem(verbose=not args.quiet)
    
    # Executa análise principal
    analysis = system.analyze_project(
        args.project_path,
        file_patterns=args.patterns
    )
    
    if not analysis:
        print("❌ Análise falhou")
        sys.exit(1)
    
    # Análise de impacto
    if args.impact:
        system.analyze_impact(args.impact)
    
    # Comparação de versões
    if args.compare:
        system.compare_versions(args.project_path, args.compare, args.patterns)
    
    # Visualização
    if args.visualize:
        system.visualize(args.visualize, args.output)
    
    # Exportação
    if args.export:
        system.export_analysis(args.export, args.output)
    
    # Relatório completo
    if args.report:
        system.generate_report()
    
    print("\n✅ Análise concluída com sucesso!")


if __name__ == "__main__":
    main()
