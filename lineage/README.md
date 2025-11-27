# 🚀 Data Lineage AI Agent

Um agente de IA especializado em identificar, analisar e visualizar a linhagem de dados em pipelines complexos. Suporta múltiplos formatos de arquivo e oferece visualizações interativas avançadas.

## ✨ Características Principais

### 🔍 Análise Inteligente
- **Multi-formato**: Python, SQL, Terraform, Databricks, Spark
- **Detecção Automática**: Identifica assets e transformações automaticamente
- **Análise de Impacto**: Simula mudanças e mostra impactos upstream/downstream
- **Comparação de Versões**: Compara diferentes versões de pipelines

### 📊 Visualizações Interativas
- **Force-Directed Graph**: Visualização dinâmica com física de forças
- **Hierárquica**: Árvore de dependências
- **Sankey Diagram**: Fluxo de dados entre componentes
- **3D Graph**: Visualização tridimensional interativa
- **Radial Layout**: Vista centrada em assets específicos
- **Dashboard Completo**: Múltiplas métricas em uma única tela
- **Impact Analysis**: Destaque visual de áreas afetadas

### 🎯 Recursos Avançados
- **Detecção de Ciclos**: Identifica dependências circulares
- **Métricas de Complexidade**: Densidade, conectividade, componentes
- **Documentação Automática**: Gera documentação detalhada
- **Exportação Flexível**: JSON, Markdown, HTML
- **Relatórios Completos**: HTML interativo com todas as visualizações

## 📦 Instalação

### Requisitos
- Python 3.8+
- pip

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

### Dependências Principais
- `networkx`: Análise de grafos
- `plotly`: Visualizações interativas
- `sqlparse`: Parsing de SQL
- `pandas`: Manipulação de dados
- `matplotlib`: Visualizações estáticas
- `requests`: Integração opcional com LLM para extração contextual de linhagem

### Setup rápido com ambiente virtual

```bash
# 1) Crie e ative um ambiente virtual chamado dgagentkit
python -m venv dgagentkit
source dgagentkit/bin/activate  # Linux/macOS
# .\\dgagentkit\\Scripts\\activate  # Windows PowerShell

# 2) Instale as dependências dentro do venv
pip install -r requirements.txt
```

### Integração Opcional com LLM
- Defina `OPENAI_API_KEY` para habilitar o fallback de extração contextual (modelo padrão `gpt-4o-mini`).
- Variáveis opcionais: `DATA_LINEAGE_LLM_MODEL` (nome do modelo) e `OPENAI_API_URL` (endpoint compatível com OpenAI). Sem token, o parser continua usando apenas regras determinísticas.

Para configurar tokens antes de rodar a análise e aproveitar o fallback contextual:

```bash
# 3) Configure as variáveis de ambiente para o LLM
export OPENAI_API_KEY="seu_token"
export DATA_LINEAGE_LLM_MODEL="gpt-4o-mini"          # opcional
export OPENAI_API_URL="https://api.openai.com/v1/chat/completions"  # opcional

# 4) Execute a análise completa (CLI) após configurar o LLM
python lineage_system.py /caminho/para/projeto --visualize dashboard --report
```

## 🚀 Uso Rápido

### 1. Uso via CLI

```bash
# Análise básica
python lineage_system.py /caminho/para/projeto

# Com visualização específica
python lineage_system.py /caminho/para/projeto --visualize dashboard

# Análise de impacto
python lineage_system.py /caminho/para/projeto --impact table1 table2

# Comparação de versões
python lineage_system.py /versao/antiga --compare /versao/nova

# Relatório completo
python lineage_system.py /caminho/para/projeto --report

# Exportar resultados
python lineage_system.py /caminho/para/projeto --export json --output results.json
```

### 2. Uso Programático

```python
from lineage_system import DataLineageSystem

# Inicializa o sistema
system = DataLineageSystem(verbose=True)

# Analisa projeto
analysis = system.analyze_project(
    '/caminho/para/projeto',
    file_patterns=['*.py', '*.sql', '*.tf']
)

# Análise de impacto
impact = system.analyze_impact(['tabela_modificada'])

# Gera visualização
system.visualize('dashboard', 'output.html')

# Gera relatório completo
system.generate_report()
```

### 3. Exemplo Rápido

```bash
# Executa demonstração completa com dados de exemplo
python example_usage.py
```

## 📊 Tipos de Visualização

### Force-Directed Graph
Grafo interativo com simulação de forças físicas. Ideal para explorar conexões complexas.

```python
system.visualize('force', 'force_graph.html')
```

### Hierarchical View
Visualização em árvore, mostrando hierarquia de dependências.

```python
system.visualize('hierarchical', 'hierarchy.html')
```

### Sankey Diagram
Fluxo de dados entre componentes, mostrando volume e direção.

```python
system.visualize('sankey', 'data_flow.html')
```

### Impact Analysis
Destaca visualmente assets afetados por mudanças.

```python
system.visualize('impact', 'impact.html', changed_nodes=['table1', 'table2'])
```

### 3D Visualization
Exploração tridimensional do grafo de dependências.

```python
system.visualize('3d', '3d_graph.html')
```

### Radial Layout
Vista centrada em um asset específico.

```python
system.visualize('radial', 'radial.html', center_node='main_table')
```

### Dashboard
Visão geral com múltiplas métricas e mini-visualizações.

```python
system.visualize('dashboard', 'dashboard.html')
```

## 🔧 Análise de Impacto

O sistema oferece análise detalhada de impacto para mudanças planejadas:

```python
# Identifica assets que serão modificados
changed_assets = ['dim_customer', 'fact_sales']

# Executa análise de impacto
impact = system.analyze_impact(changed_assets)

# Resultados incluem:
# - directly_affected: Assets modificados diretamente
# - downstream_affected: Assets impactados downstream
# - upstream_dependencies: Dependências upstream
# - risk_level: Nível de risco (LOW/MEDIUM/HIGH)
# - recommendations: Recomendações baseadas na análise
```

## 📁 Formatos Suportados

### Python (.py)
- Pandas operations (read_csv, to_parquet, etc.)
- PySpark transformations
- SQLAlchemy queries
- Dask operations
- Polars dataframes

### SQL (.sql)
- CREATE TABLE/VIEW statements
- INSERT/UPDATE/DELETE operations
- SELECT queries com JOINs
- CTEs e subqueries
- Stored procedures

### Terraform (.tf, .json)
- AWS Glue resources
- Databricks tables
- BigQuery datasets
- Azure Data Factory
- S3/GCS buckets

### Databricks
- Notebooks Python (.py)
- Notebooks Scala (.scala)
- SQL notebooks
- Delta Lake operations
- Streaming operations

## 📈 Métricas e Estatísticas

O sistema calcula automaticamente:

- **Total de Assets**: Quantidade de tabelas, arquivos, e recursos
- **Total de Transformações**: Número de operações entre assets
- **Tipos de Assets**: Distribuição por tipo (table, file, view, etc.)
- **Tipos de Operações**: CREATE, SELECT, INSERT, UPDATE, etc.
- **Complexidade do Grafo**:
  - Densidade
  - Grau médio
  - Componentes conectados
  - Detecção de ciclos

## 🔍 Comparação de Versões

Compare duas versões de um pipeline:

```python
comparison = system.compare_versions(
    old_project_path='/v1/pipeline',
    new_project_path='/v2/pipeline'
)

# Resultados incluem:
# - added_assets: Novos assets
# - removed_assets: Assets removidos
# - modified_assets: Assets modificados
# - added_connections: Novas dependências
# - removed_connections: Dependências removidas
# - risk_assessment: Avaliação de riscos
```

## 📝 Exportação de Resultados

### JSON
```python
system.export_analysis('json', 'analysis.json')
```

### Markdown
```python
system.export_analysis('md', 'documentation.md')
```

### Graph Data
```python
system.export_analysis('graph', 'graph_data.json')
```

### HTML Report
```python
system.generate_report()  # Gera relatório HTML completo
```

## 🏗️ Arquitetura

### Componentes Principais

1. **DataLineageAgent** (`data_lineage_agent.py`)
   - Parser multi-formato
   - Extração de assets e transformações
   - Construção do grafo de dependências
   - Análise de impacto

2. **DataLineageVisualizer** (`visualization_engine.py`)
   - Engines de visualização
   - Layouts de grafo
   - Exportação HTML/JSON
   - Temas e estilos

3. **DataLineageSystem** (`lineage_system.py`)
   - Orquestração do sistema
   - Interface CLI
   - Geração de relatórios
   - Cache e otimizações

## 🎯 Casos de Uso

### 1. Documentação Automática
Gere documentação atualizada do seu pipeline de dados:

```bash
python lineage_system.py /projeto --export md --output docs/lineage.md
```

### 2. Análise de Impacto para Mudanças
Antes de modificar uma tabela, veja o impacto:

```bash
python lineage_system.py /projeto --impact dim_customer
```

### 3. Auditoria de Pipeline
Identifique pontos de falha e gargalos:

```bash
python lineage_system.py /projeto --visualize dashboard --report
```

### 4. Migração de Dados
Compare versões antiga e nova do pipeline:

```bash
python lineage_system.py /old_version --compare /new_version
```

### 5. Compliance e Governança
Rastreie a origem e transformações dos dados:

```bash
python lineage_system.py /projeto --export json --output compliance_report.json
```

## 🛠️ Configuração Avançada

### Padrões de Arquivo Personalizados
```python
system.analyze_project(
    project_path,
    file_patterns=['*.py', '*.sql', '*.scala', 'pipeline_*.json']
)
```

### Filtros de Visualização
```python
# Visualiza apenas subgrafo específico
system.visualize(
    'force',
    highlight_nodes=['table1', 'table2'],
    show_labels=True
)
```

### Threshold de Impacto
```python
# Define limites para análise de risco
impact = system.analyze_impact(
    changed_assets,
    risk_thresholds={'high': 10, 'medium': 5}
)
```

## 📊 Exemplos de Saída

### Análise de Pipeline
```
📊 RESUMO DA ANÁLISE DE LINHAGEM
==================================================
📈 Estatísticas Gerais:
  • Total de Assets: 42
  • Total de Transformações: 67

🗂️ Tipos de Assets:
  • table: 25
  • file: 10
  • view: 5
  • terraform_resource: 2

⚙️ Tipos de Operações:
  • SELECT: 30
  • CREATE: 15
  • INSERT: 12
  • UPDATE: 10

🔧 Métricas de Complexidade:
  • Nós no grafo: 42
  • Arestas no grafo: 67
  • Densidade: 0.039
  • Grau médio: 3.19
```

### Análise de Impacto
```
💥 ANÁLISE DE IMPACTO
==================================================
📍 Assets diretamente modificados: 2
  • dim_customer
  • fact_sales

⬇️ Impacto Downstream (15 assets):
  • sales_summary
  • mv_daily_kpis
  • customer_segments
  • ml_prepared_data
  • churn_predictions
  ... e 10 outros

⚠️ Nível de Risco: HIGH

💡 Recomendações:
  ⚠️ Alto impacto detectado. Considere testes extensivos.
  📊 Recomenda-se análise detalhada dos pipelines críticos.
  🎯 Pipelines críticos afetados: sales_summary, mv_daily_kpis
```

## 🐛 Troubleshooting

### Problema: "Formato não suportado"
**Solução**: Verifique se o arquivo tem extensão correta (.py, .sql, .tf, etc.)

### Problema: "Nenhum asset detectado"
**Solução**: Verifique se os arquivos contêm operações de dados reconhecíveis

### Problema: Visualização não carrega
**Solução**: Certifique-se de que plotly está instalado: `pip install plotly`

### Problema: Análise muito lenta
**Solução**: Use padrões de arquivo mais específicos ou analise por diretório

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- NetworkX pela biblioteca de grafos
- Plotly pela engine de visualização
- SQLParse pelo parser SQL
- Comunidade Python pelos pacotes essenciais

## 📞 Suporte

Para suporte, abra uma issue no repositório ou entre em contato através do email do projeto.

---

**Desenvolvido com ❤️ por Claude AI Assistant**

*Última atualização: Novembro 2024*
