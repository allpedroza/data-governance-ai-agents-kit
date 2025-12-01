# Novos Parsers: Azure Synapse/Data Factory e Parquet

Este documento descreve os dois novos parsers adicionados ao sistema de governança de dados:

1. **synapse_parser.py** - Parser para JSON do Azure Data Factory e Synapse Analytics
2. **parquet_parser.py** - Parser para arquivos Parquet

## 1. Synapse/Data Factory Parser

### Visão Geral

O `SynapseParser` analisa definições JSON de pipelines, datasets, linked services e data flows do Azure Data Factory e Azure Synapse Analytics, extraindo informações de lineage de dados.

### Características

- ✅ Parse de pipelines JSON com atividades e dependências
- ✅ Parse de datasets (JSON, Parquet, SQL, CSV, etc.)
- ✅ Parse de linked services (conexões)
- ✅ Parse de data flows (fluxos de transformação)
- ✅ Extração de lineage para atividades Copy
- ✅ Suporte a atividades Script, DataFlow, Lookup
- ✅ Análise de complexidade de pipelines
- ✅ Geração de diagramas Mermaid

### Classes Principais

#### SynapsePipeline
Representa um pipeline completo com suas atividades.

```python
@dataclass
class SynapsePipeline:
    name: str
    description: Optional[str]
    activities: Dict[str, SynapseActivity]
    parameters: Dict[str, Any]
    variables: Dict[str, Any]
    annotations: List[str]
    source_file: str
```

#### SynapseActivity
Representa uma atividade dentro de um pipeline (Copy, DataFlow, Script, etc.).

```python
@dataclass
class SynapseActivity:
    name: str
    type: str  # Copy, Lookup, Script, DataFlow, etc.
    pipeline_name: str
    dependencies: List[str]
    inputs: List[str]
    outputs: List[str]
    source_dataset: Optional[str]
    sink_dataset: Optional[str]
    sql_query: Optional[str]
```

#### SynapseDataset
Representa um dataset (tabela, arquivo, etc.).

```python
@dataclass
class SynapseDataset:
    name: str
    type: str  # Json, Parquet, AzureSqlTable, etc.
    linked_service: str
    location: Optional[Dict[str, Any]]
    schema: Optional[List[Dict[str, Any]]]
```

### Uso Básico

#### Parsear um único pipeline

```python
from parsers.synapse_parser import parse_synapse_pipeline

# Parse um pipeline JSON
pipeline = parse_synapse_pipeline('path/to/pipeline.json')

print(f"Pipeline: {pipeline.name}")
print(f"Atividades: {len(pipeline.activities)}")

for activity_name, activity in pipeline.activities.items():
    print(f"  - {activity_name} ({activity.type})")
    if activity.inputs:
        print(f"    Inputs: {activity.inputs}")
    if activity.outputs:
        print(f"    Outputs: {activity.outputs}")
```

#### Parsear um diretório completo

```python
from parsers.synapse_parser import parse_synapse_directory

# Parse todos os arquivos JSON em uma estrutura Data Factory
parser = parse_synapse_directory('path/to/datafactory')

print(f"Pipelines encontrados: {len(parser.pipelines)}")
print(f"Datasets encontrados: {len(parser.datasets)}")
print(f"Linked Services: {len(parser.linked_services)}")
print(f"Data Flows: {len(parser.data_flows)}")
```

#### Análise completa com lineage

```python
from parsers.synapse_parser import analyze_synapse_project

# Análise completa do projeto
result = analyze_synapse_project('path/to/datafactory')

parser = result['parser']
lineage = result['lineage_graph']

# Exibir grafo de lineage
print(f"\nNós no grafo: {len(lineage['nodes'])}")
print(f"Arestas no grafo: {len(lineage['edges'])}")

# Análise de complexidade por pipeline
for pipeline_name, analysis in result['pipeline_analyses'].items():
    print(f"\nPipeline: {pipeline_name}")
    print(f"  Total de atividades: {analysis['total_activities']}")
    print(f"  Atividades Copy: {analysis['copy_activities']}")
    print(f"  Profundidade máxima: {analysis['max_dependency_depth']}")
    print(f"  Pontos de entrada: {analysis['entry_points']}")
```

#### Gerar diagrama Mermaid

```python
parser = parse_synapse_directory('path/to/datafactory')

# Gerar diagrama para um pipeline específico
diagram = parser.generate_mermaid_diagram('MeuPipeline')
print(diagram)
```

### Estrutura de Diretórios Esperada

O parser suporta a estrutura padrão do Azure Data Factory:

```
datafactory/
├── pipeline/
│   ├── pipeline1.json
│   └── pipeline2.json
├── dataset/
│   ├── dataset1.json
│   └── dataset2.json
├── linkedService/
│   ├── AzureBlobStorage.json
│   └── AzureSqlDatabase.json
└── dataflow/
    └── dataflow1.json
```

### Tipos de Atividades Suportadas

- **Copy**: Cópia de dados entre source e sink
- **DataFlow**: Transformações de dados
- **ExecuteDataFlow**: Execução de data flow
- **Script**: Execução de scripts SQL
- **Lookup**: Consulta de dados
- **SqlServerStoredProcedure**: Execução de stored procedures
- **GetMetadata**: Obtenção de metadados

### Exemplo de JSON de Pipeline

```json
{
  "name": "CopyPipeline",
  "properties": {
    "activities": [
      {
        "name": "CopyData",
        "type": "Copy",
        "inputs": [
          {
            "referenceName": "SourceDataset",
            "type": "DatasetReference"
          }
        ],
        "outputs": [
          {
            "referenceName": "SinkDataset",
            "type": "DatasetReference"
          }
        ],
        "typeProperties": {
          "source": {
            "type": "AzureSqlSource",
            "sqlReaderQuery": "SELECT * FROM table1"
          },
          "sink": {
            "type": "ParquetSink"
          }
        }
      }
    ]
  }
}
```

---

## 2. Parquet Parser

### Visão Geral

O `ParquetParser` analisa arquivos Parquet para extrair schema, metadados, estatísticas e informações de particionamento.

### Características

- ✅ Leitura de schema completo com tipos de dados
- ✅ Extração de metadados e estatísticas por coluna
- ✅ Suporte a datasets particionados (Hive-style)
- ✅ Detecção de compressão
- ✅ Comparação de schemas entre arquivos
- ✅ Detecção automática de relacionamentos (foreign keys)
- ✅ Geração de grafo de lineage

### Pré-requisitos

O parser Parquet requer a biblioteca `pyarrow`:

```bash
pip install pyarrow
```

### Classes Principais

#### ParquetFile
Representa um arquivo Parquet individual.

```python
@dataclass
class ParquetFile:
    file_path: str
    name: str
    schema: ParquetSchema
    num_rows: int
    num_row_groups: int
    file_size: int
    compression: Optional[str]
    format_version: Optional[str]
    partition_info: Dict[str, str]
```

#### ParquetColumn
Representa uma coluna no schema.

```python
@dataclass
class ParquetColumn:
    name: str
    type: str
    physical_type: Optional[str]
    logical_type: Optional[str]
    nullable: bool
    statistics: Dict[str, Any]
```

#### ParquetDataset
Representa um conjunto de arquivos Parquet (dataset particionado).

```python
@dataclass
class ParquetDataset:
    name: str
    root_path: str
    files: List[ParquetFile]
    common_schema: Optional[ParquetSchema]
    partition_columns: List[str]
    total_rows: int
    total_size: int
```

### Uso Básico

#### Parsear um único arquivo

```python
from parsers.parquet_parser import parse_parquet_file

# Parse um arquivo Parquet
parquet_file = parse_parquet_file('path/to/file.parquet')

print(f"Arquivo: {parquet_file.name}")
print(f"Linhas: {parquet_file.num_rows:,}")
print(f"Colunas: {len(parquet_file.schema.columns)}")
print(f"Tamanho: {parquet_file.file_size / (1024*1024):.2f} MB")
print(f"Compressão: {parquet_file.compression}")

# Exibir schema
print("\nSchema:")
for col in parquet_file.schema.columns:
    print(f"  - {col.name}: {col.type} (nullable: {col.nullable})")
    if col.statistics:
        print(f"    Stats: {col.statistics}")
```

#### Parsear um dataset particionado

```python
from parsers.parquet_parser import parse_parquet_directory

# Parse todos os arquivos Parquet em um diretório
dataset = parse_parquet_directory('path/to/dataset', recursive=True)

print(f"Dataset: {dataset.name}")
print(f"Arquivos: {len(dataset.files)}")
print(f"Total de linhas: {dataset.total_rows:,}")
print(f"Tamanho total: {dataset.total_size / (1024*1024*1024):.2f} GB")
print(f"Colunas de partição: {dataset.partition_columns}")

# Exibir informações de cada arquivo
for file in dataset.files:
    print(f"\n  {file.name}")
    print(f"    Partições: {file.partition_info}")
    print(f"    Linhas: {file.num_rows:,}")
```

#### Comparar schemas entre arquivos

```python
from parsers.parquet_parser import ParquetParser

parser = ParquetParser()

# Comparar dois arquivos
comparison = parser.compare_schemas(
    'path/to/file1.parquet',
    'path/to/file2.parquet'
)

print(f"Schemas compatíveis: {comparison['schemas_match']}")
print(f"Colunas apenas no arquivo 1: {comparison['only_in_first']}")
print(f"Colunas apenas no arquivo 2: {comparison['only_in_second']}")
print(f"Diferenças de tipo: {comparison['type_differences']}")
```

#### Análise completa com detecção de relacionamentos

```python
from parsers.parquet_parser import analyze_parquet_dataset

# Análise completa
result = analyze_parquet_dataset('path/to/dataset')

dataset = result['dataset']
relationships = result['relationships']

print(f"\nDataset: {dataset.name}")
print(result['summary'])

# Relacionamentos detectados
print("\nRelacionamentos detectados:")
for rel in relationships:
    print(f"  {rel['foreign_key_column']} -> {rel['potential_primary_table']}")
    print(f"    Presente em: {rel['in_files']}")
    print(f"    Confiança: {rel['confidence']}")
```

### Detecção de Particionamento

O parser detecta automaticamente particionamento estilo Hive:

```
dataset/
├── year=2024/
│   ├── month=01/
│   │   ├── day=01/
│   │   │   └── data.parquet
│   │   └── day=02/
│   │       └── data.parquet
│   └── month=02/
│       └── day=01/
│           └── data.parquet
```

```python
dataset = parse_parquet_directory('dataset')

print(f"Colunas de partição: {dataset.partition_columns}")
# Output: ['year', 'month', 'day']

for file in dataset.files:
    print(f"{file.name}: {file.partition_info}")
# Output: data: {'year': '2024', 'month': '01', 'day': '01'}
```

### Mapeamento de Tipos de Dados

O parser mapeia tipos Parquet para tipos comuns:

| Tipo Parquet | Tipo Comum |
|--------------|------------|
| BOOLEAN | boolean |
| INT32 | int32 |
| INT64 | int64 |
| FLOAT | float |
| DOUBLE | double |
| STRING/UTF8 | string |
| DECIMAL | decimal |
| DATE | date |
| TIMESTAMP | timestamp |
| LIST | array |
| MAP | map |
| STRUCT | struct |

---

## Integração com DataLineageAgent

Ambos os parsers podem ser integrados ao `DataLineageAgent` existente:

```python
from data_lineage_agent import DataLineageAgent
from parsers.synapse_parser import SynapseParser
from parsers.parquet_parser import ParquetParser

# Criar agente
agent = DataLineageAgent()

# Parse arquivos Synapse/Data Factory
synapse_parser = SynapseParser()
synapse_parser.parse_directory('path/to/datafactory')

# Converter para DataAssets
for pipeline_name, pipeline in synapse_parser.pipelines.items():
    for activity_name, activity in pipeline.activities.items():
        if activity.type == 'Copy':
            # Criar transformação para atividade Copy
            agent.add_transformation(
                source=activity.source_dataset or 'unknown',
                target=activity.sink_dataset or 'unknown',
                transformation_type='copy',
                details={'pipeline': pipeline_name, 'activity': activity_name}
            )

# Parse arquivos Parquet
parquet_parser = ParquetParser()
dataset = parquet_parser.parse_directory('path/to/parquet_data')

# Adicionar arquivos Parquet como assets
for file in dataset.files:
    agent.add_asset(
        name=file.name,
        asset_type='parquet_file',
        location=file.file_path,
        metadata={
            'num_rows': file.num_rows,
            'num_columns': len(file.schema.columns),
            'compression': file.compression
        }
    )
```

## Exemplos de Uso Avançado

### 1. Análise de Pipeline Synapse com Visualização

```python
from parsers.synapse_parser import analyze_synapse_project

result = analyze_synapse_project('datafactory')
parser = result['parser']

# Para cada pipeline, gerar diagrama e análise
for pipeline_name in parser.pipelines.keys():
    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline_name}")
    print(f"{'='*60}")

    # Análise de complexidade
    analysis = parser.analyze_pipeline_complexity(pipeline_name)
    print(f"\nMétricas:")
    print(f"  Atividades totais: {analysis['total_activities']}")
    print(f"  Atividades Copy: {analysis['copy_activities']}")
    print(f"  Profundidade: {analysis['max_dependency_depth']}")
    print(f"  Parâmetros: {analysis['parameter_count']}")

    # Diagrama Mermaid
    print(f"\nDiagrama Mermaid:")
    print(parser.generate_mermaid_diagram(pipeline_name))
```

### 2. Auditoria de Schemas Parquet

```python
from parsers.parquet_parser import ParquetParser

parser = ParquetParser()
dataset = parser.parse_directory('data/warehouse')

# Verificar consistência de schemas
schemas_by_name = {}
for file in dataset.files:
    base_name = file.name.split('_')[0]  # Agrupar por prefixo

    if base_name not in schemas_by_name:
        schemas_by_name[base_name] = []

    schemas_by_name[base_name].append(file)

# Verificar inconsistências
for name, files in schemas_by_name.items():
    if len(files) > 1:
        print(f"\nVerificando grupo: {name} ({len(files)} arquivos)")

        reference = files[0]
        for file in files[1:]:
            comparison = parser.compare_schemas(
                reference.file_path,
                file.file_path
            )

            if not comparison['schemas_match']:
                print(f"  ⚠️  Inconsistência: {file.name}")
                print(f"     Colunas extras: {comparison['only_in_second']}")
                print(f"     Colunas faltando: {comparison['only_in_first']}")
                print(f"     Tipos diferentes: {comparison['type_differences']}")
```

### 3. Rastreamento de Lineage End-to-End

```python
from parsers.synapse_parser import SynapseParser
from parsers.parquet_parser import ParquetParser

# Parse pipeline e dados
synapse = SynapseParser()
synapse.parse_directory('datafactory')

parquet = ParquetParser()
parquet.parse_directory('data/output')

# Conectar lineage
print("Lineage End-to-End:\n")

for pipeline_name, pipeline in synapse.pipelines.items():
    for activity_name, activity in pipeline.activities.items():
        if activity.type == 'Copy' and activity.sink_dataset:
            # Verificar se o output existe como Parquet
            sink_name = activity.sink_dataset

            matching_files = [
                f for f in parquet.files.values()
                if sink_name.lower() in f.name.lower()
            ]

            if matching_files:
                print(f"Pipeline: {pipeline_name}")
                print(f"  Atividade: {activity_name}")
                print(f"  Source: {activity.source_dataset}")
                print(f"  Sink: {activity.sink_dataset}")
                print(f"  Arquivos Parquet gerados:")
                for file in matching_files:
                    print(f"    - {file.name} ({file.num_rows:,} linhas)")
                print()
```

## Referências

- [Azure Data Factory JSON Format Documentation](https://docs.microsoft.com/azure/data-factory/format-json)
- [Apache Parquet Documentation](https://parquet.apache.org/docs/)
- [PyArrow Parquet Documentation](https://arrow.apache.org/docs/python/parquet.html)

## Contribuindo

Para adicionar suporte a novos tipos de atividades ou formatos:

1. Adicione o padrão em `SynapseParser.DATA_ACTIVITY_TYPES`
2. Implemente método `_parse_<activity_type>_activity()`
3. Atualize a documentação

Para novos tipos de dados Parquet:

1. Adicione mapeamento em `ParquetParser.TYPE_MAPPING`
2. Atualize `_map_arrow_type()` se necessário
