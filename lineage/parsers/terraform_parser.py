"""
Advanced Terraform Parser for Data Lineage Analysis
Analyzes Terraform configurations to extract data infrastructure dependencies
"""

import re
import json
import hcl2
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import ast


@dataclass
class TerraformResource:
    """Represents a Terraform resource"""
    type: str  # aws_s3_bucket, google_bigquery_dataset, etc
    name: str
    provider: str  # aws, google, azure, databricks
    attributes: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    data_assets: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0
    
    def __hash__(self):
        return hash(f"{self.type}.{self.name}")
    
    def get_full_name(self):
        return f"{self.type}.{self.name}"


class TerraformParser:
    """
    Advanced parser for Terraform configurations
    Extracts data infrastructure and dependencies
    """
    
    # Data-related resource types by provider
    DATA_RESOURCES = {
        'aws': {
            's3_bucket': 'storage',
            'glue_catalog_database': 'database',
            'glue_catalog_table': 'table',
            'kinesis_stream': 'stream',
            'dynamodb_table': 'table',
            'rds_cluster': 'database',
            'redshift_cluster': 'database',
            'emr_cluster': 'compute',
            'athena_database': 'database',
            'athena_workgroup': 'compute',
            'lakeformation_resource': 'catalog',
            'msk_cluster': 'stream',  # Kafka
            'elasticsearch_domain': 'search',
            'timestream_database': 'database',
            'sagemaker_endpoint': 'ml_model'
        },
        'google': {
            'storage_bucket': 'storage',
            'bigquery_dataset': 'database',
            'bigquery_table': 'table',
            'pubsub_topic': 'stream',
            'pubsub_subscription': 'stream',
            'dataflow_job': 'compute',
            'dataproc_cluster': 'compute',
            'composer_environment': 'orchestrator',
            'bigtable_instance': 'database',
            'spanner_database': 'database',
            'cloud_sql_database': 'database',
            'vertex_ai_endpoint': 'ml_model'
        },
        'azure': {
            'storage_account': 'storage',
            'storage_container': 'storage',
            'data_lake_store': 'storage',
            'sql_database': 'database',
            'synapse_workspace': 'database',
            'synapse_sql_pool': 'database',
            'eventhub_namespace': 'stream',
            'stream_analytics_job': 'compute',
            'databricks_workspace': 'compute',
            'cosmosdb_account': 'database',
            'data_factory': 'orchestrator',
            'machine_learning_workspace': 'ml_platform'
        },
        'databricks': {
            'cluster': 'compute',
            'job': 'job',
            'notebook': 'code',
            'dbfs_file': 'storage',
            'table': 'table',
            'schema': 'database',
            'catalog': 'catalog',
            'sql_endpoint': 'compute',
            'delta_table': 'table',
            'pipeline': 'pipeline',
            'model_serving_endpoint': 'ml_model'
        },
        'snowflake': {
            'database': 'database',
            'schema': 'schema',
            'table': 'table',
            'view': 'view',
            'stage': 'storage',
            'pipe': 'pipeline',
            'stream': 'stream',
            'task': 'job',
            'warehouse': 'compute'
        }
    }
    
    def __init__(self):
        self.resources: Dict[str, TerraformResource] = {}
        self.data_sources: Dict[str, TerraformResource] = {}
        self.variables: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.modules: Dict[str, Any] = {}
        
    def parse(self, content: str, file_path: str) -> Tuple[List[TerraformResource], List[Dict]]:
        """
        Parse Terraform configuration file
        Returns resources and their dependencies
        """
        resources = []
        dependencies = []
        
        try:
            # Tenta HCL2 primeiro (formato mais recente)
            config = self._parse_hcl2(content)
        except:
            try:
                # Fallback para JSON
                config = json.loads(content)
            except:
                # Tenta parsing manual
                config = self._manual_parse(content)
        
        if not config:
            return resources, dependencies
            
        # Extrai recursos
        if 'resource' in config:
            for resource_type, resource_configs in config['resource'].items():
                provider = resource_type.split('_')[0]
                
                for resource_name, resource_config in resource_configs.items():
                    resource = self._extract_resource(
                        resource_type, 
                        resource_name, 
                        resource_config,
                        provider,
                        file_path
                    )
                    
                    if resource:
                        resources.append(resource)
                        self.resources[resource.get_full_name()] = resource
        
        # Extrai data sources
        if 'data' in config:
            for data_type, data_configs in config['data'].items():
                for data_name, data_config in data_configs.items():
                    # Data sources são tratados como recursos de leitura
                    provider = data_type.split('_')[0]
                    resource = TerraformResource(
                        type=f"data.{data_type}",
                        name=data_name,
                        provider=provider,
                        attributes=data_config,
                        source_file=file_path
                    )
                    self.data_sources[resource.get_full_name()] = resource
        
        # Extrai variáveis
        if 'variable' in config:
            self.variables.update(config['variable'])
        
        # Extrai outputs
        if 'output' in config:
            self.outputs.update(config['output'])
            
        # Extrai módulos
        if 'module' in config:
            for module_name, module_config in config['module'].items():
                self.modules[module_name] = module_config
                # Módulos podem conter recursos de dados
                if 'source' in module_config:
                    self._analyze_module(module_name, module_config)
        
        # Analisa dependências
        dependencies = self._extract_dependencies()
        
        return resources, dependencies
    
    def _parse_hcl2(self, content: str) -> Dict:
        """Parse HCL2 format"""
        import hcl2
        import io
        
        with io.StringIO(content) as file:
            return hcl2.load(file)
    
    def _manual_parse(self, content: str) -> Dict:
        """Manual parsing for simple Terraform configurations"""
        config = {
            'resource': {},
            'data': {},
            'variable': {},
            'output': {},
            'module': {}
        }
        
        # Regex patterns
        resource_pattern = r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{([^}]+)\}'
        data_pattern = r'data\s+"([^"]+)"\s+"([^"]+)"\s*\{([^}]+)\}'
        variable_pattern = r'variable\s+"([^"]+)"\s*\{([^}]+)\}'
        output_pattern = r'output\s+"([^"]+)"\s*\{([^}]+)\}'
        module_pattern = r'module\s+"([^"]+)"\s*\{([^}]+)\}'
        
        # Extract resources
        for match in re.finditer(resource_pattern, content, re.DOTALL):
            resource_type, resource_name, resource_body = match.groups()
            if resource_type not in config['resource']:
                config['resource'][resource_type] = {}
            config['resource'][resource_type][resource_name] = self._parse_block(resource_body)
        
        # Extract data sources
        for match in re.finditer(data_pattern, content, re.DOTALL):
            data_type, data_name, data_body = match.groups()
            if data_type not in config['data']:
                config['data'][data_type] = {}
            config['data'][data_type][data_name] = self._parse_block(data_body)
        
        # Extract variables
        for match in re.finditer(variable_pattern, content, re.DOTALL):
            var_name, var_body = match.groups()
            config['variable'][var_name] = self._parse_block(var_body)
        
        # Extract outputs
        for match in re.finditer(output_pattern, content, re.DOTALL):
            output_name, output_body = match.groups()
            config['output'][output_name] = self._parse_block(output_body)
            
        # Extract modules
        for match in re.finditer(module_pattern, content, re.DOTALL):
            module_name, module_body = match.groups()
            config['module'][module_name] = self._parse_block(module_body)
        
        return config
    
    def _parse_block(self, block_content: str) -> Dict:
        """Parse a Terraform block content"""
        result = {}
        
        # Simple key-value extraction
        kv_pattern = r'(\w+)\s*=\s*"([^"]+)"'
        for match in re.finditer(kv_pattern, block_content):
            key, value = match.groups()
            result[key] = value
        
        # Extract references (${...})
        ref_pattern = r'\$\{([^}]+)\}'
        refs = re.findall(ref_pattern, block_content)
        if refs:
            result['_references'] = refs
        
        return result
    
    def _extract_resource(self, resource_type: str, resource_name: str, 
                         resource_config: Dict, provider: str, file_path: str) -> Optional[TerraformResource]:
        """Extract resource information"""
        
        # Check if it's a data-related resource
        resource_key = resource_type.replace(f"{provider}_", "")
        
        is_data_resource = False
        for prov, resources in self.DATA_RESOURCES.items():
            if provider == prov and resource_key in resources:
                is_data_resource = True
                break
        
        if not is_data_resource:
            # Check for generic data patterns
            data_keywords = ['data', 'table', 'database', 'bucket', 'storage', 
                           'stream', 'queue', 'lake', 'warehouse', 'catalog']
            is_data_resource = any(keyword in resource_type.lower() for keyword in data_keywords)
        
        if not is_data_resource:
            return None
        
        resource = TerraformResource(
            type=resource_type,
            name=resource_name,
            provider=provider,
            attributes=resource_config,
            source_file=file_path
        )
        
        # Extract data assets
        self._extract_data_assets(resource)
        
        # Extract dependencies
        self._extract_resource_dependencies(resource)
        
        return resource
    
    def _extract_data_assets(self, resource: TerraformResource):
        """Extract data assets from resource attributes"""
        
        # Common patterns for data asset names
        data_patterns = {
            'bucket': ['bucket', 'bucket_name', 'source_bucket', 'target_bucket'],
            'table': ['table', 'table_name', 'source_table', 'target_table'],
            'database': ['database', 'database_name', 'catalog_id'],
            'schema': ['schema', 'schema_name'],
            'topic': ['topic', 'topic_name'],
            'stream': ['stream_name', 'kinesis_stream', 'event_stream'],
            'cluster': ['cluster_name', 'cluster_id'],
            'dataset': ['dataset', 'dataset_id'],
            'path': ['path', 's3_path', 'location', 'uri']
        }
        
        for asset_type, patterns in data_patterns.items():
            for pattern in patterns:
                if pattern in resource.attributes:
                    value = resource.attributes[pattern]
                    if isinstance(value, str) and value:
                        resource.data_assets.append(f"{asset_type}:{value}")
        
        # Special handling for S3 paths
        if 's3' in resource.type.lower():
            for key, value in resource.attributes.items():
                if isinstance(value, str) and value.startswith('s3://'):
                    resource.data_assets.append(f"s3_path:{value}")
        
        # Handle ARNs
        for key, value in resource.attributes.items():
            if isinstance(value, str) and 'arn:' in value:
                resource.data_assets.append(f"arn:{value}")
    
    def _extract_resource_dependencies(self, resource: TerraformResource):
        """Extract dependencies from resource references"""
        
        # Look for references in attributes
        if '_references' in resource.attributes:
            for ref in resource.attributes['_references']:
                # Parse reference: aws_s3_bucket.data.id
                parts = ref.split('.')
                if len(parts) >= 2:
                    dep_type = parts[0]
                    dep_name = parts[1]
                    resource.dependencies.append(f"{dep_type}.{dep_name}")
        
        # Look for depends_on
        if 'depends_on' in resource.attributes:
            deps = resource.attributes['depends_on']
            if isinstance(deps, list):
                resource.dependencies.extend(deps)
            elif isinstance(deps, str):
                resource.dependencies.append(deps)
        
        # Look for common dependency patterns
        dependency_keys = ['source', 'target', 'input', 'output', 
                          'source_arn', 'target_arn', 'role_arn']
        
        for key in dependency_keys:
            if key in resource.attributes:
                value = resource.attributes[key]
                if isinstance(value, str):
                    # Check if it's a reference
                    if '.' in value and not value.startswith('s3://'):
                        resource.dependencies.append(value)
    
    def _extract_dependencies(self) -> List[Dict]:
        """Extract all dependencies between resources"""
        dependencies = []
        
        for resource_name, resource in self.resources.items():
            for dep in resource.dependencies:
                if dep in self.resources or dep in self.data_sources:
                    dependencies.append({
                        'source': dep,
                        'target': resource_name,
                        'type': 'terraform_dependency'
                    })
        
        # Analyze outputs and their consumers
        for output_name, output_config in self.outputs.items():
            if 'value' in output_config:
                value = output_config['value']
                if isinstance(value, str):
                    # Look for resource references
                    ref_match = re.search(r'(\w+\.\w+)', value)
                    if ref_match:
                        resource_ref = ref_match.group(1)
                        if resource_ref in self.resources:
                            dependencies.append({
                                'source': resource_ref,
                                'target': f"output.{output_name}",
                                'type': 'output_dependency'
                            })
        
        return dependencies
    
    def _analyze_module(self, module_name: str, module_config: Dict):
        """Analyze module for data resources"""
        # Modules can contain data resources
        # This is a simplified analysis - in practice, you'd need to
        # recursively analyze the module source
        
        if 'source' in module_config:
            source = module_config['source']
            
            # Check for known data modules
            data_module_patterns = [
                'data-lake', 'etl', 'pipeline', 'warehouse',
                'analytics', 'bigquery', 's3', 'storage'
            ]
            
            for pattern in data_module_patterns:
                if pattern in source.lower():
                    # Create a virtual resource for the module
                    module_resource = TerraformResource(
                        type='module',
                        name=module_name,
                        provider='module',
                        attributes=module_config
                    )
                    self.resources[f"module.{module_name}"] = module_resource
                    break
    
    def get_infrastructure_graph(self) -> Dict:
        """Generate infrastructure dependency graph"""
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': {}
        }
        
        # Group by provider
        providers = {}
        for resource_name, resource in self.resources.items():
            if resource.provider not in providers:
                providers[resource.provider] = []
            providers[resource.provider].append(resource)
        
        # Create nodes
        for provider, resources in providers.items():
            graph['clusters'][provider] = {
                'label': provider.upper(),
                'resources': []
            }
            
            for resource in resources:
                node = {
                    'id': resource.get_full_name(),
                    'label': resource.name,
                    'type': resource.type,
                    'provider': provider,
                    'data_assets': resource.data_assets,
                    'attributes': resource.attributes
                }
                graph['nodes'].append(node)
                graph['clusters'][provider]['resources'].append(node['id'])
        
        # Create edges from dependencies
        for resource_name, resource in self.resources.items():
            for dep in resource.dependencies:
                if dep in self.resources:
                    graph['edges'].append({
                        'source': dep,
                        'target': resource_name,
                        'type': 'depends_on'
                    })
        
        return graph
    
    def analyze_blast_radius(self, changed_resource: str) -> Dict:
        """
        Analyze the blast radius of a resource change
        """
        if changed_resource not in self.resources:
            return {'error': f'Resource {changed_resource} not found'}
        
        affected = {
            'direct': [],
            'indirect': [],
            'data_assets_affected': [],
            'services_affected': set()
        }
        
        # Find direct dependencies
        for resource_name, resource in self.resources.items():
            if changed_resource in resource.dependencies:
                affected['direct'].append(resource_name)
                affected['services_affected'].add(resource.provider)
                affected['data_assets_affected'].extend(resource.data_assets)
        
        # Find indirect dependencies (2nd level)
        for direct_dep in affected['direct']:
            for resource_name, resource in self.resources.items():
                if direct_dep in resource.dependencies and resource_name not in affected['direct']:
                    affected['indirect'].append(resource_name)
                    affected['services_affected'].add(resource.provider)
        
        affected['services_affected'] = list(affected['services_affected'])
        
        return affected


# Utility functions for integration
def parse_terraform_file(file_path: str) -> Tuple[List[TerraformResource], List[Dict]]:
    """Parse a single Terraform file"""
    parser = TerraformParser()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return parser.parse(content, file_path)


def parse_terraform_directory(directory: str) -> Dict:
    """Parse all Terraform files in a directory"""
    parser = TerraformParser()
    all_resources = []
    all_dependencies = []
    
    from pathlib import Path
    tf_files = list(Path(directory).glob('**/*.tf'))
    tf_files.extend(Path(directory).glob('**/*.tf.json'))
    
    for tf_file in tf_files:
        with open(tf_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        resources, dependencies = parser.parse(content, str(tf_file))
        all_resources.extend(resources)
        all_dependencies.extend(dependencies)
    
    return {
        'resources': all_resources,
        'dependencies': all_dependencies,
        'graph': parser.get_infrastructure_graph()
    }
