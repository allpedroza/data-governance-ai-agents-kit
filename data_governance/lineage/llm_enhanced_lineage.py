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
"""
LLM-Enhanced Data Lineage Analysis Module
Integrates Large Language Models for advanced code understanding,
documentation generation, and intelligent recommendations
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import hashlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"  # For local models like Llama


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    context_window: int = 8192
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key and self.provider != LLMProvider.LOCAL:
            env_key = f"{self.provider.value.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)


@dataclass
class CodeAnalysis:
    """Results from LLM code analysis"""
    summary: str
    purpose: str
    complexity_explanation: str
    data_flows: List[Dict[str, Any]]
    potential_issues: List[str]
    optimization_suggestions: List[str]
    business_logic: str
    dependencies_explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentationResult:
    """Generated documentation from LLM"""
    overview: str
    technical_details: str
    data_dictionary: Dict[str, str]
    business_glossary: Dict[str, str]
    usage_examples: List[str]
    api_documentation: Optional[str] = None
    migration_guide: Optional[str] = None


class LLMDataLineageEnhancer:
    """
    Enhances data lineage analysis with LLM capabilities
    """
    
    # Prompt templates for different analysis tasks
    PROMPTS = {
        'code_analysis': """
Analyze the following {language} code for data lineage:

```{language}
{code}
```

Provide a detailed analysis including:
1. **Purpose**: What does this code do?
2. **Data Sources**: What data sources are being read?
3. **Data Targets**: Where is data being written?
4. **Transformations**: What transformations are applied?
5. **Business Logic**: What business rules are implemented?
6. **Dependencies**: What external systems/libraries are used?
7. **Performance Considerations**: Any potential bottlenecks?
8. **Data Quality**: Any data validation or quality checks?

Format your response as JSON with these keys: purpose, sources, targets, transformations, business_logic, dependencies, performance, quality_checks.
""",

        'documentation_generation': """
Generate comprehensive documentation for this data pipeline:

Pipeline Components:
{components}

Data Flow:
{data_flow}

Generate:
1. Executive Summary (2-3 sentences)
2. Technical Overview
3. Data Dictionary with business definitions
4. Step-by-step process flow
5. Error handling and recovery procedures
6. Monitoring recommendations

Make it clear and accessible to both technical and business users.
""",

        'impact_analysis': """
Analyze the impact of the following change in a data pipeline:

Change Type: {change_type}
Asset: {asset_name}
Current Dependencies:
{dependencies}

Provide:
1. Direct impacts (immediate downstream effects)
2. Indirect impacts (cascading effects)
3. Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
4. Potential data quality issues
5. Recommended validation steps
6. Rollback strategy

Consider both technical and business impacts.
""",

        'optimization_recommendations': """
Review this data pipeline for optimization opportunities:

Current Pipeline:
{pipeline_description}

Metrics:
- Complexity: {complexity}
- Processing Time: {processing_time}
- Data Volume: {data_volume}

Suggest optimizations for:
1. Performance improvements
2. Cost reduction
3. Scalability enhancements
4. Code maintainability
5. Data quality improvements
6. Security hardening

Prioritize recommendations by impact and effort.
""",

        'anomaly_detection': """
Analyze this data lineage graph for potential anomalies or anti-patterns:

Graph Structure:
{graph_structure}

Patterns to look for:
1. Circular dependencies
2. Orphaned nodes
3. Excessive complexity
4. Data silos
5. Missing documentation
6. Security vulnerabilities
7. Compliance issues

Provide severity levels and remediation suggestions.
""",

        'sql_optimization': """
Optimize the following SQL query:

```sql
{sql_query}
```

Consider:
1. Index usage
2. Join optimization
3. Subquery elimination
4. Partition pruning
5. Statistics updates
6. Query rewriting

Provide optimized query and explanation of changes.
""",

        'natural_language_query': """
User Question: {question}

Data Pipeline Context:
{context}

Available Assets:
{assets}

Provide a helpful, accurate response about this data pipeline. Include specific asset names and technical details where relevant.
""",

        'semantic_search': """
Find all components in this data pipeline related to: "{search_query}"

Pipeline Components:
{components}

Return relevant components with relevance scores and explanations.
""",

        'compliance_check': """
Analyze this data pipeline for compliance with {regulation} requirements:

Pipeline Details:
{pipeline_details}

Check for:
1. Data privacy controls
2. Audit logging
3. Data retention policies
4. Access controls
5. Encryption requirements
6. Data residency
7. Consent management

Provide compliance status and required actions.
""",

        'data_quality_rules': """
Generate data quality rules for this dataset:

Schema:
{schema}

Sample Data:
{sample_data}

Generate:
1. Completeness rules
2. Uniqueness constraints
3. Referential integrity checks
4. Business rule validations
5. Statistical anomaly detection
6. Format validations

Provide executable validation code.
"""
    }
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM enhancer with configuration"""
        self.config = config
        self.client = self._initialize_client()
        self.cache = {}  # Cache LLM responses
        
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == LLMProvider.OPENAI:
            try:
                import openai
                return openai.Client(api_key=self.config.api_key)
            except ImportError:
                logger.warning("OpenAI library not installed")
                return None
                
        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                return anthropic.Client(api_key=self.config.api_key)
            except ImportError:
                logger.warning("Anthropic library not installed")
                return None
                
        elif self.config.provider == LLMProvider.LOCAL:
            # For local models, could use transformers, llama.cpp, etc.
            logger.info("Using local model configuration")
            return None
            
        return None
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    async def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM with prompt and return response"""
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.config.provider == LLMProvider.OPENAI:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.choices[0].message.content
                
            elif self.config.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    system=system_prompt if system_prompt else "",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.content[0].text
                
            else:
                # Fallback to mock response for demo
                result = self._generate_mock_response(prompt)
            
            # Cache the response
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing without LLM"""
        if "code_analysis" in prompt:
            return json.dumps({
                "purpose": "Data transformation pipeline",
                "sources": ["source_table"],
                "targets": ["target_table"],
                "transformations": ["aggregation", "filtering"],
                "business_logic": "Business rules applied",
                "dependencies": ["pandas", "sqlalchemy"],
                "performance": "Consider indexing on join columns",
                "quality_checks": ["null checks", "duplicate detection"]
            })
        return "Mock response for: " + prompt[:100]
    
    async def analyze_code(
        self, 
        code: str, 
        language: str = "python",
        context: Dict[str, Any] = None
    ) -> CodeAnalysis:
        """
        Analyze code using LLM for deeper understanding
        """
        prompt = self.PROMPTS['code_analysis'].format(
            code=code,
            language=language
        )
        
        system_prompt = """You are an expert data engineer analyzing code for data lineage. 
        Focus on data flows, transformations, and business logic. 
        Always respond with valid JSON."""
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            parsed = json.loads(response)
            return CodeAnalysis(
                summary=parsed.get('purpose', ''),
                purpose=parsed.get('purpose', ''),
                complexity_explanation=parsed.get('performance', ''),
                data_flows=[
                    {"source": s, "target": t} 
                    for s, t in zip(
                        parsed.get('sources', []), 
                        parsed.get('targets', [])
                    )
                ],
                potential_issues=[parsed.get('performance', '')],
                optimization_suggestions=[],
                business_logic=parsed.get('business_logic', ''),
                dependencies_explanation=str(parsed.get('dependencies', [])),
                metadata=parsed
            )
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return CodeAnalysis(
                summary=response[:200],
                purpose="Analysis failed",
                complexity_explanation="",
                data_flows=[],
                potential_issues=[],
                optimization_suggestions=[],
                business_logic="",
                dependencies_explanation=""
            )
    
    async def generate_documentation(
        self,
        lineage_graph: Any,
        components: List[Dict[str, Any]],
        style: str = "technical"
    ) -> DocumentationResult:
        """
        Generate comprehensive documentation using LLM
        """
        # Prepare component descriptions
        component_desc = "\n".join([
            f"- {c.get('name', 'Unknown')}: {c.get('type', 'Unknown type')}"
            for c in components[:20]  # Limit to avoid token limits
        ])
        
        # Prepare data flow description
        data_flow = self._describe_data_flow(lineage_graph)
        
        prompt = self.PROMPTS['documentation_generation'].format(
            components=component_desc,
            data_flow=data_flow
        )
        
        response = await self._call_llm(prompt)
        
        # Parse response into structured documentation
        lines = response.split('\n')
        overview = ""
        technical_details = ""
        
        # Simple parsing - in production, use better parsing
        for i, line in enumerate(lines):
            if 'summary' in line.lower() or 'overview' in line.lower():
                overview = ' '.join(lines[i+1:i+3])
            elif 'technical' in line.lower():
                technical_details = ' '.join(lines[i+1:i+5])
        
        return DocumentationResult(
            overview=overview or response[:200],
            technical_details=technical_details or response[200:500],
            data_dictionary={},
            business_glossary={},
            usage_examples=[]
        )
    
    async def analyze_impact(
        self,
        change_type: str,
        asset_name: str,
        dependencies: List[str],
        graph: Any = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze impact of changes
        """
        deps_str = "\n".join([f"- {d}" for d in dependencies[:30]])
        
        prompt = self.PROMPTS['impact_analysis'].format(
            change_type=change_type,
            asset_name=asset_name,
            dependencies=deps_str
        )
        
        response = await self._call_llm(prompt)
        
        # Parse response to extract risk level
        risk_level = "MEDIUM"  # Default
        if "CRITICAL" in response.upper():
            risk_level = "CRITICAL"
        elif "HIGH" in response.upper():
            risk_level = "HIGH"
        elif "LOW" in response.upper():
            risk_level = "LOW"
        
        return {
            "asset": asset_name,
            "change_type": change_type,
            "risk_level": risk_level,
            "analysis": response,
            "affected_count": len(dependencies),
            "recommendations": self._extract_recommendations(response)
        }
    
    async def suggest_optimizations(
        self,
        pipeline_description: str,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions from LLM
        """
        prompt = self.PROMPTS['optimization_recommendations'].format(
            pipeline_description=pipeline_description[:1000],
            complexity=metrics.get('complexity', 'Unknown'),
            processing_time=metrics.get('processing_time', 'Unknown'),
            data_volume=metrics.get('data_volume', 'Unknown')
        )
        
        response = await self._call_llm(prompt)
        
        # Parse suggestions
        suggestions = []
        lines = response.split('\n')
        for line in lines:
            if line.strip() and any(
                keyword in line.lower() 
                for keyword in ['improve', 'optimize', 'reduce', 'enhance']
            ):
                suggestions.append({
                    "type": self._categorize_suggestion(line),
                    "description": line.strip(),
                    "priority": self._estimate_priority(line)
                })
        
        return suggestions[:10]  # Return top 10 suggestions
    
    async def detect_anomalies(
        self,
        graph_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to detect anomalies in data lineage
        """
        # Prepare graph description
        graph_desc = {
            "nodes": len(graph_structure.get('nodes', [])),
            "edges": len(graph_structure.get('edges', [])),
            "components": graph_structure.get('components', 1),
            "max_degree": max(
                [len(n.get('edges', [])) for n in graph_structure.get('nodes', [])],
                default=0
            )
        }
        
        prompt = self.PROMPTS['anomaly_detection'].format(
            graph_structure=json.dumps(graph_desc, indent=2)
        )
        
        response = await self._call_llm(prompt)
        
        anomalies = []
        if "circular" in response.lower():
            anomalies.append({
                "type": "circular_dependency",
                "severity": "HIGH",
                "description": "Potential circular dependencies detected"
            })
        
        if "orphan" in response.lower():
            anomalies.append({
                "type": "orphaned_nodes",
                "severity": "MEDIUM",
                "description": "Orphaned nodes found in pipeline"
            })
        
        return anomalies
    
    async def optimize_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Optimize SQL query using LLM
        """
        prompt = self.PROMPTS['sql_optimization'].format(sql_query=sql_query)
        
        response = await self._call_llm(prompt)
        
        # Try to extract optimized query from response
        optimized_query = sql_query  # Default to original
        
        # Look for SQL code blocks in response
        sql_pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(sql_pattern, response, re.DOTALL)
        if matches:
            optimized_query = matches[0]
        
        return {
            "original": sql_query,
            "optimized": optimized_query,
            "explanation": response,
            "improvements": self._extract_improvements(response)
        }
    
    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any],
        assets: List[str]
    ) -> str:
        """
        Answer natural language questions about the pipeline
        """
        prompt = self.PROMPTS['natural_language_query'].format(
            question=question,
            context=json.dumps(context, indent=2)[:2000],
            assets=", ".join(assets[:50])
        )
        
        response = await self._call_llm(prompt)
        return response
    
    async def semantic_search(
        self,
        query: str,
        components: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across pipeline components
        """
        comp_desc = json.dumps(components[:30], indent=2)
        
        prompt = self.PROMPTS['semantic_search'].format(
            search_query=query,
            components=comp_desc
        )
        
        response = await self._call_llm(prompt)
        
        # Mock relevance scoring
        results = []
        for comp in components[:10]:
            if query.lower() in str(comp).lower():
                results.append({
                    "component": comp,
                    "relevance_score": 0.8,
                    "explanation": f"Matches query: {query}"
                })
        
        return results
    
    async def check_compliance(
        self,
        regulation: str,
        pipeline_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check pipeline compliance with regulations
        """
        prompt = self.PROMPTS['compliance_check'].format(
            regulation=regulation,
            pipeline_details=json.dumps(pipeline_details, indent=2)[:2000]
        )
        
        response = await self._call_llm(prompt)
        
        return {
            "regulation": regulation,
            "status": "PARTIAL" if "required" in response.lower() else "COMPLIANT",
            "findings": response,
            "required_actions": self._extract_actions(response)
        }
    
    async def generate_quality_rules(
        self,
        schema: Dict[str, Any],
        sample_data: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate data quality rules using LLM
        """
        prompt = self.PROMPTS['data_quality_rules'].format(
            schema=json.dumps(schema, indent=2),
            sample_data=json.dumps(sample_data[:5], indent=2) if sample_data else "None"
        )
        
        response = await self._call_llm(prompt)
        
        rules = []
        
        # Extract rules from response
        if "null" in response.lower():
            rules.append({
                "type": "completeness",
                "rule": "Check for null values",
                "severity": "HIGH"
            })
        
        if "unique" in response.lower():
            rules.append({
                "type": "uniqueness",
                "rule": "Ensure unique identifiers",
                "severity": "HIGH"
            })
        
        if "format" in response.lower():
            rules.append({
                "type": "format",
                "rule": "Validate data formats",
                "severity": "MEDIUM"
            })
        
        return rules
    
    # Helper methods
    
    def _describe_data_flow(self, graph: Any) -> str:
        """Generate textual description of data flow"""
        if hasattr(graph, 'edges'):
            edges = list(graph.edges())[:10]
            return "Data flows from: " + " -> ".join([
                f"{e[0]} to {e[1]}" for e in edges[:5]
            ])
        return "Complex data flow graph"
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from LLM response"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should']):
                recommendations.append(line.strip())
        return recommendations[:5]
    
    def _categorize_suggestion(self, suggestion: str) -> str:
        """Categorize optimization suggestion"""
        suggestion_lower = suggestion.lower()
        if 'performance' in suggestion_lower:
            return 'performance'
        elif 'cost' in suggestion_lower:
            return 'cost'
        elif 'security' in suggestion_lower:
            return 'security'
        elif 'quality' in suggestion_lower:
            return 'quality'
        else:
            return 'general'
    
    def _estimate_priority(self, text: str) -> str:
        """Estimate priority from text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['critical', 'urgent', 'immediate']):
            return 'HIGH'
        elif any(word in text_lower for word in ['important', 'should']):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _extract_improvements(self, text: str) -> List[str]:
        """Extract improvements from optimization response"""
        improvements = []
        patterns = [
            'index', 'join', 'partition', 'cache', 'parallel',
            'batch', 'optimize', 'reduce', 'eliminate'
        ]
        
        for pattern in patterns:
            if pattern in text.lower():
                improvements.append(pattern)
        
        return improvements
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract required actions from compliance response"""
        actions = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['must', 'require', 'need']):
                actions.append(line.strip())
        return actions[:10]


class LLMIntegration:
    """
    Main integration class for LLM-enhanced data lineage
    """
    
    def __init__(self, lineage_agent, llm_config: LLMConfig):
        """Initialize integration with lineage agent and LLM config"""
        self.lineage_agent = lineage_agent
        self.llm_enhancer = LLMDataLineageEnhancer(llm_config)
        self.analysis_cache = {}
        
    async def enhanced_analysis(
        self,
        file_paths: List[str],
        include_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Perform enhanced analysis with LLM insights
        """
        # First, do standard lineage analysis
        standard_analysis = self.lineage_agent.analyze_pipeline(file_paths)
        
        if not include_llm:
            return standard_analysis
        
        # Enhance with LLM analysis
        enhanced_results = {
            **standard_analysis,
            'llm_insights': {}
        }
        
        # Analyze each code file with LLM
        code_insights = []
        for file_path in file_paths[:5]:  # Limit to avoid too many API calls
            with open(file_path, 'r') as f:
                code = f.read()
            
            language = Path(file_path).suffix[1:]  # Get language from extension
            analysis = await self.llm_enhancer.analyze_code(code, language)
            
            code_insights.append({
                'file': file_path,
                'analysis': analysis
            })
        
        enhanced_results['llm_insights']['code_analysis'] = code_insights
        
        # Generate documentation
        doc_result = await self.llm_enhancer.generate_documentation(
            standard_analysis.get('graph'),
            standard_analysis.get('assets', [])
        )
        enhanced_results['llm_insights']['documentation'] = doc_result
        
        # Get optimization suggestions
        metrics = {
            'complexity': len(standard_analysis.get('transformations', [])),
            'nodes': standard_analysis['graph'].number_of_nodes() if 'graph' in standard_analysis else 0
        }
        
        suggestions = await self.llm_enhancer.suggest_optimizations(
            "Data pipeline with multiple transformations",
            metrics
        )
        enhanced_results['llm_insights']['optimizations'] = suggestions
        
        # Detect anomalies
        if 'graph' in standard_analysis:
            graph_structure = {
                'nodes': list(standard_analysis['graph'].nodes()),
                'edges': list(standard_analysis['graph'].edges())
            }
            anomalies = await self.llm_enhancer.detect_anomalies(graph_structure)
            enhanced_results['llm_insights']['anomalies'] = anomalies
        
        return enhanced_results
    
    async def interactive_query(
        self,
        question: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Answer questions about the data pipeline
        """
        if not context:
            context = self.analysis_cache
        
        assets = list(self.lineage_agent.assets.keys())
        
        return await self.llm_enhancer.answer_question(
            question,
            context,
            assets
        )
    
    async def smart_impact_analysis(
        self,
        asset_name: str,
        change_type: str = "schema_change"
    ) -> Dict[str, Any]:
        """
        Perform smart impact analysis with LLM insights
        """
        # Get standard impact analysis
        standard_impact = self.lineage_agent.analyze_change_impact([asset_name])
        
        # Enhance with LLM
        dependencies = standard_impact.get('downstream_affected', [])
        
        llm_impact = await self.llm_enhancer.analyze_impact(
            change_type,
            asset_name,
            dependencies
        )
        
        return {
            **standard_impact,
            'llm_analysis': llm_impact
        }
    
    async def generate_data_quality_rules(
        self,
        asset_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generate data quality rules for an asset
        """
        asset = self.lineage_agent.assets.get(asset_name)
        if not asset:
            return []
        
        schema = asset.metadata.get('schema', {})
        
        return await self.llm_enhancer.generate_quality_rules(schema)
    
    async def optimize_queries(
        self,
        sql_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Optimize SQL queries in the pipeline
        """
        optimizations = []
        
        for sql_file in sql_files:
            with open(sql_file, 'r') as f:
                sql_content = f.read()
            
            # Extract individual queries (simple split by semicolon)
            queries = sql_content.split(';')
            
            for query in queries[:3]:  # Limit to avoid too many API calls
                if query.strip():
                    optimization = await self.llm_enhancer.optimize_sql(query)
                    optimization['file'] = sql_file
                    optimizations.append(optimization)
        
        return optimizations


# Utility functions for easy integration

def create_llm_config(
    provider: str = "openai",
    model: str = "gpt-4",
    api_key: str = None
) -> LLMConfig:
    """Create LLM configuration"""
    return LLMConfig(
        provider=LLMProvider(provider),
        model_name=model,
        api_key=api_key
    )


async def enhance_lineage_with_llm(
    lineage_agent,
    file_paths: List[str],
    llm_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Quick function to enhance lineage analysis with LLM
    """
    config = create_llm_config(provider=llm_provider)
    integration = LLMIntegration(lineage_agent, config)
    
    return await integration.enhanced_analysis(file_paths)


# Example usage
if __name__ == "__main__":
    import asyncio
    from data_lineage_agent import DataLineageAgent
    
    async def demo():
        # Create lineage agent
        agent = DataLineageAgent()
        
        # Create LLM config
        config = create_llm_config(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create integration
        integration = LLMIntegration(agent, config)
        
        # Example files
        files = ["example.py", "transform.sql"]
        
        # Run enhanced analysis
        results = await integration.enhanced_analysis(files)
        
        print("Enhanced Analysis Results:")
        print(json.dumps(results.get('llm_insights', {}), indent=2))
        
        # Ask questions
        answer = await integration.interactive_query(
            "What tables are being used in this pipeline?"
        )
        print(f"\nAnswer: {answer}")
    
    # Run the demo
    # asyncio.run(demo())
