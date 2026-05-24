"""Data Engineering / InfraOps Agent.

Consumes a canonical :class:`TaxonomyDocument` and emits aligned engineering
artifacts (DDL, dbt models, dbt schema.yml, CLI commands) plus a copilot
interface that turns natural intents into ready-to-run asset plans.

The agent's purpose is to *guarantee* that new services / assets are born
already compliant with the company's taxonomy and architectural pattern —
either via direct artifact generation or via the CLI commands an
operator / CI pipeline must run.
"""
try:
    from .agent import DataEngineeringAgent
    from .copilot import CopilotResponse, DataEngineeringCopilot
    from .generators.cli import CLICommand, CLICommandGenerator, CLIPlan
    from .generators.dbt import DbtAssets, DbtGenerator
    from .generators.ddl import DDLGenerationOptions, DDLGenerator

    _available = True
except ImportError:
    _available = False

__all__ = [
    "DataEngineeringAgent",
    "DataEngineeringCopilot",
    "CopilotResponse",
    "DDLGenerator", "DDLGenerationOptions",
    "DbtGenerator", "DbtAssets",
    "CLICommandGenerator", "CLIPlan", "CLICommand",
]
