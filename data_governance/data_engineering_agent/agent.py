"""Data Engineering Agent — consumes a taxonomy to scaffold engineering assets.

The agent reuses the :class:`TaxonomyDocument` as a single source of truth
and the :class:`ArchitectureProfile` to make cloud/pattern-aware decisions.
Phase 1 emits DDL and dbt artifacts; later phases will add a validator
for existing assets and a catalog publisher.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..taxonomy.context import ArchitectureProfile
from ..taxonomy.models import TaxonomyDocument
from .copilot import CopilotResponse, DataEngineeringCopilot
from .generators.cli import CLICommandGenerator, CLIPlan
from .generators.dbt import DbtAssets, DbtGenerator
from .generators.ddl import DDLGenerationOptions, DDLGenerator


class DataEngineeringAgent:
    """High-level façade over the asset generators."""

    def __init__(
        self,
        taxonomy: TaxonomyDocument,
        profile: Optional[ArchitectureProfile] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.profile = profile or ArchitectureProfile.from_metadata(taxonomy.metadata)
        self._ddl = DDLGenerator(taxonomy, profile=self.profile)
        self._dbt = DbtGenerator(taxonomy, profile=self.profile)
        self._cli = CLICommandGenerator(taxonomy, profile=self.profile)
        self._copilot = DataEngineeringCopilot(taxonomy, profile=self.profile)

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------
    def generate_ddl(self, options: DDLGenerationOptions) -> str:
        return self._ddl.generate(options)

    # ------------------------------------------------------------------
    # dbt
    # ------------------------------------------------------------------
    def generate_dbt(
        self,
        model_name: str,
        concept_names: Sequence[str],
        source_table: str,
        source_schema: Optional[str] = None,
        materialization: str = "table",
        unique_key: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> DbtAssets:
        return self._dbt.generate(
            model_name=model_name,
            concept_names=concept_names,
            source_table=source_table,
            source_schema=source_schema,
            materialization=materialization,
            unique_key=unique_key,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # CLI commands (copilot-friendly)
    # ------------------------------------------------------------------
    def generate_cli_plan(
        self,
        options: DDLGenerationOptions,
        project: Optional[str] = None,
        warehouse: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CLIPlan:
        return self._cli.create_table_plan(
            options, project=project, warehouse=warehouse, location=location,
        )

    def generate_zone_plan(
        self,
        zone_name: str,
        project: Optional[str] = None,
        region: str = "us-central1",
        bucket_prefix: str = "company",
    ) -> CLIPlan:
        return self._cli.create_zone_plan(
            zone_name=zone_name, project=project, region=region, bucket_prefix=bucket_prefix,
        )

    # ------------------------------------------------------------------
    # Copilot — natural-intent → full asset plan
    # ------------------------------------------------------------------
    def suggest(self, *args, **kwargs) -> CopilotResponse:
        """Delegate to :class:`DataEngineeringCopilot.suggest`."""
        return self._copilot.suggest(*args, **kwargs)
