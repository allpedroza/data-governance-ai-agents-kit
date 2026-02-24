# /// script
# dependencies = [
#   "openlineage-python>=1.0.0",
#   "requests>=2.31.0",
# ]
# ///
"""
OpenLineage Parser - Integração com o padrão OpenLineage

Suporta:
- Leitura de eventos OpenLineage em formato NDJSON (emitidos por Spark, dbt, Airflow, etc.)
- Conexão com a API REST do Marquez (implementação de referência do OpenLineage)
- Conversão dos modelos OpenLineage (Job, Dataset, RunEvent) para DataAsset e Transformation

Referência do padrão: https://openlineage.io/
Especificação: https://openlineage.io/spec/2-0-2/OpenLineage.json
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelos internos do OpenLineage (subset do spec)
# ---------------------------------------------------------------------------

@dataclass
class OLDataset:
    """Representa um Dataset no padrão OpenLineage."""
    name: str
    namespace: str
    facets: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}.{self.name}"

    @classmethod
    def from_dict(cls, data: Dict) -> "OLDataset":
        return cls(
            name=data.get("name", "unknown"),
            namespace=data.get("namespace", "default"),
            facets=data.get("facets", {}),
        )


@dataclass
class OLJob:
    """Representa um Job no padrão OpenLineage."""
    name: str
    namespace: str
    facets: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}/{self.name}"

    @classmethod
    def from_dict(cls, data: Dict) -> "OLJob":
        return cls(
            name=data.get("name", "unknown"),
            namespace=data.get("namespace", "default"),
            facets=data.get("facets", {}),
        )


@dataclass
class OLRun:
    """Representa uma Run no padrão OpenLineage."""
    run_id: str
    facets: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "OLRun":
        return cls(
            run_id=data.get("runId", ""),
            facets=data.get("facets", {}),
        )


@dataclass
class OLRunEvent:
    """Representa um RunEvent no padrão OpenLineage."""
    event_type: str          # START | RUNNING | COMPLETE | ABORT | FAIL | OTHER
    event_time: str
    job: OLJob
    run: OLRun
    inputs: List[OLDataset] = field(default_factory=list)
    outputs: List[OLDataset] = field(default_factory=list)
    producer: str = ""
    schema_url: str = ""
    source_file: str = ""    # origem interna: arquivo NDJSON ou "api:<url>"

    @classmethod
    def from_dict(cls, data: Dict, source_file: str = "") -> Optional["OLRunEvent"]:
        try:
            return cls(
                event_type=data.get("eventType", "OTHER"),
                event_time=data.get("eventTime", ""),
                job=OLJob.from_dict(data.get("job", {})),
                run=OLRun.from_dict(data.get("run", {})),
                inputs=[OLDataset.from_dict(d) for d in data.get("inputs", [])],
                outputs=[OLDataset.from_dict(d) for d in data.get("outputs", [])],
                producer=data.get("producer", ""),
                schema_url=data.get("schemaURL", ""),
                source_file=source_file,
            )
        except Exception as exc:
            logger.warning("Falha ao parsear RunEvent: %s — %s", exc, data)
            return None


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class OpenLineageParser:
    """
    Parser que converte eventos OpenLineage para os modelos internos
    DataAsset e Transformation usados pelo DataLineageAgent.

    Mapeamento:
        OLDataset  → DataAsset  (type='dataset', com metadata do facet de schema)
        OLJob      → DataAsset  (type='job')
        OLRunEvent → Transformations:
            Para cada input dataset → job  (operation='read')
            Para cada output dataset ← job (operation='write')
    """

    def __init__(self):
        self.events: List[OLRunEvent] = []
        self._assets_cache: Dict[str, Any] = {}    # qualified_name → DataAsset
        self._transforms_seen: set = set()

    # ------------------------------------------------------------------
    # Leitura de arquivos NDJSON
    # ------------------------------------------------------------------

    def parse_events_file(self, file_path: str) -> Tuple[List[Any], List[Any]]:
        """
        Lê um arquivo NDJSON de eventos OpenLineage e retorna
        (assets, transformations) prontos para o DataLineageAgent.

        O arquivo pode conter um evento JSON por linha (NDJSON)
        ou uma lista JSON com múltiplos eventos.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Arquivo não encontrado: %s", file_path)
            return [], []

        raw_events = []
        content = path.read_text(encoding="utf-8").strip()

        if content.startswith("["):
            # JSON array
            try:
                raw_events = json.loads(content)
            except json.JSONDecodeError as exc:
                logger.error("Erro ao parsear JSON array em %s: %s", file_path, exc)
                return [], []
        else:
            # NDJSON — uma linha por evento
            for lineno, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Linha %d ignorada em %s: %s", lineno, file_path, exc)

        events = [
            ev for raw in raw_events
            if (ev := OLRunEvent.from_dict(raw, source_file=str(path))) is not None
        ]
        self.events.extend(events)
        logger.info("Lidos %d eventos de %s", len(events), file_path)
        return self._events_to_assets_and_transformations(events, source_file=str(path))

    def parse_events(self, raw_events: List[Dict],
                     source_file: str = "openlineage:memory") -> Tuple[List[Any], List[Any]]:
        """
        Parseia uma lista de dicionários de eventos OpenLineage já carregados
        em memória e retorna (assets, transformations).
        """
        events = [
            ev for raw in raw_events
            if (ev := OLRunEvent.from_dict(raw, source_file=source_file)) is not None
        ]
        self.events.extend(events)
        return self._events_to_assets_and_transformations(events, source_file=source_file)

    # ------------------------------------------------------------------
    # Conexão com a API do Marquez (implementação de referência OpenLineage)
    # ------------------------------------------------------------------

    def fetch_from_api(
        self,
        base_url: str,
        namespace: Optional[str] = None,
        api_key: Optional[str] = None,
        limit: int = 200,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Busca linhagem diretamente da API REST do Marquez / OpenLineage backend.

        Args:
            base_url:  URL base, ex: "http://localhost:5000"
            namespace: Namespace a filtrar. Se None, busca todos os namespaces.
            api_key:   Token Bearer para autenticação (opcional).
            limit:     Número máximo de jobs/datasets por namespace.

        Returns:
            (assets, transformations) prontos para o DataLineageAgent.
        """
        session = requests.Session()
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        session.headers["Accept"] = "application/json"

        base_url = base_url.rstrip("/")
        all_assets: List[Any] = []
        all_transforms: List[Any] = []

        try:
            namespaces = self._api_get_namespaces(session, base_url, namespace)
            for ns in namespaces:
                logger.info("Buscando jobs do namespace '%s'…", ns)
                jobs = self._api_get_jobs(session, base_url, ns, limit)
                for job in jobs:
                    assets, transforms = self._api_job_to_lineage(
                        session, base_url, ns, job
                    )
                    all_assets.extend(assets)
                    all_transforms.extend(transforms)
        except requests.RequestException as exc:
            logger.error("Erro ao conectar com a API OpenLineage em %s: %s", base_url, exc)

        return self._deduplicate(all_assets, all_transforms)

    # ------------------------------------------------------------------
    # Helpers de API
    # ------------------------------------------------------------------

    def _api_get_namespaces(
        self, session: requests.Session, base_url: str, namespace: Optional[str]
    ) -> List[str]:
        if namespace:
            return [namespace]
        url = f"{base_url}/api/v1/namespaces"
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [ns["name"] for ns in data.get("namespaces", [])]

    def _api_get_jobs(
        self, session: requests.Session, base_url: str, namespace: str, limit: int
    ) -> List[Dict]:
        url = f"{base_url}/api/v1/namespaces/{namespace}/jobs"
        resp = session.get(url, params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        return resp.json().get("jobs", [])

    def _api_job_to_lineage(
        self,
        session: requests.Session,
        base_url: str,
        namespace: str,
        job_data: Dict,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Converte um job da API Marquez para DataAssets e Transformations.
        Usa os campos 'latestRun.inputDatasets' e 'latestRun.outputDatasets'.
        """
        job_name = job_data.get("name", "unknown")
        source_id = f"api:{base_url}/namespaces/{namespace}/jobs/{job_name}"

        # Monta evento sintético a partir da resposta da API
        synthetic_event_data = {
            "eventType": "COMPLETE",
            "eventTime": job_data.get("updatedAt", datetime.utcnow().isoformat()),
            "job": {"name": job_name, "namespace": namespace,
                    "facets": job_data.get("facets", {})},
            "run": {"runId": job_data.get("latestRun", {}).get("id", "api-run")},
            "inputs": job_data.get("inputs", []),
            "outputs": job_data.get("outputs", []),
            "producer": base_url,
        }

        event = OLRunEvent.from_dict(synthetic_event_data, source_file=source_id)
        if event is None:
            return [], []

        return self._events_to_assets_and_transformations([event], source_file=source_id)

    # ------------------------------------------------------------------
    # Conversão OL → DataAsset / Transformation
    # ------------------------------------------------------------------

    def _events_to_assets_and_transformations(
        self, events: List[OLRunEvent], source_file: str
    ) -> Tuple[List[Any], List[Any]]:
        """
        Converte uma lista de RunEvents para (assets, transformations).

        Filtra apenas eventos COMPLETE ou RUNNING para garantir
        que inputs/outputs estejam disponíveis.
        """
        # Importação lazy para evitar dependência circular
        from data_lineage_agent import DataAsset, Transformation

        assets: Dict[str, DataAsset] = {}
        transformations: List[Transformation] = []

        relevant_types = {"COMPLETE", "RUNNING", "OTHER"}

        for event in events:
            if event.event_type.upper() not in relevant_types:
                # Ignora START, ABORT, FAIL — não têm datasets completos
                if event.event_type.upper() not in {"START"}:
                    logger.debug(
                        "Evento %s para job '%s' ignorado (tipo=%s)",
                        event.run.run_id, event.job.qualified_name, event.event_type
                    )
                continue

            job_asset = self._make_job_asset(event, source_file)
            assets[job_asset.name] = job_asset

            for dataset in event.inputs:
                ds_asset = self._make_dataset_asset(dataset, source_file)
                assets[ds_asset.name] = ds_asset

                transform = self._make_transformation(
                    source=ds_asset,
                    target=job_asset,
                    operation="READ",
                    event=event,
                    source_file=source_file,
                )
                if transform:
                    transformations.append(transform)

            for dataset in event.outputs:
                ds_asset = self._make_dataset_asset(dataset, source_file)
                assets[ds_asset.name] = ds_asset

                transform = self._make_transformation(
                    source=job_asset,
                    target=ds_asset,
                    operation="WRITE",
                    event=event,
                    source_file=source_file,
                )
                if transform:
                    transformations.append(transform)

        return list(assets.values()), transformations

    def _make_job_asset(self, event: OLRunEvent, source_file: str) -> Any:
        from data_lineage_agent import DataAsset

        job = event.job
        metadata: Dict[str, Any] = {
            "namespace": job.namespace,
            "run_id": event.run.run_id,
            "event_time": event.event_time,
            "producer": event.producer,
            "openlineage_type": "job",
        }

        # Extrai SQL do facet SqlJobFacet, se disponível
        sql_facet = job.facets.get("sql", {})
        if sql_facet:
            metadata["sql"] = sql_facet.get("query", "")

        return DataAsset(
            name=job.qualified_name,
            type="job",
            source_file=source_file,
            metadata=metadata,
        )

    def _make_dataset_asset(self, dataset: OLDataset, source_file: str) -> Any:
        from data_lineage_agent import DataAsset

        schema: Dict[str, str] = {}
        schema_facet = dataset.facets.get("schema", {})
        for field_def in schema_facet.get("fields", []):
            fname = field_def.get("name", "")
            ftype = field_def.get("type", "unknown")
            if fname:
                schema[fname] = ftype

        metadata: Dict[str, Any] = {
            "namespace": dataset.namespace,
            "openlineage_type": "dataset",
        }

        # Tipo do asset derivado de facets
        ds_type = _infer_dataset_type(dataset)

        return DataAsset(
            name=dataset.qualified_name,
            type=ds_type,
            source_file=source_file,
            schema=schema,
            metadata=metadata,
        )

    def _make_transformation(
        self,
        source: Any,
        target: Any,
        operation: str,
        event: OLRunEvent,
        source_file: str,
    ) -> Optional[Any]:
        from data_lineage_agent import Transformation

        key = (source.name, target.name, operation)
        if key in self._transforms_seen:
            return None
        self._transforms_seen.add(key)

        logic = f"OpenLineage RunEvent — Job: {event.job.qualified_name}"
        sql_facet = event.job.facets.get("sql", {})
        if sql_facet:
            logic = sql_facet.get("query", logic)[:500]

        return Transformation(
            source=source,
            target=target,
            operation=operation,
            transformation_logic=logic,
            source_file=source_file,
            confidence_score=1.0,
        )

    # ------------------------------------------------------------------
    # Deduplicação
    # ------------------------------------------------------------------

    def _deduplicate(
        self, assets: List[Any], transforms: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        seen_assets: Dict[str, Any] = {}
        for a in assets:
            seen_assets[a.name] = a

        seen_transforms: set = set()
        unique_transforms: List[Any] = []
        for t in transforms:
            key = (t.source.name, t.target.name, t.operation)
            if key not in seen_transforms:
                seen_transforms.add(key)
                unique_transforms.append(t)

        return list(seen_assets.values()), unique_transforms


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _infer_dataset_type(dataset: OLDataset) -> str:
    """
    Infere o tipo do dataset com base nos facets OpenLineage.
    Ex: SymlinksDatasetFacet com type=TABLE → 'table'
    """
    symlinks = dataset.facets.get("symlinks", {})
    for identifier in symlinks.get("identifiers", []):
        ds_type = identifier.get("type", "").lower()
        if ds_type in {"table", "view", "file", "stream", "topic"}:
            return ds_type

    datasource = dataset.facets.get("dataSource", {})
    uri = datasource.get("uri", "").lower()
    if uri.startswith("s3://") or uri.startswith("gs://") or uri.startswith("abfss://"):
        return "file"
    if uri.startswith("kafka://") or "kafka" in uri:
        return "stream"

    # Fallback por nome
    name = dataset.name.lower()
    if any(ext in name for ext in [".parquet", ".csv", ".json", ".orc", ".avro"]):
        return "file"

    return "dataset"


def parse_openlineage_file(file_path: str) -> Tuple[List[Any], List[Any]]:
    """Conveniência: parseia um arquivo NDJSON OpenLineage."""
    parser = OpenLineageParser()
    return parser.parse_events_file(file_path)


def fetch_openlineage_from_marquez(
    marquez_url: str,
    namespace: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Any], List[Any]]:
    """Conveniência: busca linhagem da API do Marquez."""
    parser = OpenLineageParser()
    return parser.fetch_from_api(marquez_url, namespace=namespace, api_key=api_key)
