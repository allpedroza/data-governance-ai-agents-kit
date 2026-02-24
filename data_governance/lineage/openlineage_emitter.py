# /// script
# dependencies = [
#   "openlineage-python>=1.0.0",
#   "requests>=2.31.0",
# ]
# ///
"""
OpenLineage Emitter - Emite eventos OpenLineage a partir da análise de linhagem

Converte os modelos internos DataAsset e Transformation para o padrão
OpenLineage (https://openlineage.io/) e os envia para:
  - Um backend compatível (Marquez, Atlan, DataHub, etc.) via HTTP POST
  - Um arquivo local no formato NDJSON

Referência da especificação: https://openlineage.io/spec/2-0-2/OpenLineage.json
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# URL do schema OpenLineage que identifica a versão do spec
_SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json"
_PRODUCER = "https://github.com/allpedroza/data-governance-ai-agents-kit"


class OpenLineageEmitter:
    """
    Emite eventos OpenLineage no formato padrão.

    Uso básico:
        emitter = OpenLineageEmitter(
            backend_url="http://localhost:5000",  # Marquez
            namespace="my_pipeline",
        )
        emitter.emit_from_analysis(agent.assets, agent.transformations)
        emitter.save_to_file("events.ndjson")
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
        namespace: str = "default",
        api_key: Optional[str] = None,
        producer: str = _PRODUCER,
    ):
        """
        Args:
            backend_url: URL do backend OpenLineage (ex: http://localhost:5000).
                         Se None, apenas salva localmente.
            namespace:   Namespace padrão para jobs e datasets emitidos.
            api_key:     Token Bearer para autenticação (opcional).
            producer:    URL do producer que identifica esta ferramenta.
        """
        self.backend_url = backend_url.rstrip("/") if backend_url else None
        self.namespace = namespace
        self.api_key = api_key
        self.producer = producer
        self._emitted_events: List[Dict] = []

        if backend_url:
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            if api_key:
                self._session.headers["Authorization"] = f"Bearer {api_key}"
        else:
            self._session = None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def emit_from_analysis(
        self,
        assets: Dict[str, Any],
        transformations: List[Any],
        run_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Gera e emite eventos OpenLineage a partir dos resultados do DataLineageAgent.

        Agrupa transformações por job (operação) e emite um RunEvent COMPLETE
        para cada grupo, com os datasets de entrada (inputs) e saída (outputs).

        Args:
            assets:          Dicionário de DataAsset (de DataLineageAgent.assets).
            transformations: Lista de Transformation.
            run_id:          UUID da run. Se None, gera um novo.

        Returns:
            Lista de eventos emitidos (dict no formato OpenLineage).
        """
        run_id = run_id or str(uuid.uuid4())
        events_by_job: Dict[str, Dict] = {}

        for t in transformations:
            # Identifica o job com base na origem da transformação
            job_name = _derive_job_name(t)

            if job_name not in events_by_job:
                events_by_job[job_name] = {
                    "inputs": set(),
                    "outputs": set(),
                    "source_file": t.source_file,
                    "operation": t.operation,
                }

            if t.operation.upper() in {"READ", "SELECT", "INPUT"}:
                events_by_job[job_name]["inputs"].add(t.source.name)
                events_by_job[job_name]["outputs"].add(t.target.name)
            else:  # WRITE, INSERT, CREATE, etc.
                events_by_job[job_name]["inputs"].add(t.source.name)
                events_by_job[job_name]["outputs"].add(t.target.name)

        emitted = []
        now = datetime.now(timezone.utc).isoformat()

        for job_name, job_data in events_by_job.items():
            input_datasets = [
                self._make_dataset_payload(
                    name, assets.get(name)
                )
                for name in job_data["inputs"]
            ]
            output_datasets = [
                self._make_dataset_payload(
                    name, assets.get(name)
                )
                for name in job_data["outputs"]
            ]

            event = self._make_run_event(
                event_type="COMPLETE",
                event_time=now,
                job_name=job_name,
                run_id=run_id,
                inputs=input_datasets,
                outputs=output_datasets,
                source_file=job_data["source_file"],
            )
            emitted.append(event)
            self._emitted_events.append(event)

            if self.backend_url:
                self._post_event(event)

        logger.info("Emitidos %d eventos OpenLineage para %d jobs.", len(emitted), len(events_by_job))
        return emitted

    def emit_single_transformation(
        self,
        source_name: str,
        target_name: str,
        job_name: str,
        operation: str = "WRITE",
        source_asset: Optional[Any] = None,
        target_asset: Optional[Any] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
        """
        Emite um único evento de transformação entre dois assets.

        Útil para integração em pipelines onde cada step emite eventos individuais.
        """
        run_id = run_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        event = self._make_run_event(
            event_type="COMPLETE",
            event_time=now,
            job_name=job_name,
            run_id=run_id,
            inputs=[self._make_dataset_payload(source_name, source_asset)],
            outputs=[self._make_dataset_payload(target_name, target_asset)],
        )
        self._emitted_events.append(event)

        if self.backend_url:
            self._post_event(event)

        return event

    def save_to_file(self, output_path: str = "openlineage_events.ndjson") -> str:
        """
        Salva todos os eventos emitidos em um arquivo NDJSON local.

        Args:
            output_path: Caminho do arquivo de saída.

        Returns:
            Caminho absoluto do arquivo criado.
        """
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as fh:
            for event in self._emitted_events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        logger.info("Eventos salvos em %s (%d eventos)", path, len(self._emitted_events))
        return str(path.resolve())

    def get_events(self) -> List[Dict]:
        """Retorna todos os eventos emitidos na sessão atual."""
        return list(self._emitted_events)

    # ------------------------------------------------------------------
    # Construtores de payload OpenLineage
    # ------------------------------------------------------------------

    def _make_run_event(
        self,
        event_type: str,
        event_time: str,
        job_name: str,
        run_id: str,
        inputs: List[Dict],
        outputs: List[Dict],
        source_file: str = "",
    ) -> Dict:
        facets: Dict[str, Any] = {}
        if source_file:
            facets["sourceCode"] = {
                "_producer": self.producer,
                "_schemaURL": f"{_SCHEMA_URL}#/definitions/SourceCodeJobFacet",
                "language": _infer_language(source_file),
                "sourceFile": source_file,
            }

        return {
            "eventType": event_type,
            "eventTime": event_time,
            "producer": self.producer,
            "schemaURL": _SCHEMA_URL,
            "job": {
                "namespace": self.namespace,
                "name": job_name,
                "facets": facets,
            },
            "run": {
                "runId": run_id,
                "facets": {},
            },
            "inputs": inputs,
            "outputs": outputs,
        }

    def _make_dataset_payload(
        self, qualified_name: str, asset: Optional[Any] = None
    ) -> Dict:
        """
        Converte um DataAsset para o formato de dataset OpenLineage.
        Se o asset não estiver disponível, usa apenas o nome qualificado.
        """
        # Separa namespace e nome do dataset
        namespace, ds_name = _split_qualified_name(qualified_name, self.namespace)

        facets: Dict[str, Any] = {}

        if asset is not None:
            # Schema facet
            if asset.schema:
                facets["schema"] = {
                    "_producer": self.producer,
                    "_schemaURL": f"{_SCHEMA_URL}#/definitions/SchemaDatasetFacet",
                    "fields": [
                        {"name": fname, "type": ftype}
                        for fname, ftype in asset.schema.items()
                    ],
                }

            # DataSource facet
            if asset.source_file:
                facets["dataSource"] = {
                    "_producer": self.producer,
                    "_schemaURL": f"{_SCHEMA_URL}#/definitions/DatasourceDatasetFacet",
                    "name": asset.source_file,
                    "uri": _to_uri(asset.source_file),
                }

        return {
            "namespace": namespace,
            "name": ds_name,
            "facets": facets,
        }

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _post_event(self, event: Dict) -> bool:
        """Envia um evento para o backend via HTTP POST."""
        if not self._session or not self.backend_url:
            return False

        url = f"{self.backend_url}/api/v1/lineage"
        try:
            resp = self._session.post(url, json=event, timeout=10)
            resp.raise_for_status()
            logger.debug("Evento enviado para %s (status=%s)", url, resp.status_code)
            return True
        except requests.RequestException as exc:
            logger.warning("Falha ao enviar evento para %s: %s", url, exc)
            return False


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _derive_job_name(transformation: Any) -> str:
    """Deriva um nome de job a partir de uma Transformation."""
    if hasattr(transformation, "source_file") and transformation.source_file:
        source = Path(transformation.source_file).stem
    else:
        source = "unknown"

    op = getattr(transformation, "operation", "TRANSFORM").upper()
    target = getattr(transformation.target, "name", "output")
    return f"{source}:{op}:{Path(target).stem}"


def _split_qualified_name(qualified_name: str, default_namespace: str) -> tuple:
    """Separa um nome qualificado em (namespace, name)."""
    if "/" in qualified_name:
        parts = qualified_name.split("/", 1)
        return parts[0], parts[1]
    if "." in qualified_name:
        # Ex: "schema.table" → namespace="schema", name="table"
        idx = qualified_name.rfind(".")
        return qualified_name[:idx] or default_namespace, qualified_name[idx + 1:]
    return default_namespace, qualified_name


def _to_uri(path_or_name: str) -> str:
    """Converte um caminho ou nome de asset para uma URI."""
    if path_or_name.startswith(("http://", "https://", "s3://", "gs://", "abfss://")):
        return path_or_name
    p = Path(path_or_name)
    if p.is_absolute():
        return p.as_uri()
    return f"file://{p}"


def _infer_language(source_file: str) -> str:
    ext_map = {
        ".py": "python",
        ".sql": "sql",
        ".scala": "scala",
        ".tf": "terraform",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    ext = Path(source_file).suffix.lower()
    return ext_map.get(ext, "unknown")
