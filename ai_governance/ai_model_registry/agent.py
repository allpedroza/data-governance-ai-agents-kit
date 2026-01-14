"""
AI Model Registry Agent

Inventário vivo de modelos/LLMs com rastreamento de versões, stages, owners e propósito.
Objetivo: eliminar "shadow AI" e servir como base para todos os demais controles de AI Governance.

Author: Data Governance AI Agents Kit
Version: 1.0.0
"""

import json
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict


# =============================================================================
# ENUMS
# =============================================================================

class ModelType(Enum):
    """Tipos de modelos de AI/ML suportados."""
    LLM = "llm"
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERING = "clustering"
    RECOMMENDER = "recommender"
    NER = "ner"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    AGENT = "agent"
    RAG = "rag"
    FINE_TUNED = "fine_tuned"
    CUSTOM = "custom"


class ModelStage(Enum):
    """Estágios do ciclo de vida do modelo."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ModelStatus(Enum):
    """Status operacional do modelo."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskLevel(Enum):
    """Nível de risco do modelo baseado em EU AI Act e regulamentações."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"
    UNASSESSED = "unassessed"


class DataSensitivity(Enum):
    """Sensibilidade dos dados usados pelo modelo."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"
    PCI = "pci"


class LicenseType(Enum):
    """Tipos de licenciamento de modelos."""
    PROPRIETARY = "proprietary"
    OPEN_SOURCE = "open_source"
    COMMERCIAL = "commercial"
    ACADEMIC = "academic"
    INTERNAL = "internal"
    API_SERVICE = "api_service"


class ChangeType(Enum):
    """Tipos de mudanças no registro de auditoria."""
    CREATED = "created"
    UPDATED = "updated"
    VERSION_ADDED = "version_added"
    STAGE_CHANGED = "stage_changed"
    STATUS_CHANGED = "status_changed"
    OWNER_CHANGED = "owner_changed"
    ARCHIVED = "archived"
    DELETED = "deleted"
    METADATA_UPDATED = "metadata_updated"
    RISK_ASSESSED = "risk_assessed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelOwner:
    """Proprietário/responsável pelo modelo."""
    name: str
    email: str
    team: str
    department: str
    role: str = "owner"  # owner, maintainer, contributor
    contact_phone: Optional[str] = None
    slack_channel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    """Métricas de performance e uso do modelo."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    daily_requests: Optional[int] = None
    monthly_cost_usd: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelVersion:
    """Versão específica de um modelo."""
    version: str
    created_at: datetime
    created_by: str
    stage: ModelStage
    status: ModelStatus
    description: str = ""
    changelog: str = ""
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    framework: Optional[str] = None  # pytorch, tensorflow, sklearn, etc.
    framework_version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    training_data_version: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['stage'] = self.stage.value
        data['status'] = self.status.value
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()
        return data


@dataclass
class ModelDependency:
    """Dependência entre modelos."""
    model_id: str
    model_name: str
    dependency_type: str  # uses, calls, feeds_into, inherits_from
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSource:
    """Fonte de dados usada pelo modelo."""
    name: str
    type: str  # database, api, file, stream
    location: str
    sensitivity: DataSensitivity
    description: str = ""
    refresh_frequency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['sensitivity'] = self.sensitivity.value
        return data


@dataclass
class AuditEntry:
    """Entrada de auditoria para rastreamento de mudanças."""
    timestamp: datetime
    change_type: ChangeType
    changed_by: str
    model_id: str
    model_name: str
    version: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['change_type'] = self.change_type.value
        return data


@dataclass
class RegisteredModel:
    """Modelo registrado no inventário."""
    model_id: str
    name: str
    model_type: ModelType
    purpose: str
    description: str
    owner: ModelOwner
    created_at: datetime
    updated_at: datetime
    risk_level: RiskLevel
    license_type: LicenseType
    versions: List[ModelVersion] = field(default_factory=list)
    current_version: Optional[str] = None
    data_sources: List[DataSource] = field(default_factory=list)
    data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    dependencies: List[ModelDependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # IDs de modelos que dependem deste
    use_cases: List[str] = field(default_factory=list)
    business_units: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    source_code_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_shadow_ai: bool = False  # Flag para modelos descobertos não registrados oficialmente
    discovery_source: Optional[str] = None  # Como o modelo foi descoberto

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'model_id': self.model_id,
            'name': self.name,
            'model_type': self.model_type.value,
            'purpose': self.purpose,
            'description': self.description,
            'owner': self.owner.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'risk_level': self.risk_level.value,
            'license_type': self.license_type.value,
            'versions': [v.to_dict() for v in self.versions],
            'current_version': self.current_version,
            'data_sources': [ds.to_dict() for ds in self.data_sources],
            'data_sensitivity': self.data_sensitivity.value,
            'dependencies': [d.to_dict() for d in self.dependencies],
            'dependents': self.dependents,
            'use_cases': self.use_cases,
            'business_units': self.business_units,
            'applications': self.applications,
            'tags': self.tags,
            'documentation_url': self.documentation_url,
            'source_code_url': self.source_code_url,
            'api_endpoint': self.api_endpoint,
            'sla_requirements': self.sla_requirements,
            'compliance_requirements': self.compliance_requirements,
            'metadata': self.metadata,
            'is_shadow_ai': self.is_shadow_ai,
            'discovery_source': self.discovery_source,
        }
        return data

    def get_production_version(self) -> Optional[ModelVersion]:
        """Retorna a versão em produção, se houver."""
        for v in self.versions:
            if v.stage == ModelStage.PRODUCTION and v.status == ModelStatus.ACTIVE:
                return v
        return None

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Retorna uma versão específica."""
        for v in self.versions:
            if v.version == version:
                return v
        return None


@dataclass
class RegistryStatistics:
    """Estatísticas do registro de modelos."""
    total_models: int
    models_by_type: Dict[str, int]
    models_by_stage: Dict[str, int]
    models_by_status: Dict[str, int]
    models_by_risk_level: Dict[str, int]
    models_by_owner_team: Dict[str, int]
    shadow_ai_count: int
    models_in_production: int
    models_pending_approval: int
    total_versions: int
    avg_versions_per_model: float
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class SearchResult:
    """Resultado de busca no registro."""
    models: List[RegisteredModel]
    total_count: int
    query: str
    filters_applied: Dict[str, Any]
    search_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'models': [m.to_dict() for m in self.models],
            'total_count': self.total_count,
            'query': self.query,
            'filters_applied': self.filters_applied,
            'search_time_ms': self.search_time_ms,
        }


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class AIModelRegistryAgent:
    """
    Agente de Registro de Modelos de AI.

    Mantém um inventário vivo de todos os modelos/LLMs da organização,
    incluindo versões, stages, owners e propósito. Objetivo principal
    é eliminar "shadow AI" e servir como base para todos os controles
    de AI Governance.

    Funcionalidades principais:
    - Registro e catalogação de modelos
    - Gerenciamento de versões e ciclo de vida
    - Rastreamento de ownership e responsabilidades
    - Descoberta e marcação de shadow AI
    - Auditoria completa de mudanças
    - Busca e filtros avançados
    - Análise de dependências entre modelos
    - Relatórios e estatísticas
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        enable_audit: bool = True,
    ):
        """
        Inicializa o agente de registro de modelos.

        Args:
            storage_path: Caminho para persistência do registro (JSON)
            auto_save: Salvar automaticamente após cada operação
            enable_audit: Habilitar auditoria de mudanças
        """
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.enable_audit = enable_audit

        # Armazenamento em memória
        self._models: Dict[str, RegisteredModel] = {}
        self._audit_log: List[AuditEntry] = []
        self._name_index: Dict[str, str] = {}  # name -> model_id
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> set of model_ids

        # Carregar dados existentes se houver
        if storage_path:
            self._load_from_storage()

    # =========================================================================
    # REGISTRO DE MODELOS
    # =========================================================================

    def register_model(
        self,
        name: str,
        model_type: ModelType,
        purpose: str,
        description: str,
        owner: ModelOwner,
        risk_level: RiskLevel = RiskLevel.UNASSESSED,
        license_type: LicenseType = LicenseType.INTERNAL,
        initial_version: Optional[str] = None,
        data_sources: Optional[List[DataSource]] = None,
        data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL,
        use_cases: Optional[List[str]] = None,
        business_units: Optional[List[str]] = None,
        applications: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        documentation_url: Optional[str] = None,
        source_code_url: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        compliance_requirements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        registered_by: str = "system",
    ) -> RegisteredModel:
        """
        Registra um novo modelo no inventário.

        Args:
            name: Nome único do modelo
            model_type: Tipo do modelo (LLM, classifier, etc.)
            purpose: Propósito/objetivo do modelo
            description: Descrição detalhada
            owner: Proprietário/responsável
            risk_level: Nível de risco (EU AI Act)
            license_type: Tipo de licenciamento
            initial_version: Versão inicial (opcional)
            data_sources: Fontes de dados usadas
            data_sensitivity: Sensibilidade dos dados
            use_cases: Casos de uso
            business_units: Unidades de negócio que usam
            applications: Aplicações que utilizam o modelo
            tags: Tags para categorização
            documentation_url: URL da documentação
            source_code_url: URL do código fonte
            api_endpoint: Endpoint da API (se aplicável)
            compliance_requirements: Requisitos de compliance
            metadata: Metadados adicionais
            registered_by: Quem está registrando

        Returns:
            RegisteredModel: O modelo registrado

        Raises:
            ValueError: Se já existir um modelo com o mesmo nome
        """
        # Validar nome único
        if name in self._name_index:
            raise ValueError(f"Modelo com nome '{name}' já existe. Use update_model() para atualizar.")

        # Gerar ID único
        model_id = self._generate_model_id(name)
        now = datetime.now()

        # Criar modelo
        model = RegisteredModel(
            model_id=model_id,
            name=name,
            model_type=model_type,
            purpose=purpose,
            description=description,
            owner=owner,
            created_at=now,
            updated_at=now,
            risk_level=risk_level,
            license_type=license_type,
            data_sources=data_sources or [],
            data_sensitivity=data_sensitivity,
            use_cases=use_cases or [],
            business_units=business_units or [],
            applications=applications or [],
            tags=tags or [],
            documentation_url=documentation_url,
            source_code_url=source_code_url,
            api_endpoint=api_endpoint,
            compliance_requirements=compliance_requirements or [],
            metadata=metadata or {},
        )

        # Adicionar versão inicial se fornecida
        if initial_version:
            version = ModelVersion(
                version=initial_version,
                created_at=now,
                created_by=registered_by,
                stage=ModelStage.DEVELOPMENT,
                status=ModelStatus.ACTIVE,
                description="Versão inicial",
            )
            model.versions.append(version)
            model.current_version = initial_version

        # Armazenar
        self._models[model_id] = model
        self._name_index[name] = model_id

        # Atualizar índice de tags
        for tag in model.tags:
            self._tag_index[tag].add(model_id)

        # Auditoria
        self._log_audit(
            change_type=ChangeType.CREATED,
            changed_by=registered_by,
            model_id=model_id,
            model_name=name,
            new_value=model.to_dict(),
            reason="Modelo registrado no inventário",
        )

        # Auto-save
        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model

    def register_shadow_ai(
        self,
        name: str,
        model_type: ModelType,
        purpose: str,
        discovery_source: str,
        discovered_by: str,
        owner: Optional[ModelOwner] = None,
        description: str = "",
        applications: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredModel:
        """
        Registra um modelo descoberto como Shadow AI.

        Use este método para registrar modelos que foram descobertos
        em uso mas não estavam oficialmente registrados.

        Args:
            name: Nome identificado do modelo
            model_type: Tipo do modelo
            purpose: Propósito identificado
            discovery_source: Como foi descoberto (scan de rede, logs, etc.)
            discovered_by: Quem descobriu
            owner: Proprietário (se identificado)
            description: Descrição
            applications: Aplicações onde foi encontrado
            metadata: Metadados adicionais

        Returns:
            RegisteredModel: O modelo registrado como shadow AI
        """
        # Owner padrão para shadow AI
        if owner is None:
            owner = ModelOwner(
                name="Não Identificado",
                email="unknown@company.com",
                team="Desconhecido",
                department="Desconhecido",
                role="unknown",
            )

        model = self.register_model(
            name=f"[SHADOW] {name}",
            model_type=model_type,
            purpose=purpose,
            description=description or f"Shadow AI descoberto via {discovery_source}",
            owner=owner,
            risk_level=RiskLevel.UNASSESSED,
            license_type=LicenseType.PROPRIETARY,
            applications=applications,
            tags=["shadow-ai", "needs-review", "discovered"],
            metadata=metadata or {},
            registered_by=discovered_by,
        )

        # Marcar como shadow AI
        model.is_shadow_ai = True
        model.discovery_source = discovery_source

        # Auditoria específica
        self._log_audit(
            change_type=ChangeType.CREATED,
            changed_by=discovered_by,
            model_id=model.model_id,
            model_name=model.name,
            new_value={"discovery_source": discovery_source},
            reason=f"Shadow AI descoberto via {discovery_source}",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model

    # =========================================================================
    # GERENCIAMENTO DE VERSÕES
    # =========================================================================

    def add_version(
        self,
        model_id_or_name: str,
        version: str,
        created_by: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        status: ModelStatus = ModelStatus.ACTIVE,
        description: str = "",
        changelog: str = "",
        artifact_path: Optional[str] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        python_version: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metrics: Optional[ModelMetrics] = None,
        training_data_version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        set_as_current: bool = False,
    ) -> ModelVersion:
        """
        Adiciona uma nova versão a um modelo existente.

        Args:
            model_id_or_name: ID ou nome do modelo
            version: Número/string da versão
            created_by: Quem criou a versão
            stage: Estágio inicial (development, staging, production)
            status: Status inicial
            description: Descrição da versão
            changelog: Log de mudanças
            artifact_path: Caminho para artefatos do modelo
            framework: Framework usado (pytorch, tensorflow, etc.)
            framework_version: Versão do framework
            python_version: Versão do Python
            dependencies: Lista de dependências
            metrics: Métricas de performance
            training_data_version: Versão dos dados de treino
            hyperparameters: Hiperparâmetros usados
            tags: Tags da versão
            set_as_current: Definir como versão atual

        Returns:
            ModelVersion: A versão criada
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        # Verificar se versão já existe
        if model.get_version(version):
            raise ValueError(f"Versão '{version}' já existe para o modelo '{model.name}'")

        now = datetime.now()

        # Calcular hash do artefato se fornecido
        artifact_hash = None
        if artifact_path:
            artifact_hash = self._calculate_artifact_hash(artifact_path)

        # Criar versão
        model_version = ModelVersion(
            version=version,
            created_at=now,
            created_by=created_by,
            stage=stage,
            status=status,
            description=description,
            changelog=changelog,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            framework=framework,
            framework_version=framework_version,
            python_version=python_version,
            dependencies=dependencies or [],
            metrics=metrics,
            training_data_version=training_data_version,
            hyperparameters=hyperparameters or {},
            tags=tags or [],
        )

        model.versions.append(model_version)
        model.updated_at = now

        if set_as_current:
            model.current_version = version

        # Auditoria
        self._log_audit(
            change_type=ChangeType.VERSION_ADDED,
            changed_by=created_by,
            model_id=model.model_id,
            model_name=model.name,
            version=version,
            new_value=model_version.to_dict(),
            reason=f"Nova versão {version} adicionada",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model_version

    def promote_version(
        self,
        model_id_or_name: str,
        version: str,
        new_stage: ModelStage,
        promoted_by: str,
        reason: str = "",
    ) -> ModelVersion:
        """
        Promove uma versão para um novo estágio.

        Args:
            model_id_or_name: ID ou nome do modelo
            version: Versão a promover
            new_stage: Novo estágio
            promoted_by: Quem está promovendo
            reason: Razão da promoção

        Returns:
            ModelVersion: A versão atualizada
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        model_version = model.get_version(version)
        if model_version is None:
            raise ValueError(f"Versão '{version}' não encontrada")

        old_stage = model_version.stage
        model_version.stage = new_stage
        model.updated_at = datetime.now()

        # Se promovido para produção, definir como versão atual
        if new_stage == ModelStage.PRODUCTION:
            model.current_version = version
            model_version.status = ModelStatus.ACTIVE

        # Auditoria
        self._log_audit(
            change_type=ChangeType.STAGE_CHANGED,
            changed_by=promoted_by,
            model_id=model.model_id,
            model_name=model.name,
            version=version,
            old_value=old_stage.value,
            new_value=new_stage.value,
            reason=reason or f"Versão promovida de {old_stage.value} para {new_stage.value}",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model_version

    def update_version_status(
        self,
        model_id_or_name: str,
        version: str,
        new_status: ModelStatus,
        updated_by: str,
        reason: str = "",
    ) -> ModelVersion:
        """
        Atualiza o status de uma versão.

        Args:
            model_id_or_name: ID ou nome do modelo
            version: Versão a atualizar
            new_status: Novo status
            updated_by: Quem está atualizando
            reason: Razão da mudança

        Returns:
            ModelVersion: A versão atualizada
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        model_version = model.get_version(version)
        if model_version is None:
            raise ValueError(f"Versão '{version}' não encontrada")

        old_status = model_version.status
        model_version.status = new_status
        model.updated_at = datetime.now()

        # Auditoria
        self._log_audit(
            change_type=ChangeType.STATUS_CHANGED,
            changed_by=updated_by,
            model_id=model.model_id,
            model_name=model.name,
            version=version,
            old_value=old_status.value,
            new_value=new_status.value,
            reason=reason or f"Status alterado de {old_status.value} para {new_status.value}",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model_version

    def update_version_metrics(
        self,
        model_id_or_name: str,
        version: str,
        metrics: ModelMetrics,
        updated_by: str,
    ) -> ModelVersion:
        """
        Atualiza as métricas de uma versão.

        Args:
            model_id_or_name: ID ou nome do modelo
            version: Versão a atualizar
            metrics: Novas métricas
            updated_by: Quem está atualizando

        Returns:
            ModelVersion: A versão atualizada
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        model_version = model.get_version(version)
        if model_version is None:
            raise ValueError(f"Versão '{version}' não encontrada")

        old_metrics = model_version.metrics.to_dict() if model_version.metrics else None
        model_version.metrics = metrics
        model.updated_at = datetime.now()

        # Auditoria
        self._log_audit(
            change_type=ChangeType.METADATA_UPDATED,
            changed_by=updated_by,
            model_id=model.model_id,
            model_name=model.name,
            version=version,
            old_value=old_metrics,
            new_value=metrics.to_dict(),
            reason="Métricas atualizadas",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model_version

    # =========================================================================
    # BUSCA E CONSULTAS
    # =========================================================================

    def get_model(self, model_id_or_name: str) -> Optional[RegisteredModel]:
        """
        Busca um modelo por ID ou nome.

        Args:
            model_id_or_name: ID ou nome do modelo

        Returns:
            RegisteredModel ou None se não encontrado
        """
        # Tentar por ID primeiro
        if model_id_or_name in self._models:
            return self._models[model_id_or_name]

        # Tentar por nome
        if model_id_or_name in self._name_index:
            model_id = self._name_index[model_id_or_name]
            return self._models.get(model_id)

        return None

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None,
        status: Optional[ModelStatus] = None,
        risk_level: Optional[RiskLevel] = None,
        owner_team: Optional[str] = None,
        business_unit: Optional[str] = None,
        tag: Optional[str] = None,
        include_shadow_ai: bool = True,
        only_shadow_ai: bool = False,
    ) -> List[RegisteredModel]:
        """
        Lista modelos com filtros opcionais.

        Args:
            model_type: Filtrar por tipo
            stage: Filtrar por estágio (da versão atual)
            status: Filtrar por status (da versão atual)
            risk_level: Filtrar por nível de risco
            owner_team: Filtrar por time do owner
            business_unit: Filtrar por unidade de negócio
            tag: Filtrar por tag
            include_shadow_ai: Incluir shadow AI nos resultados
            only_shadow_ai: Retornar apenas shadow AI

        Returns:
            Lista de modelos que atendem aos critérios
        """
        results = []

        for model in self._models.values():
            # Filtro shadow AI
            if only_shadow_ai and not model.is_shadow_ai:
                continue
            if not include_shadow_ai and model.is_shadow_ai:
                continue

            # Filtro por tipo
            if model_type and model.model_type != model_type:
                continue

            # Filtro por risco
            if risk_level and model.risk_level != risk_level:
                continue

            # Filtro por time do owner
            if owner_team and model.owner.team != owner_team:
                continue

            # Filtro por unidade de negócio
            if business_unit and business_unit not in model.business_units:
                continue

            # Filtro por tag
            if tag and tag not in model.tags:
                continue

            # Filtro por stage/status (baseado na versão atual ou qualquer versão)
            if stage or status:
                version_match = False
                for v in model.versions:
                    stage_ok = stage is None or v.stage == stage
                    status_ok = status is None or v.status == status
                    if stage_ok and status_ok:
                        version_match = True
                        break
                if not version_match:
                    continue

            results.append(model)

        return results

    def search(
        self,
        query: str,
        search_in: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
    ) -> SearchResult:
        """
        Busca textual em modelos.

        Args:
            query: Texto para buscar
            search_in: Campos para buscar (name, purpose, description, tags)
            filters: Filtros adicionais
            limit: Limite de resultados

        Returns:
            SearchResult: Resultados da busca
        """
        import time
        start_time = time.time()

        search_in = search_in or ['name', 'purpose', 'description', 'tags']
        filters = filters or {}
        query_lower = query.lower()

        matches = []

        for model in self._models.values():
            # Aplicar filtros primeiro
            if filters:
                if 'model_type' in filters and model.model_type.value != filters['model_type']:
                    continue
                if 'risk_level' in filters and model.risk_level.value != filters['risk_level']:
                    continue
                if 'owner_team' in filters and model.owner.team != filters['owner_team']:
                    continue

            # Busca textual
            match = False
            if 'name' in search_in and query_lower in model.name.lower():
                match = True
            if 'purpose' in search_in and query_lower in model.purpose.lower():
                match = True
            if 'description' in search_in and query_lower in model.description.lower():
                match = True
            if 'tags' in search_in:
                for tag in model.tags:
                    if query_lower in tag.lower():
                        match = True
                        break

            if match:
                matches.append(model)
                if len(matches) >= limit:
                    break

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResult(
            models=matches,
            total_count=len(matches),
            query=query,
            filters_applied=filters,
            search_time_ms=search_time_ms,
        )

    def get_models_by_tag(self, tag: str) -> List[RegisteredModel]:
        """
        Retorna todos os modelos com uma tag específica.

        Args:
            tag: Tag para buscar

        Returns:
            Lista de modelos com a tag
        """
        model_ids = self._tag_index.get(tag, set())
        return [self._models[mid] for mid in model_ids if mid in self._models]

    def get_production_models(self) -> List[RegisteredModel]:
        """Retorna todos os modelos com versão em produção."""
        return [
            model for model in self._models.values()
            if model.get_production_version() is not None
        ]

    def get_shadow_ai_models(self) -> List[RegisteredModel]:
        """Retorna todos os modelos marcados como Shadow AI."""
        return [model for model in self._models.values() if model.is_shadow_ai]

    def get_high_risk_models(self) -> List[RegisteredModel]:
        """Retorna todos os modelos de alto risco ou risco inaceitável."""
        return [
            model for model in self._models.values()
            if model.risk_level in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]
        ]

    # =========================================================================
    # GERENCIAMENTO DE DEPENDÊNCIAS
    # =========================================================================

    def add_dependency(
        self,
        model_id_or_name: str,
        depends_on_id_or_name: str,
        dependency_type: str,
        description: str = "",
        added_by: str = "system",
    ) -> None:
        """
        Adiciona uma dependência entre modelos.

        Args:
            model_id_or_name: Modelo que depende
            depends_on_id_or_name: Modelo do qual depende
            dependency_type: Tipo de dependência (uses, calls, feeds_into, inherits_from)
            description: Descrição da dependência
            added_by: Quem está adicionando
        """
        model = self.get_model(model_id_or_name)
        depends_on = self.get_model(depends_on_id_or_name)

        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")
        if depends_on is None:
            raise ValueError(f"Modelo '{depends_on_id_or_name}' não encontrado")

        # Adicionar dependência
        dep = ModelDependency(
            model_id=depends_on.model_id,
            model_name=depends_on.name,
            dependency_type=dependency_type,
            description=description,
        )
        model.dependencies.append(dep)
        model.updated_at = datetime.now()

        # Adicionar como dependente no outro modelo
        if model.model_id not in depends_on.dependents:
            depends_on.dependents.append(model.model_id)
            depends_on.updated_at = datetime.now()

        # Auditoria
        self._log_audit(
            change_type=ChangeType.UPDATED,
            changed_by=added_by,
            model_id=model.model_id,
            model_name=model.name,
            new_value={"dependency": dep.to_dict()},
            reason=f"Dependência adicionada: {model.name} -> {depends_on.name}",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

    def get_dependency_graph(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Retorna o grafo de dependências entre modelos.

        Returns:
            Dicionário com model_id -> lista de dependências
        """
        graph = {}

        for model in self._models.values():
            deps = []
            for dep in model.dependencies:
                deps.append({
                    'model_id': dep.model_id,
                    'model_name': dep.model_name,
                    'type': dep.dependency_type,
                })
            graph[model.model_id] = deps

        return graph

    def get_impact_analysis(self, model_id_or_name: str) -> Dict[str, Any]:
        """
        Analisa o impacto de mudanças em um modelo.

        Retorna todos os modelos que seriam afetados se este modelo
        fosse modificado ou desativado.

        Args:
            model_id_or_name: ID ou nome do modelo

        Returns:
            Análise de impacto
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        # Encontrar todos os dependentes recursivamente
        affected = set()
        to_process = [model.model_id]

        while to_process:
            current_id = to_process.pop()
            current_model = self._models.get(current_id)
            if current_model:
                for dependent_id in current_model.dependents:
                    if dependent_id not in affected:
                        affected.add(dependent_id)
                        to_process.append(dependent_id)

        affected_models = [self._models[mid] for mid in affected if mid in self._models]

        return {
            'model_id': model.model_id,
            'model_name': model.name,
            'direct_dependents': len(model.dependents),
            'total_affected': len(affected),
            'affected_models': [
                {
                    'model_id': m.model_id,
                    'model_name': m.name,
                    'risk_level': m.risk_level.value,
                    'stage': m.get_production_version().stage.value if m.get_production_version() else 'N/A',
                }
                for m in affected_models
            ],
            'risk_summary': {
                'high_risk_affected': sum(1 for m in affected_models if m.risk_level == RiskLevel.HIGH),
                'production_affected': sum(1 for m in affected_models if m.get_production_version()),
            }
        }

    # =========================================================================
    # ATUALIZAÇÃO DE MODELOS
    # =========================================================================

    def update_model(
        self,
        model_id_or_name: str,
        updated_by: str,
        purpose: Optional[str] = None,
        description: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        owner: Optional[ModelOwner] = None,
        use_cases: Optional[List[str]] = None,
        business_units: Optional[List[str]] = None,
        applications: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        documentation_url: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        compliance_requirements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredModel:
        """
        Atualiza informações de um modelo.

        Args:
            model_id_or_name: ID ou nome do modelo
            updated_by: Quem está atualizando
            ... outros campos para atualizar ...

        Returns:
            RegisteredModel: O modelo atualizado
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        changes = {}

        if purpose is not None and purpose != model.purpose:
            changes['purpose'] = {'old': model.purpose, 'new': purpose}
            model.purpose = purpose

        if description is not None and description != model.description:
            changes['description'] = {'old': model.description, 'new': description}
            model.description = description

        if risk_level is not None and risk_level != model.risk_level:
            changes['risk_level'] = {'old': model.risk_level.value, 'new': risk_level.value}
            model.risk_level = risk_level

            # Auditoria específica para mudança de risco
            self._log_audit(
                change_type=ChangeType.RISK_ASSESSED,
                changed_by=updated_by,
                model_id=model.model_id,
                model_name=model.name,
                old_value=changes['risk_level']['old'],
                new_value=changes['risk_level']['new'],
                reason="Nível de risco reavaliado",
            )

        if owner is not None:
            changes['owner'] = {'old': model.owner.to_dict(), 'new': owner.to_dict()}
            model.owner = owner

            # Auditoria específica para mudança de owner
            self._log_audit(
                change_type=ChangeType.OWNER_CHANGED,
                changed_by=updated_by,
                model_id=model.model_id,
                model_name=model.name,
                old_value=changes['owner']['old'],
                new_value=changes['owner']['new'],
                reason="Ownership transferido",
            )

        if use_cases is not None:
            changes['use_cases'] = {'old': model.use_cases, 'new': use_cases}
            model.use_cases = use_cases

        if business_units is not None:
            changes['business_units'] = {'old': model.business_units, 'new': business_units}
            model.business_units = business_units

        if applications is not None:
            changes['applications'] = {'old': model.applications, 'new': applications}
            model.applications = applications

        if tags is not None:
            # Atualizar índice de tags
            old_tags = set(model.tags)
            new_tags = set(tags)

            for tag in old_tags - new_tags:
                self._tag_index[tag].discard(model.model_id)
            for tag in new_tags - old_tags:
                self._tag_index[tag].add(model.model_id)

            changes['tags'] = {'old': model.tags, 'new': tags}
            model.tags = tags

        if documentation_url is not None:
            changes['documentation_url'] = {'old': model.documentation_url, 'new': documentation_url}
            model.documentation_url = documentation_url

        if api_endpoint is not None:
            changes['api_endpoint'] = {'old': model.api_endpoint, 'new': api_endpoint}
            model.api_endpoint = api_endpoint

        if compliance_requirements is not None:
            changes['compliance_requirements'] = {'old': model.compliance_requirements, 'new': compliance_requirements}
            model.compliance_requirements = compliance_requirements

        if metadata is not None:
            changes['metadata'] = {'old': model.metadata, 'new': metadata}
            model.metadata.update(metadata)

        if changes:
            model.updated_at = datetime.now()

            # Auditoria geral
            self._log_audit(
                change_type=ChangeType.UPDATED,
                changed_by=updated_by,
                model_id=model.model_id,
                model_name=model.name,
                old_value={k: v['old'] for k, v in changes.items()},
                new_value={k: v['new'] for k, v in changes.items()},
                reason="Modelo atualizado",
            )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model

    def archive_model(
        self,
        model_id_or_name: str,
        archived_by: str,
        reason: str = "",
    ) -> RegisteredModel:
        """
        Arquiva um modelo (soft delete).

        Args:
            model_id_or_name: ID ou nome do modelo
            archived_by: Quem está arquivando
            reason: Razão do arquivamento

        Returns:
            RegisteredModel: O modelo arquivado
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        # Arquivar todas as versões
        for version in model.versions:
            version.stage = ModelStage.ARCHIVED
            version.status = ModelStatus.INACTIVE

        model.tags.append("archived")
        model.updated_at = datetime.now()

        # Auditoria
        self._log_audit(
            change_type=ChangeType.ARCHIVED,
            changed_by=archived_by,
            model_id=model.model_id,
            model_name=model.name,
            reason=reason or "Modelo arquivado",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model

    def legitimize_shadow_ai(
        self,
        model_id_or_name: str,
        new_owner: ModelOwner,
        legitimized_by: str,
        purpose: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> RegisteredModel:
        """
        Legitima um modelo de Shadow AI, removendo a flag e atualizando informações.

        Args:
            model_id_or_name: ID ou nome do modelo
            new_owner: Novo proprietário responsável
            legitimized_by: Quem está legitimando
            purpose: Propósito atualizado
            risk_level: Nível de risco avaliado

        Returns:
            RegisteredModel: O modelo legitimado
        """
        model = self.get_model(model_id_or_name)
        if model is None:
            raise ValueError(f"Modelo '{model_id_or_name}' não encontrado")

        if not model.is_shadow_ai:
            raise ValueError(f"Modelo '{model.name}' não é um Shadow AI")

        # Atualizar informações
        model.is_shadow_ai = False
        model.owner = new_owner
        model.name = model.name.replace("[SHADOW] ", "")  # Remover prefixo

        if purpose:
            model.purpose = purpose
        if risk_level:
            model.risk_level = risk_level

        # Atualizar tags
        model.tags = [t for t in model.tags if t not in ['shadow-ai', 'needs-review', 'discovered']]
        model.tags.append("legitimized")

        model.updated_at = datetime.now()

        # Atualizar índice de nomes
        old_name = f"[SHADOW] {model.name}"
        if old_name in self._name_index:
            del self._name_index[old_name]
        self._name_index[model.name] = model.model_id

        # Auditoria
        self._log_audit(
            change_type=ChangeType.UPDATED,
            changed_by=legitimized_by,
            model_id=model.model_id,
            model_name=model.name,
            new_value={
                'is_shadow_ai': False,
                'owner': new_owner.to_dict(),
            },
            reason="Shadow AI legitimado e incorporado ao registro oficial",
        )

        if self.auto_save and self.storage_path:
            self._save_to_storage()

        return model

    # =========================================================================
    # ESTATÍSTICAS E RELATÓRIOS
    # =========================================================================

    def get_statistics(self) -> RegistryStatistics:
        """
        Retorna estatísticas do registro de modelos.

        Returns:
            RegistryStatistics: Estatísticas completas
        """
        models = list(self._models.values())

        # Contagens por tipo
        by_type = defaultdict(int)
        for m in models:
            by_type[m.model_type.value] += 1

        # Contagens por stage (baseado na versão atual ou mais recente)
        by_stage = defaultdict(int)
        for m in models:
            if m.versions:
                current = m.get_version(m.current_version) if m.current_version else m.versions[-1]
                by_stage[current.stage.value] += 1

        # Contagens por status
        by_status = defaultdict(int)
        for m in models:
            if m.versions:
                current = m.get_version(m.current_version) if m.current_version else m.versions[-1]
                by_status[current.status.value] += 1

        # Contagens por risco
        by_risk = defaultdict(int)
        for m in models:
            by_risk[m.risk_level.value] += 1

        # Contagens por time
        by_team = defaultdict(int)
        for m in models:
            by_team[m.owner.team] += 1

        # Outras estatísticas
        total_versions = sum(len(m.versions) for m in models)
        avg_versions = total_versions / len(models) if models else 0

        return RegistryStatistics(
            total_models=len(models),
            models_by_type=dict(by_type),
            models_by_stage=dict(by_stage),
            models_by_status=dict(by_status),
            models_by_risk_level=dict(by_risk),
            models_by_owner_team=dict(by_team),
            shadow_ai_count=sum(1 for m in models if m.is_shadow_ai),
            models_in_production=sum(1 for m in models if m.get_production_version()),
            models_pending_approval=sum(
                1 for m in models
                if any(v.status == ModelStatus.PENDING_APPROVAL for v in m.versions)
            ),
            total_versions=total_versions,
            avg_versions_per_model=round(avg_versions, 2),
            last_updated=datetime.now(),
        )

    def generate_report(self, format: str = "markdown") -> str:
        """
        Gera um relatório completo do registro.

        Args:
            format: Formato do relatório (markdown, json)

        Returns:
            Relatório no formato especificado
        """
        stats = self.get_statistics()
        shadow_ai = self.get_shadow_ai_models()
        high_risk = self.get_high_risk_models()
        production = self.get_production_models()

        if format == "json":
            return json.dumps({
                'statistics': stats.to_dict(),
                'shadow_ai': [m.to_dict() for m in shadow_ai],
                'high_risk': [m.to_dict() for m in high_risk],
                'production': [m.to_dict() for m in production],
                'generated_at': datetime.now().isoformat(),
            }, indent=2, ensure_ascii=False)

        # Markdown format
        report = []
        report.append("# AI Model Registry Report")
        report.append(f"\n**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Summary Statistics")
        report.append(f"- **Total Models:** {stats.total_models}")
        report.append(f"- **Models in Production:** {stats.models_in_production}")
        report.append(f"- **Shadow AI Detected:** {stats.shadow_ai_count}")
        report.append(f"- **Pending Approval:** {stats.models_pending_approval}")
        report.append(f"- **Total Versions:** {stats.total_versions}")
        report.append(f"- **Avg Versions/Model:** {stats.avg_versions_per_model}")

        report.append("\n## Models by Type")
        for model_type, count in sorted(stats.models_by_type.items()):
            report.append(f"- {model_type}: {count}")

        report.append("\n## Models by Risk Level")
        for risk, count in sorted(stats.models_by_risk_level.items()):
            emoji = {"unacceptable": "🔴", "high": "🟠", "limited": "🟡", "minimal": "🟢", "unassessed": "⚪"}.get(risk, "")
            report.append(f"- {emoji} {risk}: {count}")

        report.append("\n## Models by Stage")
        for stage, count in sorted(stats.models_by_stage.items()):
            report.append(f"- {stage}: {count}")

        report.append("\n## Models by Team")
        for team, count in sorted(stats.models_by_owner_team.items(), key=lambda x: -x[1]):
            report.append(f"- {team}: {count}")

        if shadow_ai:
            report.append("\n## ⚠️ Shadow AI Models (Require Review)")
            for m in shadow_ai:
                report.append(f"- **{m.name}**")
                report.append(f"  - Discovery Source: {m.discovery_source}")
                report.append(f"  - Applications: {', '.join(m.applications) if m.applications else 'Unknown'}")

        if high_risk:
            report.append("\n## 🔴 High Risk Models")
            for m in high_risk:
                report.append(f"- **{m.name}** ({m.risk_level.value})")
                report.append(f"  - Purpose: {m.purpose}")
                report.append(f"  - Owner: {m.owner.name} ({m.owner.team})")

        report.append("\n## Production Models")
        for m in production:
            prod_version = m.get_production_version()
            report.append(f"- **{m.name}** v{prod_version.version if prod_version else 'N/A'}")
            report.append(f"  - Type: {m.model_type.value}")
            report.append(f"  - Owner: {m.owner.name} ({m.owner.team})")

        return "\n".join(report)

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Gera relatório de compliance para auditoria regulatória.

        Returns:
            Relatório de compliance
        """
        models = list(self._models.values())

        # Análise de risco
        risk_analysis = {
            'total_assessed': sum(1 for m in models if m.risk_level != RiskLevel.UNASSESSED),
            'total_unassessed': sum(1 for m in models if m.risk_level == RiskLevel.UNASSESSED),
            'high_risk_in_production': [
                m.name for m in models
                if m.risk_level in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]
                and m.get_production_version()
            ],
        }

        # Shadow AI
        shadow_analysis = {
            'total_shadow_ai': sum(1 for m in models if m.is_shadow_ai),
            'shadow_ai_in_production': [
                m.name for m in models if m.is_shadow_ai and m.get_production_version()
            ],
        }

        # Ownership
        ownership_analysis = {
            'models_without_clear_owner': [
                m.name for m in models if m.owner.name == "Não Identificado"
            ],
        }

        # Dados sensíveis
        data_sensitivity_analysis = {
            'models_with_pii': [m.name for m in models if m.data_sensitivity == DataSensitivity.PII],
            'models_with_phi': [m.name for m in models if m.data_sensitivity == DataSensitivity.PHI],
            'models_with_pci': [m.name for m in models if m.data_sensitivity == DataSensitivity.PCI],
        }

        return {
            'generated_at': datetime.now().isoformat(),
            'total_models': len(models),
            'risk_analysis': risk_analysis,
            'shadow_ai_analysis': shadow_analysis,
            'ownership_analysis': ownership_analysis,
            'data_sensitivity_analysis': data_sensitivity_analysis,
            'compliance_gaps': {
                'unassessed_models': risk_analysis['total_unassessed'],
                'shadow_ai_count': shadow_analysis['total_shadow_ai'],
                'high_risk_production': len(risk_analysis['high_risk_in_production']),
                'unknown_owners': len(ownership_analysis['models_without_clear_owner']),
            },
            'recommendations': self._generate_compliance_recommendations(
                risk_analysis, shadow_analysis, ownership_analysis
            ),
        }

    def _generate_compliance_recommendations(
        self,
        risk_analysis: Dict,
        shadow_analysis: Dict,
        ownership_analysis: Dict,
    ) -> List[str]:
        """Gera recomendações baseadas na análise de compliance."""
        recommendations = []

        if risk_analysis['total_unassessed'] > 0:
            recommendations.append(
                f"URGENTE: {risk_analysis['total_unassessed']} modelo(s) não avaliados quanto ao risco. "
                "Realize avaliação de risco conforme EU AI Act."
            )

        if risk_analysis['high_risk_in_production']:
            recommendations.append(
                f"ATENÇÃO: {len(risk_analysis['high_risk_in_production'])} modelo(s) de alto risco em produção. "
                "Verifique conformidade com requisitos regulatórios."
            )

        if shadow_analysis['total_shadow_ai'] > 0:
            recommendations.append(
                f"CRÍTICO: {shadow_analysis['total_shadow_ai']} Shadow AI detectado(s). "
                "Legitime ou desative urgentemente para eliminar riscos não controlados."
            )

        if ownership_analysis['models_without_clear_owner']:
            recommendations.append(
                f"IMPORTANTE: {len(ownership_analysis['models_without_clear_owner'])} modelo(s) sem proprietário definido. "
                "Atribua ownership para garantir accountability."
            )

        if not recommendations:
            recommendations.append("✅ Nenhuma recomendação crítica. O registro está em conformidade.")

        return recommendations

    # =========================================================================
    # AUDITORIA
    # =========================================================================

    def get_audit_log(
        self,
        model_id_or_name: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        changed_by: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Retorna entradas do log de auditoria com filtros.

        Args:
            model_id_or_name: Filtrar por modelo
            change_type: Filtrar por tipo de mudança
            changed_by: Filtrar por quem fez a mudança
            start_date: Data inicial
            end_date: Data final
            limit: Limite de resultados

        Returns:
            Lista de entradas de auditoria
        """
        results = []

        # Resolver model_id se nome foi fornecido
        model_id = None
        if model_id_or_name:
            model = self.get_model(model_id_or_name)
            if model:
                model_id = model.model_id

        for entry in reversed(self._audit_log):  # Mais recentes primeiro
            if model_id and entry.model_id != model_id:
                continue
            if change_type and entry.change_type != change_type:
                continue
            if changed_by and entry.changed_by != changed_by:
                continue
            if start_date and entry.timestamp < start_date:
                continue
            if end_date and entry.timestamp > end_date:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def _log_audit(
        self,
        change_type: ChangeType,
        changed_by: str,
        model_id: str,
        model_name: str,
        version: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        reason: str = "",
    ) -> None:
        """Registra uma entrada de auditoria."""
        if not self.enable_audit:
            return

        entry = AuditEntry(
            timestamp=datetime.now(),
            change_type=change_type,
            changed_by=changed_by,
            model_id=model_id,
            model_name=model_name,
            version=version,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self._audit_log.append(entry)

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def _save_to_storage(self) -> None:
        """Salva o registro para o arquivo de storage."""
        if not self.storage_path:
            return

        data = {
            'models': {mid: m.to_dict() for mid, m in self._models.items()},
            'audit_log': [e.to_dict() for e in self._audit_log],
            'saved_at': datetime.now().isoformat(),
        }

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_storage(self) -> None:
        """Carrega o registro do arquivo de storage."""
        import os

        if not self.storage_path or not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruir modelos
            for model_id, model_data in data.get('models', {}).items():
                model = self._dict_to_model(model_data)
                self._models[model_id] = model
                self._name_index[model.name] = model_id
                for tag in model.tags:
                    self._tag_index[tag].add(model_id)

            # Reconstruir audit log
            for entry_data in data.get('audit_log', []):
                entry = AuditEntry(
                    timestamp=datetime.fromisoformat(entry_data['timestamp']),
                    change_type=ChangeType(entry_data['change_type']),
                    changed_by=entry_data['changed_by'],
                    model_id=entry_data['model_id'],
                    model_name=entry_data['model_name'],
                    version=entry_data.get('version'),
                    old_value=entry_data.get('old_value'),
                    new_value=entry_data.get('new_value'),
                    reason=entry_data.get('reason', ''),
                )
                self._audit_log.append(entry)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load storage file: {e}")

    def _dict_to_model(self, data: Dict) -> RegisteredModel:
        """Converte um dicionário para RegisteredModel."""
        # Reconstruir owner
        owner_data = data['owner']
        owner = ModelOwner(**owner_data)

        # Reconstruir data sources
        data_sources = []
        for ds_data in data.get('data_sources', []):
            ds_data['sensitivity'] = DataSensitivity(ds_data['sensitivity'])
            data_sources.append(DataSource(**ds_data))

        # Reconstruir dependencies
        dependencies = [ModelDependency(**d) for d in data.get('dependencies', [])]

        # Reconstruir versions
        versions = []
        for v_data in data.get('versions', []):
            v_data['created_at'] = datetime.fromisoformat(v_data['created_at'])
            v_data['stage'] = ModelStage(v_data['stage'])
            v_data['status'] = ModelStatus(v_data['status'])

            if v_data.get('metrics'):
                v_data['metrics'] = ModelMetrics(**v_data['metrics'])

            versions.append(ModelVersion(**v_data))

        return RegisteredModel(
            model_id=data['model_id'],
            name=data['name'],
            model_type=ModelType(data['model_type']),
            purpose=data['purpose'],
            description=data['description'],
            owner=owner,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            risk_level=RiskLevel(data['risk_level']),
            license_type=LicenseType(data['license_type']),
            versions=versions,
            current_version=data.get('current_version'),
            data_sources=data_sources,
            data_sensitivity=DataSensitivity(data['data_sensitivity']),
            dependencies=dependencies,
            dependents=data.get('dependents', []),
            use_cases=data.get('use_cases', []),
            business_units=data.get('business_units', []),
            applications=data.get('applications', []),
            tags=data.get('tags', []),
            documentation_url=data.get('documentation_url'),
            source_code_url=data.get('source_code_url'),
            api_endpoint=data.get('api_endpoint'),
            sla_requirements=data.get('sla_requirements', {}),
            compliance_requirements=data.get('compliance_requirements', []),
            metadata=data.get('metadata', {}),
            is_shadow_ai=data.get('is_shadow_ai', False),
            discovery_source=data.get('discovery_source'),
        )

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    def _generate_model_id(self, name: str) -> str:
        """Gera um ID único para o modelo."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"model_{name_hash}_{timestamp}"

    def _calculate_artifact_hash(self, artifact_path: str) -> Optional[str]:
        """Calcula o hash SHA256 de um artefato."""
        import os

        if not os.path.exists(artifact_path):
            return None

        sha256 = hashlib.sha256()
        try:
            with open(artifact_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return None

    def export_to_json(self) -> str:
        """Exporta todo o registro para JSON."""
        return json.dumps({
            'models': {mid: m.to_dict() for mid, m in self._models.items()},
            'statistics': self.get_statistics().to_dict(),
            'exported_at': datetime.now().isoformat(),
        }, indent=2, ensure_ascii=False)

    def export_model_catalog(self) -> List[Dict[str, Any]]:
        """
        Exporta um catálogo simplificado de modelos.

        Útil para integração com ferramentas externas.
        """
        catalog = []

        for model in self._models.values():
            prod_version = model.get_production_version()

            catalog.append({
                'id': model.model_id,
                'name': model.name,
                'type': model.model_type.value,
                'purpose': model.purpose,
                'owner': model.owner.name,
                'owner_team': model.owner.team,
                'risk_level': model.risk_level.value,
                'current_version': model.current_version,
                'production_version': prod_version.version if prod_version else None,
                'is_shadow_ai': model.is_shadow_ai,
                'tags': model.tags,
                'last_updated': model.updated_at.isoformat(),
            })

        return catalog
