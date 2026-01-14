"""
Data Contract Agent

Creates, validates, and manages data contracts for ingestion pipelines.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    ContractField,
    ContractValidationFinding,
    ContractValidationReport,
    DataContract,
    DataContractSLA,
    DataQualityRule,
)


class DataContractAgent:
    """Agent responsible for contract creation and validation."""

    def __init__(self, contract_store: str = "./data_contracts") -> None:
        self.contract_store = Path(contract_store)
        self.contract_store.mkdir(parents=True, exist_ok=True)

    def create_contract(
        self,
        name: str,
        version: str,
        owner: str,
        domain: Optional[str] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        fields: Optional[List[ContractField]] = None,
        quality_rules: Optional[List[DataQualityRule]] = None,
        sla: Optional[DataContractSLA] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataContract:
        return DataContract(
            name=name,
            version=version,
            owner=owner,
            domain=domain,
            description=description,
            source=source,
            destination=destination,
            fields=fields or [],
            quality_rules=quality_rules or [],
            sla=sla,
            tags=tags or [],
            metadata=metadata or {},
        )

    def save_contract(self, contract: DataContract, filename: Optional[str] = None) -> Path:
        safe_name = (filename or f"{contract.name}_v{contract.version}").replace(" ", "_")
        path = self.contract_store / f"{safe_name}.json"
        path.write_text(contract.to_json(), encoding="utf-8")
        return path

    def load_contract(self, path: str | Path) -> DataContract:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self._contract_from_dict(payload)

    def infer_schema_from_file(self, path: str, sample_rows: int = 200) -> List[ContractField]:
        df = self._load_dataframe(path, sample_rows=sample_rows)
        return self.infer_schema_from_dataframe(df)

    def infer_schema_from_dataframe(self, df: Any) -> List[ContractField]:
        fields: List[ContractField] = []
        for column in df.columns:
            dtype = str(df[column].dtype)
            fields.append(
                ContractField(
                    name=str(column),
                    data_type=self._normalize_dtype(dtype),
                    required=False,
                    description="",
                )
            )
        return fields

    def validate_dataframe(self, contract: DataContract, df: Any) -> ContractValidationReport:
        findings: List[ContractValidationFinding] = []
        row_count = len(df)
        column_count = len(df.columns)
        metrics: Dict[str, Any] = {
            "row_count": row_count,
            "column_count": column_count,
            "rules_evaluated": 0,
            "rules_failed": 0,
        }

        missing_columns = [field.name for field in contract.fields if field.name not in df.columns]
        for missing in missing_columns:
            findings.append(
                ContractValidationFinding(
                    level="error",
                    message=f"Missing required column: {missing}",
                    column=missing,
                )
            )

        for field in contract.fields:
            if field.name not in df.columns:
                continue
            series = df[field.name]
            if field.required:
                null_pct = series.isna().mean() if row_count else 0
                if null_pct > 0:
                    findings.append(
                        ContractValidationFinding(
                            level="error",
                            message=f"Column {field.name} has {null_pct:.1%} null values.",
                            column=field.name,
                        )
                    )

            inferred = self._normalize_dtype(str(series.dtype))
            if field.data_type and inferred and field.data_type != inferred:
                findings.append(
                    ContractValidationFinding(
                        level="warning",
                        message=(
                            f"Column {field.name} expected type {field.data_type} "
                            f"but inferred {inferred}."
                        ),
                        column=field.name,
                    )
                )

            if field.constraints:
                metrics["rules_evaluated"] += 1
                constraint_findings = self._validate_constraints(field, series)
                if constraint_findings:
                    metrics["rules_failed"] += len(constraint_findings)
                    findings.extend(constraint_findings)

        for rule in contract.quality_rules:
            if rule.column not in df.columns:
                findings.append(
                    ContractValidationFinding(
                        level="error",
                        message=f"Quality rule {rule.name} references missing column {rule.column}.",
                        column=rule.column,
                    )
                )
                metrics["rules_failed"] += 1
                continue

            metrics["rules_evaluated"] += 1
            rule_findings = self._evaluate_quality_rule(rule, df[rule.column])
            if rule_findings:
                metrics["rules_failed"] += len(rule_findings)
                findings.extend(rule_findings)

        status = "passed" if not any(f.level == "error" for f in findings) else "failed"
        if status == "passed" and any(f.level == "warning" for f in findings):
            status = "warning"

        return ContractValidationReport(
            contract_name=contract.name,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
            row_count=row_count,
            column_count=column_count,
            findings=findings,
            metrics=metrics,
        )

    def _evaluate_quality_rule(
        self,
        rule: DataQualityRule,
        series: Any,
    ) -> List[ContractValidationFinding]:
        findings: List[ContractValidationFinding] = []
        rule_type = rule.rule_type
        params = rule.parameters or {}
        column = rule.column

        if rule_type == "not_null":
            null_pct = series.isna().mean() if len(series) else 0
            threshold = params.get("max_null_pct", 0.0)
            if null_pct > threshold:
                findings.append(
                    ContractValidationFinding(
                        level=self._severity(rule),
                        message=(
                            f"Rule {rule.name}: {column} has {null_pct:.1%} nulls "
                            f"(allowed {threshold:.1%})."
                        ),
                        column=column,
                    )
                )
        elif rule_type == "unique":
            duplicate_pct = 1 - series.nunique(dropna=False) / max(len(series), 1)
            threshold = params.get("max_duplicate_pct", 0.0)
            if duplicate_pct > threshold:
                findings.append(
                    ContractValidationFinding(
                        level=self._severity(rule),
                        message=(
                            f"Rule {rule.name}: {column} has {duplicate_pct:.1%} duplicates "
                            f"(allowed {threshold:.1%})."
                        ),
                        column=column,
                    )
                )
        elif rule_type == "range":
            min_value = params.get("min")
            max_value = params.get("max")
            if min_value is not None:
                below_min = (series < min_value).mean() if len(series) else 0
                if below_min > 0:
                    findings.append(
                        ContractValidationFinding(
                            level=self._severity(rule),
                            message=f"Rule {rule.name}: {column} has values below {min_value}.",
                            column=column,
                        )
                    )
            if max_value is not None:
                above_max = (series > max_value).mean() if len(series) else 0
                if above_max > 0:
                    findings.append(
                        ContractValidationFinding(
                            level=self._severity(rule),
                            message=f"Rule {rule.name}: {column} has values above {max_value}.",
                            column=column,
                        )
                    )
        elif rule_type == "regex":
            pattern = params.get("pattern")
            if pattern:
                invalid = (~series.astype(str).str.match(pattern)).mean() if len(series) else 0
                if invalid > 0:
                    findings.append(
                        ContractValidationFinding(
                            level=self._severity(rule),
                            message=(
                                f"Rule {rule.name}: {column} has {invalid:.1%} values not matching {pattern}."
                            ),
                            column=column,
                        )
                    )
        elif rule_type == "allowed_values":
            values = set(params.get("values", []))
            if values:
                invalid = (~series.isin(values)).mean() if len(series) else 0
                if invalid > 0:
                    findings.append(
                        ContractValidationFinding(
                            level=self._severity(rule),
                            message=(
                                f"Rule {rule.name}: {column} has {invalid:.1%} values outside {values}."
                            ),
                            column=column,
                        )
                    )

        return findings

    def _validate_constraints(
        self,
        field: ContractField,
        series: Any,
    ) -> List[ContractValidationFinding]:
        findings: List[ContractValidationFinding] = []
        constraints = field.constraints
        column = field.name

        if "min" in constraints:
            invalid = (series < constraints["min"]).mean() if len(series) else 0
            if invalid > 0:
                findings.append(
                    ContractValidationFinding(
                        level="warning",
                        message=f"Column {column} has values below {constraints['min']}.",
                        column=column,
                    )
                )
        if "max" in constraints:
            invalid = (series > constraints["max"]).mean() if len(series) else 0
            if invalid > 0:
                findings.append(
                    ContractValidationFinding(
                        level="warning",
                        message=f"Column {column} has values above {constraints['max']}.",
                        column=column,
                    )
                )
        if "regex" in constraints:
            pattern = constraints["regex"]
            invalid = (~series.astype(str).str.match(pattern)).mean() if len(series) else 0
            if invalid > 0:
                findings.append(
                    ContractValidationFinding(
                        level="warning",
                        message=f"Column {column} has values not matching regex {pattern}.",
                        column=column,
                    )
                )

        return findings

    def _load_dataframe(self, path: str, sample_rows: int = 200):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to infer schema. Install with: pip install pandas") from exc

        file_path = Path(path)
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, nrows=sample_rows)
        else:
            df = pd.read_parquet(file_path)
            if sample_rows:
                df = df.head(sample_rows)
        return df

    def _normalize_dtype(self, dtype: str) -> str:
        dtype = dtype.lower()
        if "int" in dtype:
            return "integer"
        if "float" in dtype:
            return "float"
        if "bool" in dtype:
            return "boolean"
        if "datetime" in dtype or "date" in dtype:
            return "datetime"
        if "object" in dtype or "string" in dtype:
            return "string"
        return dtype

    def _contract_from_dict(self, payload: Dict[str, Any]) -> DataContract:
        fields = [ContractField(**field) for field in payload.get("fields", [])]
        rules = []
        for rule in payload.get("quality_rules", []) or []:
            rules.append(DataQualityRule(**rule))
        sla_payload = payload.get("sla")
        sla = DataContractSLA(**sla_payload) if sla_payload else None
        return DataContract(
            name=payload.get("name", "Unnamed Contract"),
            version=payload.get("version", "1.0"),
            owner=payload.get("owner", ""),
            domain=payload.get("domain"),
            description=payload.get("description"),
            source=payload.get("source"),
            destination=payload.get("destination"),
            fields=fields,
            quality_rules=rules,
            sla=sla,
            tags=payload.get("tags", []) or [],
            metadata=payload.get("metadata", {}) or {},
        )

    def _severity(self, rule: DataQualityRule) -> str:
        return "error" if rule.severity == "critical" else "warning"
