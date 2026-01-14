"""
Model Catalog Provider

Connector for loading model cards (metadata) to index into discovery.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class ModelCatalogProvider:
    """Load model metadata from JSON files or directories."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load_models(self) -> List[Dict[str, Any]]:
        """
        Load model cards from a JSON file or directory.

        Expected JSON format: list of model metadata dicts.
        """
        if self.path.is_dir():
            models: List[Dict[str, Any]] = []
            for json_file in sorted(self.path.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    models.extend(data)
                elif isinstance(data, dict):
                    models.append(data)
            return models

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]

        raise ValueError("Invalid model catalog format. Expected list or dict.")
