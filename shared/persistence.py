"""
JSON persistence mixin for classes that store data in a local directory.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonStorageMixin:
    """Mixin para persistência JSON em disco.

    Provê inicialização de diretório e helpers de leitura/escrita de arquivos
    JSON, eliminando a repetição desse padrão nas classes que mantêm estado
    local em disco.

    Usage::

        class MyStore(JsonStorageMixin):
            def __init__(self, persist_dir: str = "./my_data"):
                self._init_persist_dir(persist_dir)

            def save(self, key: str, data: dict) -> None:
                self._save_json(self.persist_dir / f"{key}.json", data)

            def load(self, key: str) -> Optional[dict]:
                return self._load_json(self.persist_dir / f"{key}.json")
    """

    def _init_persist_dir(self, persist_dir: str) -> None:
        """Inicializa self.persist_dir criando o diretório se necessário."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Escreve *data* como JSON formatado em *path*."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Lê e retorna JSON de *path*, ou None se o arquivo não existir."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _list_json_files(self, directory: Path) -> List[Path]:
        """Retorna todos os arquivos *.json em *directory*, ordenados."""
        return sorted(directory.glob("*.json"))
