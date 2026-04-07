"""
Serialization mixin for dataclasses across the governance agents.
"""

import json
from typing import Any, Dict


class SerializableMixin:
    """Mixin que provê serialização JSON para dataclasses que implementam to_dict().

    Usage::

        @dataclass
        class MyModel(SerializableMixin):
            name: str

            def to_dict(self) -> Dict[str, Any]:
                return {"name": self.name}

        obj = MyModel("example")
        obj.to_json()          # '{"name": "example"}'
        obj.to_json(indent=4)  # pretty-printed with 4 spaces
    """

    def to_json(self, indent: int = 2) -> str:
        """Serialize this object to a JSON string via to_dict()."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
