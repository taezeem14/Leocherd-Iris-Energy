"""
Extensible component registry for models, morphologies, exporters, and algorithms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry pattern supporting decorators and dynamic lookup."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: Dict[str, Type[T] | Callable[..., T]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, key: str, **metadata: Any) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls_or_func: Type[T]) -> Type[T]:
            lower_key = key.lower()
            if lower_key in self._entries:
                pass  # Allow overwrite or warn
            self._entries[lower_key] = cls_or_func
            self._metadata[lower_key] = metadata
            return cls_or_func

        return decorator

    def get(self, key: str) -> Optional[Type[T] | Callable[..., T]]:
        return self._entries.get(key.lower())

    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self._metadata.get(key.lower(), {})

    def list_keys(self) -> List[str]:
        return sorted(list(self._entries.keys()))

    def contains(self, key: str) -> bool:
        return key.lower() in self._entries
