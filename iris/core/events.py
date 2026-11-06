"""
Event bus and execution telemetry for Leocherd Iris Energy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_COMPLETED = "PIPELINE_COMPLETED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_PROGRESS = "STAGE_PROGRESS"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    STAGE_FAILED = "STAGE_FAILED"
    ASSET_EXPORTED = "ASSET_EXPORTED"


@dataclass
class PipelineEvent:
    event_type: EventType
    stage_name: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[PipelineEvent], None]


class EventBus:
    """Thread-safe event subscriber and publisher."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[EventHandler]] = {et: [] for et in EventType}
        self._global_subscribers: List[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        self._global_subscribers.append(handler)

    def emit(self, event: PipelineEvent) -> None:
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                pass
        for handler in self._global_subscribers:
            try:
                handler(event)
            except Exception:
                pass
