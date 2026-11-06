"""
Biological and procedural morphology subsystem.
"""

from typing import Any, Dict, List, Optional, Type

from iris.core.exceptions import MorphologyError
from iris.morphology.base import BaseMorphology, BoneDefinition, JointLimit
from iris.morphology.biped import BipedMorphology
from iris.morphology.custom import CustomMorphology
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology


class MorphologyRegistry:
    """Central factory for supported biological morphologies."""

    _PLUGINS: Dict[str, Type[BaseMorphology]] = {
        "human": HumanMorphology,
        "quadruped": QuadrupedMorphology,
        "biped": BipedMorphology,
        "serpentine": SerpentineMorphology,
        "tentacular": TentacularMorphology,
    }

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseMorphology:
        lower_name = name.lower()
        if lower_name == "custom":
            return CustomMorphology(kwargs.get("config", {}))

        plugin_cls = cls._PLUGINS.get(lower_name)
        if not plugin_cls:
            raise MorphologyError(
                f"Unknown morphology '{name}'. Available: {sorted(list(cls._PLUGINS.keys())) + ['custom']}"
            )
        return plugin_cls(**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        return sorted(list(cls._PLUGINS.keys())) + ["custom"]


__all__ = [
    "BaseMorphology",
    "BoneDefinition",
    "JointLimit",
    "HumanMorphology",
    "QuadrupedMorphology",
    "BipedMorphology",
    "SerpentineMorphology",
    "TentacularMorphology",
    "CustomMorphology",
    "MorphologyRegistry",
]
