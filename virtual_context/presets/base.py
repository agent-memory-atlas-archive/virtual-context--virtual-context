"""Preset registry: register, lookup, and list presets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Preset:
    name: str
    description: str
    config_dict: dict
    template: str
    # What to do after ``virtual-context init``; empty uses the generic steps.
    next_steps: list[str] = field(default_factory=list)


_PRESETS: dict[str, Preset] = {}


def register_preset(preset: Preset) -> None:
    _PRESETS[preset.name] = preset


def get_preset(name: str) -> Preset | None:
    return _PRESETS.get(name)


def list_presets() -> list[Preset]:
    return list(_PRESETS.values())
