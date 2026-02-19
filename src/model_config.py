"""モデル設定の解決を担当"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class ModelSettings:
    supports_system: bool = True
    chat_format: str | None = None


def _build_settings(data: dict[str, Any], base: ModelSettings) -> ModelSettings:
    return ModelSettings(
        supports_system=data.get("supports_system", base.supports_system),
        chat_format=data.get("chat_format", base.chat_format),
    )


def resolve_model_settings(
    model_path: str,
    config_path: str = "models.toml",
) -> ModelSettings:
    config_file = Path(config_path)
    if not config_file.exists():
        return ModelSettings()

    data = tomllib.loads(config_file.read_text(encoding="utf-8"))
    defaults = _build_settings(data.get("defaults", {}), ModelSettings())

    model_name = Path(model_path).name
    model_data = data.get("models", {}).get(model_name, {})
    return _build_settings(model_data, defaults)
