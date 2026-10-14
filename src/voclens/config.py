"""설정 파일 로딩.

설정 폴더는 VOCLENS_CONFIG_DIR 환경변수로 바꿀 수 있고,
없으면 저장소 루트의 configs/를 쓴다 (editable 설치 기준).
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

def _find_repo_root() -> Path:
    """configs/가 있는 저장소 루트. editable 설치면 패키지 기준 두 단계 위,
    일반 설치(배포 환경)면 현재 작업 폴더를 쓴다."""
    package_based = Path(__file__).resolve().parents[2]
    if (package_based / "configs").is_dir():
        return package_based
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def config_dir() -> Path:
    return Path(os.environ.get("VOCLENS_CONFIG_DIR", REPO_ROOT / "configs"))


@lru_cache(maxsize=None)
def load_yaml(relative_path: str) -> dict[str, Any]:
    path = config_dir() / relative_path
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(override_path: str | Path | None = None) -> dict[str, Any]:
    settings = copy.deepcopy(load_yaml("default.yaml"))
    if override_path:
        with Path(override_path).open(encoding="utf-8") as f:
            settings = _deep_merge(settings, yaml.safe_load(f) or {})
    return settings
