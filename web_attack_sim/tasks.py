import copy
import json
from pathlib import Path
from typing import Any


def load_task_config(task: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, dict):
        return copy.deepcopy(task)

    path = Path(task)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def task_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tasks"


def bundled_task_paths() -> list[Path]:
    return sorted(task_dir().glob("*.json"))

