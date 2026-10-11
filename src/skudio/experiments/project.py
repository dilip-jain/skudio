"""Local project directory: on-disk layout for datasets, IR, runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ProjectMeta(BaseModel):
    """On-disk metadata written to `project.json`."""
    
    model_config = ConfigDict(extra="forbid")

    name: str
    project_version: Literal[1] = 1


class Project:
    """A local project directory."""

    def __init__(self, root: str | Path) -> None:
        """Bind to `root`; does not touch the filesystem until `create` or `open`."""
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, root: str | Path, *, name: str) -> Project:
        """Create the project directory layout under `root` and return the handle."""
        p = cls(root)
        p.root.mkdir(parents=True, exist_ok=True)
        p.datasets_dir.mkdir(exist_ok=True)
        p.experiments_dir.mkdir(exist_ok=True)
        p._meta_path.write_text(ProjectMeta(name=name).model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def open(cls, root: str | Path) -> Project:
        """Open an existing project directory; raises if `project.json` is missing."""
        p = cls(root)
        if not p._meta_path.exists():
            raise FileNotFoundError(f"not a project directory: {p.root}")
        return p

    @property
    def datasets_dir(self) -> Path:
        """Return the `datasets/` subdirectory under the project root."""
        return self.root / "datasets"

    @property
    def experiments_dir(self) -> Path:
        """Return the `experiments/` subdirectory under the project root."""
        return self.root / "experiments"

    @property
    def _meta_path(self) -> Path:
        return self.root / "project.json"

    def meta(self) -> ProjectMeta:
        """Read and return `project.json` as a `ProjectMeta`."""
        return ProjectMeta.model_validate_json(self._meta_path.read_text(encoding="utf-8"))

    def run_dir(self, run_id: str) -> Path:
        """Return the on-disk directory for a run (creates it lazily elsewhere)."""
        return self._safe(self.experiments_dir / run_id)

    def dataset_dir(self, dataset_id: str) -> Path:
        """Return the on-disk directory for a dataset id."""
        return self._safe(self.datasets_dir / dataset_id)

    def _safe(self, path: Path) -> Path:
        """Reject paths outside the project root."""
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc
        return resolved


def load_json(path: Path) -> object:
    """Read and decode a JSON file at `path`."""
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: object) -> None:
    """Serialise `obj` to JSON at `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
