"""Local generation jobs shared by the HTTP and stdio servers."""

from __future__ import annotations

import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

from filelock import FileLock, Timeout

from synthdata import sdk
from synthdata.core.generator import SENTINEL
from synthdata.core.provenance import MANIFEST_NAME, read_manifest


class QueueFull(RuntimeError):
    """This server already has the maximum number of outstanding jobs."""


class RunNotReady(RuntimeError):
    """The requested run does not have a completed dataset."""


def validate(design: dict) -> dict:
    model = sdk.validate_design(design)
    return {"valid": True, "design": model.model_dump(mode="json", exclude_none=True)}


def preview_png(design: dict) -> bytes:
    with io.BytesIO() as output:
        sdk.preview(design).save(output, format="PNG")
        return output.getvalue()


class RunService:
    """A single render worker per server. Completed runs survive restarts."""

    def __init__(self, root: Path | None = None, *, capacity: int = 8):
        default = Path.home() / ".local" / "share" / "fovea-synthdata" / "runs"
        self.root = Path(root or os.environ.get("SYNTHDATA_OUTPUT_DIR", default)).expanduser().resolve()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthdata")
        self.slots = BoundedSemaphore(capacity)

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def submit(self, design: dict, writer: str | None = None) -> dict:
        model = sdk.validate_design(design)
        writers = sdk.describe_generator(model.generator)["writers"]
        if writer is not None and writer not in writers:
            raise ValueError(f"Unknown writer {writer!r}. Available writers: {', '.join(writers)}")
        if not self.slots.acquire(blocking=False):
            raise QueueFull("Generation queue is full. Retry once a run finishes.")
        run_id = uuid4().hex
        run = self.root / run_id
        owner = FileLock(run / "worker.lock", thread_local=False)
        try:
            run.mkdir(parents=True, mode=0o700)
            owner.acquire()
            self.pool.submit(self._render, run, model, writer, owner)
        except BaseException:
            owner.release()
            self.slots.release()
            raise
        return {"id": run_id, "status": "queued"}

    def _render(self, run, model, writer, owner):
        try:
            (run / "running").touch()
            sdk.generate(model, run / "dataset", writer=writer, workers=1)
        except Exception as exc:
            temporary = run / "error.tmp"
            temporary.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            temporary.replace(run / "error")
        finally:
            owner.release()
            self.slots.release()

    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise KeyError("Unknown run")
        run = self.root / run_id
        if run.is_symlink() or not run.is_dir():
            raise KeyError(f"Unknown run {run_id}")
        return run

    @staticmethod
    def _terminal(run: Path) -> dict | None:
        if (run / "error").is_file():
            return {"id": run.name, "status": "failed", "error": (run / "error").read_text(encoding="utf-8")}
        dataset = run / "dataset"
        if (dataset / MANIFEST_NAME).is_file():
            manifest = read_manifest(dataset)
            paths = sorted(path for path in manifest.files if path != SENTINEL)
            if (run / "checked").is_file():
                paths.extend(["check/report.json", "check/report.md"])
            return {"id": run.name, "status": "succeeded", "artifacts": [MANIFEST_NAME, *paths]}
        return None

    def status(self, run_id: str) -> dict:
        run = self._run_dir(run_id)
        result = self._terminal(run)
        if result is not None:
            return result
        # A lock also covers queued work. An unlocked unfinished job lost its
        # worker (e.g. on restart). Re-read once under the lock to avoid a race.
        try:
            with FileLock(run / "worker.lock", timeout=0):
                return self._terminal(run) or {
                    "id": run_id, "status": "failed", "error": "Generation was interrupted."
                }
        except Timeout:
            return {"id": run_id, "status": "running" if (run / "running").is_file() else "queued"}

    def _ready(self, run_id: str) -> tuple[Path, dict]:
        status = self.status(run_id)
        if status["status"] != "succeeded":
            raise RunNotReady(f"Run {run_id} is {status['status']}")
        return self.root / run_id, status

    def check(self, run_id: str) -> dict:
        run, _ = self._ready(run_id)
        dataset = run / "dataset"
        marker = run / "checked"
        if not marker.is_file():
            try:
                with FileLock(self.root / "check.lock", timeout=0):
                    if not marker.is_file():
                        sdk.check(dataset)
                        marker.touch()
            except Timeout as exc:
                raise QueueFull("Another dataset check is running. Retry once it finishes.") from exc
        return json.loads((dataset / "check" / "report.json").read_text(encoding="utf-8"))

    def artifact(self, run_id: str, path: str) -> Path:
        run, status = self._ready(run_id)
        if path not in status["artifacts"]:
            raise KeyError(f"Unknown artifact {path!r}")
        dataset = run / "dataset"
        file = (dataset / path).resolve()
        if not file.is_relative_to(dataset) or not file.is_file():
            raise KeyError(f"Artifact {path!r} is unavailable")
        return file
