"""HTTP access to the generator catalog, designs, and local dataset runs."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from synthdata import __version__, sdk
from synthdata.service import QueueFull, RunNotReady, RunService, preview_png, validate


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design: dict[str, Any]


class GenerationRequest(DesignRequest):
    writer: str | None = None


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "failed", "succeeded"]
    artifacts: list[str] | None = None
    error: str | None = None


def create_app(root: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = RunService(root)
        app.state.runs = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="Fovea Lab Synthetic Data", version=__version__, lifespan=lifespan)

    def error_handler(code: int):
        def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=code)
        return handle

    for error, code in ((KeyError, 404), (ValueError, 422), (ImportError, 503),
                        (RunNotReady, 409), (QueueFull, 429)):
        app.add_exception_handler(error, error_handler(code))

    @app.get("/generators")
    def generators() -> list[str]:
        """List the registered generator names."""
        return sdk.list_generators()

    @app.get("/generators/{name}")
    def describe(name: str) -> dict:
        """Return settings schema, available presets, writers and operations."""
        return sdk.describe_generator(name)

    @app.get("/generators/{name}/presets/{preset}")
    def preset_design(name: str, preset: str) -> dict:
        """Return an editable starter design."""
        return sdk.load_preset(f"{name}/{preset}")

    @app.post("/validate")
    def validate_design(payload: DesignRequest) -> dict:
        """Validate the design and return its settings with defaults applied."""
        return validate(payload.design)

    @app.post("/preview", response_class=Response, responses={200: {"content": {"image/png": {}}}})
    def preview_design(payload: DesignRequest) -> Response:
        """Render a PNG contact sheet."""
        return Response(preview_png(payload.design), media_type="image/png")

    @app.post("/runs", status_code=202, response_model=Job, response_model_exclude_none=True)
    def generate(payload: GenerationRequest, request: Request) -> dict:
        """Queue a dataset and return its run ID."""
        return request.app.state.runs.submit(payload.design, payload.writer)

    @app.get("/runs/{run_id}", response_model=Job, response_model_exclude_none=True)
    def run_status(run_id: str, request: Request) -> dict:
        """Return the generation status and the available files on success."""
        return request.app.state.runs.status(run_id)

    @app.post("/runs/{run_id}/check")
    def check_dataset(run_id: str, request: Request) -> dict:
        """Check a completed dataset and return its recoverability report."""
        return request.app.state.runs.check(run_id)

    @app.get("/runs/{run_id}/artifacts/{path:path}", response_class=FileResponse)
    def artifact(run_id: str, path: str, request: Request) -> FileResponse:
        """Download one of the files listed in the run's status."""
        file = request.app.state.runs.artifact(run_id, path)
        return FileResponse(file, filename=file.name)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
