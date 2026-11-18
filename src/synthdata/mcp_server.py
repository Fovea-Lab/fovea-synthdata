"""Model Context Protocol tools for local synthetic datasets, served over stdio."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from synthdata import sdk
from synthdata.service import RunService, preview_png, validate


def create_server(service: RunService) -> FastMCP:
    server = FastMCP("Fovea Lab Synthetic Data")

    @server.tool()
    def list_generators() -> list[str]:
        """List registered synthetic dataset generators."""
        return sdk.list_generators()

    @server.tool()
    def describe_generator(name: str) -> dict[str, Any]:
        """Get the settings schema, presets, writers and operations for a generator."""
        return sdk.describe_generator(name)

    @server.tool()
    def get_preset(generator: str, preset: str) -> dict[str, Any]:
        """Load an editable design, for example generator particle_plate and preset small."""
        return sdk.load_preset(f"{generator}/{preset}")

    @server.tool()
    def validate_design(design: dict[str, Any]) -> dict[str, Any]:
        """Validate a design and return its settings with defaults applied."""
        return validate(design)

    @server.tool(structured_output=False)
    async def preview_design(design: dict[str, Any]) -> Image:
        """Preview a design as a PNG contact sheet."""
        data = await asyncio.to_thread(preview_png, design)
        return Image(data=data, format="png")

    @server.tool()
    def generate_dataset(design: dict[str, Any], writer: str | None = None) -> dict[str, Any]:
        """Queue a generation run. Poll get_run with the returned ID."""
        return service.submit(design, writer)

    @server.tool()
    def get_run(run_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        """Get status and a page of artifact paths. Pass next_offset to see the next page."""
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 500")
        result = service.status(run_id)
        if result["status"] == "succeeded":
            paths = result["artifacts"]
            end = offset + limit
            result.update(artifacts=paths[offset:end], artifact_count=len(paths),
                          next_offset=end if end < len(paths) else None)
        return result

    @server.tool()
    async def check_dataset(run_id: str) -> dict[str, Any]:
        """Check the completed dataset against its ground truth and return the report."""
        return await asyncio.to_thread(service.check, run_id)

    @server.tool()
    def get_artifact(run_id: str, path: str, offset: int = 0, limit: int = 65536) -> dict[str, Any]:
        """Read a listed artifact, returning its local file URI and a base64 chunk.

        The next_offset is null at the end of the file. Maximum chunk size is 1 MiB.
        """
        if offset < 0 or not 1 <= limit <= 1048576:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 1048576")
        file = service.artifact(run_id, path)
        size = file.stat().st_size
        if offset > size:
            raise ValueError(f"offset exceeds artifact size {size}")
        with file.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(limit)
        end = offset + len(data)
        return {"path": path, "uri": file.as_uri(), "size": size, "offset": offset,
                "next_offset": end if end < size else None,
                "data_base64": base64.b64encode(data).decode("ascii")}

    return server


def main() -> None:
    service = RunService()
    try:
        create_server(service).run(transport="stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
