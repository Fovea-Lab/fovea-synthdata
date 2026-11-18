"""Run directly with a wheel installation including particle_plate, cell_plate, api and mcp."""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import httpx
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from synthdata import sdk


def executable(name):
    path = shutil.which(name, path=sysconfig.get_path("scripts"))
    if path is None:
        raise RuntimeError(f"Missing installed entry point {name}")
    return path


def cli(*args):
    subprocess.run([executable("synthdata"), *args], check=True, timeout=90)


def edit_design(design):
    """Keep short, nonempty fields with the same distributions as the public presets."""
    design["plate"]["sites_per_well"] = 1
    if design["generator"] == "particle_plate":
        design["plate"]["format"] = {"rows": 2, "cols": 4}
        design["conditions"]["large"]["diameter"]["mean"] = 24
        design["layout"] = {"kind": "by_row", "order": ["control", "large"]}
        design["expectations"] = {"reference": "control", "differs": {"large": ["diameter"]}}
    else:
        design["plate"]["format"] = {"rows": 2, "cols": 2}
        design["compounds"] = {"DMSO": design["compounds"]["DMSO"]}
        design["layout"] = {"fixed": {"A01": "DMSO", "A02": "DMSO", "B01": "DMSO"}, "blanks": 1}
        design.pop("qc_defects")
        design.pop("expectations")  # the cell count and label checks apply without a compound contrast


@contextmanager
def serve_http(root):
    # Port zero lets the operating system pick an available port while binding it.
    log_path = root / "api.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [executable("synthdata-api"), "--port", "0"], stdout=log, stderr=log,
            env={**os.environ, "SYNTHDATA_OUTPUT_DIR": str(root / "http")},
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process.poll() is None:
                address = re.search(r"Uvicorn running on (http://127\.0\.0\.1:\d+)", log_path.read_text())
                if address:
                    yield address[1]
                    return
                time.sleep(0.1)
            raise RuntimeError(f"HTTP server failed to start. {log_path.read_text()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


async def mcp_call(session, name, **arguments):
    reply = await session.call_tool(name, arguments)
    assert not reply.isError, (name, reply.content)
    return reply.structuredContent


async def mcp_download(session, run_id, path):
    data = bytearray()
    offset = 0
    while offset is not None:
        part = await mcp_call(session, "get_artifact", run_id=run_id, path=path, offset=offset)
        assert part["path"] == path and part["offset"] == len(data)
        data.extend(base64.b64decode(part["data_base64"], validate=True))
        offset = part["next_offset"]
        assert offset is None or offset == len(data) > part["offset"]
    assert len(data) == part["size"]
    return bytes(data)


async def wait_for_run(get_status):
    async with asyncio.timeout(90):
        while True:
            status = await get_status()
            if status["status"] == "succeeded":
                return status
            assert status["status"] in {"queued", "running"}, status
            await asyncio.sleep(0.1)


async def exercise(name, root, http, session):
    key = f"{name}/small"
    design_file = root / f"{name}.yaml"
    cli("preset", "export", key, "-o", design_file)
    design = yaml.safe_load(design_file.read_text())
    assert design == sdk.load_preset(key)
    assert design == (await http.get(f"/generators/{name}/presets/small")).raise_for_status().json()
    assert design == await mcp_call(session, "get_preset", generator=name, preset="small")
    edit_design(design)
    design_file.write_text(yaml.safe_dump(design, sort_keys=False))

    sdk_out, cli_out = root / f"{name}-sdk", root / f"{name}-cli"
    expected = sdk.generate(design, sdk_out, writer="tiff")
    assert sdk.check(sdk_out).passed
    cli("generate", design_file, "-o", cli_out, "--writer", "tiff", "--workers", "1")
    cli("check", cli_out)
    assert json.loads((cli_out / "check/report.json").read_text())["passed"]

    payload = {"design": design, "writer": "tiff"}
    reply = await http.post("/runs", json=payload)
    assert reply.status_code == 202, reply.text
    http_path = f"/runs/{reply.json()['id']}"
    mcp_id = (await mcp_call(session, "generate_dataset", **payload))["id"]

    async def http_status():
        return (await http.get(http_path)).raise_for_status().json()

    http_run, mcp_run = await asyncio.gather(
        wait_for_run(http_status), wait_for_run(lambda: mcp_call(session, "get_run", run_id=mcp_id))
    )
    assert (await http.post(f"{http_path}/check")).raise_for_status().json()["passed"]
    assert (await mcp_call(session, "check_dataset", run_id=mcp_id))["passed"]

    async def http_download(path):
        return (await http.get(f"{http_path}/artifacts/{path}")).raise_for_status().content

    assert "manifest.json" in http_run["artifacts"] and "manifest.json" in mcp_run["artifacts"]
    manifests = [json.loads((out / "manifest.json").read_text()) for out in (sdk_out, cli_out)]
    manifests += [json.loads(await http_download("manifest.json")),
                  json.loads(await mcp_download(session, mcp_id, "manifest.json"))]
    for manifest in manifests:
        assert (manifest["generator"], manifest["writer"], manifest["seed"]) == (name, "tiff", design["seed"])
        assert manifest["synthdata_version"] == version("fovea-synthdata")
        assert manifest["design_sha256"] == expected.design_sha256
        assert manifest["extra"] == expected.extra and manifest["files"] == expected.files
    tiffs = [path for path in expected.files if path.endswith(".tif")]
    assert any(path.startswith("images/") for path in tiffs)
    assert any(path.startswith(("masks/", "truth/labels/")) for path in tiffs)
    for path in [*tiffs, "design.yaml", "truth/images.parquet"]:
        assert path in http_run["artifacts"] and path in mcp_run["artifacts"]
        downloads = await asyncio.gather(http_download(path), mcp_download(session, mcp_id, path))
        for data in [(sdk_out / path).read_bytes(), (cli_out / path).read_bytes(), *downloads]:
            assert hashlib.sha256(data).hexdigest() == expected.files[path], (name, path)
    print(f"{name}: verified {len(tiffs)} TIFFs and four recoverability checks", flush=True)


async def main():
    with tempfile.TemporaryDirectory(prefix="synthdata-release-") as directory:
        root = Path(directory)
        server = StdioServerParameters(command=executable("synthdata-mcp"),
                                       env={"SYNTHDATA_OUTPUT_DIR": str(root / "mcp")})
        with serve_http(root) as url:
            async with httpx.AsyncClient(base_url=url, timeout=90, trust_env=False) as http:
                async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
                    async with asyncio.timeout(300):
                        await session.initialize()
                        for name in ("particle_plate", "cell_plate"):
                            await exercise(name, root, http, session)


if __name__ == "__main__":
    asyncio.run(main())
