import threading
import time
from uuid import uuid4

import pytest

pytest.importorskip("filelock")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from filelock import FileLock

from synthdata import sdk
from synthdata.http_api import create_app
from synthdata.service import QueueFull, RunNotReady, RunService


def test_worker_failure_and_queue(tmp_path, monkeypatch):
    design = sdk.load_preset("particle_plate/small")
    entered = threading.Event()
    resume = threading.Event()

    def failed_render(*args, **kwargs):
        entered.set()
        assert resume.wait(timeout=15)
        raise RuntimeError("render failed")

    monkeypatch.setattr(sdk, "generate", failed_render)
    service = RunService(tmp_path, capacity=1)
    try:
        job = service.submit(design)
        assert entered.wait(timeout=15)
        assert service.status(job["id"])["status"] == "running"
        with pytest.raises(RunNotReady):
            service.artifact(job["id"], "design.yaml")
        with pytest.raises(QueueFull):
            service.submit(design)
    finally:
        resume.set()
        service.close()
    assert service.status(job["id"]) == {
        "id": job["id"], "status": "failed", "error": "RuntimeError: render failed"
    }


def test_interrupted_run_reports_failure_after_restart(tmp_path):
    run = tmp_path / uuid4().hex
    run.mkdir()
    (run / "running").touch()
    service = RunService(tmp_path)
    try:
        assert service.status(run.name)["status"] == "failed"
    finally:
        service.close()


def test_http_job_status_and_artifacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert "cell_plate" in client.get("/generators").json()
        assert "settings_schema" in client.get("/generators/particle_plate").json()
        assert client.get("/generators/unknown").status_code == 404
        assert client.post("/validate", json={"design": {}}).status_code == 422
        design = client.get("/generators/particle_plate/presets/small").json()
        design["plate"].update(format={"rows": 1, "cols": 2}, sites_per_well=1)
        design["plate"]["image"].update(width=128, height=128)
        design["conditions"] = {"control": {"role": "control"}}
        design["layout"] = {"kind": "by_row", "order": ["control"]}
        del design["expectations"]
        payload = {"design": design}
        assert client.post("/validate", json=payload).json()["valid"]
        assert client.post("/preview", json=payload).content.startswith(b"\x89PNG")
        assert client.post("/runs", json={**payload, "writer": "invalid"}).status_code == 422
        response = client.post("/runs", json={**payload, "writer": "tiff"})
        assert response.status_code == 202
        url = "/runs/" + response.json()["id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.get(url).json()
            if status["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert status["status"] == "succeeded", status
        assert "manifest.json" in status["artifacts"]
        assert client.get(url + "/artifacts/manifest.json").json()["writer"] == "tiff"
        assert client.get(url + "/artifacts/.synthdata").status_code == 404
        assert client.get(url + "/artifacts/check/report.json").status_code == 404
        with FileLock(tmp_path / "check.lock"):
            assert client.post(url + "/check").status_code == 429
        report = client.post(url + "/check").json()
        assert isinstance(report["passed"], bool)
        assert client.get(url + "/artifacts/check/report.json").json() == report
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get(url).json()["status"] == "succeeded"
        assert restarted.post(url + "/check").json() == report
