import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import app
from io import BytesIO
from PIL import Image

client = TestClient(app.app)

def make_jpeg_bytes():
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

# --- Root and Health Endpoints ---
def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()

def test_health(monkeypatch):
    monkeypatch.setattr(app, "SMALL_MODEL_NAME", "dummy-small")
    monkeypatch.setattr(app, "LARGE_MODEL_NAME", "dummy-large")
    monkeypatch.setattr(app, "get_config", lambda: {})
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0, stdout='{"name": "dummy-small"}\n{"name": "dummy-large"}', stderr=''))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service_status"] == "ok"

# --- Orchestration Endpoints ---
@patch("app.orchestrator")
def test_start_and_trigger(mock_orch):
    mock_orch._config = MagicMock()
    data = {
        "source": "test.mp4",
        "trigger": "a test trigger"
    }
    resp = client.post("/start", json=data)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Now test trigger update
    mock_orch._config = MagicMock()
    resp = client.put("/trigger", json={"trigger": "new trigger"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

@patch("app.orchestrator")
def test_trigger_no_active_orchestration(mock_orch):
    mock_orch._config = None
    resp = client.put("/trigger", json={"trigger": "new trigger"})
    assert resp.status_code == 404

# --- Inference Endpoints ---
@patch("app.run_small_inference_direct", return_value="yes")
def test_infer_small_success(mock_inf):
    img_bytes = make_jpeg_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"trigger": "a test trigger"}
    resp = client.post("/infer/small", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json() == "yes"

@patch("app.run_small_inference_direct", return_value=None)
def test_infer_small_failure(mock_inf):
    img_bytes = make_jpeg_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"trigger": "a test trigger"}
    resp = client.post("/infer/small", data=data, files=files)
    assert resp.status_code == 500

@patch("app.run_large_inference_direct", return_value=MagicMock(result="yes", detailed_analysis="details"))
def test_infer_large_success(mock_inf):
    img_bytes = make_jpeg_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    resp = client.post("/infer/large", files=files)
    assert resp.status_code == 200
    assert resp.json()["result"] == "yes"

@patch("app.run_large_inference_direct", return_value=None)
def test_infer_large_failure(mock_inf):
    img_bytes = make_jpeg_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    resp = client.post("/infer/large", files=files)
    assert resp.status_code == 500

# --- Debug Endpoints ---
@patch("app.orchestrator")
def test_debug_endpoints(mock_orch):
    mock_orch.get_latest_small_inference_frame.return_value = None
    resp = client.get("/latest-small-inference-frame")
    assert resp.status_code == 404
    mock_orch.get_results_status.return_value = {"small": False, "large": False}
    resp = client.get("/results_status")
    assert resp.status_code == 200
    mock_orch.get_latest_small_inference_result.return_value = {"result": "yes"}
    resp = client.get("/latest-small-inference-result")
    assert resp.status_code == 200
    mock_orch.get_latest_large_inference_result.return_value = {"result": "yes"}
    resp = client.get("/latest-large-inference-result")
    assert resp.status_code == 200