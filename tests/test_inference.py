"""
Unit tests for the unified inference system.
- Covers mocked Ollama engine (initialization, process_frame, error handling).
- Tests unified inference logic and error propagation.
- Includes orchestrator error logging test (mocked).
- Does not require real backend services; all network calls are mocked.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
import aiohttp
from inference import (
    UnifiedVLMInference,
    TriggerConfig,
    TriggerType,
    OllamaConfig,
    OllamaEngine,
    OllamaError
)
import time

@pytest.fixture
def mock_response():
    """Create a mock response for Ollama API."""
    mock = MagicMock()
    mock.status = 200
    mock.json = AsyncMock(return_value={
        "response": '{"result": "yes", "detailed_analysis": "Test analysis"}'
    })
    return mock

@pytest.fixture
def mock_session(mock_response):
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.close = AsyncMock()  # Make close() an async mock
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = mock_response
    session.post.return_value = context_manager
    return session

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def test_trigger():
    """Create a test trigger configuration."""
    return TriggerConfig(
        type=TriggerType.OBJECT_DETECTION,
        description="test object"
    )

@pytest.mark.asyncio
async def test_ollama_engine_initialization():
    """Test OllamaEngine initialization."""
    config = OllamaConfig(model_name="test_model")
    engine = OllamaEngine(config)

    assert not engine.is_initialized
    await engine.initialize()
    assert engine.is_initialized
    assert engine.name == "ollama_test_model"
    await engine.shutdown()
    assert not engine.is_initialized

@pytest.mark.asyncio
async def test_ollama_engine_process_frame(monkeypatch, test_image, test_trigger, mock_session):
    """Test OllamaEngine frame processing."""
    config = OllamaConfig(model_name="test_model")
    engine = OllamaEngine(config)

    # Mock aiohttp.ClientSession
    monkeypatch.setattr(aiohttp, "ClientSession", MagicMock(return_value=mock_session))

    await engine.initialize()
    result = await engine.process_frame(test_image, test_trigger)

    assert result.result == "yes"
    assert result.detailed_analysis == "Test analysis"
    assert result.engine_name == "ollama_test_model"

    await engine.shutdown()

@pytest.mark.asyncio
async def test_unified_inference_lifecycle():
    """Test UnifiedVLMInference lifecycle management."""
    unified = UnifiedVLMInference()

    # Test initialization
    await unified.initialize_engine("ollama_small")
    with pytest.raises(ValueError):
        await unified.initialize_engine("ollama_small")  # Already initialized

    # Test shutdown
    await unified.shutdown_engine("ollama_small")
    with pytest.raises(ValueError):
        await unified.shutdown_engine("ollama_small")  # Already shut down

    # Test invalid engine type
    with pytest.raises(ValueError):
        await unified.initialize_engine("invalid_engine")

@pytest.mark.asyncio
async def test_unified_inference_process_frame(monkeypatch, test_image, test_trigger, mock_session):
    """Test UnifiedVLMInference frame processing."""
    unified = UnifiedVLMInference()

    # Mock aiohttp.ClientSession
    monkeypatch.setattr(aiohttp, "ClientSession", MagicMock(return_value=mock_session))

    # Initialize engine and process frame
    await unified.initialize_engine("ollama_small")
    result = await unified.process_frame("ollama_small", test_image, test_trigger)

    assert result.result == "yes"
    assert result.detailed_analysis == "Test analysis"
    assert "ollama" in result.engine_name

    # Test with uninitialized engine
    with pytest.raises(ValueError):
        await unified.process_frame("invalid_engine", test_image, test_trigger)

    await unified.shutdown_all()

@pytest.mark.asyncio
async def test_ollama_engine_error_handling(monkeypatch, test_image, test_trigger):
    """Test OllamaEngine error handling."""
    config = OllamaConfig(
        model_name="test_model",
        retry_count=2,
        retry_cooldown=1  # Minimum valid value
    )
    engine = OllamaEngine(config)

    # Mock session that always fails
    failed_session = MagicMock()
    failed_session.close = AsyncMock()  # Make close() an async mock
    context_manager = AsyncMock()
    context_manager.__aenter__.side_effect = aiohttp.ClientError("Test error")
    failed_session.post.return_value = context_manager

    monkeypatch.setattr(aiohttp, "ClientSession", MagicMock(return_value=failed_session))

    await engine.initialize()
    with pytest.raises(OllamaError):
        await engine.process_frame(test_image, test_trigger)

    await engine.shutdown()

def test_orchestrator_engine_error_logging(monkeypatch):
    """
    Test that orchestrator logs and returns engine errors in the result payload.
    Simulates an engine error and checks that the error is present in the output queue payload.
    """
    import multiprocessing
    import queue
    from inference.ollama_engine import OllamaError
    from orchestrator import inference_worker_process

    # Simulate input and output queues
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    config = {"batch_mode_enabled": False, "batch_size": 1}

    # Dummy frame and trigger
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_trigger = "test trigger"
    timestamp = time.time()

    # Patch run_unified_inference to always raise OllamaError
    async def fake_run_unified_inference(*args, **kwargs):
        raise OllamaError("Simulated engine failure for test.")
    monkeypatch.setattr("inference.unified.run_unified_inference", fake_run_unified_inference)

    # Put a single item in the input queue
    input_queue.put((dummy_frame, dummy_trigger, timestamp))
    # Set stop_event after one iteration
    def stop_after_one():
        time.sleep(0.5)
        stop_event.set()
    import threading
    threading.Thread(target=stop_after_one, daemon=True).start()

    # Run the worker process (in the same process for test)
    inference_worker_process(input_queue, output_queue, stop_event, config)

    # Check output queue for error
    result = output_queue.get(timeout=2)
    print("[DEBUG] Orchestrator result payload:", result)
    assert "error" in result, "Result payload missing 'error' key"
    assert result["error"] is not None, "Result 'error' should not be None when engine fails"
    assert "Inference failed" in result["error"], f"Unexpected error string: {result['error']}"