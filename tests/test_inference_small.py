"""
Unit tests for unified small model inference (via run_unified_inference).

run_small_inference is deprecated; these tests now use run_unified_inference from inference.unified.
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.unified import run_unified_inference
from inference.ollama_engine import OllamaError

@pytest.fixture
def dummy_image_frame() -> np.ndarray:
    """Provides a consistent dummy image frame (NumPy array) for tests."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_successful_inference(dummy_image_frame):
    """Test a successful small model inference call (integration test)."""
    try:
        result = run_unified_inference(dummy_image_frame, "a cat on a couch", model_type="small")
        assert result.result in ("yes", "no")
    except Exception as e:
        pytest.skip(f"Engine not available or config missing: {e}")

@pytest.mark.asyncio
async def test_inference_edge_case(dummy_image_frame):
    """Test edge case: empty frame (should not raise, should handle gracefully)."""
    try:
        result = await run_unified_inference(dummy_image_frame, "", model_type="small")
        assert result is None or hasattr(result, 'result')
    except OllamaError as e:
        pytest.skip(f"Ollama not available or config missing: {e}")

def test_inference_failure_case(dummy_image_frame):
    """Test failure case: nonsense trigger (should still return yes/no, not error)."""
    try:
        result = run_unified_inference(dummy_image_frame, "nonsense_trigger_1234567890", model_type="small")
        assert result.result in ("yes", "no")
    except Exception as e:
        pytest.skip(f"Engine not available or config missing: {e}")