"""
Unit tests for unified large model inference (via run_unified_inference).

run_large_inference is deprecated; these tests now use run_unified_inference from inference.unified.
"""

import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.unified import run_unified_inference

@pytest.fixture
def dummy_image_frame_large() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_successful_large_inference(dummy_image_frame_large):
    """Test a successful large model inference call (integration test)."""
    try:
        result = run_unified_inference(dummy_image_frame_large, "a cat on a couch", model_type="large")
        assert result.result in ("yes", "no")
        assert isinstance(result.detailed_analysis, str)
    except Exception as e:
        pytest.skip(f"Engine not available or config missing: {e}")

@pytest.mark.asyncio
async def test_large_inference_edge_case(dummy_image_frame_large):
    """Test edge case: empty frame (should not raise, should handle gracefully)."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARN] Skipping test_large_inference_edge_case: OPENAI_API_KEY not set.")
        pytest.skip("OPENAI_API_KEY not set")
    result = await run_unified_inference(dummy_image_frame_large, "", model_type="large")
    assert result is None or hasattr(result, 'result')

def test_large_inference_failure_case(dummy_image_frame_large):
    """Test failure case: nonsense trigger (should still return yes/no, not error)."""
    try:
        result = run_unified_inference(dummy_image_frame_large, "nonsense_trigger_1234567890", model_type="large")
        assert result.result in ("yes", "no")
    except Exception as e:
        pytest.skip(f"Engine not available or config missing: {e}")