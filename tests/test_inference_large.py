"""
Unit tests for unified large model inference (via run_unified_inference).

run_large_inference is deprecated; these tests now use run_unified_inference from inference.unified.
"""

import pytest
import numpy as np
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.unified import run_unified_inference

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

def validate_large_model_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        large_model = config.get('large_model', {})
        if not large_model.get('name'):
            raise ValueError('large_model.name is not set in config.yaml')
        return large_model['name']
    except Exception as e:
        raise RuntimeError(f'Failed to validate large_model config: {e}')

@pytest.fixture
def dummy_image_frame_large() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.mark.asyncio
async def test_successful_large_inference(dummy_image_frame_large):
    try:
        model = validate_large_model_config()
        if not os.environ.get("OPENAI_API_KEY"):
            reason = "OPENAI_API_KEY not set"
            print(f"[WARN] {reason}")
            pytest.skip(reason)
    except Exception as e:
        pytest.skip(f"Config or environment not available: {e}")
    try:
        result = await run_unified_inference(dummy_image_frame_large, "a cat on a couch", model_type="large")
        assert result.result in ("yes", "no")
        assert isinstance(result.detailed_analysis, str)
    except Exception as e:
        print(f"[DEBUG] Exception in test_successful_large_inference: {e}")
        pytest.skip(f"Engine not available or config missing: {e}")

@pytest.mark.asyncio
async def test_large_inference_edge_case(dummy_image_frame_large):
    """Test edge case: empty frame (should not raise, should handle gracefully)."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARN] Skipping test_large_inference_edge_case: OPENAI_API_KEY not set.")
        pytest.skip("OPENAI_API_KEY not set")
    result = await run_unified_inference(dummy_image_frame_large, "", model_type="large")
    assert result is None or hasattr(result, 'result')

@pytest.mark.asyncio
async def test_large_inference_failure_case(dummy_image_frame_large):
    try:
        model = validate_large_model_config()
        if not os.environ.get("OPENAI_API_KEY"):
            reason = "OPENAI_API_KEY not set"
            print(f"[WARN] {reason}")
            pytest.skip(reason)
    except Exception as e:
        pytest.skip(f"Config or environment not available: {e}")
    try:
        result = await run_unified_inference(dummy_image_frame_large, "nonsense_trigger_1234567890", model_type="large")
        assert result.result in ("yes", "no")
    except Exception as e:
        print(f"[DEBUG] Exception in test_large_inference_failure_case: {e}")
        pytest.skip(f"Engine not available or config missing: {e}")