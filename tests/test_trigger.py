"""
Unit tests for trigger.py.

Covers expected, edge, and failure cases for the evaluate_trigger function, including:
- 'yes' and 'no' inference results
- None as inference result (failure/edge case)

Uses pytest for parameterized and explicit tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

# Assuming trigger.py and inference_small.py (for SmallInferenceOutput) are importable.
# Adjust path if necessary, e.g., from ..trigger import evaluate_trigger
from trigger import evaluate_trigger

@pytest.mark.parametrize(
    "inference_result, expected_result",
    [
        ("yes", True),
        ("no", False),
        (None, False),
    ]
)
def test_evaluate_trigger_with_valid_inference_result(
    inference_result,
    expected_result
):
    """
    Tests evaluate_trigger with various valid string inference results.
    """
    trigger_desc = "a test trigger description"
    result = evaluate_trigger(inference_result, trigger_desc)
    assert result == expected_result

def test_evaluate_trigger_with_none_inference_result():
    """
    Tests evaluate_trigger when the inference_result is None (e.g., inference failed).
    """
    trigger_desc = "a test trigger description"
    result = evaluate_trigger(None, trigger_desc)
    assert result is False