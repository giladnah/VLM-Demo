"""Unit tests for the trigger.py module."""

import pytest

# Assuming trigger.py and inference_small.py (for SmallInferenceOutput) are importable.
# Adjust path if necessary, e.g., from ..trigger import evaluate_trigger
from trigger import evaluate_trigger
from inference_small import SmallInferenceOutput # Used to create test data

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

# Example of a more specific test if logic were to change
# def test_evaluate_trigger_custom_logic_example():
#     """
#     Placeholder for a test if trigger logic becomes more complex
#     (e.g., involving trigger_description keywords and labels).
#     For now, it relies purely on inference_result.detection.
#     """
#     # Assuming future logic: if detection is false but specific keyword in labels matches trigger_description
#     # This test would need evaluate_trigger to be modified.
#     inference_input = SmallInferenceOutput(detection=False, labels=["important_keyword"])
#     trigger_desc_with_keyword = "This trigger has an important_keyword"
#
#     # current_result = evaluate_trigger(inference_input, trigger_desc_with_keyword)
#     # assert current_result is False # Based on current logic
#     pass # Keep as a reminder for potential future enhancements