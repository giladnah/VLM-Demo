"""Module for evaluating trigger conditions based on VLM inference results."""

from typing import Optional

def evaluate_trigger(
    inference_result: Optional[str],
    trigger_description: str
) -> bool:
    """
    Evaluates whether the trigger condition is met based on the inference result string.

    Args:
        inference_result (Optional[str]): The result from the VLM inference ('yes' or 'no').
        trigger_description (str): The textual description of the trigger (unused, for future logic).

    Returns:
        bool: True if the trigger is met (i.e., result is 'yes'), False otherwise.
    """
    return inference_result == "yes"

if __name__ == '__main__':
    print("--- Testing evaluate_trigger ---   ")

    # Test case 1: Detection is True
    result_true = "yes"
    desc1 = "a dog in the park"
    eval1 = evaluate_trigger(result_true, desc1)
    print(f"Trigger: '{desc1}', Inference: 'yes' -> Evaluates to: {eval1} (Expected: True)")
    assert eval1 is True

    # Test case 2: Detection is False
    result_false = "no"
    desc2 = "a dog in the park"
    eval2 = evaluate_trigger(result_false, desc2)
    print(f"Trigger: '{desc2}', Inference: 'no' -> Evaluates to: {eval2} (Expected: False)")
    assert eval2 is False

    # Test case 3: Inference result is None (e.g., inference failed)
    desc3 = "anything"
    eval3 = evaluate_trigger(None, desc3)
    print(f"Trigger: '{desc3}', Inference: None -> Evaluates to: {eval3} (Expected: False)")
    assert eval3 is False

    # Test case 4: Detection is True, empty labels
    result_true_no_labels = "yes"
    desc4 = "something specific"
    eval4 = evaluate_trigger(result_true_no_labels, desc4)
    print(f"Trigger: '{desc4}', Inference: 'yes' -> Evaluates to: {eval4} (Expected: True)")
    assert eval4 is True

    print("\n--- End of evaluate_trigger tests ---   ")