"""Module for running inference using a large Vision Language Model via Ollama CLI."""

import cv2
import subprocess
import json
import tempfile
import os
from typing import List, Optional, Union, Dict, Any, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Type alias for an image frame, expected to be a NumPy array from OpenCV.
ImageFrame: TypeAlias = np.ndarray

# Consistent with 'ollama pull qwen2.5vl:7b' in TASKS.md
MODEL_NAME_DEFAULT_LARGE = "qwen2.5vl:7b"

class LargeInferenceOutput(BaseModel):
    """Data model for the structured output of the large VLM inference."""
    caption: str
    tags: List[str] = Field(default_factory=list)
    detailed_analysis: str # Could be Dict[str, Any] if a more structured analysis is expected

class OllamaError(Exception):
    """Custom exception for Ollama related errors encountered during inference."""
    pass # Re-using the same exception type as in inference_small for consistency

def run_large_inference(
    frames: Union[ImageFrame, List[ImageFrame]],
    model_name: str = MODEL_NAME_DEFAULT_LARGE,
    timeout_seconds: int = 90 # Larger models might take more time
) -> Optional[LargeInferenceOutput]:
    """
    Runs inference using a large VLM (via Ollama CLI) on a given image frame or list of frames.

    If a list of frames is provided, the most recent frame (the last one in the list)
    is used for inference. The function saves the selected frame to a temporary image file,
    constructs a prompt for the Ollama model, executes the Ollama CLI, parses the JSON output,
    and validates it against the LargeInferenceOutput model.

    Args:
        frames (Union[ImageFrame, List[ImageFrame]]):
            A single image frame (NumPy array) or a list of image frames.
            If a list, the last frame is processed.
        model_name (str, optional):
            The name of the Ollama large model to use.
            Defaults to MODEL_NAME_DEFAULT_LARGE.
        timeout_seconds (int, optional):
            Timeout for the Ollama CLI command. Defaults to 90 seconds.

    Returns:
        Optional[LargeInferenceOutput]: A LargeInferenceOutput object if successful,
                                      None if any error occurs or if input is invalid.
    """
    temp_image_path: Optional[str] = None
    image_to_process: Optional[ImageFrame] = None

    if isinstance(frames, list):
        if not frames:
            print("Error: Received an empty list of frames for large inference.")
            return None
        image_to_process = frames[-1] # Process the last frame (most recent)
    elif isinstance(frames, np.ndarray):
        image_to_process = frames
    else:
        print(f"Error: Invalid input type for frames. Expected np.ndarray or list, got {type(frames)}.")
        return None

    if image_to_process is None: # Should not happen if logic above is correct
        print("Error: No image frame determined for processing in large inference.")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file: # Using PNG for potentially better quality
            temp_image_path = tmp_file.name

        if not cv2.imwrite(temp_image_path, image_to_process):
            raise OllamaError(f"Failed to write temporary image to {temp_image_path} for large inference.")

        prompt = (
            f"Provide a detailed analysis of this image: <img>{temp_image_path}</img>. "
            "Respond ONLY with a JSON object containing these keys: "
            "'caption' (a descriptive sentence about the image), "
            "'tags' (a list of relevant keywords or detected objects/attributes, max 10 tags), and "
            "'detailed_analysis' (a concise textual description or notable insights about the scene, 2-3 sentences)."
        )

        command = ["ollama", "run", model_name, prompt, "--format", "json"]

        process_result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )

        if process_result.returncode != 0:
            error_message = (
                f"Ollama CLI error (large model: {model_name}, returncode: {process_result.returncode}). "
                f"Stderr: {process_result.stderr.strip() if process_result.stderr else 'N/A'}"
            )
            raise OllamaError(error_message)

        try:
            output_json = json.loads(process_result.stdout.strip())
        except json.JSONDecodeError as e:
            error_message = (
                f"Failed to decode JSON from Ollama (large model: {model_name}): {e}. "
                f"Raw Output: '{process_result.stdout.strip()}'"
            )
            raise OllamaError(error_message) from e

        try:
            inference_result = LargeInferenceOutput(**output_json)
            return inference_result
        except ValidationError as e:
            error_message = (
                f"Ollama output validation error (large model: {model_name}): {e}. "
                f"Received JSON: {output_json}"
            )
            raise OllamaError(error_message) from e

    except subprocess.TimeoutExpired:
        print(f"Error: Ollama CLI command timed out after {timeout_seconds}s (large model: {model_name})")
        return None
    except FileNotFoundError:
        print("Error: Ollama CLI executable not found. Ensure Ollama is installed and in your system's PATH.")
        raise
    except OllamaError as e:
        print(f"OllamaError (large inference): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error in run_large_inference (model: {model_name}): {type(e).__name__} - {e}")
        return None
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e_remove:
                print(f"Warning: Failed to remove temporary image file {temp_image_path} (large inference): {e_remove}")

if __name__ == '__main__':
    print(f"--- Testing run_large_inference with model: {MODEL_NAME_DEFAULT_LARGE} ---")

    test_image_filename = "test_image_large.png" # Use a different name or same if content is fine
    current_image_frame: Optional[ImageFrame] = None

    try:
        if os.path.exists(test_image_filename):
            current_image_frame = cv2.imread(test_image_filename)
            if current_image_frame is not None:
                print(f"Successfully loaded test image: '{test_image_filename}'.")
            else:
                print(f"Warning: Could not read '{test_image_filename}'. Creating a dummy one.")

        if current_image_frame is None:
            print(f"Creating a dummy image for large inference: '{test_image_filename}'.")
            dummy_frame_data = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.circle(dummy_frame_data, (150,150), 100, (0,255,0), -1) # Green circle
            cv2.putText(dummy_frame_data, "TEST", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            if cv2.imwrite(test_image_filename, dummy_frame_data):
                print(f"Dummy image with circle saved as '{test_image_filename}'.")
                current_image_frame = dummy_frame_data
            else:
                print(f"Error: Failed to save dummy image. Using fallback black image.")
                current_image_frame = np.zeros((100,100,3),dtype=np.uint8)

    except Exception as e_img:
        print(f"Error during large test image setup: {e_img}")
        current_image_frame = np.zeros((100,100,3),dtype=np.uint8)

    if current_image_frame is not None:
        print("\n--- Test 1: Single frame input ---")
        result_single = run_large_inference(current_image_frame)
        if result_single:
            print("Large inference (single frame) successful:")
            print(f"  Caption: {result_single.caption}")
            print(f"  Tags: {result_single.tags}")
            print(f"  Analysis: {result_single.detailed_analysis}")
        else:
            print("Large inference (single frame) failed or returned no result.")
            print("  Check Ollama status and model availability.")

        print("\n--- Test 2: List of frames input (processes last frame) ---")
        # Create a dummy list of frames, where the last one is our test image
        dummy_older_frame = np.zeros((50,50,3), dtype=np.uint8)
        frames_list = [dummy_older_frame, current_image_frame]
        result_list = run_large_inference(frames_list)
        if result_list:
            print("Large inference (list of frames) successful:")
            print(f"  Caption: {result_list.caption}")
            print(f"  Tags: {result_list.tags}")
            print(f"  Analysis: {result_list.detailed_analysis}")
        else:
            print("Large inference (list of frames) failed or returned no result.")

        print("\n--- Test 3: Empty list input ---")
        result_empty_list = run_large_inference([])
        if result_empty_list is None:
            print("Large inference with empty list correctly returned None.")
        else:
            print(f"Large inference with empty list unexpectedly returned: {result_empty_list}")

    else:
        print("Could not obtain an image frame for large inference testing.")

    print("\n--- End of run_large_inference test ---   ")