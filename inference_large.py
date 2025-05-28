"""Module for running inference using a large Vision Language Model via Ollama CLI."""

import cv2
import requests
import json
import base64
import time
import tempfile
import os
from typing import List, Optional, Union, Dict, Any, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Type alias for an image frame, expected to be a NumPy array from OpenCV.
ImageFrame: TypeAlias = np.ndarray

# Consistent with 'ollama pull qwen2.5vl:7b' in TASKS.md
MODEL_NAME_DEFAULT_LARGE = "qwen2.5vl:7b"
OLLAMA_URL_DEFAULT = "http://localhost:11434"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class LargeInferenceOutput(BaseModel):
    """Data model for the structured output of the large VLM inference."""
    caption: str
    tags: List[str] = Field(default_factory=list)
    detailed_analysis: str # Could be Dict[str, Any] if a more structured analysis is expected

class OllamaError(Exception):
    """Custom exception for Ollama related errors encountered during inference."""
    pass # Re-using the same exception type as in inference_small for consistency

def run_large_inference(
    frame: ImageFrame,
    model_name: str = MODEL_NAME_DEFAULT_LARGE,
    timeout_seconds: int = 90,
    ollama_url: str = OLLAMA_URL_DEFAULT,
    retry_count: int = MAX_RETRIES,
    max_image_size: int = 640,
    jpeg_quality: int = 90,
    retry_cooldown: int = 5,
    reset_between_retries: bool = False
) -> Optional[LargeInferenceOutput]:
    """
    Runs inference using a large VLM (via Ollama REST API) on a given image frame.
    Args:
        frame (ImageFrame): The image frame (NumPy array from OpenCV) to process.
        model_name (str, optional): The name of the Ollama model to use.
        timeout_seconds (int, optional): Timeout for the Ollama REST API call.
        ollama_url (str, optional): The base URL for the Ollama server.
        retry_count (int, optional): Number of retry attempts for transient errors.
        max_image_size (int, optional): Maximum dimension (width or height) for resizing.
        jpeg_quality (int, optional): JPEG encoding quality (0-100).
        retry_cooldown (int, optional): Seconds to wait between retries to let server recover.
        reset_between_retries (bool, optional): Not used for REST API, kept for interface compatibility.
    Returns:
        Optional[LargeInferenceOutput]: A LargeInferenceOutput object if successful, None if all retries fail.
    """
    for attempt in range(1, retry_count + 1):
        try:
            # 1. Resize the image to reduce payload size while maintaining aspect ratio
            h, w = frame.shape[:2]
            original_size = f"{w}x{h}"
            if max(h, w) > max_image_size:
                scale = max_image_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w, h

            # 2. Encode the frame as JPEG in memory with specified quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            success, img_encoded = cv2.imencode('.jpg', frame, encode_params)
            if not success:
                raise OllamaError("Failed to encode image frame as JPEG.")

            image_bytes = img_encoded.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # 3. Construct the prompt for the VLM
            prompt = (
                "Provide a detailed analysis of this image. "
                "Respond ONLY with a JSON object containing these keys: "
                "'caption' (a descriptive sentence about the image), "
                "'tags' (a list of relevant keywords or detected objects/attributes, max 10 tags), and "
                "'detailed_analysis' (a concise textual description or notable insights about the scene, 2-3 sentences)."
            )

            # 4. Prepare the payload with the base64-encoded image
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_b64],
                "format": "json",
                "stream": False
            }

            print(f"[LargeInference] Sending request to Ollama API: {ollama_url}/api/generate (attempt {attempt}/{retry_count})")
            start_time = time.time()

            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=timeout_seconds
            )

            if response.status_code >= 500:
                print(f"[LargeInference] Server error (status code {response.status_code}): {response.text}")
                if attempt < retry_count:
                    print(f"[LargeInference] Waiting {retry_cooldown} seconds before retry...")
                    time.sleep(retry_cooldown)
                    continue
                else:
                    print("[LargeInference] All retry attempts failed due to server errors")
                    return None

            response.raise_for_status()

            elapsed_time = time.time() - start_time
            print(f"[LargeInference] Request completed in {elapsed_time:.2f} seconds")

            response_json = response.json()
            generated_text = response_json.get("response", "")

            try:
                output_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                print(f"[LargeInference] Failed to decode JSON from Ollama REST API (model: {model_name}): {e}. Raw Output: '{generated_text}'")
                return None

            try:
                inference_result = LargeInferenceOutput(**output_json)
                return inference_result
            except ValidationError as e:
                print(f"[LargeInference] Ollama output validation error (large model: {model_name}): {e}. Received JSON: {output_json}")
                return None

        except Exception as e:
            print(f"[LargeInference] Unexpected error: {type(e).__name__} - {e}")
            if attempt < retry_count:
                print(f"[LargeInference] Waiting {retry_cooldown} seconds before retry...")
                time.sleep(retry_cooldown)
                continue
            else:
                return None
    return None

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