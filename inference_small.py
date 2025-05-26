"""Module for running inference using a small Vision Language Model via Ollama REST API."""

import cv2
import requests
import json
import base64
import time
from typing import List, Optional, TypeAlias, Dict, Any
import logging

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type alias for an image frame, expected to be a NumPy array from OpenCV.
ImageFrame: TypeAlias = np.ndarray

# Consistent with 'ollama pull qwen2.5vl:3b' in TASKS.md
MODEL_NAME_DEFAULT = "qwen2.5vl:3b"
OLLAMA_URL_DEFAULT = "http://localhost:11434"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class OllamaError(Exception):
    """Custom exception for Ollama related errors encountered during inference."""
    pass

def reset_server(ollama_url: str = OLLAMA_URL_DEFAULT) -> bool:
    """
    Attempts to reset the Ollama server state by sending a lightweight request.
    This can help clear GPU memory between retries.

    Args:
        ollama_url (str): The Ollama server URL.

    Returns:
        bool: True if reset appears successful, False otherwise.
    """
    try:
        # Simple text-only query to help clear state
        reset_payload = {
            "model": MODEL_NAME_DEFAULT,
            "prompt": "Reset server state. Reply with just 'OK'.",
            "stream": False,
            "raw": True  # Use raw mode for faster processing
        }

        logger.debug("Sending reset request to Ollama server...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=reset_payload,
            timeout=5  # Short timeout for reset request
        )

        if response.status_code == 200:
            logger.debug("Reset request completed")
            return True
        else:
            logger.warning(f"Reset request failed with status {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"Error during server reset attempt: {e}")
        return False

def run_small_inference(
    frame: ImageFrame,
    trigger_description: str,
    model_name: str = MODEL_NAME_DEFAULT,
    timeout_seconds: int = 60,
    ollama_url: str = OLLAMA_URL_DEFAULT,
    retry_count: int = MAX_RETRIES,
    max_image_size: int = 224,
    jpeg_quality: int = 90,
    retry_cooldown: int = 5,  # Seconds to wait between retries to let server recover
    reset_between_retries: bool = True  # Whether to try resetting server between retries
) -> Optional[str]:
    """
    Runs inference using a small VLM (via Ollama REST API) on a given image frame.

    This function encodes the image as base64, sends it to the Ollama REST API,
    collects the response, and parses the JSON result. It includes
    retry logic for transient failures.

    Args:
        frame (ImageFrame): The image frame (NumPy array from OpenCV) to process.
        trigger_description (str): A textual description of what to detect
                                  (e.g., "a dog on a couch").
        model_name (str, optional): The name of the Ollama model to use.
                                   Defaults to MODEL_NAME_DEFAULT.
        timeout_seconds (int, optional): Timeout for the Ollama REST API call.
                                        Defaults to 60 seconds.
        ollama_url (str, optional): The base URL for the Ollama server.
                                   Defaults to OLLAMA_URL_DEFAULT.
        retry_count (int, optional): Number of retry attempts for transient errors.
                                    Defaults to MAX_RETRIES.
        max_image_size (int, optional): Maximum dimension (width or height) for resizing.
                                       Defaults to 640px.
        jpeg_quality (int, optional): JPEG encoding quality (0-100).
                                     Defaults to 85.
        retry_cooldown (int, optional): Seconds to wait between retries to let server recover.
                                       Defaults to 5 seconds.
        reset_between_retries (bool, optional): Whether to try resetting server between retries.
                                              Defaults to True.

    Returns:
        Optional[str]: 'yes' if the trigger is present, 'no' if not, or None if all retries fail.
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
                logger.debug(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 2. Encode the frame as JPEG in memory with specified quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            success, img_encoded = cv2.imencode('.jpg', frame, encode_params)
            if not success:
                raise OllamaError("Failed to encode image frame as JPEG.")

            # Get size in KB for logging
            image_size_kb = len(img_encoded) / 1024
            logger.info(f"Image size: {original_size} → {new_w}x{new_h}, JPEG size: {image_size_kb:.1f}KB, quality: {jpeg_quality}%")

            image_bytes = img_encoded.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # 3. Construct the prompt for the VLM
            prompt = (
                f"Analyze this image. Based on the image, is there a '{trigger_description}' present? "
                "Respond ONLY with a JSON object containing a single key 'result': "
                "If the trigger is present, respond with {\"result\": \"yes\"}. If not, respond with {\"result\": \"no\"}. "
                "Do not include any other text or explanation."
            )

            # 4. Prepare the payload with the base64-encoded image
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_b64],
                "format": "json",
                "stream": False
            }

            logger.info(f"Sending request to Ollama API: {ollama_url}/api/generate (attempt {attempt}/{retry_count})")
            start_time = time.time()

            # 5. Send the request to the Ollama REST API
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=timeout_seconds
            )

            # Check for server-side errors and handle them
            if response.status_code >= 500:
                error_message = f"Server error (status code {response.status_code}): {response.text}"
                logger.warning(f"Attempt {attempt}/{retry_count}: {error_message}")

                if attempt < retry_count:
                    if reset_between_retries:
                        reset_server(ollama_url)
                    logger.info(f"Waiting {retry_cooldown} seconds before retry...")
                    time.sleep(retry_cooldown)
                    continue
                else:
                    logger.error("All retry attempts failed due to server errors")
                    return None

            response.raise_for_status()

            elapsed_time = time.time() - start_time
            logger.info(f"Request completed in {elapsed_time:.2f} seconds")

            # 6. Process the response (non-streaming)
            response_json = response.json()
            generated_text = response_json.get("response", "")

            # 7. Parse the JSON output
            try:
                output_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                error_message = (
                    f"Failed to decode JSON from Ollama REST API (model: {model_name}): {e}. "
                    f"Raw Output: '{generated_text}'"
                )
                raise OllamaError(error_message) from e

            # 8. Extract the 'result' key
            result_value = output_json.get("result", None)
            if result_value in ("yes", "no"):
                return result_value
            else:
                logger.error(f"Unexpected output JSON: {output_json}")
                return None

        except requests.Timeout:
            logger.warning(f"Attempt {attempt}/{retry_count}: Ollama REST API request timed out after {timeout_seconds}s")
            if attempt < retry_count:
                if reset_between_retries:
                    reset_server(ollama_url)
                logger.info(f"Waiting {retry_cooldown} seconds before retry...")
                time.sleep(retry_cooldown)
            else:
                logger.error("All retry attempts failed due to timeout")
                return None

        except requests.ConnectionError as e:
            logger.warning(f"Attempt {attempt}/{retry_count}: Connection error: {e}")
            if attempt < retry_count:
                if reset_between_retries:
                    reset_server(ollama_url)
                logger.info(f"Waiting {retry_cooldown} seconds before retry...")
                time.sleep(retry_cooldown)
            else:
                logger.error("All retry attempts failed due to connection error")
                return None

        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{retry_count}: Ollama REST API request failed: {e}")
            if attempt < retry_count:
                if reset_between_retries:
                    reset_server(ollama_url)
                logger.info(f"Waiting {retry_cooldown} seconds before retry...")
                time.sleep(retry_cooldown)
            else:
                logger.error("All retry attempts failed due to request exception")
                return None

        except OllamaError as e:
            logger.error(f"OllamaError: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in run_small_inference: {type(e).__name__} - {e}")
            return None

def check_ollama_server_health(
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout_seconds: int = 10,
    test_model: bool = True
) -> Dict[str, Any]:
    """
    Checks if the Ollama server is responsive and reports basic health information.
    Can optionally run a basic test on the model to verify it's working.

    Args:
        ollama_url (str): The base URL for the Ollama server.
        timeout_seconds (int): Timeout for the API call.
        test_model (bool): Whether to run a basic test on the model.

    Returns:
        Dict[str, Any]: A dictionary containing health status information.
    """
    result = {
        "status": "error",
        "url": ollama_url,
        "checks": {}
    }

    # Step 1: Check server connectivity
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()

        result["checks"]["server_connectivity"] = {
            "status": "passed",
            "message": "Server is responsive"
        }

        # Check if our required model is available
        models_response = response.json()
        models = models_response.get("models", [])
        model_names = [model.get("name") for model in models]

        result["available_models"] = model_names
        result["required_model_available"] = MODEL_NAME_DEFAULT in model_names

        if MODEL_NAME_DEFAULT in model_names:
            result["checks"]["model_availability"] = {
                "status": "passed",
                "message": f"Required model '{MODEL_NAME_DEFAULT}' is available"
            }
        else:
            result["checks"]["model_availability"] = {
                "status": "failed",
                "message": f"Required model '{MODEL_NAME_DEFAULT}' is not available"
            }
            result["status"] = "warning"

        # Step 2: If requested, run a simple test to confirm the model works
        if test_model and MODEL_NAME_DEFAULT in model_names:
            try:
                # Test model with minimal payload (simple text prompt)
                test_payload = {
                    "model": MODEL_NAME_DEFAULT,
                    "prompt": "What is 2+2? Respond with just the number.",
                    "stream": False
                }

                test_response = requests.post(
                    f"{ollama_url}/api/generate",
                    json=test_payload,
                    timeout=timeout_seconds
                )
                test_response.raise_for_status()

                # If we got a successful response, model is working
                result["checks"]["model_basic_test"] = {
                    "status": "passed",
                    "message": "Model responds to basic prompts"
                }

                # If all checks have passed, mark as healthy
                all_passed = all(check["status"] == "passed" for check in result["checks"].values())
                result["status"] = "healthy" if all_passed else "warning"

            except requests.RequestException as e:
                result["checks"]["model_basic_test"] = {
                    "status": "failed",
                    "message": f"Model test failed: {str(e)}"
                }
                result["status"] = "warning"

    except requests.RequestException as e:
        result["checks"]["server_connectivity"] = {
            "status": "failed",
            "message": f"Failed to connect to Ollama server: {str(e)}"
        }
        result["status"] = "error"

    return result

if __name__ == '__main__':
    # This block provides a basic way to test the run_small_inference function locally.
    # Requirements for this test:
    # 1. OpenCV (`pip install opencv-python`) and NumPy (`pip install numpy`) installed.
    # 2. Ollama service must be running locally.
    # 3. The model specified by MODEL_NAME_DEFAULT (e.g., "qwen2.5vl:3b") must be pulled
    #    in Ollama (e.g., using `ollama pull qwen2.5vl:3b`).
    # 4. A 'test_examples' directory with .jpg, .jpeg, or .png images in the same directory as this script.

    import os

    print(f"--- Testing run_small_inference with model: {MODEL_NAME_DEFAULT} ---")
    print("First, checking Ollama server health:")
    health = check_ollama_server_health()
    print(json.dumps(health, indent=2))

    if health["status"] == "error":
        print("ERROR: Ollama server is not available. Please start the Ollama service.")
        exit(1)

    if not health.get("required_model_available", False):
        print(f"WARNING: Model '{MODEL_NAME_DEFAULT}' is not available.")
        print(f"Please run: ollama pull {MODEL_NAME_DEFAULT}")
        print("Available models:", ", ".join(health.get("available_models", [])))
        exit(1)

    test_examples_dir = "test_examples"
    user_trigger = "a person have fallen and need help" # Your specified trigger

    if not os.path.isdir(test_examples_dir):
        print(f"Error: Directory '{test_examples_dir}' not found. Please create it and add images.")
    else:
        image_files = [
            f for f in os.listdir(test_examples_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            print(f"No image files (.png, .jpg, .jpeg) found in '{test_examples_dir}'.")
        else:
            print(f"Found images: {image_files}")
            print(f"Using trigger: '{user_trigger}'")

            for image_file in image_files:
                image_path = os.path.join(test_examples_dir, image_file)
                print(f"--- Processing image: {image_path} ---")
                current_image_frame: Optional[ImageFrame] = cv2.imread(image_path)

                if current_image_frame is None:
                    print(f"Warning: Could not read image '{image_path}' with OpenCV.")
                    continue

                result = run_small_inference(current_image_frame, user_trigger)

                if result:
                    print("Inference successful:")
                    print(f"  Result: {result}")
                else:
                    print("Inference failed or returned no result.")
                    print("  Troubleshooting tips:")
                    print("    - Is the Ollama service running? Check with 'systemctl status ollama'")
                    print(f"    - Is the model '{MODEL_NAME_DEFAULT}' pulled? Check with 'ollama list'")
                    print("    - Check Ollama logs for more details with 'journalctl -u ollama'")
                    print("    - Try increasing the timeout value if the model is large")
                print("-" * 30)

    print("--- End of run_small_inference test ---")