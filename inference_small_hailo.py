"""Module for running inference using a small Vision Language Model on a Hailo device."""

import os
import threading
import logging
import tempfile
from typing import Optional, TypeAlias
import numpy as np
from PIL import Image
# from image_preprocess import image_preprocess
from raw_connection import ConnectionContextWrapper, RawConnectionWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type alias for an image frame, expected to be a NumPy array from OpenCV or PIL.
ImageFrame: TypeAlias = np.ndarray

# MODEL_DIR_DEFAULT = "./hefs_and_embeddings"
MODEL_DIR_DEFAULT = "/home/giladn/tappas_apps/repos/QWEN2-2B_VL_Demo/hefs_and_embeddings"
MODEL_FILES = {
    "qwen": "qwen2_combined_shared_weights.hef.v3.optimized",
    "embed": "token_embed_after_rot_uint16.npy",
    "vision": "qwen2_vl_vision_224x224_w_pre.hef"
}


RESIZE_SHAPE = (224, 224)

def image_preprocess(image_path):
    image = Image.open(image_path)
    image = image.resize(RESIZE_SHAPE, resample=Image.Resampling.BICUBIC)
    image = np.array(image)
    return image

class HailoInferenceSession:
    """
    Singleton-style class to manage Hailo device connection and model loading.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_dir: str = MODEL_DIR_DEFAULT):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: str = MODEL_DIR_DEFAULT):
        self._initialized = getattr(self, '_initialized', False)
        if self._initialized:
            return
        self.model_dir = model_dir
        self.connection = None
        self._init_connection()
        self._initialized = True

    def _init_connection(self):
        context = ConnectionContextWrapper(False)
        self.connection = RawConnectionWrapper(context)
        self.connection.__enter__()  # Manually enter context
        logger.info("Hailo device connection established.")
        # Load model files
        self._send_model_file(MODEL_FILES["qwen"])
        self._send_model_file(MODEL_FILES["vision"])
        self._send_model_file(MODEL_FILES["embed"])

    def _send_model_file(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        with open(path, "rb") as f:
            data = f.read()
        self.connection.write(data)
        res_read = self.connection.read()
        output = res_read.decode("utf-8", errors="replace")
        logger.info(f"Loaded model file {filename}: {output}")

    def send_image(self, patches: np.ndarray):
        self.connection.write(b"new_image")
        self.connection.write(patches.tobytes())
        res_read = self.connection.read()
        output = res_read.decode("utf-8", errors="replace")
        logger.info(f"New image acknowledgment: {output}")
        return output

    def send_prompt(self, prompt: str) -> str:
        self.connection.write(prompt.encode("utf-8"))
        response = ""
        while True:
            res_read = self.connection.read()
            output = res_read.decode("utf-8", errors="replace")
            if output in ["<|endoftext|>"]:
                break
            response += output
        # Remove trailing <|im_end|> if present
        response = response.strip()
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")].rstrip()
        return response

    def close(self):
        if self.connection:
            self.connection.__exit__()
            self.connection = None
            logger.info("Hailo device connection closed.")


def run_small_inference(
    frame: ImageFrame,
    trigger_description: str,
    model_dir: str = MODEL_DIR_DEFAULT,
    timeout_seconds: int = 60
) -> Optional[str]:
    """
    Runs inference using a small VLM on a Hailo device on a given image frame.

    This function preprocesses the image, sends it to the Hailo device,
    sends the prompt, and collects the response.

    Args:
        frame (ImageFrame): The image frame (NumPy array from OpenCV or PIL) to process.
        trigger_description (str): A textual description of what to detect
                                  (e.g., "a dog on a couch").
        model_dir (str, optional): Directory containing model files.
        timeout_seconds (int, optional): Timeout for the inference call (not used, for API compatibility).

    Returns:
        Optional[str]: The model's response as a string, or None if an error occurs.
    """
    try:
        # 1. Preprocess the image using the same pipeline as app.py
        if isinstance(frame, Image.Image):
            # Convert PIL Image to file for preprocessing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                frame.save(tmp_file.name)
                patches = image_preprocess(tmp_file.name)
            os.unlink(tmp_file.name)
        else:
            # Assume frame is a numpy array (OpenCV)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                Image.fromarray(frame).save(tmp_file.name)
                patches = image_preprocess(tmp_file.name)
            os.unlink(tmp_file.name)

        # 2. Get or create the Hailo inference session
        session = HailoInferenceSession(model_dir)

        # 3. Send the image patches to the device
        session.send_image(patches)

        # 4. Construct the prompt
        prompt = (
            f"Analyze this image. Based on the image, is there a '{trigger_description}' present? "
            "Respond ONLY with a JSON object containing a single key 'result': "
            "If the trigger is present, respond with {\"result\": \"yes\"}. If not, respond with {\"result\": \"no\"}. "
            "Do not include any other text or explanation."
        )
        logger.info(f"Prompt sent to model: {prompt}")

        # 5. Send the prompt and get the response
        response = session.send_prompt(prompt)
        logger.info(f"Model response: {response}")

        # 6. Parse the response for 'yes' or 'no'
        response_lower = response.lower()
        if "yes" in response_lower:
            return "yes"
        elif "no" in response_lower:
            return "no"
        else:
            logger.error(f"Unexpected model response: {response}")
            return None

    except Exception as e:
        logger.error(f"Error during Hailo inference: {e}")
        return None