"""Ollama-based inference engine implementation."""

import cv2
import json
import base64
import time
import asyncio
import aiohttp
from typing import Union, List, Optional, Dict, Any

import numpy as np
from pydantic import BaseModel, Field

from .engine import (
    InferenceEngine,
    InferenceResult,
    TriggerConfig,
    InferenceConfig,
    ImageFrame
)

class OllamaConfig(InferenceConfig):
    """Ollama-specific configuration."""
    model_name: str = "qwen2.5vl:7b"
    ollama_url: str = "http://localhost:11434"
    reset_between_retries: bool = False

class OllamaError(Exception):
    """Custom exception for Ollama-related errors."""
    pass

class OllamaEngine(InferenceEngine):
    """Ollama-based inference engine implementation."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._initialized = False
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return f"ollama_{self.config.model_name}"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the aiohttp session."""
        if not self._initialized:
            self._session = aiohttp.ClientSession()
            self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False

    async def process_frame(
        self,
        frame: Union[ImageFrame, List[ImageFrame]],
        trigger: TriggerConfig
    ) -> InferenceResult:
        """Process a frame using Ollama."""
        if not self._initialized or not self._session:
            raise OllamaError("Engine not initialized")

        # Use last frame if list is provided
        if isinstance(frame, list):
            if not frame:
                raise OllamaError("Empty frame list provided")
            frame = frame[-1]

        start_time = time.time()

        try:
            # Resize image if needed
            h, w = frame.shape[:2]
            if max(h, w) > self.config.max_image_size:
                scale = self.config.max_image_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode image
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            success, img_encoded = cv2.imencode('.jpg', frame, encode_params)
            if not success:
                raise OllamaError("Failed to encode image")

            image_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

            # Prepare prompt based on trigger type
            prompt = self._create_prompt(trigger)

            # Prepare payload
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "format": "json",
                "stream": False
            }

            # Make request with retries
            response_json = await self._make_request_with_retries(payload)

            # Parse response
            try:
                output_json = json.loads(response_json.get("response", ""))

                return InferenceResult(
                    result=output_json["result"],
                    detailed_analysis=output_json["detailed_analysis"],
                    confidence_score=output_json.get("confidence", None),
                    raw_model_output=response_json,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    engine_name=self.name
                )
            except (json.JSONDecodeError, KeyError) as e:
                raise OllamaError(f"Failed to parse Ollama response: {e}")

        except Exception as e:
            raise OllamaError(f"Inference failed: {str(e)}")

    def _create_prompt(self, trigger: TriggerConfig) -> str:
        """Create appropriate prompt based on trigger type."""
        base_prompt = (
            f"Analyze this image. Based on the image, is there a '{trigger.description}' present? "
            "Respond ONLY with a JSON object containing: "
            "'result': 'yes' or 'no' (whether the trigger is present), "
            "and 'detailed_analysis': a one sentence concise explanation of why."
        )

        if trigger.type == "object_detection":
            base_prompt = f"Focus on detecting objects. {base_prompt}"
        elif trigger.type == "scene_analysis":
            base_prompt = f"Focus on the overall scene and context. {base_prompt}"

        return base_prompt

    async def _make_request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Ollama with retries."""
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                async with self._session.post(
                    f"{self.config.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout_seconds
                ) as response:
                    if response.status >= 500:
                        raise OllamaError(f"Server error (status {response.status})")

                    response.raise_for_status()
                    return await response.json()

            except Exception as e:
                last_error = e
                if isinstance(e, aiohttp.ClientConnectorError):
                    print(f"[WARN] Ollama server not available at {self.config.ollama_url}: {e}")
                if attempt < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_cooldown)
                    continue

        raise OllamaError(f"All retry attempts failed: {last_error}")