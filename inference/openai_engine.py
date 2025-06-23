"""OpenAI GPT-4V based inference engine implementation."""

import cv2
import base64
import time
import asyncio
import os
from typing import Union, List, Optional, Dict, Any
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

import numpy as np
from pydantic import BaseModel, Field

from .engine import (
    InferenceEngine,
    InferenceResult,
    TriggerConfig,
    InferenceConfig,
    ImageFrame
)

class OpenAIConfig(InferenceConfig):
    """OpenAI-specific configuration.
    If api_key is not provided, will attempt to load from the OPENAI_API_KEY environment variable.
    As of 2024, 'gpt-4o' is the recommended model for vision-language tasks (VLM).
    """
    api_key: Optional[str] = None
    model: str = "gpt-4o"  # Default to GPT-4o (4o-mini, vision-capable)
    max_tokens: int = 300
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    organization_id: Optional[str] = None

class OpenAIError(Exception):
    """Custom exception for OpenAI-related errors."""
    pass

class OpenAIEngine(InferenceEngine):
    """OpenAI GPT-4V based inference engine implementation."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize the OpenAI engine with configuration."""
        api_key = None
        if config and config.api_key:
            api_key = config.api_key
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIError("OpenAI API key is required. Set it in OpenAIConfig or as the OPENAI_API_KEY environment variable.")
        if config:
            config.api_key = api_key
        else:
            config = OpenAIConfig(api_key=api_key)
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the name of the engine."""
        return f"openai_{self.config.model}"

    @property
    def is_initialized(self) -> bool:
        """Return whether the engine is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self._initialized:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization_id
            )
            self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False

    async def process_frame(
        self,
        frame: Union[ImageFrame, List[ImageFrame]],
        trigger: TriggerConfig
    ) -> InferenceResult:
        """Process a frame using OpenAI's GPT-4V."""
        if not self._initialized or not self._client:
            raise OpenAIError("Engine not initialized")

        # Use last frame if list is provided
        if isinstance(frame, list):
            if not frame:
                raise OpenAIError("Empty frame list provided")
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
                raise OpenAIError("Failed to encode image")

            image_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

            # Create system message based on trigger type
            system_message = self._create_system_message(trigger)

            # Create user message with the image
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._create_user_prompt(trigger)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            # Make request with retries
            response = await self._make_request_with_retries(messages)

            # Parse response
            try:
                content = response.choices[0].message.content
                output_json = self._parse_response(content)

                return InferenceResult(
                    result=output_json["result"],
                    detailed_analysis=output_json["detailed_analysis"],
                    confidence_score=output_json.get("confidence", None),
                    raw_model_output={"response": content},
                    processing_time_ms=(time.time() - start_time) * 1000,
                    engine_name=self.name
                )
            except (KeyError, ValueError) as e:
                raise OpenAIError(f"Failed to parse OpenAI response: {e}")

        except Exception as e:
            raise OpenAIError(f"Inference failed: {str(e)}")

    def _create_system_message(self, trigger: TriggerConfig) -> str:
        """Create appropriate system message based on trigger type."""
        base_message = (
            "You are a computer vision analysis system. "
            "Your task is to analyze images and determine if specific conditions are met. "
            "Respond ONLY with a JSON object containing: "
            "'result': 'yes' or 'no' (whether the condition is met), "
            "'detailed_analysis': a one sentence explanation, "
            "and optionally 'confidence': a float between 0 and 1."
        )

        if trigger.type == "object_detection":
            return f"{base_message} Focus specifically on detecting and identifying objects in the image."
        elif trigger.type == "scene_analysis":
            return f"{base_message} Focus on analyzing the overall scene, context, and relationships between elements."
        else:
            return base_message

    def _create_user_prompt(self, trigger: TriggerConfig) -> str:
        """Create the user prompt based on trigger configuration."""
        return f"Based on this image, is there {trigger.description}? Remember to respond only with the specified JSON format."

    async def _make_request_with_retries(self, messages: List[Dict[str, Any]]) -> ChatCompletion:
        """Make request to OpenAI with retries."""
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                return await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"}
                )

            except Exception as e:
                last_error = e
                if attempt < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_cooldown)
                    continue

        raise OpenAIError(f"All retry attempts failed: {last_error}")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the response content into a structured format."""
        import json
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OpenAIError(f"Failed to parse JSON response: {e}")