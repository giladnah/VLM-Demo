"""Unified VLM inference system coordinating different engines."""

from typing import Dict, Type, Optional, Any, Union
from .engine import InferenceEngine, InferenceConfig, TriggerConfig, InferenceResult, ImageFrame
from .ollama_engine import OllamaEngine, OllamaConfig
from .openai_engine import OpenAIEngine, OpenAIConfig
import asyncio
import os
from config import get_config

class UnifiedVLMInference:
    """Main class coordinating different inference engines."""

    def __init__(self):
        self._engines: Dict[str, InferenceEngine] = {}

    async def initialize_engine(
        self,
        engine_type: str,
        config: Optional[InferenceConfig] = None
    ) -> None:
        """Initialize a specific type of inference engine."""
        if engine_type in self._engines:
            raise ValueError(f"Engine {engine_type} already initialized")

        engine = self._create_engine(engine_type, config)
        await engine.initialize()
        self._engines[engine_type] = engine

    async def shutdown_engine(self, engine_type: str) -> None:
        """Shutdown a specific engine."""
        if engine_type not in self._engines:
            raise ValueError(f"Engine {engine_type} not initialized")

        await self._engines[engine_type].shutdown()
        del self._engines[engine_type]

    async def shutdown_all(self) -> None:
        """Shutdown all engines."""
        for engine_type in list(self._engines.keys()):
            await self.shutdown_engine(engine_type)

    async def process_frame(
        self,
        engine_type: str,
        frame: ImageFrame,
        trigger: TriggerConfig
    ) -> InferenceResult:
        """Process a frame using the specified engine."""
        if engine_type not in self._engines:
            raise ValueError(f"Engine {engine_type} not initialized")

        return await self._engines[engine_type].process_frame(frame, trigger)

    def _create_engine(
        self,
        engine_type: str,
        config: Optional[InferenceConfig] = None
    ) -> InferenceEngine:
        """Create an instance of the specified engine type."""
        engine_map = {
            "ollama_small": (OllamaEngine, OllamaConfig(model_name="qwen2.5vl:2b")),
            "ollama_large": (OllamaEngine, OllamaConfig(model_name="qwen2.5vl:7b")),
            "openai_gpt4v": (OpenAIEngine, OpenAIConfig()),
            # Add other engine types here as they are implemented
        }

        if engine_type not in engine_map:
            raise ValueError(f"Unsupported engine type: {engine_type}")

        engine_class, default_config = engine_map[engine_type]
        engine_config = config if config else default_config

        return engine_class(config=engine_config)

async def run_unified_inference(
    frame: ImageFrame,
    trigger_description: str,
    model_type: str = "small",
    config_dict: Optional[Dict[str, Any]] = None,
    trigger_type: Optional[str] = None,
    confidence_threshold: float = 0.7,
) -> InferenceResult:
    """
    Async entry point for VLM inference (small or large model) using the configured engine (Ollama, OpenAI, ...).

    Args:
        frame (ImageFrame): The image/frame to process (numpy array).
        trigger_description (str): The textual description of the trigger (e.g., "a person lying on the floor").
        model_type (str): 'small' or 'large'. Determines which model/engine config to use.
        config_dict (dict, optional): If provided, overrides config.yaml for engine/model selection.
        trigger_type (str, optional): Type of trigger (e.g., 'object_detection', 'scene_analysis', etc.).
        confidence_threshold (float): Confidence threshold for the trigger.

    Returns:
        InferenceResult: Standardized inference result from the selected engine.

    Usage:
        from inference.unified import run_unified_inference
        result = await run_unified_inference(frame, "a person lying on the floor", model_type="large")

    Note:
        This is the recommended entry point for all inference calls. The old run_small_inference/run_large_inference are deprecated.
    """
    # Load config
    config = config_dict or get_config()
    model_key = f"{model_type}_model"
    model_cfg = config[model_key]
    engine = model_cfg.get("engine", "ollama").lower()
    model_name = model_cfg["name"]

    # Build engine config
    if engine == "ollama":
        from .ollama_engine import OllamaConfig
        engine_config = OllamaConfig(
            model_name=model_name,
            ollama_url=model_cfg.get("ollama_server", "http://localhost:11434")
        )
        engine_type = f"ollama_{model_type}"
    elif engine == "openai":
        from .openai_engine import OpenAIConfig
        engine_config = OpenAIConfig(
            model=model_name,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        engine_type = "openai_gpt4v"
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    # Build trigger config
    trigger_type_val = trigger_type or "custom_prompt"
    trigger = TriggerConfig(
        type=trigger_type_val,
        description=trigger_description,
        confidence_threshold=confidence_threshold
    )

    vlm = UnifiedVLMInference()
    await vlm.initialize_engine(engine_type, engine_config)
    result = await vlm.process_frame(engine_type, frame, trigger)
    await vlm.shutdown_all()
    return result