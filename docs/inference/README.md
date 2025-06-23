# Unified VLM Inference System

The Unified VLM Inference System provides a flexible and extensible architecture for running inference using multiple Vision Language Models (VLMs). It supports different inference engines and trigger types, making it easy to switch between models or combine their outputs.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Supported Engines](#supported-engines)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Examples](#examples)
6. [API Reference](#api-reference)

## Architecture Overview

The system is built around three main components:

1. **Inference Engines**: Implementations of different VLM backends (Ollama, OpenAI, etc.)
2. **Trigger System**: Configurable triggers for different types of detections
3. **Unified Interface**: Central coordination of engines and standardized I/O

```python
from inference import UnifiedVLMInference, TriggerConfig, TriggerType, OpenAIConfig

# Create the unified inference system
vlm = UnifiedVLMInference()

# Initialize engines
await vlm.initialize_engine("ollama_small")  # 2B parameter model
await vlm.initialize_engine("ollama_large")  # 7B parameter model

# Initialize OpenAI GPT-4V engine
openai_config = OpenAIConfig(api_key="sk-...", model="gpt-4-vision-preview")
await vlm.initialize_engine("openai_gpt4v", openai_config)

# Process frames with different engines
result = await vlm.process_frame(
    engine_type="openai_gpt4v",
    frame=image,
    trigger=TriggerConfig(
        type=TriggerType.OBJECT_DETECTION,
        description="a person falling"
    )
)
```

## Supported Engines

### Ollama Engine
- Supports both small (2B) and large (7B) Qwen VL models
- Local inference with configurable retries and timeouts
- Automatic image resizing and optimization

### OpenAI GPT-4V Engine
- Cloud-based inference using OpenAI's GPT-4V
- Requires API key and internet connection
- Supports advanced vision analysis and flexible prompt engineering

### Coming Soon
- Hailo Hardware-Accelerated Engine

## Quick Start

1. Install the package:
```bash
pip install -r requirements.txt
```

2. Basic usage (OpenAI):
```python
import asyncio
from inference import UnifiedVLMInference, TriggerConfig, TriggerType, OpenAIConfig
import cv2

async def main():
    # Initialize the system
    vlm = UnifiedVLMInference()
    openai_config = OpenAIConfig(api_key="sk-...", model="gpt-4-vision-preview")
    await vlm.initialize_engine("openai_gpt4v", openai_config)

    # Load an image
    image = cv2.imread("test.jpg")

    # Configure trigger
    trigger = TriggerConfig(
        type=TriggerType.OBJECT_DETECTION,
        description="a cat sitting on a windowsill",
        confidence_threshold=0.7
    )

    # Process the image
    result = await vlm.process_frame("openai_gpt4v", image, trigger)
    print(f"Result: {result.result}")
    print(f"Analysis: {result.detailed_analysis}")

    # Clean up
    await vlm.shutdown_all()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Base Configuration
All engines support these basic parameters:
```python
from inference import InferenceConfig

config = InferenceConfig(
    max_image_size=616,      # Maximum image dimension
    jpeg_quality=90,         # JPEG encoding quality
    timeout_seconds=90,      # Request timeout
    retry_count=3,           # Number of retry attempts
    retry_cooldown=5         # Seconds between retries
)
```

### Ollama-Specific Configuration
```python
from inference import OllamaConfig

config = OllamaConfig(
    model_name="qwen2.5vl:7b",
    ollama_url="http://localhost:11434",
    reset_between_retries=False,
    **base_config_params
)
```

### OpenAI-Specific Configuration
```python
from inference import OpenAIConfig

openai_config = OpenAIConfig(
    api_key="sk-...",
    model="gpt-4-vision-preview",
    max_tokens=300,
    temperature=0.0
)
```

## Examples

### Object Detection (OpenAI)
```python
trigger = TriggerConfig(
    type=TriggerType.OBJECT_DETECTION,
    description="a red car",
    confidence_threshold=0.7
)
result = await vlm.process_frame("openai_gpt4v", image, trigger)
```

### Scene Analysis (OpenAI)
```python
trigger = TriggerConfig(
    type=TriggerType.SCENE_ANALYSIS,
    description="a sunny beach with palm trees",
    confidence_threshold=0.6
)
result = await vlm.process_frame("openai_gpt4v", image, trigger)
```

### Custom Prompts (OpenAI)
```python
trigger = TriggerConfig(
    type=TriggerType.CUSTOM_PROMPT,
    description="an unusual or out-of-place object in the scene",
    confidence_threshold=0.8,
    additional_params={
        "focus_areas": ["background", "foreground"],
        "context": "security monitoring"
    }
)
result = await vlm.process_frame("openai_gpt4v", image, trigger)
```

## API Reference

### UnifiedVLMInference
The main class coordinating different inference engines.

Methods:
- `initialize_engine(engine_type: str, config: Optional[InferenceConfig] = None) -> None`
- `shutdown_engine(engine_type: str) -> None`
- `shutdown_all() -> None`
- `process_frame(engine_type: str, frame: ImageFrame, trigger: TriggerConfig) -> InferenceResult`

### OpenAIConfig
Configuration for the OpenAI GPT-4V engine.

Attributes:
- `api_key: str` (required)
- `model: str` (default: "gpt-4-vision-preview")
- `max_tokens: int` (default: 300)
- `temperature: float` (default: 0.0)
- `organization_id: Optional[str]`

### TriggerConfig
Configuration for different types of triggers.

Attributes:
- `type: TriggerType`
- `description: str`
- `confidence_threshold: float = 0.7`
- `additional_params: Dict[str, Any] = {}`

### InferenceResult
Standardized output format for all inference engines.

Attributes:
- `result: str` - 'yes' or 'no'
- `detailed_analysis: str`
- `confidence_score: Optional[float]`
- `raw_model_output: Optional[Dict[str, Any]]`
- `processing_time_ms: float`
- `engine_name: str`

For more detailed information about specific components, see:
- [Engine Protocol](./engine.md)
- [Trigger System](./triggers.md)
- [Configuration Guide](./configuration.md)