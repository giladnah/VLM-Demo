# Configuration Guide

This guide covers the configuration system used by the unified inference system, including base configuration options and engine-specific settings.

## Base Configuration

All inference engines use `InferenceConfig` as their base configuration class:

```python
class InferenceConfig(BaseModel):
    """Base configuration for inference engines."""
    max_image_size: int = Field(default=616, gt=0)
    jpeg_quality: int = Field(default=90, ge=0, le=100)
    timeout_seconds: int = Field(default=90, gt=0)
    retry_count: int = Field(default=3, ge=0)
    retry_cooldown: int = Field(default=5, gt=0)
    additional_params: Dict[str, Any] = Field(default_factory=dict)
```

### Parameters

#### max_image_size
- **Type**: `int`
- **Default**: 616
- **Validation**: > 0
- **Purpose**: Maximum dimension (width or height) for image processing
- **Impact**: Larger images are automatically resized while maintaining aspect ratio

#### jpeg_quality
- **Type**: `int`
- **Default**: 90
- **Range**: 0-100
- **Purpose**: Quality setting for JPEG encoding
- **Impact**: Higher values = better quality but larger payload size

#### timeout_seconds
- **Type**: `int`
- **Default**: 90
- **Validation**: > 0
- **Purpose**: Maximum time to wait for inference response
- **Impact**: Affects reliability vs latency trade-off

#### retry_count
- **Type**: `int`
- **Default**: 3
- **Validation**: >= 0
- **Purpose**: Number of retry attempts for failed requests
- **Impact**: Higher values increase reliability but may increase latency

#### retry_cooldown
- **Type**: `int`
- **Default**: 5
- **Validation**: > 0
- **Purpose**: Seconds to wait between retry attempts
- **Impact**: Affects recovery from transient failures

#### additional_params
- **Type**: `Dict[str, Any]`
- **Default**: `{}`
- **Purpose**: Engine-specific additional configuration
- **Impact**: Allows for extensibility and custom settings

## Engine-Specific Configurations

### Ollama Engine

```python
class OllamaConfig(InferenceConfig):
    """Ollama-specific configuration."""
    model_name: str = "qwen2.5vl:7b"
    ollama_url: str = "http://localhost:11434"
    reset_between_retries: bool = False
```

#### Parameters

##### model_name
- **Type**: `str`
- **Default**: "qwen2.5vl:7b"
- **Options**:
  - "qwen2.5vl:2b" (small model)
  - "qwen2.5vl:7b" (large model)
- **Purpose**: Specifies which Ollama model to use

##### ollama_url
- **Type**: `str`
- **Default**: "http://localhost:11434"
- **Purpose**: URL of the Ollama server
- **Note**: Can be changed for remote Ollama instances

##### reset_between_retries
- **Type**: `bool`
- **Default**: False
- **Purpose**: Whether to reset the model between retry attempts
- **Impact**: May help with stuck models but increases retry latency

### Usage Examples

#### Basic Configuration
```python
from inference import UnifiedVLMInference, OllamaConfig

config = OllamaConfig(
    max_image_size=512,
    timeout_seconds=60,
    retry_count=2
)

vlm = UnifiedVLMInference()
await vlm.initialize_engine("ollama_small", config)
```

#### Custom Server Configuration
```python
config = OllamaConfig(
    model_name="qwen2.5vl:7b",
    ollama_url="http://inference-server:11434",
    timeout_seconds=120,
    retry_count=5,
    retry_cooldown=10
)
```

#### High-Quality Configuration
```python
config = OllamaConfig(
    max_image_size=1024,
    jpeg_quality=95,
    timeout_seconds=180,
    additional_params={
        "priority": "quality"
    }
)
```

## Best Practices

1. **Image Size**
   - Balance between quality and performance
   - Consider network bandwidth limitations
   - Test with representative images

2. **Timeouts and Retries**
   - Set timeouts based on model size and hardware
   - Use more retries for critical applications
   - Adjust cooldown based on failure patterns

3. **Quality Settings**
   - Use higher JPEG quality for detailed analysis
   - Lower quality for real-time applications
   - Test impact on accuracy vs performance

4. **Server Configuration**
   - Use local server for development
   - Configure remote servers for production
   - Consider load balancing for multiple instances

## Environment Variables

The system also supports configuration through environment variables:

```bash
# Base configuration
export VLM_MAX_IMAGE_SIZE=616
export VLM_JPEG_QUALITY=90
export VLM_TIMEOUT_SECONDS=90
export VLM_RETRY_COUNT=3
export VLM_RETRY_COOLDOWN=5

# Ollama configuration
export OLLAMA_URL="http://localhost:11434"
```

## Configuration Precedence

1. Explicitly passed configuration
2. Environment variables
3. Default values

## Future Engine Configurations

### OpenAI Engine (Coming Soon)
```python
class OpenAIConfig(InferenceConfig):
    api_key: str
    model: str = "gpt-4-vision-preview"
    max_tokens: int = 300
```

### Hailo Engine (Coming Soon)
```python
class HailoConfig(InferenceConfig):
    device_id: int = 0
    power_mode: str = "high_performance"
```