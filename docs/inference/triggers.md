# Trigger System Documentation

The trigger system provides a flexible way to configure what the VLM should detect in images. It supports different types of triggers and allows for customization through additional parameters.

## Trigger Types

```python
class TriggerType(str, Enum):
    """Types of triggers supported by the system."""
    OBJECT_DETECTION = "object_detection"
    SCENE_ANALYSIS = "scene_analysis"
    CUSTOM_PROMPT = "custom_prompt"
```

### Object Detection
- Focused on identifying specific objects in the image
- Best for concrete, visible items
- Example: "a red car", "a person wearing a hat"

### Scene Analysis
- Analyzes the overall context and composition
- Good for complex scenes or situations
- Example: "a crowded street", "a peaceful garden"

### Custom Prompt
- Allows for more flexible and nuanced queries
- Can combine multiple aspects
- Example: "an unusual activity in a workplace setting"

## Configuration

### TriggerConfig Class
```python
class TriggerConfig(BaseModel):
    """Configuration for different types of triggers."""
    type: TriggerType = TriggerType.CUSTOM_PROMPT
    description: str
    confidence_threshold: float = 0.7
    additional_params: Dict[str, Any] = {}
```

### Parameters

#### type
- **Type**: `TriggerType`
- **Default**: `CUSTOM_PROMPT`
- **Purpose**: Determines how the prompt is formatted and interpreted

#### description
- **Type**: `str`
- **Required**: Yes
- **Purpose**: The actual detection query
- **Guidelines**:
  - Be specific and clear
  - Use natural language
  - Focus on visible characteristics

#### confidence_threshold
- **Type**: `float`
- **Range**: 0.0 to 1.0
- **Default**: 0.7
- **Purpose**: Minimum confidence level for a positive detection

#### additional_params
- **Type**: `Dict[str, Any]`
- **Default**: `{}`
- **Purpose**: Engine-specific parameters or extra context

## Usage Examples

### Basic Object Detection
```python
trigger = TriggerConfig(
    type=TriggerType.OBJECT_DETECTION,
    description="a person wearing a hard hat",
    confidence_threshold=0.8
)
```

### Complex Scene Analysis
```python
trigger = TriggerConfig(
    type=TriggerType.SCENE_ANALYSIS,
    description="a workplace safety violation",
    confidence_threshold=0.7,
    additional_params={
        "focus_areas": ["equipment", "personnel"],
        "safety_rules": ["PPE", "restricted areas"]
    }
)
```

### Custom Analysis
```python
trigger = TriggerConfig(
    type=TriggerType.CUSTOM_PROMPT,
    description="signs of water damage or mold",
    confidence_threshold=0.6,
    additional_params={
        "context": "building inspection",
        "areas": ["walls", "ceilings", "corners"],
        "indicators": ["discoloration", "peeling", "moisture"]
    }
)
```

## Best Practices

1. **Choosing Trigger Types**
   - Use `OBJECT_DETECTION` for specific, visible items
   - Use `SCENE_ANALYSIS` for overall context or situations
   - Use `CUSTOM_PROMPT` for complex or nuanced queries

2. **Writing Descriptions**
   - Be specific and unambiguous
   - Focus on visible characteristics
   - Use natural language
   - Keep descriptions concise

3. **Setting Confidence Thresholds**
   - Higher (>0.8) for critical safety applications
   - Medium (0.6-0.8) for general detection
   - Lower (<0.6) for exploratory or broad queries

4. **Using Additional Parameters**
   - Include relevant context
   - Specify areas of interest
   - Define specific criteria
   - Add domain-specific information

## Engine-Specific Considerations

### Ollama Engine
- Automatically formats prompts based on trigger type
- Adds focus directives for different trigger types
- Supports all standard parameters

### Future Engines
- OpenAI: Will support advanced prompt engineering
- Hailo: Will optimize for hardware acceleration