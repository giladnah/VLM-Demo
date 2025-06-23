"""Base interfaces and models for the unified inference system."""

from enum import Enum
from typing import Protocol, Dict, Any, Optional, Union, List
import numpy as np
from pydantic import BaseModel, Field

# Type alias for image frame
ImageFrame = np.ndarray

class TriggerType(str, Enum):
    """Types of triggers supported by the system."""
    OBJECT_DETECTION = "object_detection"
    SCENE_ANALYSIS = "scene_analysis"
    CUSTOM_PROMPT = "custom_prompt"

class TriggerConfig(BaseModel):
    """Configuration for different types of triggers."""
    type: TriggerType = TriggerType.CUSTOM_PROMPT
    description: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class InferenceResult(BaseModel):
    """Standardized output format for all inference engines."""
    result: str  # 'yes' or 'no'
    detailed_analysis: str
    confidence_score: Optional[float] = None
    raw_model_output: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    engine_name: str

class InferenceConfig(BaseModel):
    """Base configuration for inference engines."""
    max_image_size: int = Field(default=616, gt=0)
    jpeg_quality: int = Field(default=90, ge=0, le=100)
    timeout_seconds: int = Field(default=90, gt=0)
    retry_count: int = Field(default=3, ge=0)
    retry_cooldown: int = Field(default=5, gt=0)
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class InferenceEngine(Protocol):
    """Protocol defining the interface for inference engines."""

    async def initialize(self) -> None:
        """Initialize the engine and load any necessary resources."""
        ...

    async def process_frame(
        self,
        frame: Union[ImageFrame, List[ImageFrame]],
        trigger: TriggerConfig
    ) -> InferenceResult:
        """Process a frame or list of frames with the given trigger configuration."""
        ...

    async def shutdown(self) -> None:
        """Clean up resources and shut down the engine."""
        ...

    @property
    def name(self) -> str:
        """Return the name of the engine."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Return whether the engine is initialized and ready for inference."""
        ...