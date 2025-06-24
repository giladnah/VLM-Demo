"""
Unit tests for the OpenAI inference engine.
- Covers initialization, process_frame, error handling, and invalid response handling.
- All network calls are mocked; does not require real OpenAI API access.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError
from inference import (
    OpenAIEngine,
    OpenAIConfig,
    OpenAIError,
    TriggerConfig,
    TriggerType
)

@pytest.fixture
def mock_response():
    """Create a mock response for OpenAI API."""
    message = MagicMock()
    message.content = '{"result": "yes", "detailed_analysis": "Test analysis", "confidence": 0.95}'
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.index = 0
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response

@pytest.fixture
def mock_client(mock_response):
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.close = AsyncMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def test_trigger():
    """Create a test trigger configuration."""
    return TriggerConfig(
        type=TriggerType.OBJECT_DETECTION,
        description="test object"
    )

@pytest.mark.asyncio
async def test_openai_engine_initialization():
    """Test OpenAIEngine initialization."""
    config = OpenAIConfig(api_key="test_key")
    engine = OpenAIEngine(config)

    assert not engine.is_initialized
    with patch('inference.openai_engine.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.close = AsyncMock()
        await engine.initialize()
        assert engine.is_initialized
        assert engine.name == "openai_gpt-4o"
        mock_openai.assert_called_once_with(
            api_key="test_key",
            organization=None
        )
    await engine.shutdown()
    assert not engine.is_initialized

@pytest.mark.asyncio
async def test_openai_engine_process_frame(monkeypatch, test_image, test_trigger, mock_client):
    """Test OpenAIEngine frame processing."""
    config = OpenAIConfig(api_key="test_key")
    engine = OpenAIEngine(config)

    # Mock OpenAI client creation
    with patch('inference.openai_engine.AsyncOpenAI', return_value=mock_client):
        await engine.initialize()
        result = await engine.process_frame(test_image, test_trigger)

        assert result.result == "yes"
        assert result.detailed_analysis == "Test analysis"
        assert result.confidence_score == 0.95
        assert result.engine_name == "openai_gpt-4o"

        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["max_tokens"] == 300
        assert call_args["temperature"] == 0.0
        assert call_args["response_format"] == {"type": "json_object"}

    await engine.shutdown()

@pytest.mark.asyncio
async def test_openai_engine_error_handling(monkeypatch, test_image, test_trigger):
    """Test OpenAIEngine error handling."""
    config = OpenAIConfig(
        api_key="test_key",
        retry_count=2,
        retry_cooldown=1
    )
    engine = OpenAIEngine(config)

    # Mock client that always fails
    failed_client = MagicMock()
    failed_client.close = AsyncMock()
    failed_client.chat = MagicMock()
    failed_client.chat.completions = MagicMock()
    failed_client.chat.completions.create = AsyncMock(side_effect=Exception("Test error"))

    with patch('inference.openai_engine.AsyncOpenAI', return_value=failed_client):
        await engine.initialize()
        with pytest.raises(OpenAIError):
            await engine.process_frame(test_image, test_trigger)

    await engine.shutdown()

@pytest.mark.asyncio
async def test_openai_engine_invalid_response(monkeypatch, test_image, test_trigger):
    """Test OpenAIEngine handling of invalid responses."""
    config = OpenAIConfig(api_key="test_key")
    engine = OpenAIEngine(config)

    # Create mock response with invalid JSON
    message = MagicMock()
    message.content = 'invalid json'
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.index = 0
    choice.message = message
    invalid_response = MagicMock()
    invalid_response.choices = [choice]

    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=invalid_response)

    with patch('inference.openai_engine.AsyncOpenAI', return_value=mock_client):
        await engine.initialize()
        with pytest.raises(OpenAIError):
            await engine.process_frame(test_image, test_trigger)

    await engine.shutdown()