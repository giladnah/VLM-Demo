"""
Unit tests for inference_large.py.

Covers expected, edge, and failure cases for the run_large_inference function, including:
- Successful inference (valid result)
- API/server errors
- Malformed/invalid JSON responses
- Missing/invalid keys in the response
- Request exceptions
- Edge case: invalid input type

Uses pytest and unittest.mock for isolation from external dependencies.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import requests
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference_large import run_large_inference, MODEL_NAME_DEFAULT_LARGE, ImageFrame

@pytest.fixture
def dummy_image_frame_large() -> ImageFrame:
    return np.zeros((100, 100, 3), dtype=np.uint8)

@patch('requests.post')
def test_successful_large_inference(mock_post, dummy_image_frame_large):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": json.dumps({
        "result": "yes",
        "detailed_analysis": "A cat is sitting comfortably on a couch."
    })}
    mock_post.return_value = mock_response
    result = run_large_inference(dummy_image_frame_large, MODEL_NAME_DEFAULT_LARGE)
    assert result is not None
    assert hasattr(result, 'result')
    assert result.result == "yes"

@patch('requests.post')
def test_large_inference_api_error(mock_post, dummy_image_frame_large):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response
    result = run_large_inference(dummy_image_frame_large, MODEL_NAME_DEFAULT_LARGE)
    assert result is None

@patch('requests.post')
def test_large_inference_json_decode_error(mock_post, dummy_image_frame_large):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "not a json string"}
    mock_post.return_value = mock_response
    result = run_large_inference(dummy_image_frame_large, MODEL_NAME_DEFAULT_LARGE)
    assert result is None

@patch('requests.post')
def test_large_inference_missing_caption_key(mock_post, dummy_image_frame_large):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": json.dumps({"unexpected": True})}
    mock_post.return_value = mock_response
    result = run_large_inference(dummy_image_frame_large, MODEL_NAME_DEFAULT_LARGE)
    assert result is None

@patch('requests.post', side_effect=requests.RequestException("API error"))
def test_large_inference_request_exception(mock_post, dummy_image_frame_large):
    result = run_large_inference(dummy_image_frame_large, MODEL_NAME_DEFAULT_LARGE)
    assert result is None

# Edge case: invalid input type

def test_invalid_frames_type_input():
    result = run_large_inference("not_an_image_or_list", "dummy trigger") # type: ignore
    assert result is None