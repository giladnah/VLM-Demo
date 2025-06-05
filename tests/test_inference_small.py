"""Unit tests for the inference_small.py module."""

import pytest
import numpy as np
import cv2 # Though cv2.imwrite is mocked, importing for type consistency if needed
import json
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
import requests
import sys

# Assuming inference_small.py is in the parent directory or PYTHONPATH is set up
# For a typical project structure, you might need to adjust imports:
# from ..inference_small import run_small_inference, SmallInferenceOutput, OllamaError, MODEL_NAME_DEFAULT
# For now, using a simpler import path assuming flat structure or correct PYTHONPATH for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference_small import run_small_inference, OllamaError, MODEL_NAME_DEFAULT, ImageFrame

@pytest.fixture
def dummy_image_frame() -> ImageFrame:
    """Provides a consistent dummy image frame (NumPy array) for tests."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_temp_file_path() -> str:
    """Returns a dummy path for the temporary image file."""
    return "/tmp/dummy_test_image.jpg"

@patch('requests.post')
def test_successful_inference(mock_post, dummy_image_frame):
    """Test a successful REST API inference call returning 'yes'."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": json.dumps({"result": "yes"})}
    mock_post.return_value = mock_response
    result = run_small_inference(dummy_image_frame, "a cat on a couch")
    assert result == "yes"

@patch('requests.post')
def test_inference_returns_no(mock_post, dummy_image_frame):
    """Test a successful REST API inference call returning 'no'."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": json.dumps({"result": "no"})}
    mock_post.return_value = mock_response
    result = run_small_inference(dummy_image_frame, "a dog on a couch")
    assert result == "no"

@patch('requests.post')
def test_inference_api_error(mock_post, dummy_image_frame):
    """Test REST API returns error status code."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response
    result = run_small_inference(dummy_image_frame, "a cat on a couch")
    assert result is None

@patch('requests.post')
def test_inference_json_decode_error(mock_post, dummy_image_frame):
    """Test REST API returns invalid JSON in 'response'."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "not a json string"}
    mock_post.return_value = mock_response
    result = run_small_inference(dummy_image_frame, "a cat on a couch")
    assert result is None

@patch('requests.post')
def test_inference_missing_result_key(mock_post, dummy_image_frame):
    """Test REST API returns JSON missing 'result' key."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": json.dumps({"unexpected": True})}
    mock_post.return_value = mock_response
    result = run_small_inference(dummy_image_frame, "a cat on a couch")
    assert result is None

@patch('requests.post', side_effect=requests.RequestException("API error"))
def test_inference_request_exception(mock_post, dummy_image_frame):
    """Test REST API raises a RequestException."""
    result = run_small_inference(dummy_image_frame, "a cat on a couch")
    assert result is None