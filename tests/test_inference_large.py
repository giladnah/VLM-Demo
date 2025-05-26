"""Unit tests for the inference_large.py module."""

import pytest
import numpy as np
import cv2 # For type consistency if needed, though imwrite is mocked
import json
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock
from typing import List, Union

from inference_large import (
    run_large_inference,
    LargeInferenceOutput,
    OllamaError, # Assuming same OllamaError is used or defined
    MODEL_NAME_DEFAULT_LARGE,
    ImageFrame
)

@pytest.fixture
def dummy_image_frame_large() -> ImageFrame:
    """Provides a consistent dummy image frame for large inference tests."""
    return np.zeros((200, 200, 3), dtype=np.uint8)

@pytest.fixture
def mock_temp_file_path_large() -> str:
    """Returns a dummy path for the temporary image file for large inference tests."""
    return "/tmp/dummy_test_image_large.png"

class TestRunLargeInference:

    @patch('inference_large.os.remove')
    @patch('inference_large.os.path.exists')
    @patch('inference_large.subprocess.run')
    @patch('inference_large.cv2.imwrite')
    @patch('inference_large.tempfile.NamedTemporaryFile')
    def test_successful_inference_single_frame(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame_large: ImageFrame,
        mock_temp_file_path_large: str
    ):
        """Test successful large inference with a single image frame."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path_large
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        expected_output_dict = {
            "caption": "A test image with a green circle.",
            "tags": ["test", "circle", "green"],
            "detailed_analysis": "This is a detailed analysis of the test image."
        }
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(expected_output_dict), stderr=''
        )

        result = run_large_inference(dummy_image_frame_large)

        assert result is not None
        assert isinstance(result, LargeInferenceOutput)
        assert result.caption == expected_output_dict["caption"]
        assert result.tags == expected_output_dict["tags"]
        assert result.detailed_analysis == expected_output_dict["detailed_analysis"]

        mock_named_temp_file.assert_called_once_with(suffix=".png", delete=False)
        mock_imwrite.assert_called_once_with(mock_temp_file_path_large, dummy_image_frame_large)
        mock_subprocess_run.assert_called_once()
        cmd_args = mock_subprocess_run.call_args[0][0]
        assert f"<img>{mock_temp_file_path_large}</img>" in cmd_args[3]
        mock_os_remove.assert_called_once_with(mock_temp_file_path_large)

    @patch('inference_large.os.remove')
    @patch('inference_large.os.path.exists')
    @patch('inference_large.subprocess.run')
    @patch('inference_large.cv2.imwrite')
    @patch('inference_large.tempfile.NamedTemporaryFile')
    def test_successful_inference_list_of_frames(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame_large: ImageFrame,
        mock_temp_file_path_large: str
    ):
        """Test successful large inference with a list of frames (uses the last one)."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path_large
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        expected_output_dict = {"caption": "Last frame analysis", "tags": ["last"], "detailed_analysis": "Processed last."}
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(expected_output_dict), stderr=''
        )

        older_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frames_list: List[ImageFrame] = [older_frame, dummy_image_frame_large]
        result = run_large_inference(frames_list)

        assert result is not None
        assert result.caption == expected_output_dict["caption"]
        # Check that cv2.imwrite was called with the LAST frame from the list
        # np.array_equal is needed for comparing numpy arrays if not comparing object id
        assert mock_imwrite.call_args[0][0] == mock_temp_file_path_large
        assert np.array_equal(mock_imwrite.call_args[0][1], dummy_image_frame_large)
        mock_os_remove.assert_called_once_with(mock_temp_file_path_large)

    def test_empty_list_input(self, dummy_image_frame_large: ImageFrame):
        """Test run_large_inference with an empty list of frames."""
        result = run_large_inference([])
        assert result is None

    def test_invalid_frames_type_input(self):
        """Test run_large_inference with an invalid type for the frames argument."""
        result = run_large_inference(frames="not_an_image_or_list") # type: ignore
        assert result is None

    @patch('inference_large.os.remove')
    @patch('inference_large.os.path.exists')
    @patch('inference_large.subprocess.run')
    @patch('inference_large.cv2.imwrite')
    @patch('inference_large.tempfile.NamedTemporaryFile')
    def test_ollama_cli_error_large_model(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame_large: ImageFrame,
        mock_temp_file_path_large: str
    ):
        """Test Ollama CLI error with the large model."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path_large
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout='', stderr='Large model error'
        )
        result = run_large_inference(dummy_image_frame_large)
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path_large)

    # Add more tests similar to TestRunSmallInference for:
    # - CLI timeout
    # - CLI not found (FileNotFoundError)
    # - Image write failure (cv2.imwrite returns False)
    # - JSON decode error
    # - Pydantic validation error
    # - Unexpected exceptions
    # - Temp file cleanup scenarios (e.g., if file doesn't exist for removal)

    @patch('inference_large.os.remove')
    @patch('inference_large.os.path.exists')
    @patch('inference_large.subprocess.run')
    @patch('inference_large.cv2.imwrite')
    @patch('inference_large.tempfile.NamedTemporaryFile')
    def test_json_decode_error_large_model(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame_large: ImageFrame,
        mock_temp_file_path_large: str
    ):
        """Test handling of invalid JSON output from the large model."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path_large
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='Malformed JSON String', stderr=''
        )
        result = run_large_inference(dummy_image_frame_large)
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path_large)

    @patch('inference_large.os.remove')
    @patch('inference_large.os.path.exists')
    @patch('inference_large.subprocess.run')
    @patch('inference_large.cv2.imwrite')
    @patch('inference_large.tempfile.NamedTemporaryFile')
    def test_pydantic_validation_error_large_model(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame_large: ImageFrame,
        mock_temp_file_path_large: str
    ):
        """Test output that is JSON but doesn't match LargeInferenceOutput schema."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path_large
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True
        invalid_json_output = json.dumps({"wrong_caption_key": "A caption", "tags": []}) # Missing detailed_analysis
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=invalid_json_output, stderr=''
        )
        result = run_large_inference(dummy_image_frame_large)
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path_large)