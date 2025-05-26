"""Unit tests for the inference_small.py module."""

import pytest
import numpy as np
import cv2 # Though cv2.imwrite is mocked, importing for type consistency if needed
import json
import subprocess
import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

# Assuming inference_small.py is in the parent directory or PYTHONPATH is set up
# For a typical project structure, you might need to adjust imports:
# from ..inference_small import run_small_inference, SmallInferenceOutput, OllamaError, MODEL_NAME_DEFAULT
# For now, using a simpler import path assuming flat structure or correct PYTHONPATH for tests
from inference_small import run_small_inference, OllamaError, MODEL_NAME_DEFAULT, ImageFrame

@pytest.fixture
def dummy_image_frame() -> ImageFrame:
    """Provides a consistent dummy image frame (NumPy array) for tests."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_temp_file_path() -> str:
    """Returns a dummy path for the temporary image file."""
    return "/tmp/dummy_test_image.jpg"

# Using a class to group tests for inference_small module
class TestRunSmallInference:

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_successful_inference(
        self,
        mock_named_temp_file: MagicMock,
        mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock,
        dummy_image_frame: ImageFrame,
        mock_temp_file_path: str
    ):
        """Test a successful inference call."""
        # Setup mocks for tempfile context manager
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True # So os.remove is attempted

        expected_output_dict = {"result": "yes"}
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=['ollama', 'run', MODEL_NAME_DEFAULT, 'prompt', '--format', 'json'],
            returncode=0,
            stdout=json.dumps(expected_output_dict),
            stderr=''
        )

        result = run_small_inference(dummy_image_frame, "a cat on a couch")

        assert result is not None
        assert result == "yes"

        mock_named_temp_file.assert_called_once_with(suffix=".jpg", delete=False)
        mock_imwrite.assert_called_once_with(mock_temp_file_path, dummy_image_frame)
        mock_subprocess_run.assert_called_once()
        # Check the command parts (more robustly if prompt varies significantly)
        cmd_args = mock_subprocess_run.call_args[0][0]
        assert cmd_args[0] == "ollama"
        assert cmd_args[1] == "run"
        assert cmd_args[2] == MODEL_NAME_DEFAULT
        assert f"<img>{mock_temp_file_path}</img>" in cmd_args[3]
        assert "a cat on a couch" in cmd_args[3]
        assert cmd_args[4] == "--format"
        assert cmd_args[5] == "json"

        mock_os_path_exists.assert_called_with(mock_temp_file_path)
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_ollama_cli_error_return_code(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test Ollama CLI returning a non-zero exit code."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout='', stderr='Ollama internal error'
        )

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_ollama_cli_timeout(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test Ollama CLI command timing out."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=30)

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.subprocess.run') # Only need to mock subprocess for this
    @patch('inference_small.cv2.imwrite', return_value=True) # Assume imwrite works
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_ollama_cli_not_found(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test scenario where Ollama CLI executable is not found."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

        mock_subprocess_run.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'ollama'")

        with pytest.raises(FileNotFoundError):
            run_small_inference(dummy_image_frame, "test trigger")
        # Check if cleanup was attempted for the temp file (it should be, via finally)
        # This requires mocking os.remove and os.path.exists as well if we want to assert them.

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run') # Not called
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_image_write_failure(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test failure during cv2.imwrite."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = False # Simulate imwrite failure
        mock_os_path_exists.return_value = True # Assume file was created before write failed

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_subprocess_run.assert_not_called()
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_json_decode_error(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test handling of invalid JSON output from Ollama."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='This is not JSON', stderr=''
        )

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_pydantic_validation_error(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_os_path_exists: MagicMock,
        mock_os_remove: MagicMock, dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test output that is JSON but doesn't match SmallInferenceOutput schema."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True

        invalid_json_output = json.dumps({"wrong_key": True, "other_labels": []})
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=invalid_json_output, stderr=''
        )

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists')
    @patch('inference_small.json.loads') # Mock json.loads to raise an unexpected error
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_unexpected_exception_handling(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock, mock_json_loads: MagicMock,
        mock_os_path_exists: MagicMock, mock_os_remove: MagicMock,
        dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test that a generic unexpected exception is caught and returns None."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True
        mock_os_path_exists.return_value = True # File exists for cleanup check

        # Mock subprocess.run to return a seemingly valid result
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps({"detection": True, "labels": []}), stderr=''
        )
        # Make json.loads raise an unexpected error (e.g., a TypeError simulation)
        mock_json_loads.side_effect = TypeError("Unexpected internal error in json.loads")

        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None
        mock_os_remove.assert_called_once_with(mock_temp_file_path)

    @patch('inference_small.os.remove')
    @patch('inference_small.os.path.exists', return_value=False) # File does not exist for removal
    @patch('inference_small.subprocess.run')
    @patch('inference_small.cv2.imwrite')
    @patch('inference_small.tempfile.NamedTemporaryFile')
    def test_temp_file_already_deleted_or_not_created_for_cleanup(
        self,
        mock_named_temp_file: MagicMock, mock_imwrite: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_os_path_exists_after_error: MagicMock, # Renamed for clarity
        mock_os_remove_after_error: MagicMock, # Renamed for clarity
        dummy_image_frame: ImageFrame, mock_temp_file_path: str
    ):
        """Test that os.remove is not called if temp file doesn't exist at cleanup."""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = mock_temp_file_path
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_imwrite.return_value = True # Successful write

        # Simulate an error after file write, e.g., Ollama CLI error
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout='', stderr='Ollama error'
        )

        # mock_os_path_exists is already patched at class/method level,
        # its return_value is False, so os.remove should not be called.
        result = run_small_inference(dummy_image_frame, "test trigger")
        assert result is None # Due to subprocess error

        # os.path.exists(temp_image_path) is called in the finally block.
        mock_os_path_exists_after_error.assert_called_with(mock_temp_file_path)
        mock_os_remove_after_error.assert_not_called() # Because os.path.exists returned False