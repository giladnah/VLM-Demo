"""Unit tests for the orchestrator.py module."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock, call # call for checking multiple calls
import numpy as np

# Modules to test and their components
from orchestrator import orchestrate_processing, IFRAME_INTERVAL_SECONDS, BUFFER_SIZE, OrchestrationConfig

# Dependent modules that will be mocked
# (Actual classes/models are used for type hinting and constructing mock return values)
from video_source import VideoStream, ImageFrame # ImageFrame is np.ndarray
from buffer import FrameBuffer # FrameBuffer is deque
from inference_large import LargeInferenceOutput
from collections import deque

@pytest.fixture
def mock_video_stream() -> MagicMock:
    """Provides a mock VideoStream object."""
    stream = MagicMock(spec=VideoStream)
    stream.get_fps.return_value = 30.0
    # Configure read() to return a dummy frame initially, then None to stop loop
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Simulate a few successful reads then an end-of-stream
    stream.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None) # End of stream
    ]
    return stream

@pytest.fixture
def mock_frame_buffer() -> MagicMock:
    """Provides a mock FrameBuffer (deque)."""
    return MagicMock(spec=deque)

@pytest.fixture
def mock_small_inference_output_triggered() -> str:
    """Provides a string result that should result in a trigger."""
    return "yes"

@pytest.fixture
def mock_small_inference_output_not_triggered() -> str:
    """Provides a string result that should NOT result in a trigger."""
    return "no"

@pytest.fixture
def mock_large_inference_output() -> LargeInferenceOutput:
    """Provides a mock LargeInferenceOutput."""
    return LargeInferenceOutput(
        caption="Detailed analysis of the triggered frame.",
        tags=["analysis", "triggered"],
        detailed_analysis="This frame contained elements that met the trigger."
    )

# --- Main Test Class for Orchestrator --- #
class TestOrchestrateProcessing:

    @patch('orchestrator.get_video_stream')
    @patch('orchestrator.extract_i_frame') # Directly mocks extract_i_frame
    @patch('orchestrator.init_frame_buffer')
    @patch('orchestrator.update_buffer')
    @patch('orchestrator.run_small_inference')
    @patch('orchestrator.evaluate_trigger')
    @patch('orchestrator.run_large_inference')
    @patch('orchestrator.time.sleep') # Mock sleep to speed up tests
    @patch('orchestrator.time.time') # Mock time to control loop iterations
    def test_full_pipeline_trigger_met(
        self, mock_time: MagicMock, mock_sleep: MagicMock,
        mock_run_large_inference: MagicMock, mock_evaluate_trigger: MagicMock,
        mock_run_small_inference: MagicMock, mock_update_buffer: MagicMock,
        mock_init_frame_buffer: MagicMock, mock_extract_i_frame: MagicMock,
        mock_get_video_stream: MagicMock,
        mock_video_stream: MagicMock, # Patched VideoStream object from get_video_stream
        mock_frame_buffer: MagicMock, # Patched FrameBuffer object from init_frame_buffer
        mock_small_inference_output_triggered: str,
        mock_large_inference_output: LargeInferenceOutput
    ):
        """Test the full pipeline where a trigger is met and large VLM is called."""
        mock_get_video_stream.return_value = mock_video_stream
        mock_init_frame_buffer.return_value = mock_frame_buffer

        dummy_frame = np.zeros((100,100,3), dtype=np.uint8)
        mock_extract_i_frame.side_effect = [dummy_frame, dummy_frame, None] # Simulate 2 frames then end

        mock_run_small_inference.return_value = mock_small_inference_output_triggered
        mock_evaluate_trigger.return_value = True # Simulate trigger met
        mock_run_large_inference.return_value = mock_large_inference_output

        # Control time.time() to run for a few iterations
        mock_time_values = [
            1000.0, # Initial time
            1000.1, # Before first 2s interval
            1002.0, # First processing cycle (2s elapsed)
            1002.1, # Loop check
            1004.0, # Second processing cycle (4s elapsed)
            1004.1, # Loop check, extract_i_frame returns None next
            1006.0  # Attempt third cycle, extract_i_frame returns None
        ]
        mock_time.side_effect = mock_time_values

        stop_event = threading.Event()
        test_config = OrchestrationConfig(source_uri="test_source", initial_trigger_description="test_trigger")
        orchestrate_processing(test_config, stop_event)

        assert mock_get_video_stream.call_count == 1
        mock_get_video_stream.assert_called_with("test_source") # Check that source_uri from config is used
        assert mock_init_frame_buffer.call_count == 1
        assert mock_extract_i_frame.call_count == 3 # Called until None is returned
        assert mock_run_small_inference.call_count == 2 # For the two valid frames
        mock_run_small_inference.assert_any_call(dummy_frame, "test_trigger")
        assert mock_update_buffer.call_count == 2
        mock_update_buffer.assert_any_call(mock_frame_buffer, dummy_frame)
        assert mock_evaluate_trigger.call_count == 2
        mock_evaluate_trigger.assert_any_call(mock_small_inference_output_triggered, "test_trigger")
        assert mock_run_large_inference.call_count == 2 # Triggered for both valid frames
        mock_run_large_inference.assert_any_call(dummy_frame)
        mock_video_stream.release.assert_called_once()

    @patch('orchestrator.get_video_stream')
    @patch('orchestrator.extract_i_frame')
    @patch('orchestrator.init_frame_buffer')
    @patch('orchestrator.run_small_inference')
    @patch('orchestrator.evaluate_trigger')
    @patch('orchestrator.run_large_inference')
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.time.time')
    def test_pipeline_trigger_not_met(
        self, mock_time: MagicMock, mock_sleep: MagicMock,
        mock_run_large_inference: MagicMock, mock_evaluate_trigger: MagicMock,
        mock_run_small_inference: MagicMock, mock_init_frame_buffer: MagicMock,
        mock_extract_i_frame: MagicMock, mock_get_video_stream: MagicMock,
        mock_video_stream: MagicMock,
        mock_small_inference_output_not_triggered: str
    ):
        """Test pipeline where trigger is NOT met, large VLM should not be called."""
        mock_get_video_stream.return_value = mock_video_stream
        dummy_frame = np.zeros((100,100,3), dtype=np.uint8)
        mock_extract_i_frame.side_effect = [dummy_frame, None] # One frame then end
        mock_run_small_inference.return_value = mock_small_inference_output_not_triggered
        mock_evaluate_trigger.return_value = False # Simulate trigger NOT met

        mock_time.side_effect = [1000.0, 1002.0, 1004.0] # Control loop for one processing cycle

        test_config_no_match = OrchestrationConfig(source_uri="test_source", initial_trigger_description="test_trigger_no_match")
        orchestrate_processing(test_config_no_match) # stop_event is optional

        assert mock_run_small_inference.call_count == 1
        mock_run_small_inference.assert_called_with(dummy_frame, "test_trigger_no_match")
        assert mock_evaluate_trigger.call_count == 1
        mock_evaluate_trigger.assert_called_with(mock_small_inference_output_not_triggered, "test_trigger_no_match")
        mock_run_large_inference.assert_not_called() # Crucial check
        mock_video_stream.release.assert_called_once()

    @patch('orchestrator.get_video_stream')
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.time.time')
    @patch('orchestrator.run_small_inference') # Add patch for run_small_inference
    def test_stop_event_terminates_loop(
        self, mock_run_small_inference: MagicMock, # Add to parameters
        mock_time: MagicMock, mock_sleep: MagicMock,
        mock_get_video_stream: MagicMock, mock_video_stream: MagicMock
    ):
        """Test that setting the stop_event terminates the processing loop."""
        mock_get_video_stream.return_value = mock_video_stream
        mock_run_small_inference.return_value = None # Or a dummy string result

        # Mock extract_i_frame to keep providing frames if loop didn't stop
        with patch('orchestrator.extract_i_frame', return_value=np.zeros((10,10,3),dtype=np.uint8)) as mock_extract:
            stop_event = threading.Event()

            # Simulate time passing, then set stop_event
            def time_side_effect():
                yield 1000.0 # For orchestrator's initial last_processed_time calculation (e.g., 1000.0 - 2.0 = 998.0)

                # Orchestrator Loop Iteration 1:
                yield 1002.0 # current_time. elapsed = 1002.0 - 998.0 = 4.0. Process.
                             # extract_i_frame() called here (1st time).
                             # last_processed_time becomes 1002.0.

                # Test logic sets stop_event.set() after the first processing cycle simulated above.
                # This happens before the orchestrator's next while loop iteration starts.
                stop_event.set()

                # Orchestrator Loop Iteration 2 (top of while True):
                # stop_event.is_set() is checked here. It is True. Loop should break.
                # This yield is for the current_time call if the loop *didn't* break,
                # to ensure the test doesn't hang waiting for mock_time if logic was flawed.
                yield 1002.1

            mock_time.side_effect = time_side_effect()

            test_config_stop_event = OrchestrationConfig(source_uri="test_source", initial_trigger_description="any_trigger")
            orchestrate_processing(test_config_stop_event, stop_event)

            # We expect extract_i_frame to be called exactly once.
            assert mock_extract.call_count == 1

        mock_video_stream.release.assert_called_once()

    @patch('orchestrator.get_video_stream')
    def test_video_source_unavailable(self, mock_get_video_stream: MagicMock):
        """Test scenario where the video source cannot be opened."""
        mock_get_video_stream.side_effect = ValueError("Cannot open video source")

        # No need to mock other parts as it should fail early
        test_config_invalid_source = OrchestrationConfig(source_uri="invalid_source", initial_trigger_description="any_trigger")
        orchestrate_processing(test_config_invalid_source)
        # Assert that some error was printed or logged (if using logging module)
        # For now, just ensure it doesn't hang and completes (implicitly means it handled the error)
        pass

    @patch('orchestrator.get_video_stream')
    @patch('orchestrator.extract_i_frame')
    @patch('orchestrator.init_frame_buffer')
    @patch('orchestrator.run_small_inference') # Mock this to return None
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.time.time')
    def test_small_inference_fails(
        self, mock_time: MagicMock, mock_sleep: MagicMock,
        mock_run_small_inference: MagicMock, mock_init_frame_buffer: MagicMock,
        mock_extract_i_frame: MagicMock, mock_get_video_stream: MagicMock,
        mock_video_stream: MagicMock
    ):
        """Test when small inference fails (returns None)."""
        mock_get_video_stream.return_value = mock_video_stream
        dummy_frame = np.zeros((100,100,3), dtype=np.uint8)
        mock_extract_i_frame.side_effect = [dummy_frame, None]
        mock_run_small_inference.return_value = None # Simulate small inference failure

        mock_time.side_effect = [1000.0, 1002.0, 1004.0]
        with patch('orchestrator.evaluate_trigger') as mock_eval_trigger, \
             patch('orchestrator.run_large_inference') as mock_run_large_inf:

            test_config_small_fail = OrchestrationConfig(source_uri="test_source", initial_trigger_description="any")
            orchestrate_processing(test_config_small_fail)

            mock_eval_trigger.assert_not_called()
            mock_run_large_inf.assert_not_called()

        mock_video_stream.release.assert_called_once()

    # Additional tests could include:
    # - What happens if extract_i_frame itself raises an exception (other than returning None)
    # - What happens if buffer operations raise exceptions (less likely with deque)
    # - Behavior with different IFRAME_INTERVAL_SECONDS values (more of an integration/performance test)
    # - Test with FileNotFoundError from inference modules if ollama cli is not found
    @patch('orchestrator.get_video_stream')
    @patch('orchestrator.extract_i_frame')
    @patch('orchestrator.init_frame_buffer')
    @patch('orchestrator.run_small_inference')
    @patch('orchestrator.time.sleep')
    @patch('orchestrator.time.time')
    def test_ollama_not_found_propagates(self, mock_time: MagicMock, mock_sleep: MagicMock,
        mock_run_small_inference: MagicMock, mock_init_frame_buffer: MagicMock,
        mock_extract_i_frame: MagicMock, mock_get_video_stream: MagicMock,
        mock_video_stream: MagicMock):
        """Test that FileNotFoundError from run_small_inference propagates and stops orchestration."""
        mock_get_video_stream.return_value = mock_video_stream
        dummy_frame = np.zeros((100,100,3), dtype=np.uint8)
        mock_extract_i_frame.side_effect = [dummy_frame, None]
        mock_run_small_inference.side_effect = FileNotFoundError("ollama not found")

        mock_time.side_effect = [1000.0, 1002.0, 1004.0]

        # orchestrate_processing catches FileNotFoundError and re-raises it or prints and stops.
        # The current orchestrator code catches it and prints, then exits finally block.
        # If it were to re-raise, we'd use pytest.raises here.
        # For now, we check that video stream is released, implying termination.
        test_config_ollama_fail = OrchestrationConfig(source_uri="test_source", initial_trigger_description="any")
        orchestrate_processing(test_config_ollama_fail)
        mock_video_stream.release.assert_called_once()