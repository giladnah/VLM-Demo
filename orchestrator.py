"""
Orchestrator for the VLM Camera Service video processing pipeline.

Handles video stream acquisition, frame grabbing, buffering, inference (small and large models), and trigger logic using threading and multiprocessing.
"""

import time
import threading
from typing import Optional, List, Any
import os
import subprocess
import cv2
import queue
import multiprocessing
import asyncio
import traceback
from inference.ollama_engine import OllamaError
from inference.openai_engine import OpenAIError

from video_source import get_video_stream, extract_i_frame, VideoStream, ImageFrame
from buffer import init_frame_buffer, update_buffer, FrameBuffer
from inference.unified import run_unified_inference
from trigger import evaluate_trigger
from config import get_config

BUFFER_SIZE = 10
FRAME_GRAB_INTERVAL = 1/30
BUFFER_GRAB_INTERVAL = 1/5  # Example: update buffer every 0.2s (5 FPS), adjust as needed

# --- Batch mode config (set in start) ---
BATCH_MODE_ENABLED = False
BATCH_SIZE = 1
BATCH_MIN_TIME_DIFF = 0.0

class OrchestrationConfig:
    """
    Configuration for an orchestration task, allowing dynamic updates.
    """
    def __init__(self, source_uri: str, initial_trigger_description: str):
        self.source_uri = source_uri
        self._trigger_description = initial_trigger_description
        self._lock = threading.Lock()

    def get_trigger_description(self) -> str:
        """
        Returns the current trigger description (thread-safe).

        Returns:
            str: The current trigger description.
        """
        with self._lock:
            return self._trigger_description

    def set_trigger_description(self, new_description: str) -> None:
        """
        Updates the trigger description if it has changed (thread-safe).

        Args:
            new_description (str): The new trigger description.
        """
        with self._lock:
            if self._trigger_description != new_description:
                self._trigger_description = new_description
                print(f"[OrchestrationConfig] INFO: Trigger description updated to: '{new_description}'")
            else:
                print(f"[OrchestrationConfig] INFO: Trigger description is already '{new_description}'. No change made.")

# --- Top-level inference worker process for multiprocessing ---
def inference_worker_process(input_queue, output_queue, stop_event, config):
    """
    Worker process for running small and large inference in a separate process.
    Uses the unified inference system (run_unified_inference).
    Old run_small_inference/run_large_inference are deprecated.
    """
    import cv2
    import os
    # --- Batch mode config from orchestrator ---
    batch_mode_enabled = config.get("batch_mode_enabled", False)
    batch_size = config.get("batch_size", 1)
    # Debug save frames feature
    debug_save_frames = False
    try:
        debug_save_frames = get_config().get('debug_save_frames', False)
    except Exception:
        pass
    # Local buffer for batch mode
    rolling_frame_buffer = []
    while not stop_event.is_set():
        engine_err = None  # Ensure always defined
        e = None
        try:
            frame, trigger_description, timestamp = input_queue.get(timeout=1)
        except Exception:
            continue
        # Maintain rolling buffer for batch mode
        if batch_mode_enabled and batch_size > 1:
            rolling_frame_buffer.append(frame)
            if len(rolling_frame_buffer) > batch_size:
                rolling_frame_buffer = rolling_frame_buffer[-batch_size:]
        # Run small inference (always single frame)
        print(f"[Orchestrator] [InferenceProcess] Running small inference for trigger: '{trigger_description}' (unified engine)")
        try:
            small_result = asyncio.run(run_unified_inference(
                frame,
                trigger_description,
                model_type="small"
            ))
            small_inference_out = small_result.result
            # Debug: Save latest small inference frame
            if debug_save_frames:
                try:
                    cv2.imwrite("small_infer_latest.jpg", frame)  # type: ignore[attr-defined]
                except Exception as e_inner:
                    print(f"[DEBUG] Failed to save small inference frame: {e_inner}")
        except (OllamaError, OpenAIError) as err:
            print(f"[Orchestrator] [InferenceProcess] ENGINE ERROR (small): {err}")
            traceback.print_exc()
            small_inference_out = None
            small_result = None
            engine_err = err
        except Exception as err:
            print(f"[Orchestrator] [InferenceProcess] UNEXPECTED ERROR (small): {err}")
            traceback.print_exc()
            small_inference_out = None
            small_result = None
            e = err
        # Immediately send small inference result
        result_payload = {
            "result": small_inference_out,
            "timestamp": time.time(),
            "frame": frame,
            "trigger": trigger_description,
            "raw_result": getattr(small_result, 'raw_model_output', None) if small_inference_out else None,
            "error": str(engine_err) if engine_err is not None else (str(e) if e is not None else None)
        }
        output_queue.put(result_payload)
        # If trigger is met, run large inference and send a second result
        if small_inference_out == "yes":
            print(f"[Orchestrator] [InferenceProcess] Trigger MET. Running large inference (unified engine)")
            try:
                if batch_mode_enabled and batch_size > 1:
                    batch_frames = list(rolling_frame_buffer)
                    print(f"[Orchestrator] [InferenceProcess] Sending batch of {len(batch_frames)} frames for large inference.")
                    # Debug: Save all batch frames
                    if debug_save_frames:
                        for idx, f in enumerate(batch_frames):
                            try:
                                cv2.imwrite(f"large_infer_batch_{idx}.jpg", f)  # type: ignore[attr-defined]
                            except Exception as e:
                                print(f"[DEBUG] Failed to save large batch frame {idx}: {e}")
                    large_result = asyncio.run(run_unified_inference(
                        batch_frames,
                        trigger_description,
                        model_type="large"
                    ))
                else:
                    # Debug: Save single large inference frame
                    if debug_save_frames:
                        try:
                            cv2.imwrite("large_infer_latest.jpg", frame)  # type: ignore[attr-defined]
                        except Exception as e:
                            print(f"[DEBUG] Failed to save large inference frame: {e}")
                    large_result = asyncio.run(run_unified_inference(
                        frame,
                        trigger_description,
                        model_type="large"
                    ))
                print(f"[Orchestrator] [InferenceProcess] Large inference result - Result: '{large_result.result}', Analysis: '{large_result.detailed_analysis[:100]}...'")
                # Send a second payload with large inference result
                result_payload_large = {
                    "result": small_inference_out,  # Always include small result for context
                    "timestamp": time.time(),
                    "frame": frame,
                    "trigger": trigger_description,
                    "large_inference": {
                        "result": large_result.result,
                        "detailed_analysis": large_result.detailed_analysis,
                        "confidence_score": large_result.confidence_score,
                        "raw_model_output": large_result.raw_model_output,
                        "processing_time_ms": large_result.processing_time_ms,
                        "engine_name": large_result.engine_name
                    }
                }
                output_queue.put(result_payload_large)
            except (OllamaError, OpenAIError) as engine_err:
                print(f"[Orchestrator] [InferenceProcess] ENGINE ERROR (large): {engine_err}")
                traceback.print_exc()
                result_payload_large = {
                    "result": small_inference_out,
                    "timestamp": time.time(),
                    "frame": frame,
                    "trigger": trigger_description,
                    "large_inference": None,
                    "error": str(engine_err)
                }
                output_queue.put(result_payload_large)
            except Exception as e:
                print(f"[Orchestrator] [InferenceProcess] UNEXPECTED ERROR (large): {e}")
                traceback.print_exc()
                result_payload_large = {
                    "result": small_inference_out,
                    "timestamp": time.time(),
                    "frame": frame,
                    "trigger": trigger_description,
                    "large_inference": None,
                    "error": str(e)
                }
                output_queue.put(result_payload_large)
    print("[Orchestrator] [InferenceProcess] Exiting.")

class Orchestrator:
    """
    Singleton orchestrator for the video processing pipeline.
    Manages video stream, frame grabbing, buffering, inference, and result handling.
    """
    def __init__(self) -> None:
        self.latest_small_inference_frame_path = "latest_small_inference.jpg"
        self.latest_small_inference_lock = threading.Lock()
        self.latest_small_inference_result = None
        self.latest_small_inference_result_str = None
        self.latest_small_inference_result_lock = threading.Lock()
        self.latest_large_inference_result = None
        self.latest_large_inference_result_lock = threading.Lock()
        self.new_small_result_available = False
        self.new_large_result_available = False
        self.result_status_lock = threading.Lock()
        self._processing_thread = None
        self._stop_event = None
        self._config = None
        self._buffer_lock = threading.Lock()
        self._frame_buffer = None
        self._video_stream = None
        self._inference_process = None
        self._inference_stop_event = None
        self._inference_input_queue = None
        self._inference_output_queue = None
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        self.last_inference_frame = None

    def start(self, config: OrchestrationConfig) -> None:
        """
        Starts the orchestration process with the given configuration.
        """
        self.stop()  # Stop any existing orchestration
        self._config = config
        self._stop_event = threading.Event()
        self._inference_stop_event = multiprocessing.Event()
        self._inference_input_queue = multiprocessing.Queue(maxsize=2)
        self._inference_output_queue = multiprocessing.Queue(maxsize=2)
        self._processing_thread = threading.Thread(target=self._orchestrate_processing, daemon=True)
        # --- Batch mode config logic ---
        global BUFFER_SIZE, BUFFER_GRAB_INTERVAL, BATCH_MODE_ENABLED, BATCH_SIZE, BATCH_MIN_TIME_DIFF
        config_yaml = get_config()
        large_model_cfg = config_yaml.get('large_model', {})
        engine = large_model_cfg.get('engine', 'ollama').lower()
        batch_cfg = large_model_cfg.get('batch_inference', {})
        if batch_cfg.get('enabled', False):
            if engine == 'openai':
                BATCH_MODE_ENABLED = True
                BATCH_SIZE = int(batch_cfg.get('batch_size', 5))
                BATCH_MIN_TIME_DIFF = float(batch_cfg.get('min_time_diff_seconds', 2))
                BUFFER_SIZE = max(BATCH_SIZE, 10)
                BUFFER_GRAB_INTERVAL = BATCH_MIN_TIME_DIFF
                print(f"[Orchestrator] INFO: Batch mode enabled for large inference (OpenAI). Batch size: {BATCH_SIZE}, min_time_diff: {BATCH_MIN_TIME_DIFF}s")
            else:
                print("[Orchestrator] WARNING: Batch mode enabled but engine is not OpenAI. Ignoring batch mode.")
                BATCH_MODE_ENABLED = False
                BATCH_SIZE = 1
                BATCH_MIN_TIME_DIFF = 0.0
                BUFFER_SIZE = 10
                BUFFER_GRAB_INTERVAL = 1/5
        else:
            BATCH_MODE_ENABLED = False
            BATCH_SIZE = 1
            BATCH_MIN_TIME_DIFF = 0.0
            BUFFER_SIZE = 10
            BUFFER_GRAB_INTERVAL = 1/5
        config_dict = {
            "small_model_name": getattr(config, "small_model_name", None),
            "small_ollama_server": getattr(config, "small_ollama_server", None),
            "large_model_name": getattr(config, "large_model_name", None),
            "large_ollama_server": getattr(config, "large_ollama_server", None),
            # Pass batch mode settings to inference worker
            "batch_mode_enabled": BATCH_MODE_ENABLED,
            "batch_size": BATCH_SIZE,
            "batch_min_time_diff": BATCH_MIN_TIME_DIFF,
        }
        self._inference_process = multiprocessing.Process(
            target=inference_worker_process,
            args=(self._inference_input_queue, self._inference_output_queue, self._inference_stop_event, config_dict),
            daemon=True
        )
        self._processing_thread.start()
        self._inference_process.start()

    def stop(self) -> None:
        """
        Stops the orchestration process and cleans up resources.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._inference_stop_event is not None:
            self._inference_stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
        if self._inference_process is not None:
            self._inference_process.join(timeout=5)
        self._processing_thread = None
        self._inference_process = None
        self._stop_event = None
        self._inference_stop_event = None
        self._inference_input_queue = None
        self._inference_output_queue = None
        if self._video_stream:
            self._video_stream.release()
            self._video_stream = None

    def _frame_grabber(self, video_stream: VideoStream, frame_buffer: FrameBuffer, buffer_lock: threading.Lock, stop_event: threading.Event) -> None:
        """
        Grabs frames from the video stream and updates the buffer in a separate thread.

        Args:
            video_stream (VideoStream): The video stream object.
            frame_buffer (FrameBuffer): The frame buffer.
            buffer_lock (threading.Lock): Lock for buffer access.
            stop_event (threading.Event): Event to signal stopping.
        """
        print("[FrameGrabber] INFO: Starting frame grabber thread.")
        last_buffer_update = time.time()
        while not stop_event.is_set():
            ret, frame = video_stream.read()
            if not ret:
                print("[FrameGrabber] INFO: End of stream or failed to read frame.")
                break
            # Update current_frame for inference
            with self.current_frame_lock:
                self.current_frame = frame
            # Update buffer at BUFFER_GRAB_INTERVAL
            now = time.time()
            if now - last_buffer_update >= BUFFER_GRAB_INTERVAL:
                with buffer_lock:
                    if len(frame_buffer) >= BUFFER_SIZE:
                        frame_buffer.popleft()  # Leaky: drop oldest
                    frame_buffer.append(frame)
                last_buffer_update = now
            time.sleep(FRAME_GRAB_INTERVAL)
        print("[FrameGrabber] INFO: Frame grabber thread exiting.")

    def _orchestrate_processing(self) -> None:
        """
        Main orchestration loop: connects to video source, manages frame grabbing, inference, and result handling.
        """
        try:
            source_uri = self._config.source_uri
            print(f"[Orchestrator] INFO: Attempting to connect to video source: {source_uri}")
            self._video_stream = get_video_stream(source_uri)
            print(f"[Orchestrator] INFO: Successfully connected to video source. FPS: {self._video_stream.get_fps()}")

            self._frame_buffer = init_frame_buffer(size=BUFFER_SIZE)
            print(f"[Orchestrator] INFO: Initialized frame buffer with size {BUFFER_SIZE}.")

            grabber_thread = threading.Thread(
                target=self._frame_grabber,
                args=(self._video_stream, self._frame_buffer, self._buffer_lock, self._stop_event),
                daemon=True
            )
            grabber_thread.start()

            while True:
                if self._stop_event and self._stop_event.is_set():
                    print("[Orchestrator] INFO: Stop event received, terminating processing loop.")
                    break

                current_trigger_description = self._config.get_trigger_description()
                with self.current_frame_lock:
                    current_frame = self.current_frame

                if current_frame is None:
                    print("[Orchestrator] INFO: No current frame available. Waiting for frames.")
                    time.sleep(0.1)
                    continue

                # Save the frame that will be sent to inference
                self.last_inference_frame = current_frame.copy()

                # Always update the latest inference input
                try:
                    if self._inference_input_queue is not None and not self._inference_input_queue.full():
                        self._inference_input_queue.put((self.last_inference_frame, current_trigger_description, time.time()))
                except Exception as e:
                    print(f"[Orchestrator] WARN: Could not queue frame for inference process: {e}")

                # Check for new inference results from the process
                if self._inference_output_queue is not None:
                    try:
                        while True:
                            result = self._inference_output_queue.get_nowait()
                            self._handle_inference_result(result)
                    except queue.Empty:
                        pass

                time.sleep(0.01)

        except ValueError as ve:
            print(f"[Orchestrator] ERROR: Configuration or source error - {ve}. Terminating.")
        except FileNotFoundError as fnfe:
            print(f"[Orchestrator] CRITICAL: Ollama CLI not found - {fnfe}. Ensure Ollama is installed and in PATH. Terminating.")
        except Exception as e:
            import traceback
            print(f"[Orchestrator] CRITICAL: An unexpected error occurred: {type(e).__name__} - {e}")
            print(f"[Orchestrator] Traceback: {traceback.format_exc()}")
        finally:
            if self._video_stream:
                print("[Orchestrator] INFO: Releasing video stream.")
                self._video_stream.release()
                self._video_stream = None
            print("[Orchestrator] INFO: Processing finished.")

    def _handle_inference_result(self, result: dict) -> None:
        """
        Handles inference results from the worker process and updates shared state.

        Args:
            result (dict): Inference result payload.
        """
        small_inference_out = result.get("result")
        frame = result.get("frame")
        trigger = result.get("trigger")
        # Always update small inference result if present
        if small_inference_out is not None:
            print(f"[Orchestrator] [Main] Small inference result - Result: {small_inference_out}")
            with self.latest_small_inference_result_lock:
                self.latest_small_inference_result = {
                    "result": small_inference_out,
                    "timestamp": time.time()
                }
                self.latest_small_inference_result_str = f"[Orchestrator] INFO: Small inference result - Result: {small_inference_out}"
            with self.result_status_lock:
                self.new_small_result_available = True
        # Only update large inference result if present in this payload
        if "large_inference" in result:
            large_inference_out = result["large_inference"]
            print(f"[Orchestrator] [Main] Large inference result - {large_inference_out}")
            with self.latest_large_inference_result_lock:
                self.latest_large_inference_result = large_inference_out
            with self.result_status_lock:
                self.new_large_result_available = True
        elif small_inference_out is None:
            print("[Orchestrator] [Main] WARN: Small inference failed or returned no result.")

    def get_latest_small_inference_frame(self) -> Optional[Any]:
        """
        Returns the latest frame used for small inference as a numpy array (in-memory).

        Returns:
            Optional[Any]: Latest frame or None.
        """
        return self.last_inference_frame

    def get_latest_small_inference_result(self, clear_status: bool = True) -> Optional[dict]:
        """
        Returns the latest small inference result.

        Args:
            clear_status (bool, optional): Whether to clear the new result status. Defaults to True.

        Returns:
            Optional[dict]: Latest result or None.
        """
        with self.latest_small_inference_result_lock:
            result = self.latest_small_inference_result
        if clear_status:
            with self.result_status_lock:
                self.new_small_result_available = False
        return result

    def get_latest_small_inference_result_str(self) -> Optional[str]:
        """
        Returns the latest small inference result as a string.

        Returns:
            Optional[str]: Latest result string or None.
        """
        with self.latest_small_inference_result_lock:
            return self.latest_small_inference_result_str

    def get_latest_large_inference_result(self, clear_status: bool = True) -> Optional[dict]:
        """
        Returns the latest large inference result.

        Args:
            clear_status (bool, optional): Whether to clear the new result status. Defaults to True.

        Returns:
            Optional[dict]: Latest result or None.
        """
        with self.latest_large_inference_result_lock:
            result = self.latest_large_inference_result
        if clear_status:
            with self.result_status_lock:
                self.new_large_result_available = False
        return result

    def get_results_status(self) -> dict:
        """
        Returns the status of new inference results.

        Returns:
            dict: Status of small and large inference results.
        """
        with self.result_status_lock:
            return {
                "small": self.new_small_result_available,
                "large": self.new_large_result_available
            }

# Singleton instance
orchestrator = Orchestrator()

if __name__ == '__main__':
    # This example demonstrates running the orchestrator in a background thread.
    # Prerequisites:
    # 1. All dependent modules (video_source, buffer, inference_*, trigger) are present.
    # 2. Ollama service is running with models (e.g., qwen2.5vl:3b, qwen2.5vl:7b) pulled.
    # 3. A video source: e.g., "test_video.mp4".
    #    If "test_video.mp4" does not exist, the video_source.py module's __main__
    #    block can create a dummy one if ffmpeg is installed and in PATH.
    #    Consider running `python video_source.py` once to ensure it exists.

    print("[Orchestrator Main] INFO: Starting orchestration test with OrchestrationConfig.")

    # Attempt to create a dummy video file if it doesn't exist (simplified from video_source.py)
    TEST_SOURCE_FILE = "test_video.mp4"
    if not os.path.exists(TEST_SOURCE_FILE):
        print(f"[Orchestrator Main] INFO: '{TEST_SOURCE_FILE}' not found. Attempting to create a dummy video.")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration=15:size=640x480:rate=5", TEST_SOURCE_FILE], # Increased duration for testing trigger changes
                check=True, capture_output=True, timeout=10
            )
            print(f"[Orchestrator Main] INFO: Dummy video '{TEST_SOURCE_FILE}' created successfully.")
        except Exception as e_ffmpeg:
            print(f"[Orchestrator Main] WARN: Could not create dummy video file using ffmpeg: {e_ffmpeg}.")

    initial_trigger = "a test pattern"
    config = OrchestrationConfig(source_uri=TEST_SOURCE_FILE, initial_trigger_description=initial_trigger)

    stop_processing_event = threading.Event()

    def orchestration_target():
        print(f"[Orchestrator Thread] INFO: Starting orchestrate_processing with source '{config.source_uri}' and initial trigger '{config.get_trigger_description()}'.")
        orchestrator.start(config) # Pass the config object
        print("[Orchestrator Thread] INFO: orchestrate_processing has finished.")

    processing_thread = threading.Thread(target=orchestration_target, daemon=True)

    print("[Orchestrator Main] INFO: Starting processing thread.")
    processing_thread.start()

    run_duration_seconds = 30
    change_trigger_after_seconds = 10
    new_trigger = "something else entirely"

    print(f"[Orchestrator Main] INFO: Orchestrator will run for approx {run_duration_seconds}s.")
    print(f"[Orchestrator Main] INFO: Initial trigger: '{config.get_trigger_description()}'")
    print(f"[Orchestrator Main] INFO: Trigger will be changed to '{new_trigger}' after {change_trigger_after_seconds}s.")
    print("[Orchestrator Main] INFO: Press Ctrl+C to stop earlier.")

    try:
        # Keep the main thread alive while the orchestrator runs, or use thread.join() directly
        start_time = time.time()
        trigger_changed = False
        while processing_thread.is_alive() and (time.time() - start_time) < run_duration_seconds:
            time.sleep(0.5) # Check periodically
            if not trigger_changed and (time.time() - start_time) >= change_trigger_after_seconds:
                print(f"[Orchestrator Main] INFO: Changing trigger description to '{new_trigger}'.")
                config.set_trigger_description(new_trigger)
                trigger_changed = True

        if processing_thread.is_alive():
             print(f"[Orchestrator Main] INFO: {run_duration_seconds}s elapsed. Signaling stop to orchestrator.")
        else:
             print(f"[Orchestrator Main] INFO: Orchestrator thread finished before {run_duration_seconds}s timeout.")

    except KeyboardInterrupt:
        print("[Orchestrator Main] INFO: Keyboard interrupt received. Signaling stop.")
    finally:
        if processing_thread.is_alive():
            stop_processing_event.set()
            print("[Orchestrator Main] INFO: Waiting for processing thread to join...")
            processing_thread.join(timeout=15) # Wait for graceful shutdown
            if processing_thread.is_alive():
                print("[Orchestrator Main] WARN: Processing thread did not join in time after stop signal.")
            else:
                print("[Orchestrator Main] INFO: Processing thread joined successfully.")
        else:
            print("[Orchestrator Main] INFO: Processing thread was already finished.")

    print("[Orchestrator Main] INFO: Orchestration test finished.")