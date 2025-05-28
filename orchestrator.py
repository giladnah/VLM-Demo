"""Module for orchestrating the video processing pipeline."""

import time
import threading
from typing import Optional, List # Required for List type hint if used explicitly
import os  # Fix linter error for os
import subprocess  # Fix linter error for subprocess

# Assuming ImageFrame type is consistent (np.ndarray)
from video_source import get_video_stream, extract_i_frame, VideoStream, ImageFrame
from buffer import init_frame_buffer, update_buffer, FrameBuffer
from inference_small import run_small_inference
from trigger import evaluate_trigger
from inference_large import run_large_inference, LargeInferenceOutput

# Constants based on PLANNING.md or derived for the orchestration logic
IFRAME_INTERVAL_SECONDS = 2.0
BUFFER_SIZE = 10 # As per PLANNING.md: "rolling buffer of 10 I-frames"
FRAME_GRAB_INTERVAL = 1/30  # 30 FPS (adjust as needed)

class OrchestrationConfig:
    """Configuration for an orchestration task, allowing dynamic updates."""
    def __init__(self, source_uri: str, initial_trigger_description: str):
        self.source_uri = source_uri # Typically set once at the start
        self._trigger_description = initial_trigger_description
        self._lock = threading.Lock() # To protect access to trigger_description

    def get_trigger_description(self) -> str:
        """Returns the current trigger description in a thread-safe manner."""
        with self._lock:
            return self._trigger_description

    def set_trigger_description(self, new_description: str) -> None:
        """Updates the trigger description in a thread-safe manner."""
        with self._lock:
            if self._trigger_description != new_description:
                self._trigger_description = new_description
                print(f"[OrchestrationConfig] INFO: Trigger description updated to: '{new_description}'")
            else:
                print(f"[OrchestrationConfig] INFO: Trigger description is already '{new_description}'. No change made.")

def frame_grabber(video_stream, frame_buffer, buffer_lock, stop_event):
    """
    Continuously reads frames from the video stream and fills the buffer in real time.
    """
    print("[FrameGrabber] INFO: Starting frame grabber thread.")
    while not stop_event.is_set():
        ret, frame = video_stream.read()
        if not ret:
            print("[FrameGrabber] INFO: End of stream or failed to read frame.")
            break
        with buffer_lock:
            frame_buffer.append(frame)
        time.sleep(FRAME_GRAB_INTERVAL)
    print("[FrameGrabber] INFO: Frame grabber thread exiting.")

def orchestrate_processing(
    config: OrchestrationConfig,
    stop_event: Optional[threading.Event] = None
) -> None:
    """
    Coordinates the video processing pipeline with real-time frame grabbing.
    """
    video_stream: Optional[VideoStream] = None
    buffer_lock = threading.Lock()
    try:
        source_uri = config.source_uri
        print(f"[Orchestrator] INFO: Attempting to connect to video source: {source_uri}")
        video_stream = get_video_stream(source_uri)
        print(f"[Orchestrator] INFO: Successfully connected to video source. FPS: {video_stream.get_fps()}")

        frame_buffer: FrameBuffer = init_frame_buffer(size=BUFFER_SIZE)
        print(f"[Orchestrator] INFO: Initialized frame buffer with size {BUFFER_SIZE}.")

        # Start frame grabber thread
        grabber_stop_event = stop_event or threading.Event()
        grabber_thread = threading.Thread(target=frame_grabber, args=(video_stream, frame_buffer, buffer_lock, grabber_stop_event), daemon=True)
        grabber_thread.start()

        last_processed_time = time.time() - IFRAME_INTERVAL_SECONDS

        while True:
            if stop_event and stop_event.is_set():
                print("[Orchestrator] INFO: Stop event received, terminating processing loop.")
                break

            current_trigger_description = config.get_trigger_description()
            current_time = time.time()
            elapsed_since_last_process = current_time - last_processed_time

            if elapsed_since_last_process >= IFRAME_INTERVAL_SECONDS:
                last_processed_time = current_time
                print(f"[Orchestrator] DEBUG: Cycle start at {time.strftime('%Y-%m-%d %H:%M:%S')}. Trigger: '{current_trigger_description}'.")

                # Get the latest frame from the buffer
                with buffer_lock:
                    if frame_buffer:
                        current_frame = frame_buffer[-1]
                    else:
                        current_frame = None

                if current_frame is None:
                    print("[Orchestrator] INFO: No frames in buffer. Waiting for frames.")
                    time.sleep(0.1)
                    continue

                print(f"[Orchestrator] DEBUG: Using latest frame from buffer. Shape: {current_frame.shape}")

                print(f"[Orchestrator] INFO: Running small inference for trigger: '{current_trigger_description}'")
                small_inference_out: Optional[str] = run_small_inference(current_frame, current_trigger_description)

                if small_inference_out:
                    print(f"[Orchestrator] INFO: Small inference result - Result: {small_inference_out}")
                    if small_inference_out == "yes":
                        print("[Orchestrator] INFO: Trigger MET. Running large inference.")
                        large_inference_out: Optional[LargeInferenceOutput] = run_large_inference(current_frame)
                        if large_inference_out:
                            print(f"[Orchestrator] INFO: Large inference result - Caption: '{large_inference_out.caption}', Tags: {large_inference_out.tags}, Analysis: '{large_inference_out.detailed_analysis[:100]}...'")
                        else:
                            print("[Orchestrator] WARN: Large inference failed or returned no result.")
                else:
                    print("[Orchestrator] WARN: Small inference failed or returned no result.")
            else:
                time.sleep(0.05)

    except ValueError as ve:
        print(f"[Orchestrator] ERROR: Configuration or source error - {ve}. Terminating.")
    except FileNotFoundError as fnfe:
        print(f"[Orchestrator] CRITICAL: Ollama CLI not found - {fnfe}. Ensure Ollama is installed and in PATH. Terminating.")
    except Exception as e:
        import traceback
        print(f"[Orchestrator] CRITICAL: An unexpected error occurred: {type(e).__name__} - {e}")
        print(f"[Orchestrator] Traceback: {traceback.format_exc()}")
    finally:
        if video_stream:
            print("[Orchestrator] INFO: Releasing video stream.")
            video_stream.release()
        print("[Orchestrator] INFO: Processing finished.")

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
        orchestrate_processing(config, stop_processing_event) # Pass the config object
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