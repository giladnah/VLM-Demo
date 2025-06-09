"""
Gradio UI runner for the VLM Camera Service.

This module provides a user-friendly web interface for interacting with the FastAPI backend, allowing users to select video sources, set triggers, start orchestration, and view inference results and logs.
"""

import gradio as gr
import requests
import threading
import time
import os
from PIL import Image
from io import BytesIO
import glob
from datetime import datetime
from config import get_config

CONFIG = get_config()
API_URL = CONFIG['server_ips']['backend']
DEFAULT_TRIGGERS = CONFIG.get('default_triggers', ["a person falling down"])

class VLMGradioApp:
    """
    Gradio application for the VLM Camera Service.

    Handles UI logic, polling, and backend API communication.
    """
    def __init__(self) -> None:
        """
        Initializes the Gradio app, device choices, and polling thread.
        """
        self.display_logs = []
        self.display_logs_lock = threading.Lock()
        self.rtsp_cameras = self.fetch_rtsp_cameras()
        self.video_device_choices = self.get_video_device_choices()
        self.poll_thread = threading.Thread(target=self.poll_results, daemon=True)
        self.poll_thread.start()
        self.last_frame = None
        self.last_status_md_content = "<span style='color: gray; font-size: 1.5em; font-weight: bold;'>...</span>"

    def fetch_rtsp_cameras(self) -> list:
        """
        Fetches RTSP camera configurations from the backend.

        Returns:
            list: List of RTSP camera configs.
        """
        try:
            resp = requests.get(f"{API_URL}/rtsp-cameras", timeout=3)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return []

    def get_video_device_choices(self) -> list:
        """
        Returns a list of available video device choices (webcams, RTSP, upload).

        Returns:
            list: List of device choice strings.
        """
        devices = sorted(glob.glob("/dev/video*"))
        choices = ["Upload Video"]
        for dev in devices:
            choices.append(f"Webcam: {dev}")
        # Add RTSP cameras from config
        for cam in self.rtsp_cameras:
            choices.append(f"RTSP: {cam['name']}")
        return choices

    def get_rtsp_url_by_name(self, name: str) -> str | None:
        """
        Returns the RTSP URL for a given camera name.

        Args:
            name (str): Camera name.

        Returns:
            str | None: RTSP URL or None if not found.
        """
        for cam in self.rtsp_cameras:
            if cam['name'] == name:
                # Build RTSP URL with credentials if provided
                addr = cam['address']
                user = cam.get('username', '')
                pwd = cam.get('password', '')
                if user and pwd:
                    # Insert credentials into the URL
                    prefix, rest = addr.split('://', 1)
                    return f"{prefix}://{user}:{pwd}@{rest}"
                return addr
        return None

    def start_orchestration(self, source_type: str, video_file, trigger: str) -> tuple[str, str]:
        """
        Starts orchestration by sending a request to the backend.

        Args:
            source_type (str): Video source type.
            video_file: Uploaded video file or path.
            trigger (str): Trigger description.

        Returns:
            tuple[str, str]: (Status message, logs string)
        """
        if source_type.startswith("Webcam: "):
            video_path = source_type.replace("Webcam: ", "")
        elif source_type.startswith("RTSP: "):
            cam_name = source_type.replace("RTSP: ", "")
            video_path = self.get_rtsp_url_by_name(cam_name)
            if not video_path:
                return f"RTSP camera '{cam_name}' not found in config.", "\n".join(self.display_logs)
        else:
            if video_file is None:
                return "Please upload a video file.", "\n".join(self.display_logs)
            if isinstance(video_file, dict) and "name" in video_file:
                video_path = video_file["name"]
            elif isinstance(video_file, str) and os.path.exists(video_file):
                video_path = video_file
            else:
                return "Invalid video input.", "\n".join(self.display_logs)
        try:
            payload = {
                "source": video_path,
                "trigger": trigger,
                "small_model_name": CONFIG['small_model']['name'],
                "small_ollama_server": CONFIG['small_model']['ollama_server'],
                "large_model_name": CONFIG['large_model']['name'],
                "large_ollama_server": CONFIG['large_model']['ollama_server'],
            }
            resp = requests.post(f"{API_URL}/start", json=payload)
            if resp.ok:
                msg = resp.json().get("message", "Started.")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with self.display_logs_lock:
                    self.display_logs.append(f"[{timestamp}] [START] {msg}")
                return msg, "\n".join(self.display_logs)
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with self.display_logs_lock:
                    self.display_logs.append(f"[{timestamp}] [START][ERROR] {resp.text}")
                return f"Error: {resp.text}", "\n".join(self.display_logs)
        except Exception as e:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self.display_logs_lock:
                self.display_logs.append(f"[{timestamp}] [START][EXCEPTION] {e}")
            return f"Exception: {e}", "\n".join(self.display_logs)

    def update_trigger(self, trigger: str) -> str:
        """
        Updates the trigger description in the backend.

        Args:
            trigger (str): New trigger description.

        Returns:
            str: Status message.
        """
        try:
            resp = requests.put(f"{API_URL}/trigger", json={"trigger": trigger})
            if resp.ok:
                msg = resp.json().get("message", "Trigger updated.")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with self.display_logs_lock:
                    self.display_logs.append(f"[{timestamp}] [TRIGGER] {msg}")
                return msg
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with self.display_logs_lock:
                    self.display_logs.append(f"[{timestamp}] [TRIGGER][ERROR] {resp.text}")
                return f"Error: {resp.text}"
        except Exception as e:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self.display_logs_lock:
                self.display_logs.append(f"[{timestamp}] [TRIGGER][EXCEPTION] {e}")
            return f"Exception: {e}"

    def infer_large(self, image_file) -> tuple[str | None, str]:
        """
        Runs large model inference on an uploaded image via the backend.

        Args:
            image_file: Uploaded image file or path.

        Returns:
            tuple[str | None, str]: (Image path or None, result or error message)
        """
        if image_file is None:
            return None, "Please upload an image."
        try:
            if isinstance(image_file, str) and os.path.exists(image_file):
                image_path = image_file
            elif isinstance(image_file, dict) and "name" in image_file:
                image_path = image_file["name"]
            else:
                return None, "Invalid image input."
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
                data = {
                    "model_name": CONFIG['large_model']['name'],
                    "ollama_server": CONFIG['large_model']['ollama_server']
                }
                resp = requests.post(f"{API_URL}/infer/large", files=files, data=data)
                if resp.ok:
                    try:
                        return image_path, resp.json()
                    except Exception:
                        return image_path, resp.text
                else:
                    return image_path, f"Error: {resp.text}"
        except Exception as e:
            return None, f"Exception: {e}"

    def fetch_latest_small_inference_frame(self) -> Image.Image | None:
        """
        Fetches the latest small inference frame from the backend.

        Returns:
            Image.Image | None: Latest frame as PIL Image, or None if unavailable.
        """
        try:
            resp = requests.get(f"{API_URL}/latest-small-inference-frame", timeout=2)
            if resp.ok:
                return Image.open(BytesIO(resp.content))
        except Exception:
            pass
        return None

    def fetch_latest_small_inference_result(self) -> dict:
        """
        Fetches the latest small inference result from the backend.

        Returns:
            dict: Inference result.
        """
        try:
            resp = requests.get(f"{API_URL}/latest-small-inference-result", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {}

    def fetch_latest_large_inference_result(self) -> dict:
        """
        Fetches the latest large inference result from the backend.

        Returns:
            dict: Inference result.
        """
        try:
            resp = requests.get(f"{API_URL}/latest-large-inference-result", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {}

    def fetch_results_status(self) -> dict:
        """
        Fetches the status of available inference results from the backend.

        Returns:
            dict: Status of small and large inference results.
        """
        try:
            resp = requests.get(f"{API_URL}/results_status", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {"small": False, "large": False}

    def poll_results(self) -> None:
        """
        Polls the backend for results and manages log trimming in a background thread.
        """
        while True:
            time.sleep(2)
            with self.display_logs_lock:
                if len(self.display_logs) > 10:
                    del self.display_logs[:-10]

    def refresh_small_frame_auto(self) -> tuple[Image.Image | None, str, str]:
        """
        Refreshes the small inference frame, status, and logs for the UI.

        Returns:
            tuple[Image.Image | None, str, str]: (Frame, status markdown, logs string)
        """
        status = self.fetch_results_status()
        new_logs = []
        status_md_content = self.last_status_md_content
        small_result = None
        large_result = None
        if status.get("small"):
            self.last_frame = self.fetch_latest_small_inference_frame()
            result = self.fetch_latest_small_inference_result()
            if result and result.get("result") is not None:
                small_result = result["result"]
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_log = f"[{timestamp}] [SMALL INFERENCE RESULT]:  {result['result']}"
                new_logs.append(new_log)
        if status.get("large"):
            result = self.fetch_latest_large_inference_result()
            print("DEBUG: Large inference result from backend:", result)
            if result:
                large_result = result.get("result", "")
                analysis = result.get("detailed_analysis", "")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_log = f"[{timestamp}] [LARGE INFERENCE RESULT] Result: {large_result}, Analysis: {analysis}"
                new_logs.append(new_log)
        # Status logic
        new_status_md_content = status_md_content
        if large_result == "yes":
            new_status_md_content = ("<div style='display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 180px;'>"
                                    "<span style='color: red; font-size: 4em; font-weight: bold; text-align: center;'>Alarm</span>"
                                    "</div>")
        elif small_result == "yes":
            new_status_md_content = ("<div style='display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 180px;'>"
                                    "<span style='color: orange; font-size: 4em; font-weight: bold; text-align: center;'>Alert</span>"
                                    "</div>")
        elif small_result == "no" or large_result == "no":
            new_status_md_content = ("<div style='display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 180px;'>"
                                    "<span style='color: green; font-size: 4em; font-weight: bold; text-align: center;'>OK</span>"
                                    "</div>")
        # Only update if changed
        if new_status_md_content != self.last_status_md_content:
            self.last_status_md_content = new_status_md_content
        status_md_content = self.last_status_md_content
        with self.display_logs_lock:
            for log in new_logs:
                if not self.display_logs or self.display_logs[-1] != log:
                    self.display_logs.append(log)
            self.display_logs = self.display_logs[-10:]
            logs_str = "\n".join(self.display_logs)
        return self.last_frame, status_md_content, logs_str

    def on_start(self, src_type, video, trig) -> tuple[str, str]:
        """
        Gradio callback for starting orchestration.

        Args:
            src_type: Source type.
            video: Uploaded video.
            trig: Trigger description.

        Returns:
            tuple[str, str]: (Status message, logs string)
        """
        msg, logs_str = self.start_orchestration(src_type, video, trig)
        with self.display_logs_lock:
            return msg, "\n".join(self.display_logs)

    def on_update(self, trig) -> tuple[str, str]:
        """
        Gradio callback for updating the trigger.

        Args:
            trig: New trigger description.

        Returns:
            tuple[str, str]: (Status message, logs string)
        """
        msg = self.update_trigger(trig)
        with self.display_logs_lock:
            return msg, "\n".join(self.display_logs)

    def update_logs(self):
        with self.display_logs_lock:
            return "\n".join(self.display_logs)

    def build_ui(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("# VLM Camera Service Gradio Runner", elem_id="main-title")
                with gr.Column(scale=2, min_width=120):
                    gr.Image(
                        value="hailo_logo.png",
                        interactive=False,
                        show_label=False,
                        elem_id="hailo-logo",
                        show_download_button=False,
                        show_fullscreen_button=False,
                        container=False
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    source_type = gr.Dropdown(choices=self.video_device_choices, value=self.video_device_choices[0], label="Source Type")
                    start_btn = gr.Button("Start Orchestration")
                    trigger = gr.Textbox(label="Trigger Description", value=DEFAULT_TRIGGERS[0], placeholder="Enter trigger description")
                    update_btn = gr.Button("Update Trigger")
                    video_input = gr.Video(label="Upload Video File")
                with gr.Column(scale=2):
                    gr.Markdown("### Current Inference Frame (Auto-Refresh)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            latest_small_frame = gr.Image(interactive=False, show_label=False, container=False,show_download_button=False,
                        show_fullscreen_button=False)
                        with gr.Column(scale=1):
                            status_md = gr.Markdown(label="Status")
                    logs = gr.Textbox(label="Logs", lines=10, interactive=False)

            start_btn.click(self.on_start, inputs=[source_type, video_input, trigger], outputs=[status_md, logs])
            update_btn.click(self.on_update, inputs=[trigger], outputs=[status_md, logs])
            logs.change(fn=self.update_logs, outputs=logs)

            timer = gr.Timer(value=1)
            timer.tick(
                fn=self.refresh_small_frame_auto,
                inputs=[],
                outputs=[latest_small_frame, status_md, logs],
            )
        return demo

    def run(self):
        self.build_ui().launch()

if __name__ == "__main__":
    VLMGradioApp().run()