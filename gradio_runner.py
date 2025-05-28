import gradio as gr
import requests
import threading
import time
import os
from PIL import Image
from io import BytesIO
import glob
from config import get_config

CONFIG = get_config()
API_URL = CONFIG['server_ips']['backend']
DEFAULT_TRIGGERS = CONFIG.get('default_triggers', ["a person falling down"])

class VLMGradioApp:
    def __init__(self):
        self.display_logs = []
        self.display_logs_lock = threading.Lock()
        self.video_device_choices = self.get_video_device_choices()
        self.poll_thread = threading.Thread(target=self.poll_results, daemon=True)
        self.poll_thread.start()
        self.last_frame = None

    def get_video_device_choices(self):
        devices = sorted(glob.glob("/dev/video*"))
        choices = ["Upload Video"]
        for dev in devices:
            choices.append(f"Webcam: {dev}")
        return choices

    def start_orchestration(self, source_type, video_file, trigger):
        if source_type.startswith("Webcam: "):
            video_path = source_type.replace("Webcam: ", "")
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
                with self.display_logs_lock:
                    self.display_logs.append(f"[START] {msg}")
                return msg, "\n".join(self.display_logs)
            else:
                with self.display_logs_lock:
                    self.display_logs.append(f"[START][ERROR] {resp.text}")
                return f"Error: {resp.text}", "\n".join(self.display_logs)
        except Exception as e:
            with self.display_logs_lock:
                self.display_logs.append(f"[START][EXCEPTION] {e}")
            return f"Exception: {e}", "\n".join(self.display_logs)

    def update_trigger(self, trigger):
        try:
            resp = requests.put(f"{API_URL}/trigger", json={"trigger": trigger})
            if resp.ok:
                msg = resp.json().get("message", "Trigger updated.")
                with self.display_logs_lock:
                    self.display_logs.append(f"[TRIGGER] {msg}")
                return msg
            else:
                with self.display_logs_lock:
                    self.display_logs.append(f"[TRIGGER][ERROR] {resp.text}")
                return f"Error: {resp.text}"
        except Exception as e:
            with self.display_logs_lock:
                self.display_logs.append(f"[TRIGGER][EXCEPTION] {e}")
            return f"Exception: {e}"

    def infer_large(self, image_file):
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

    def fetch_latest_small_inference_frame(self):
        try:
            resp = requests.get(f"{API_URL}/latest-small-inference-frame", timeout=2)
            if resp.ok:
                return Image.open(BytesIO(resp.content))
        except Exception:
            pass
        return None

    def fetch_latest_small_inference_result(self):
        try:
            resp = requests.get(f"{API_URL}/latest-small-inference-result", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {}

    def fetch_latest_large_inference_result(self):
        try:
            resp = requests.get(f"{API_URL}/latest-large-inference-result", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {}

    def fetch_results_status(self):
        try:
            resp = requests.get(f"{API_URL}/results_status", timeout=2)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        return {"small": False, "large": False}

    def poll_results(self):
        while True:
            time.sleep(2)
            with self.display_logs_lock:
                if len(self.display_logs) > 10:
                    del self.display_logs[:-10]

    def refresh_small_frame_auto(self):
        status = self.fetch_results_status()
        new_logs = []
        if status.get("small"):
            self.last_frame = self.fetch_latest_small_inference_frame()
            result = self.fetch_latest_small_inference_result()
            if result and result.get("result") is not None:
                new_log = f"[SMALL INFERENCE RESULT] {result}"
                new_logs.append(new_log)
        if status.get("large"):
            result = self.fetch_latest_large_inference_result()
            print("DEBUG: Large inference result from backend:", result)
            if result:
                caption = result.get("caption", "")
                tags = result.get("tags", "")
                analysis = result.get("detailed_analysis", "")
                new_log = f"[LARGE INFERENCE RESULT] Caption: {caption}, Tags: {tags}, Analysis: {analysis[:80]}..."
                new_logs.append(new_log)
        with self.display_logs_lock:
            for log in new_logs:
                if not self.display_logs or self.display_logs[-1] != log:
                    self.display_logs.append(log)
            self.display_logs = self.display_logs[-10:]
            logs_str = "\n".join(self.display_logs)
        return self.last_frame, logs_str

    def on_start(self, src_type, video, trig):
        msg, logs_str = self.start_orchestration(src_type, video, trig)
        with self.display_logs_lock:
            return msg, "\n".join(self.display_logs)

    def on_update(self, trig):
        msg = self.update_trigger(trig)
        with self.display_logs_lock:
            return msg, "\n".join(self.display_logs)

    def update_logs(self):
        with self.display_logs_lock:
            return "\n".join(self.display_logs)

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# VLM Camera Service Gradio Runner")
            with gr.Tab("Orchestration"):
                with gr.Row():
                    source_type = gr.Dropdown(choices=self.video_device_choices, value=self.video_device_choices[0], label="Source Type")
                    video_input = gr.Video(label="Upload Video File")
                    trigger = gr.Textbox(label="Trigger Description", value=DEFAULT_TRIGGERS[0], placeholder="Enter trigger description")
                with gr.Row():
                    start_btn = gr.Button("Start Orchestration")
                    update_btn = gr.Button("Update Trigger")
                status = gr.Textbox(label="Status", interactive=False)
                logs = gr.Textbox(label="Logs", lines=10, interactive=False)

                start_btn.click(self.on_start, inputs=[source_type, video_input, trigger], outputs=[status, logs])
                update_btn.click(self.on_update, inputs=[trigger], outputs=[status, logs])

                logs.change(fn=self.update_logs, outputs=logs)

                gr.Markdown("### Latest Small Inference Frame (Auto-Refresh)")
                latest_small_frame = gr.Image(label="Latest Small Inference Frame")

                timer = gr.Timer(value=1)
                timer.tick(
                    fn=self.refresh_small_frame_auto,
                    inputs=[],
                    outputs=[latest_small_frame, logs],
                )

                # Placeholder for future: video frame inference (only one at a time)
                # ...
        return demo

    def run(self):
        self.build_ui().launch()

if __name__ == "__main__":
    VLMGradioApp().run()