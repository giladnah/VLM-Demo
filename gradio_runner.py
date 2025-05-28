import gradio as gr
import requests
import threading
import time
import os

API_URL = "http://localhost:8000"

# Shared state for logs and results
display_logs = []
display_results = []

# Helper to call /start endpoint
def start_orchestration(video_file, trigger):
    try:
        if video_file is None:
            return "Please upload a video file.", "\n".join(display_logs)
        if isinstance(video_file, dict) and "name" in video_file:
            video_path = video_file["name"]
        elif isinstance(video_file, str) and os.path.exists(video_file):
            video_path = video_file
        else:
            return "Invalid video input.", "\n".join(display_logs)
        resp = requests.post(f"{API_URL}/start", json={"source": video_path, "trigger": trigger})
        if resp.ok:
            msg = resp.json().get("message", "Started.")
            display_logs.append(f"[START] {msg}")
            return msg, "\n".join(display_logs)
        else:
            display_logs.append(f"[START][ERROR] {resp.text}")
            return f"Error: {resp.text}", "\n".join(display_logs)
    except Exception as e:
        display_logs.append(f"[START][EXCEPTION] {e}")
        return f"Exception: {e}", "\n".join(display_logs)

# Helper to update trigger
def update_trigger(trigger):
    try:
        resp = requests.put(f"{API_URL}/trigger", json={"trigger": trigger})
        if resp.ok:
            msg = resp.json().get("message", "Trigger updated.")
            display_logs.append(f"[TRIGGER] {msg}")
            return msg
        else:
            display_logs.append(f"[TRIGGER][ERROR] {resp.text}")
            return f"Error: {resp.text}"
    except Exception as e:
        display_logs.append(f"[TRIGGER][EXCEPTION] {e}")
        return f"Exception: {e}"

def infer_large(image_file):
    if image_file is None:
        return None, "Please upload an image."
    try:
        # Gradio image upload returns a numpy array or a file path
        if isinstance(image_file, str) and os.path.exists(image_file):
            image_path = image_file
        elif isinstance(image_file, dict) and "name" in image_file:
            image_path = image_file["name"]
        else:
            return None, "Invalid image input."
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            resp = requests.post(f"{API_URL}/infer/large", files=files)
            if resp.ok:
                try:
                    return image_path, resp.json()
                except Exception:
                    return image_path, resp.text
            else:
                return image_path, f"Error: {resp.text}"
    except Exception as e:
        return None, f"Exception: {e}"

# Polling thread for logs/results (dummy: just shows logs for now)
def poll_results():
    # In a real app, you might poll a /results or /logs endpoint
    # Here, we just keep the last 20 log lines
    while True:
        time.sleep(2)
        # This could be extended to fetch actual inference results
        # For now, just keep logs up to date
        if len(display_logs) > 20:
            del display_logs[:-20]

# Start polling in background
t = threading.Thread(target=poll_results, daemon=True)
t.start()

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# VLM Camera Service Gradio Runner")
        with gr.Tab("Orchestration"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video File")
                trigger = gr.Textbox(label="Trigger Description", value="a person falling down")
            with gr.Row():
                start_btn = gr.Button("Start Orchestration")
                update_btn = gr.Button("Update Trigger")
            status = gr.Textbox(label="Status", interactive=False)
            logs = gr.Textbox(label="Logs", lines=10, interactive=False)

            def on_start(video, trig):
                return start_orchestration(video, trig)

            def on_update(trig):
                msg = update_trigger(trig)
                return msg, "\n".join(display_logs)

            start_btn.click(on_start, inputs=[video_input, trigger], outputs=[status, logs])
            update_btn.click(on_update, inputs=[trigger], outputs=[status, logs])

            # Periodically update logs
            def update_logs():
                return "\n".join(display_logs)
            logs.change(fn=update_logs, outputs=logs)

            # Placeholder for future: video frame inference (only one at a time)
            # with gr.Row():
            #     frame_input = gr.Image(label="Select Frame from Video (future)")
            #     frame_infer_btn = gr.Button("Run Inference on Frame")
            #     frame_status = gr.Textbox(label="Frame Inference Status", interactive=False)
            #     frame_result = gr.JSON(label="Frame Inference Result")
            #
            #     def on_frame_infer(frame):
            #         yield gr.update(interactive=False), gr.update(value="Running inference..."), None
            #         # ... call inference logic here ...
            #         yield gr.update(interactive=True), gr.update(value="Done."), {"result": "example"}
            #
            #     frame_infer_btn.click(
            #         on_frame_infer,
            #         inputs=[frame_input],
            #         outputs=[frame_infer_btn, frame_status, frame_result],
            #         queue=True,
            #         show_progress=True,
            #     )

        gr.Markdown("---")
        gr.Markdown("## Direct Large Inference (Image Upload)")
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload Image for Inference")
        with gr.Row():
            infer_btn = gr.Button("Run Large Inference")
        with gr.Row():
            image_display = gr.Image(label="Image Sent for Inference")
            result_display = gr.JSON(label="Inference Result")
        status_box = gr.Textbox(label="Inference Status", interactive=False)

        # Generator callback: disables button, shows status, then re-enables
        def on_infer(image_file):
            yield gr.update(interactive=False), gr.update(value="Running inference..."), None, None
            img_path, result = infer_large(image_file)
            yield gr.update(interactive=True), gr.update(value="Done."), img_path, result

        infer_btn.click(
            on_infer,
            inputs=[image_input],
            outputs=[infer_btn, status_box, image_display, result_display],
            queue=True,  # Only one request at a time
            show_progress=True,
        )

    return demo

if __name__ == "__main__":
    gradio_ui().launch()