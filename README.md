# VLM Camera Application

## Overview

This project implements a Vision Language Model (VLM) powered camera application. It can connect to video sources (like RTSP streams or local files), process frames, and perform inferences using both small and large VLMs based on configurable triggers. The application exposes a FastAPI backend for controlling the orchestration and performing direct inferences.

## Features

- **Video Ingestion**: Connects to RTSP streams or local video files.
- **Frame Processing**: Extracts frames from the video source.
- **Rolling Buffer**: Maintains a buffer of recent frames.
- **Small VLM Inference**: Periodically runs a small VLM (e.g., Qwen2.5-VL 3B) on extracted frames to detect trigger conditions.
- **Dynamic Triggers**: Allows updating the trigger condition for the small VLM at runtime.
- **Large VLM Inference**: If the small VLM detects a trigger, a larger VLM (e.g., Qwen2.5-VL 7B) is run on the relevant frame(s) for more detailed analysis (captioning, tagging, detailed description).
- **FastAPI Backend**:
    - `/start`: Endpoint to start/restart the video processing orchestration.
    - `/trigger`: Endpoint to update the trigger description for an active orchestration.
    - `/infer/small`: Endpoint for direct inference with the small VLM using a provided image and trigger.
    - `/infer/large`: Endpoint for direct inference with the large VLM using a provided image.
    - `/health`: Endpoint to check service status, Ollama CLI accessibility, and required model availability.
- **Modular Design**: Code is organized into modules for video source, buffering, inference, triggering, and orchestration.
- **Unit Tests**: Pytest unit tests for various modules.

## Project Structure

```
.
├── app.py                  # FastAPI application
├── orchestrator.py         # Main orchestration logic
├── video_source.py         # Video stream handling
├── buffer.py               # Frame buffer management
├── inference_small.py      # Small VLM inference logic
├── inference_large.py      # Large VLM inference logic
├── trigger.py              # Trigger evaluation logic
├── TASKS.md                # Project task tracking
├── PLANNING.md             # (Assumed, as per instructions) Project planning details
├── requirements.txt        # Python dependencies (to be generated)
├── venv/                   # Python virtual environment
└── tests/                  # Pytest unit tests
    ├── test_orchestrator.py
    ├── test_inference_small.py
    ├── test_inference_large.py
    └── test_trigger.py
```

## Setup and Installation

Refer to `TASKS.md` for detailed initial setup steps, including:

1.  **Install Ollama CLI**:
    *   Follow instructions on [ollama.com](https://ollama.com/) for your OS.
    *   Example for Linux: `sudo apt update && sudo apt install ollama`
2.  **Download VLM Models**:
    *   Small model: `ollama pull qwen2.5vl:3b`
    *   Large model: `ollama pull qwen2.5vl:7b`
    *   Verify models are available: `ollama list`
3.  **Set up Python Virtual Environment**:
    *   `python3 -m venv venv`
    *   `source venv/bin/activate`
4.  **Install Python Dependencies**:
    *   A `requirements.txt` will be provided. For now, core dependencies are:
        `pip install fastapi uvicorn pydantic opencv-python numpy pytest typeguard python-multipart`
    *   If you encounter issues with `opencv-python`, you might need `opencv-python-headless` for server environments.

## Running the Application

1.  **Ensure Ollama Service is Running**:
    *   Start the Ollama service (e.g., `sudo systemctl start ollama` on Linux, or run the Ollama desktop application).
2.  **Activate Virtual Environment**:
    *   `source venv/bin/activate`
3.  **Start the FastAPI Server**:
    *   `python app.py`
    *   Or using Uvicorn directly for more options: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
    *   The API will typically be available at `http://localhost:8000`.
    *   Check the `/health` endpoint (`http://localhost:8000/health`) to verify service and model status.

## Running Tests

1.  **Activate Virtual Environment**:
    *   `source venv/bin/activate`
2.  **Run Pytest**:
    *   From the project root directory: `python -m pytest`
    *   Or: `venv/bin/python -m pytest`

## API Endpoints

(Details can be found via the auto-generated OpenAPI docs at `http://localhost:8000/docs` when the API is running)

*   **GET /**: Welcome message.
*   **GET /health**: Health check for the service and models.
*   **POST /start**:
    *   Body: `{"source": "your_video_source_uri", "trigger": "your_trigger_description"}`
    *   Starts or restarts the video processing orchestration.
*   **PUT /trigger**:
    *   Body: `{"trigger": "new_trigger_description"}`
    *   Updates the trigger for the active orchestration.
*   **POST /infer/small**:
    *   Form data: `trigger` (string), `image` (file upload)
    *   Directly runs small VLM inference on the uploaded image.
*   **POST /infer/large**:
    *   Form data: `image` (file upload)
    *   Directly runs large VLM inference on the uploaded image.

## TODO / Future Enhancements

*   Generate `requirements.txt`.
*   Integrate with a Grad.io UI (as per `TASKS.md`).
*   Implement true I-frame detection in `video_source.py`.
*   More robust error handling and logging.
*   Database integration for storing inference results or events.

# Setting Up an Ollama Server for Remote REST API Access

To use a remote Ollama server for inference (small or large models), follow these steps:

## 1. Install Ollama on the Remote Machine
- Follow the official instructions: https://ollama.com/download

## 2. Start Ollama Listening on All Interfaces
By default, Ollama listens only on `127.0.0.1` (localhost), which is not accessible from other machines.

**To allow remote access, start Ollama with:**

```sh
OLLAMA_HOST=0.0.0.0 ollama serve
```

## 3. Open Firewall Port (if needed)
- Ensure port `11434` is open to your client machine(s).
- On Ubuntu with UFW:
  ```sh
  sudo ufw allow 11434/tcp
  ```
- On CentOS/RedHat with firewalld:
  ```sh
  sudo firewall-cmd --add-port=11434/tcp --permanent
  sudo firewall-cmd --reload
  ```

## 4. Test Remote Access
From your client machine, run:
```sh
curl -X POST http://<REMOTE_IP>:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5vl:7b", "prompt": "Describe the capabilities of this model.", "format": "json", "stream": false}'
```

You should receive a JSON response from the model.

## Troubleshooting
- If you get `Connection refused`, make sure Ollama is running and listening on `0.0.0.0:11434`.
- If you get a timeout, check firewall rules on both the server and client side.
- You can test locally on the server with:
  ```sh
  curl -X POST http://localhost:11434/api/generate ...
  ```
- If you want to restrict access, consider using a VPN or firewall rules to limit which IPs can connect.