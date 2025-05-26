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