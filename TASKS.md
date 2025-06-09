# Task List for VLM Camera Application

Use this checklist to track progress on building the VLM Camera Application.

## Environment & Setup

* [x] **Install Ollama CLI and download Qwen2-VL 2B model**

  * `sudo apt update && sudo apt install ollama`
  * `ollama pull qwen2.5vl:3b`
* [x] **Download 7B VLM model**

  * `ollama pull qwen2.5vl:7b`
* [x] **Set up Python virtual environment**

  * `python3 -m venv venv`
  * `source venv/bin/activate`
  * `pip install fastapi uvicorn pydantic opencv-python`
  * **Virtual environment created and dependencies installed (2024-06-13)**
* [ ] **Verify Grad.io UI environment and dependencies**

  * Confirm Node.js and Grad.io CLI installed
  * `npm install gradio react react-dom`

## Modular Codebase

* [x] **Create `video_source.py`** (2024-07-15)

  * Define `get_video_stream(source_uri)`: connect to RTSP or file path.
  * Define `extract_i_frame(stream)`: use OpenCV to grab I-frames.
* [x] **Create `buffer.py`** (2024-07-15)

  * Define `init_frame_buffer(size=10)`: initialize deque of fixed length.
  * Define `update_buffer(buffer, frame)`: append new frame, pop oldest.
* [x] **Create `inference_small.py`** (2024-07-15)

  * Define `run_small_inference(frame, trigger_description)`: call `ollama run qwen2-vl2b`, parse JSON output.
* [x] **Create `trigger.py`** (2024-07-15)

  * Define `evaluate_trigger(detection, trigger_description)`: compare detection labels to trigger keywords.
* [x] **Create `inference_large.py`** (2024-07-15)

  * Define `run_large_inference(frames)`: call `ollama run 7b-vlm` with batch of frames or single image.
* [x] **Create `orchestrator.py`** (2024-07-15)

  * Define `orchestrate_processing(source_uri, trigger_description)`: coordinate frame extraction, buffering, inference, and triggers.

## API Development

* [x] **Scaffold FastAPI service** (2024-07-15)

  * Create `app.py` with FastAPI instance
  * Add CORS middleware for Grad.io UI.
* [x] **Add `/start` endpoint** (2024-07-15)

  * Accept JSON: `{ "source": "rtsp://...", "trigger": "..." }`
  * Launch `orchestrate_processing` as background task.
* [x] **Add `/trigger` endpoint** (2024-07-15)

  * Allow updating trigger description at runtime.
* [x] **Implement `/infer/small` endpoint** (2024-07-15)

  * Accept image payload, call `run_small_inference`, return result.
* [x] **Implement `/infer/large` endpoint** (2024-07-15)
* [x] **Implement `/health` endpoint** (2024-07-15)

## Code Review, Documentation, and Test Coverage

* [x] **buffer.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **app.py**
  * [x] Review code and improve documentation
  * [x] Build or improve API/integration tests
* [x] **gradio_runner.py**
  * [x] Review code and improve documentation
  * [x] Build or improve smoke/integration tests
* [x] **orchestrator.py**
  * [x] Review code and improve documentation
  * [ ] Build or improve unit/integration tests
* [x] **inference_small.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **inference_large.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **trigger.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **config.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **video_source.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests