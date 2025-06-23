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
  * `pip install fastapi uvicorn pydantic opencv-python python-dotenv`
  * **Virtual environment created and dependencies installed (2024-06-13)**
* [x] **Verify Grad.io UI environment and dependencies**

  * Confirm Node.js and Grad.io CLI installed
  * `npm install gradio react react-dom`

## Modular Codebase

* [x] **Create `video_source.py`** (2024-07-15)

  * Define `get_video_stream(source_uri)`: connect to RTSP or file path.
  * Define `extract_i_frame(stream)`: use OpenCV to grab I-frames.
* [x] **Create `buffer.py`** (2024-07-15)

  * Define `init_frame_buffer(size=10)`: initialize deque of fixed length.
  * Define `update_buffer(buffer, frame)`: append new frame, pop oldest.
* [x] **Create `trigger.py`** (2024-07-15)

  * Define `evaluate_trigger(detection, trigger_description)`: compare detection labels to trigger keywords.
* [x] **Create `orchestrator.py`** (2024-07-15)

  * Define `orchestrate_processing(source_uri, trigger_description)`: coordinate frame extraction, buffering, inference, and triggers.

## Unified Inference System

* [x] **Implement Unified Inference System** (2024-08-01)
  * [x] Define `InferenceEngine` protocol and base models
  * [x] Implement `UnifiedVLMInference` class
  * [x] Implement Ollama engine (small/large)
  * [x] Implement OpenAI engine (large/small)
  * [x] Remove/deprecate old `inference_small.py` and `inference_large.py`
  * [x] Update all orchestrator and API logic to use unified system
  * [x] Update all tests to use unified system
  * [x] Add secure config and .env support for OpenAI API keys

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

  * Accept image payload, call unified inference, return result.
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
  * [x] Build or improve unit/integration tests
* [x] **trigger.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **config.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **video_source.py**
  * [x] Review code and improve documentation
  * [x] Build or improve unit tests
* [x] **Unified Inference System**
  * [x] Review code and improve documentation
  * [x] Build or improve unit/integration tests
  * [x] Remove all references to deprecated inference modules

## Documentation

* [x] **Update README.md** for unified system, OpenAI/Ollama config, and .env usage
* [x] **Update docs/inference/README.md** for unified system and engine-agnostic usage
* [x] **Update docs/inference/configuration.md** for OpenAI config and .env
* [x] **Update PLANNING.md** to reflect unified system and current architecture
* [x] **Update or remove docs/inference/engine.md** (empty or redundant)
* [x] **Update or remove references to Hailo unless specifically planned**

## Batch Large Inference (OpenAI Only)

* [x] Add `batch_inference` config section under `large_model` in `config.yaml`
* [x] Update config parsing and validation for batch mode (OpenAI only)
* [x] Update large inference logic to use batch when enabled and engine is OpenAI
* [x] Adapt OpenAI engine and prompt for batch image input
* [x] Log warning and ignore batch mode if engine is not OpenAI
* [x] Update documentation for batch inference mode
* [ ] Add unit/integration tests for batch mode and fallback logic

## Debug Features

* [x] Add debug feature to save latest small/large inference frames (and batch frames) to files when enabled in config

## Future Tasks
* [ ] **Hailo Integration** (if/when planned)
  * [ ] Research Hailo API requirements
  * [ ] Design Hailo engine implementation
  * [ ] Plan hardware acceleration strategy
  * [ ] Document hardware requirements
* [ ] **Advanced Features**
  * [ ] Add support for custom model loading
  * [ ] Implement model switching at runtime
  * [ ] Add advanced trigger combinations
  * [ ] Create plugin system for custom engines