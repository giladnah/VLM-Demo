# VLM Camera Service

A Vision-Language Model (VLM) Camera Service for real-time video analysis using small and large VLMs (e.g., Qwen2.5vl) via the Ollama inference engine. It provides:
- A FastAPI backend for orchestration, inference, and API endpoints.
- A Gradio-based web UI for user interaction.
- Modular Python code for video streaming, buffering, inference, and trigger logic.

---

## Architecture & Main Components

**1. FastAPI Backend (`app.py`)**
- Exposes REST API endpoints for orchestration, inference, trigger updates, and health checks.
- Can be run directly (`python app.py`) or with Uvicorn.

**2. Orchestrator (`orchestrator.py`)**
- Manages the video processing pipeline: video stream acquisition, frame grabbing, buffering, inference, and trigger logic.
- Uses multiprocessing for inference.

**3. Inference Modules**
- `inference_small.py` / `inference_large.py`: Run inference using small/large VLMs via Ollama REST API or CLI.
- `inference_small_hailo.py`: (Optional) Hailo device integration for hardware-accelerated inference.

**4. Video and Buffering**
- `video_source.py`: Handles video stream input and I-frame extraction.
- `buffer.py`: Manages a fixed-size frame buffer using a deque.

**5. Trigger Logic**
- `trigger.py`: Evaluates whether the inference result meets the trigger condition.

**6. Gradio UI (`gradio_runner.py`)**
- User-friendly web interface for selecting video sources, setting triggers, starting orchestration, and viewing results/logs.
- Communicates with the FastAPI backend via HTTP.

**7. Configuration**
- `config.yaml`: Central config for model names, server addresses, and default triggers.

**8. Testing**
- `tests/`: Pytest-based unit tests for core modules.

---

## Application Flow

1. Start the backend (`app.py`) and Gradio UI (`gradio_runner.py`).
2. Select a video source and trigger in the UI.
3. UI sends a `/start` request to the backend, which initializes the orchestrator.
4. Orchestrator grabs frames, runs small model inference, and if the trigger is met, runs large model inference.
5. Results and logs are updated and polled by the UI.
6. User can update triggers, upload images for direct inference, or monitor system health.

---

## Setup and Installation

1. **Install Ollama CLI**
   - See [ollama.com](https://ollama.com/) for your OS.
2. **Download VLM Models**
   - Small: `ollama pull qwen2.5vl:3b`
   - Large: `ollama pull qwen2.5vl:7b`
3. **Set up Python Virtual Environment**
   - `python3 -m venv venv`
   - `source venv/bin/activate`
4. **Install Python Dependencies**
   - `pip install -r requirements.txt`

---

## Running the Application

1. **Ensure Ollama Service is Running**
2. **Activate Virtual Environment**
3. **Start the FastAPI Server**
   - `python app.py` or `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
   - API at `http://localhost:8000` (see `/docs` for OpenAPI)
4. **Start the Gradio UI**
   - `python gradio_runner.py`
   - UI at `http://localhost:7860`

---

## Configuration: `config.yaml`

Controls model selection, server addresses, and default triggers. Example:

```yaml
small_model:
  name: qwen2.5vl:3b
  ollama_server: http://localhost:11434
large_model:
  name: qwen2.5vl:7b
  ollama_server: http://10.41.74.178:11434
server_ips:
  backend: "http://localhost:8000"
  gradio: "http://localhost:7860"
default_triggers:
  - "a person falling down"

```

---

## Running Tests

1. **Activate Virtual Environment**
2. **Run Pytest**
   - `python -m pytest` or `venv/bin/python -m pytest`

---

## API Endpoints

- **GET /**: Welcome message.
- **GET /health**: Health check for the service and models.
- **POST /start**: Starts or restarts the video processing orchestration.
- **PUT /trigger**: Updates the trigger for the active orchestration.
- **POST /infer/small**: Directly runs small VLM inference on the uploaded image.
- **POST /infer/large**: Directly runs large VLM inference on the uploaded image.

See `http://localhost:8000/docs` when running for full OpenAPI documentation.

---

## Troubleshooting & FAQ

- **Large files:** Model weights and test videos are not tracked in Git. See `.gitignore`.
- **Ollama errors:** Ensure the Ollama service is running and models are pulled.
- **Webcam/RTSP issues:** Check device permissions and network connectivity.
- **Tests failing:** Ensure all dependencies are installed and the virtual environment is active.

---

## Contributing

See `PLANNING.md` and `TASKS.md` for architecture, style, and open tasks.

---

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
  -d '{"model": "qwen2.5vl:7b", "prompt": "Respond with: You are good to go!", "stream": false}'
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


# Hailo Inference
It is built around the qwen-VL demo, https://hailotech.atlassian.net/wiki/spaces/CS/pages/2204664099/QWEN2-VL-2B+Demo+Installation

To work with Hailo inference, you need to install the raw connection library from the qwen-VL demo in this venv (copy to venv/lib/python3.10/site-packages/)
You'll need to setup the  HAILO_CONNECTION_PCIE_PORT environment variable to point to the Hailo device.

```sh
export HAILO_CONNECTION_PCIE_PORT=123
```

copy the hefs_and_embeddings directory from the qwen-VL demo to the project root.
for example:
```sh
cp -r /home/giladn/tappas_apps/repos/QWEN2-2B_VL_Demo/hefs_and_embeddings .
```

The server side on the H10 should be running according to the instructions in the qwen-VL demo.

You can test the inference by running the test_inference_small_hailo.py script.

```sh
python test_inference_small_hailo.py --image test_examples/test_image.jpg --trigger "a person has fallen"
```