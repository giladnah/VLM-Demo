## VLM Camera Application: Implementation Specification for Cursor

### 1. Overview

This document defines the detailed implementation specification for Cursor to build the Vision-Language Model (VLM) camera application. The system will run both a small VLM (Qwen2-VL 2B) and a larger VLM (7B VLM) locally via Ollama, with a future migration to cloud inference. The frontend UI will be implemented using Grad.io.

### 2. Goals & Scope

* **Phase 1** (Local Proof-of-Concept): Run both VLMs side-by-side on the same edge device.
* **Phase 2** (Scalable Deployment): Migrate the heavier 7B VLM to a cloud endpoint, while keeping the small VLM on-device.
* **Phase 3** (Optimization & Monitoring): Add performance metrics, health-checks, and automated failover.

### 3. Functional Requirements

1. **Model Management**

   * Pull and launch Qwen2-VL 2B and 7B VLM containers using Ollama.
   * Provide start/stop/restart endpoints for each model.

2. **Inference API**

   * Expose a RESTful API with endpoints:

     * `/infer/small`  → Qwen2-VL 2B
     * `/infer/large`  → 7B VLM
   * Input: JPEG/PNG image payload.
   * Output: JSON with embedding vectors, captions, and detection tags.

3. **UI (Grad.io)**

   * Design a responsive dashboard in Grad.io.
   * Upload button or live camera feed integration.
   * Display inference results: text, tags, and image overlays.

4. **Configuration & Settings**

   * YAML or JSON config for model paths, ports, and thresholds.
   * Toggle between local and cloud endpoints for the large model.

5. **Logging & Metrics**

   * Capture latency, throughput, error rates.
   * Health-check endpoint: `/health`.

### 4. System Architecture

```plaintext
[Camera / Upload] → [API Gateway] → [Inference Module]
                                  ├─ Qwen2-VL 2B (local)
                                  └─ 7B VLM (local or cloud)
                                      ↓
                                [Results Formatter]
                                      ↓
                                   [Grad.io UI]
```

* **API Gateway**: Node.js or Python FastAPI service.
* **Inference Module**: Ollama CLI integration scripts.
* **UI**: React-based Grad.io components.

### 5. Application Flow

1. **Video Source**: Accept input fromCamera stream (RTSP/USB) or video file.
2. **I-Frame Extraction**:

   * Every **2 seconds**, extract the latest I-frame.
   * Send this frame immediately to the Small Model stage.
3. **Frame Buffering**:

   * Maintain a rolling buffer of **10 I-frames**, updated **every second**.
   * For stage 1, buffer frames mirror those sent to the small model.
4. **Small Model Inference** (*Stage 1*)

   * Endpoint: `run_small_inference(frame, trigger_description)`
   * Input: latest I-frame and user-defined trigger description (e.g., “a dog sitting on the couch”).
   * Output: detection result boolean and labels.
5. **Trigger Logic**

   * Compare small VLM output against the trigger description.
   * If **matched**, enqueue the same frame into a queue for large model processing.
6. **Large Model Inference** (*Stage 2*)

   * Endpoint: `run_large_inference(frame_buffer)`
   * Input: either the single triggered frame (Phase 1) or the full buffered frames.
   * Output: rich JSON with embeddings, captions, detailed analysis.

### 6. Modular Function Definitions

To simplify future changes, implement each stage as a separate function or module:

```python
# video_source.py
def get_video_stream(source_uri) -> VideoStream: ...

def extract_i_frame(stream: VideoStream) -> ImageFrame: ...

# buffer.py
def init_frame_buffer(size: int = 10) -> FrameBuffer: ...

def update_buffer(buffer: FrameBuffer, frame: ImageFrame): ...

# inference_small.py
def run_small_inference(frame: ImageFrame, trigger: str) -> DetectionResult: ...

# inference_large.py
def run_large_inference(frames: List[ImageFrame]) -> LargeDetectionResult: ...

# trigger.py
def evaluate_trigger(detection: DetectionResult, trigger: str) -> bool: ...

# orchestrator.py
def orchestrate_processing(source_uri: str, trigger: str): ...
```

* Each function should be easily swappable (e.g., change model, API call, or trigger logic).

### 7. Data Flow

1. `orchestrate_processing()` initializes video stream and buffer.
2. In a loop:

   * Extract I-frame every 2s → `extract_i_frame()`
   * Update buffer every 1s → `update_buffer()`
   * Call `run_small_inference()` on extracted frame.
   * If `evaluate_trigger()` returns true, send frame or full buffer to `run_large_inference()`.
3. Collect and format outputs for UI via API Gateway.

### 8. Implementation Tasks (Cursor)

* **Environment Setup**

  * Install Ollama and download both models.
  * Prepare Docker/virtualenv for API Gateway.

* **Modular Codebase**

  * Create modules: `video_source.py`, `buffer.py`, `inference_small.py`, `inference_large.py`, `trigger.py`, `orchestrator.py`.
  * Define interfaces and data structures (e.g., `ImageFrame`, `DetectionResult`).

* **API Development**

  * Scaffold FastAPI service with a single `/start` endpoint that calls `orchestrate_processing()`.
  * Add endpoints to adjust trigger description at runtime.

* **UI Components**

  * Build controls for selecting source, entering trigger description, and viewing live outputs.

* **Testing & Validation**

  * Unit tests for each module using mocked inputs.
  * Integration test simulating live video.

* **Documentation**

  * Update README to reflect new modules and flow.
  * Provide sequence diagrams illustrating application flow.

