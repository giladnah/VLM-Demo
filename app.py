"""Main FastAPI application for the VLM Camera Service."""

import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import cv2
import numpy as np
import subprocess # For health check
import json # For parsing ollama list output
from fastapi.responses import FileResponse, StreamingResponse

# --- Load environment variables from .env automatically ---
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------

from orchestrator import orchestrator, OrchestrationConfig
from config import get_config
from inference.unified import run_unified_inference

# TODO: Potentially import Pydantic models for request/response if defined centrally
# TODO: Import orchestrator and other necessary components when endpoints are added

app = FastAPI(
    title="VLM Camera Service",
    description="Provides API endpoints for VLM inference and video processing orchestration.",
    version="0.1.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is important for allowing the Grad.io UI (or other web frontends)
# to communicate with the FastAPI backend if they are on different origins.
origins = [
    "http://localhost",          # Common for local development
    "http://localhost:3000",     # Default for React dev server
    "http://localhost:7860",     # Corrected: Gradio typical port
    "http://127.0.0.1:7860",   # Alternative for Gradio
    # TODO: Add the production URL of the Grad.io UI if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allows specific origins
    # allow_origin_regex="https://.*\.example\.com", # Or allow via regex
    allow_credentials=True,      # Allows cookies to be included in requests
    allow_methods=["*"],         # Allows all standard HTTP methods (GET, POST, etc.)
    allow_headers=["*"],         # Allows all headers
)

# TODO: Import orchestrator and other necessary components when endpoints are added

# Global state for the current orchestration
current_orchestration_config: Optional[OrchestrationConfig] = None
current_orchestration_stop_event: Optional[threading.Event] = None
orchestration_lock = threading.Lock() # Protects access to the global state above

CONFIG = get_config()
SMALL_MODEL_NAME = CONFIG['small_model']['name']
# Use .get to avoid KeyError if ollama_server is not present (e.g., when using OpenAI)
SMALL_MODEL_OLLAMA_SERVER = CONFIG['small_model'].get('ollama_server')
LARGE_MODEL_NAME = CONFIG['large_model']['name']
LARGE_MODEL_OLLAMA_SERVER = CONFIG['large_model'].get('ollama_server')
DEFAULT_TRIGGERS = CONFIG.get('default_triggers', ["a person falling down"])

class StartRequest(BaseModel):
    """Request model for the /start endpoint."""
    source: str = Field(..., description="The URI of the video source (e.g., RTSP URL, file path).")
    trigger: str = Field(..., description="The initial textual trigger description for VLM inference.")
    small_model_name: Optional[str] = None
    small_ollama_server: Optional[str] = None
    large_model_name: Optional[str] = None
    large_ollama_server: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "rtsp://user:pass@ip_address:port/path",
                "trigger": "a person wearing a red hat"
            }
        }
    )

class UpdateTriggerRequest(BaseModel):
    """Request model for the /trigger endpoint."""
    trigger: str = Field(..., description="The new textual trigger description.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trigger": "a blue car"
            }
        }
    )

@app.get("/", tags=["Root"], summary="Root path with a welcome message.")
async def read_root() -> dict:
    """
    Root endpoint providing a welcome message.

    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the VLM Camera Service API!"}

@app.get("/health", tags=["Service Health"], summary="Performs a health check of the service and models.")
async def health_check() -> dict:
    """
    Enhanced health check endpoint.

    Checks for Ollama CLI accessibility and the presence of required VLM models.

    Returns:
        dict: Service status, Ollama CLI accessibility, model availability, and details.
    """
    service_status = "ok"
    messages = ["FastAPI service is running."]
    ollama_accessible = False
    models_status = {
        SMALL_MODEL_NAME: {"status": "unknown", "message": "Not checked"},
        LARGE_MODEL_NAME: {"status": "unknown", "message": "Not checked"},
    }

    # 1. Check Ollama CLI accessibility
    try:
        version_process = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5, check=False
        )
        if version_process.returncode == 0:
            ollama_accessible = True
            messages.append(f"Ollama CLI accessible. Version: {version_process.stdout.strip()}")
        else:
            messages.append(f"Ollama CLI not accessible or error. Return code: {version_process.returncode}. Error: {version_process.stderr.strip()}")
            service_status = "error"
    except FileNotFoundError:
        messages.append("Ollama CLI not found in PATH.")
        service_status = "error"
    except subprocess.TimeoutExpired:
        messages.append("Ollama --version command timed out.")
        service_status = "error"
    except Exception as e:
        messages.append(f"Error checking Ollama version: {str(e)}")
        service_status = "error"

    # 2. If Ollama is accessible, check for models
    if ollama_accessible:
        try:
            list_process = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10, check=False
            )
            if list_process.returncode == 0:
                available_models = []
                # Ollama list output is typically line-by-line, each line a model info. Some versions output JSON lines.
                # Trying to handle both simple text and JSONL like format.
                try:
                    # Attempt to parse as JSONL first (multiple JSON objects, one per line)
                    parsed_models_json = []
                    for line in list_process.stdout.strip().split('\n'):
                        if line.strip(): # Ensure line is not empty
                            try:
                                parsed_models_json.append(json.loads(line))
                            except json.JSONDecodeError:
                                # If a line is not JSON, it might be the header or plain text from older ollama
                                if "NAME" in line and "ID" in line and "SIZE" in line: # Header line
                                    continue
                                # Assume model name is the first word if not JSON
                                available_models.append(line.split()[0])

                    if parsed_models_json: # If JSONL parsing worked for any line
                        available_models.extend([m.get('name') for m in parsed_models_json if m.get('name')])

                except Exception as e_parse: # Catch any error during parsing, fallback to simple split
                    messages.append(f"Could not robustly parse 'ollama list' output ({e_parse}), falling back to simple name check.")
                    # Fallback for older ollama versions or unexpected format: split lines and take first word
                    available_models = [line.split()[0] for line in list_process.stdout.strip().split('\n')[1:] if line.strip()] # Skip header

                # Remove potential duplicates if both parsing methods added same models
                available_models = sorted(list(set(available_models)))
                messages.append(f"Ollama models found: {available_models if available_models else 'None'}")

                # Check for specific small model
                if any(SMALL_MODEL_NAME in model_name for model_name in available_models):
                    models_status[SMALL_MODEL_NAME]["status"] = "available"
                    models_status[SMALL_MODEL_NAME]["message"] = "Model is available."
                else:
                    models_status[SMALL_MODEL_NAME]["status"] = "unavailable"
                    models_status[SMALL_MODEL_NAME]["message"] = f"Model '{SMALL_MODEL_NAME}' not found in `ollama list`."
                    service_status = "error" # Critical model missing

                # Check for specific large model
                if any(LARGE_MODEL_NAME in model_name for model_name in available_models):
                    models_status[LARGE_MODEL_NAME]["status"] = "available"
                    models_status[LARGE_MODEL_NAME]["message"] = "Model is available."
                else:
                    models_status[LARGE_MODEL_NAME]["status"] = "unavailable"
                    models_status[LARGE_MODEL_NAME]["message"] = f"Model '{LARGE_MODEL_NAME}' not found in `ollama list`."
                    service_status = "error" # Critical model missing
            else:
                messages.append(f"`ollama list` command failed. Return code: {list_process.returncode}. Error: {list_process.stderr.strip()}")
                service_status = "error"
                models_status[SMALL_MODEL_NAME]["message"] = "Failed to query ollama list."
                models_status[LARGE_MODEL_NAME]["message"] = "Failed to query ollama list."

        except subprocess.TimeoutExpired:
            messages.append("`ollama list` command timed out.")
            service_status = "error"
            models_status[SMALL_MODEL_NAME]["message"] = "'ollama list' timed out."
            models_status[LARGE_MODEL_NAME]["message"] = "'ollama list' timed out."
        except Exception as e_list:
            messages.append(f"Error running `ollama list`: {str(e_list)}")
            service_status = "error"
            models_status[SMALL_MODEL_NAME]["message"] = f"Error querying models: {str(e_list)}"
            models_status[LARGE_MODEL_NAME]["message"] = f"Error querying models: {str(e_list)}"
    else:
        messages.append("Skipping model checks as Ollama CLI is not accessible.")
        models_status[SMALL_MODEL_NAME]["message"] = "Ollama CLI not accessible."
        models_status[LARGE_MODEL_NAME]["message"] = "Ollama CLI not accessible."

    return {
        "service_status": service_status,
        "ollama_cli_accessible": ollama_accessible,
        "model_availability": models_status,
        "details": messages
    }

# TODO: Add other endpoints as defined in TASKS.md:
# - /start (POST): to launch orchestrate_processing
# - /trigger (POST/PUT): to update trigger description
# - /infer/small (POST): for direct small model inference
# - /infer/large (POST): for direct large model inference

@app.post("/start", tags=["Orchestration"], summary="Starts or restarts the video processing orchestration.")
async def start_processing(request: StartRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Starts the video processing pipeline in the background.

    Args:
        request (StartRequest): The request body containing source, trigger, and model/server info.
        background_tasks (BackgroundTasks): FastAPI background task manager.

    Returns:
        dict: Status and configuration of the started orchestration.
    """
    config = OrchestrationConfig(
        source_uri=request.source,
        initial_trigger_description=request.trigger
    )
    # Store model/server info in config or orchestrator as needed
    config.small_model_name = request.small_model_name or SMALL_MODEL_NAME
    config.small_ollama_server = request.small_ollama_server or SMALL_MODEL_OLLAMA_SERVER
    config.large_model_name = request.large_model_name or LARGE_MODEL_NAME
    config.large_ollama_server = request.large_ollama_server or LARGE_MODEL_OLLAMA_SERVER
    background_tasks.add_task(orchestrator.start, config)
    return {
        "status": "success",
        "message": "Orchestration process started/restarted in the background.",
        "source": config.source_uri,
        "initial_trigger": config.get_trigger_description(),
        "small_model_name": config.small_model_name,
        "small_ollama_server": config.small_ollama_server,
        "large_model_name": config.large_model_name,
        "large_ollama_server": config.large_ollama_server
    }

@app.put("/trigger", tags=["Orchestration"], summary="Updates the trigger description for the active orchestration.")
async def update_trigger(request: UpdateTriggerRequest) -> dict:
    """
    Updates the trigger description for the currently running orchestration.

    Args:
        request (UpdateTriggerRequest): The request body containing the new trigger description.

    Returns:
        dict: Status and message about the trigger update.
    """
    # No need for orchestration_lock or global config, just update via orchestrator
    if orchestrator._config is None:
        raise HTTPException(status_code=404, detail="No active orchestration process to update. Please start one using /start.")
    orchestrator._config.set_trigger_description(request.trigger)
    return {
        "status": "success",
        "message": "Trigger description updated for the active orchestration.",
        "new_trigger": request.trigger
    }

@app.post("/infer/small", tags=["Direct Inference"], summary="Performs inference using the small VLM on a single image.")
async def direct_small_inference(
    background_tasks: BackgroundTasks,
    trigger: str = Form(...),
    image: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    ollama_server: Optional[str] = Form(None)
) -> Optional[str]:
    """Accepts an image and a trigger description, then runs the small VLM for direct inference.

    - **trigger**: The textual trigger description (e.g., "a person on a bike").
    - **image**: The image file (JPEG/PNG) to process.
    """
    if not image.content_type or image.content_type.lower() not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Please upload a JPEG or PNG image.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Received an empty image file.")

        # Decode image using OpenCV
        image_np_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # type: ignore[attr-defined]

        if image_np_array is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPEG or PNG.")

    except Exception as e:
        print(f"[API /infer/small] ERROR: Failed to read or decode image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read or decode image: {str(e)}")
    finally:
        await image.close() # Ensure the UploadFile is closed

    print(f"[API /infer/small] INFO: Received image for small inference. Trigger: '{trigger}', Image size: {len(image_bytes)} bytes.")

    # Using background_tasks here if run_small_inference_direct itself could be significantly blocking.
    # The current run_small_inference uses subprocess.run which is blocking.
    # For a highly concurrent API, one might consider running this in a separate thread pool
    # explicitly, or ensure the subprocess call is handled asynchronously if possible.
    # However, FastAPI handles blocking IO in background tasks appropriately.
    # For simplicity, let's call it directly and rely on FastAPI/Uvicorn worker model.
    # If it becomes a bottleneck, can wrap with `run_in_threadpool` or use BackgroundTasks.

    # Direct call for now, as background_tasks.add_task does not return the result to the client.
    # If it needs to be non-blocking for the endpoint, an async version of run_small_inference would be ideal
    # or use something like `fastapi.concurrency.run_in_threadpool`.

    # Let's make it a blocking call for now as per standard request-response pattern.
    # If it were to use BackgroundTasks, the client would get an immediate 200 OK before result is ready.
    model_name = model_name or SMALL_MODEL_NAME
    ollama_server = ollama_server or SMALL_MODEL_OLLAMA_SERVER
    # ollama_server may be None if using OpenAI; handle accordingly in run_unified_inference
    inference_result = await run_unified_inference(image_np_array, trigger, model_type="small")

    if inference_result is None:
        print(f"[API /infer/small] WARN: Small inference direct call returned no result for trigger '{trigger}'.")
        raise HTTPException(status_code=500, detail="Small VLM inference failed or returned no result. Check server logs.")

    print(f"[API /infer/small] INFO: Small inference direct call successful. Result: {inference_result.result}")
    return inference_result.result

@app.post("/infer/large", tags=["Direct Inference"], summary="Performs inference using the large VLM on a single image.")
async def direct_large_inference(image: UploadFile = File(...), model_name: Optional[str] = Form(None), ollama_server: Optional[str] = Form(None)) -> Optional[dict]:
    """Accepts an image, then runs the large VLM for direct, detailed inference.

    - **image**: The image file (JPEG/PNG) to process.

    Note: The 'buffer payload' aspect mentioned in some planning documents is handled
    internally by the orchestrator. This direct endpoint accepts a single image.
    """
    if not image.content_type or image.content_type.lower() not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Please upload a JPEG or PNG image.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Received an empty image file.")

        image_np_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # type: ignore[attr-defined]

        if image_np_array is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPEG or PNG.")

    except Exception as e:
        print(f"[API /infer/large] ERROR: Failed to read or decode image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read or decode image: {str(e)}")
    finally:
        await image.close()

    print(f"[API /infer/large] INFO: Received image for large inference. Image size: {len(image_bytes)} bytes.")

    # Similar to /infer/small, this is a blocking call.
    # Consider fastapi.concurrency.run_in_threadpool if it becomes a performance issue.
    model_name = model_name or LARGE_MODEL_NAME
    ollama_server = ollama_server or LARGE_MODEL_OLLAMA_SERVER
    # ollama_server may be None if using OpenAI; handle accordingly in run_unified_inference
    inference_result = await run_unified_inference(image_np_array, "", model_type="large")

    if inference_result is None:
        print(f"[API /infer/large] WARN: Large inference direct call returned no result.")
        raise HTTPException(status_code=500, detail="Large VLM inference failed or returned no result. Check server logs.")

    print(f"[API /infer/large] INFO: Large inference direct call successful. Result: '{inference_result.result}', Analysis: '{inference_result.detailed_analysis[:80]}...'")
    return inference_result.dict()

@app.get("/latest-small-inference-frame", tags=["Debug"], summary="Get the latest frame used for small inference.")
def get_latest_small_inference_frame():
    # Return the in-memory numpy array as a JPEG image
    frame = orchestrator.get_latest_small_inference_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No inference frame available.")
    import cv2
    import io
    import numpy as np
    # Convert BGR to RGB if needed for display (Gradio can handle BGR, but browsers expect RGB)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, img_encoded = cv2.imencode('.jpg', frame)  # type: ignore[attr-defined]
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.get("/results_status", tags=["Debug"], summary="Get the status of available inference results.")
def get_results_status():
    return orchestrator.get_results_status()

@app.get("/latest-small-inference-result", tags=["Debug"], summary="Get the latest small inference result.")
def get_latest_small_inference_result():
    return orchestrator.get_latest_small_inference_result(clear_status=True) or {}

@app.get("/latest-large-inference-result", tags=["Debug"], summary="Get the latest large inference result.")
def get_latest_large_inference_result():
    return orchestrator.get_latest_large_inference_result(clear_status=True) or {}

@app.get("/rtsp-cameras", tags=["Config"], summary="Get the list of RTSP cameras from config.yaml.")
def get_rtsp_cameras():
    """Returns the list of RTSP cameras as defined in config.yaml."""
    return CONFIG.get('rtsp_cameras', [])

if __name__ == "__main__":
    import uvicorn
    # Suppress FastAPI/Uvicorn access logs for cleaner output
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False  # <--- disables '[FastAPI] INFO: ...' request logs
    )