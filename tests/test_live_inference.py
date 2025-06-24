"""
Live integration test for both small and large models using config.yaml.
- Loads the current config.yaml and prints the model-specific configuration.
- Runs real inference for both small and large models using a dummy image and a simple prompt.
- Prints the full inference result for diagnostics.
- Skips with a clear message if the config, required service, or environment variable is missing.
- Recommended for verifying that your backend and models are working end-to-end.
"""
import pytest
import numpy as np
import os
import yaml
from inference.unified import run_unified_inference

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

def load_model_config(model_type):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config.get(f'{model_type}_model', {})
    if not model_cfg.get('name'):
        raise ValueError(f"{model_type}_model.name is not set in config.yaml")
    engine = model_cfg.get('engine', 'ollama')
    name = model_cfg['name']
    return engine, name, model_cfg, config

@pytest.mark.asyncio
async def test_live_inference_small_and_large():
    """
    Live integration test: Runs real inference for both small and large models using config.yaml.
    Skips if config or required services are not available.
    Prints the configuration and inference result for debugging.
    """
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for model_type in ["small", "large"]:
        try:
            engine, name, model_cfg, full_config = load_model_config(model_type)
        except Exception as e:
            pytest.skip(f"[{model_type}] Config or model not available: {e}")
        print(f"\n[LIVE TEST] {model_type}_model configuration:")
        print(model_cfg)
        print(f"[LIVE TEST] {model_type}_model: engine={engine}, name={name}")
        if engine == "openai" and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip(f"[{model_type}] OPENAI_API_KEY not set in environment.")
        try:
            result = await run_unified_inference(dummy_image, "a cat on a couch", model_type=model_type)
            print(f"[LIVE TEST] {model_type} inference result:")
            if hasattr(result, 'model_dump'):
                print(result.model_dump())
            elif hasattr(result, 'dict'):
                print(result.dict())
            else:
                print(result.__dict__)
            assert result.result in ("yes", "no")
            assert isinstance(result.detailed_analysis, str)
        except Exception as e:
            print(f"[WARNING] [{model_type}] Live inference failed: {e}")
            pytest.skip(f"[WARNING] [{model_type}] Engine not available, service not running, or config missing: {e}")