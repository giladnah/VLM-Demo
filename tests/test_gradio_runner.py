import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock
import gradio_runner

def test_gradio_app_instantiation_and_ui():
    # Mock requests to avoid real HTTP calls
    with patch('gradio_runner.requests.get', return_value=MagicMock(ok=True, json=lambda: [])), \
         patch('gradio_runner.requests.post', return_value=MagicMock(ok=True, json=lambda: {"message": "Started."})), \
         patch('gradio_runner.requests.put', return_value=MagicMock(ok=True, json=lambda: {"message": "Trigger updated."})):
        app = gradio_runner.VLMGradioApp()
        ui = app.build_ui()
        assert ui is not None
        # Optionally, check that device choices are populated
        assert isinstance(app.video_device_choices, list)
        # Optionally, check that logs are accessible
        assert hasattr(app, 'display_logs')