"""
Unit tests for config.py.

Covers expected, edge, and failure cases for the configuration loader, including:
- Successful config loading and key presence
- Caching behavior
- Missing file and invalid YAML error handling

Uses pytest and monkeypatching for isolation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import yaml
from config import load_config, get_config

def test_config_loads():
    config = load_config()
    assert isinstance(config, dict)
    assert 'small_model' in config
    assert 'large_model' in config
    assert 'name' in config['small_model']
    assert 'ollama_server' in config['small_model']
    assert 'name' in config['large_model']
    # Only require 'ollama_server' for large_model if engine is ollama or not set
    engine = config['large_model'].get('engine', 'ollama').lower()
    if engine == 'ollama':
        assert 'ollama_server' in config['large_model']
    assert 'server_ips' in config
    assert 'backend' in config['server_ips']
    assert 'default_triggers' in config
    assert isinstance(config['default_triggers'], list)
    assert isinstance(config['default_triggers'][0], str)

def test_get_config_returns_same_object():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2

def test_missing_file_raises(monkeypatch):
    from config import _CONFIG_PATH
    monkeypatch.setattr('config._CONFIG_PATH', 'nonexistent.yaml')
    # Clear cached config
    monkeypatch.setattr('config._config', None)
    with pytest.raises(FileNotFoundError):
        load_config()

def test_invalid_yaml(tmp_path, monkeypatch):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not: [valid: yaml: here]")
    monkeypatch.setattr('config._CONFIG_PATH', str(bad_yaml))
    # Clear cached config
    monkeypatch.setattr('config._config', None)
    with pytest.raises(yaml.YAMLError):
        load_config()