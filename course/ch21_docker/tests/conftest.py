import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

BASE_AGENT = {
    "version": 1,
    "name": "http-test",
    "model": {
        "id": "mock",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.2,
        "max_tokens": 64,
    },
    "limits": {"max_steps": 3},
    "prompt": {"system": "You are a test agent.", "guidance": "Answer briefly."},
    "tools": {"enabled": ["read_file"]},
}


@pytest.fixture
def make_agent(tmp_path):
    def _make(**overrides):
        data = copy.deepcopy(BASE_AGENT)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        path = tmp_path / "agent.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _make
