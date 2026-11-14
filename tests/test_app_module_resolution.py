#!/usr/bin/env python3
"""The model-lifecycle endpoints must reach the RUNNING app module.

app.py is launched as __main__ (./run.sh → python app.py). ``import app`` from
server/routes.py therefore creates a second module object with its own globals:
the endpoints used to read a phantom `model_state` ('unloaded' forever) and call
`unload_model()` on a copy that owned no weights, so /api/model/unload reported
"already unloaded" while the real model stayed resident.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_app_module_resolution.py
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import routes  # noqa: E402


def test_resolves_main_when_it_is_app_py(monkeypatch):
    """Simulate `python app.py`: __main__.__file__ is app.py."""
    running = types.ModuleType("__main__")
    running.__file__ = "/repo/app.py"
    running.model_state = "ready"
    monkeypatch.setitem(sys.modules, "__main__", running)

    assert routes._app_module() is running


def test_falls_back_to_import_when_main_is_not_app_py(monkeypatch):
    """Under pytest, __main__ is pytest, so the real `app` module is used."""
    other = types.ModuleType("__main__")
    other.__file__ = "/usr/bin/pytest"
    monkeypatch.setitem(sys.modules, "__main__", other)

    resolved = routes._app_module()
    assert str(getattr(resolved, "__file__", "")).endswith("app.py")
    assert resolved is not other


def test_resolved_module_is_the_one_holding_model_state(monkeypatch):
    """The whole point: the endpoint sees the running module's state."""
    running = types.ModuleType("__main__")
    running.__file__ = "/repo/app.py"
    running.model_state = "ready"
    running.model_backend = "musetalk"
    monkeypatch.setitem(sys.modules, "__main__", running)

    mod = routes._app_module()
    assert mod.model_state == "ready"
    assert mod.model_backend == "musetalk"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
