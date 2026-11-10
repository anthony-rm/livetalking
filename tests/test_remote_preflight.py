"""Tests for remote/preflight.py — the worker dependency gate.

The point of this module is that it can REPORT a broken environment instead of
dying inside the module it validates. These tests pin that property directly:
preflight must import cleanly with every heavy dependency absent, and must name
the missing one.
"""
from __future__ import annotations

import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote import preflight


def test_preflight_imports_without_its_dependencies():
    """A checker that needs the thing it checks is useless (the original bug:
    a `--check` flag inside worker.py could never run)."""
    src = open(preflight.__file__, encoding="utf-8").read()
    module_level = []
    for line in src.splitlines():
        if line.startswith(("import ", "from ")) and "importlib" not in line:
            module_level.append(line)
    for line in module_level:
        for heavy in ("grpc", "torch", "numpy", "cv2"):
            assert heavy not in line, (
                f"{heavy} imported at module scope: the check would become "
                f"unreachable when it is missing")


def test_probe_reports_each_missing_module_individually(monkeypatch):
    """One bad module must not hide the others."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("grpc", "numpy"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rows = preflight._probe([("grpc", "transport"), ("numpy", "tensors"),
                             ("json", "stdlib works")])
    by_mod = {m: err for m, _p, err in rows}
    assert by_mod["grpc"] and "grpc" in by_mod["grpc"]
    assert by_mod["numpy"] and "numpy" in by_mod["numpy"]
    assert by_mod["json"] is None, "a present module must report no error"


def test_probe_never_raises(monkeypatch):
    """A probe is a report, not a control-flow primitive."""
    monkeypatch.setattr(builtins, "__import__",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("boom")))
    rows = preflight._probe([("anything", "test")])
    assert rows[0][2] == "ImportError: boom"


def test_required_covers_the_worker_startup_imports():
    """grpc/protobuf/numpy/cv2/tqdm/torch are what worker.py needs to bind."""
    mods = {m for m, _p in preflight.REQUIRED}
    for expected in ("grpc", "google.protobuf", "numpy", "cv2", "tqdm", "torch"):
        assert expected in mods, f"{expected} missing from REQUIRED"


def test_required_covers_what_worker_py_actually_imports():
    """Derive the list from worker.py so it cannot drift silently."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(preflight.__file__)))
    needed = set()
    for rel in ("remote/worker.py", "remote/tensors.py", "utils/image.py"):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    mod = line.split()[1].split(".")[0]
                    if mod in {"grpc", "numpy", "cv2", "tqdm", "google", "torch"}:
                        needed.add("google.protobuf" if mod == "google" else mod)
    assert needed
    declared = {m for m, _p in preflight.REQUIRED}
    for mod in needed:
        assert mod in declared, f"{mod} is imported by the worker but not preflighted"


def test_run_reports_ok_and_device_shape():
    report = preflight.run(deep=False)
    assert set(report) >= {"ok", "device", "checks", "missing"}
    assert "python" in report["device"]
    assert isinstance(report["ok"], bool)
    assert set(report["checks"]) == {"required"}


def test_json_mode_exits_2_when_required_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "REQUIRED", [("definitely_not_a_module", "test")])
    rc = preflight.main(["--json"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "definitely_not_a_module" in out


def test_text_mode_names_the_install_command(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "REQUIRED", [("definitely_not_a_module", "test")])
    rc = preflight.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "definitely_not_a_module" in captured.err
    assert "pip install" in captured.err
