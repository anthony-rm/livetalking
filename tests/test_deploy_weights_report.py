"""Tests for the deploy's weights survey.

`verify()` proves "the box restarted and serves my revision". It says nothing
about whether the box can RUN, and reporting green on a box with an empty
models/ dir is exactly the class of lie this suite exists to prevent.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import deploy_remote as dz  # noqa: E402


def _fake_target():
    return {"username": "u", "host": "h", "port": 22}


def _stub_probe(monkeypatch, stdout: str):
    monkeypatch.setattr(
        dz, "run_script",
        lambda target, script, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr=""))


def test_all_present_reports_ok(monkeypatch, capsys):
    lines = []
    for rel, _feat, _minb in dz.WEIGHT_FILES:
        lines.append(f"HAVE 50000000 {rel}")
    _stub_probe(monkeypatch, "\n".join(lines))
    dz.report_weights(_fake_target(), "~/livetalking")
    out = capsys.readouterr().out
    assert "all" in out and "present" in out
    assert "WARNING" not in out


def test_missing_weights_name_the_affected_feature(monkeypatch, capsys):
    """The point is to say WHICH feature breaks, not just that a file is gone."""
    lines = []
    for rel, _feat, _minb in dz.WEIGHT_FILES:
        if rel == "models/wav2lip.pth":
            lines.append(f"MISS 0 {rel}")
        else:
            lines.append(f"HAVE 50000000 {rel}")
    _stub_probe(monkeypatch, "\n".join(lines))
    dz.report_weights(_fake_target(), "~/livetalking")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "wav2lip inference" in out
    assert "models/wav2lip.pth" in out
    assert "fetch_worker_weights.sh" in out


def test_a_tiny_file_counts_as_incomplete(monkeypatch, capsys):
    """A half-downloaded weight loads and then fails at inference — worse than
    a missing one, so it must not be reported as present."""
    lines = []
    for rel, _feat, _minb in dz.WEIGHT_FILES:
        if rel == "models/musetalkV15/unet.pth":
            lines.append(f"HAVE 4096 {rel}")   # 4 KB of a 3.4 GB file
        else:
            lines.append(f"HAVE 50000000 {rel}")
    _stub_probe(monkeypatch, "\n".join(lines))
    dz.report_weights(_fake_target(), "~/livetalking")
    out = capsys.readouterr().out
    assert "incomplete" in out
    assert "unet.pth" in out


def test_survey_covers_every_feature_the_worker_serves():
    features = {feat for _rel, feat, _m in dz.WEIGHT_FILES}
    for expected in ("musetalk inference", "wav2lip inference",
                     "musetalk avatar creation", "video enhance"):
        assert expected in features, f"{expected} has no weights entry"


def test_tiny_config_file_is_not_reported_incomplete(monkeypatch, capsys):
    """A 748-byte musetalk.json is a real config, not a half-downloaded weight;
    the size gate must not flag legitimately small files as incomplete."""
    lines = []
    for rel, _feat, _minb in dz.WEIGHT_FILES:
        lines.append(f"HAVE {_minb} {rel}")
    _stub_probe(monkeypatch, "\n".join(lines))
    dz.report_weights(_fake_target(), "~/livetalking")
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "all" in out and "present" in out


def test_verify_survives_a_survey_failure(monkeypatch):
    """A working deploy must not be reported as failed because the survey broke."""
    monkeypatch.setattr(dz, "_probe",
                        lambda address, timeout=5.0: {
                            "ok": True, "device": "cuda", "revision": "abc",
                            "detail": "GPU"})
    monkeypatch.setattr(dz, "health_address", lambda target: "127.0.0.1:50051")

    def boom(*a, **kw):
        raise KeyError("port")
    monkeypatch.setattr(dz, "report_weights", boom)

    info = dz.verify(_fake_target(), "abc", expect_cuda=True)
    assert info["revision"] == "abc"
