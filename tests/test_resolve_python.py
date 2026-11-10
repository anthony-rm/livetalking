"""resolve_python discovery — modelled on the real box's layout.

The probing logic runs ON THE BOX, so a syntax or glob error there fails at
deploy time on a server, not in CI. These tests run the exact command string
against a fake box (shimmed ssh) with stub interpreters.

The shape that matters: the user's server has
  * env  ~/.conda/envs/livetalking/bin/python   (torch + cu124 + grpc)
  * sys  /usr/bin/python3, /usr/bin/python      (NO torch, FIRST on PATH)
so a resolver that trusts PATH starts a worker that dies on `import torch`.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fake_gpu_box  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="needs bash")


@pytest.fixture()
def box(tmp_path):
    info = fake_gpu_box.install(str(tmp_path / "box"))
    os.makedirs(os.path.join(info["home"], "livetalking", ".deploy"), exist_ok=True)
    return info


def _stub_python(path: str, *, ok: bool) -> str:
    """A fake interpreter: exits 0 for the probe imports only when `ok`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\n"
                 + ("exit 0\n" if ok else "exit 1\n"))
    os.chmod(path, 0o755)
    return path


def _shadow_path(box, *dirs: str) -> None:
    """Make the fake box's shell see only these dirs plus the OS basics."""
    with open(os.path.join(box["home"], ".bash_profile"), "w") as fh:
        fh.write(f'export PATH="{":".join(dirs)}:/usr/bin:/bin"\n')


def _run(box, monkeypatch, *, env_python=None, env_globs=None):
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    if env_python:
        monkeypatch.setenv("LT_REMOTE_PYTHON", env_python)
    else:
        monkeypatch.delenv("LT_REMOTE_PYTHON", raising=False)
    from scripts import deploy_remote as dz
    if env_globs is not None:
        # The fake box lives under $HOME; a /opt glob would reach THIS machine's
        # real conda install and make the test pass for the wrong reason.
        monkeypatch.setattr(dz, "ENV_GLOB_SH", env_globs)
    return dz.resolve_python({"username": "u", "host": "host", "port": 22},
                             "~/livetalking")


HOME_GLOBS = ['"$HOME"/.conda/envs/*/bin/python', '"$HOME"/*/envs/*/bin/python']


def test_env_override_wins(box, monkeypatch):
    monkeypatch.setenv("LT_REMOTE_PYTHON", "/custom/python")
    from scripts import deploy_remote as dz
    assert dz.resolve_python({"username": "u", "host": "h", "port": 22},
                             "~/livetalking") == "/custom/python"


def test_finds_user_level_conda_env_before_torchless_system_python(box, monkeypatch):
    """The real layout: ~/.conda/envs/livetalking must beat PATH's python3."""
    env_py = _stub_python(
        os.path.join(box["home"], ".conda/envs/livetalking/bin/python"), ok=True)
    sysbin = os.path.join(box["home"], "sysbin")
    _stub_python(os.path.join(sysbin, "python3"), ok=False)
    _stub_python(os.path.join(sysbin, "python"), ok=False)
    _shadow_path(box, sysbin)

    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert got == env_py, f"expected the conda env, got {got}"
    cached = open(os.path.join(box["home"], "livetalking/.deploy/python")).read()
    assert ".conda/envs/livetalking/bin/python" in cached


def test_conda_layout_is_found_and_cached(box, monkeypatch):
    conda = _stub_python(
        os.path.join(box["home"], "miniconda3/envs/livetalking/bin/python"), ok=True)
    sysbin = os.path.join(box["home"], "only-conda-bin")
    _stub_python(os.path.join(sysbin, "python3"), ok=False)
    _stub_python(os.path.join(sysbin, "python"), ok=False)
    _shadow_path(box, sysbin)

    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert got == conda, got
    assert os.path.isabs(got), got


def test_prefers_the_env_named_livetalking(box, monkeypatch):
    """Several envs with torch: the one called livetalking wins."""
    _stub_python(os.path.join(box["home"], ".conda/envs/aaa-other/bin/python"),
                 ok=True)
    wanted = _stub_python(
        os.path.join(box["home"], ".conda/envs/livetalking/bin/python"), ok=True)
    assert _run(box, monkeypatch, env_globs=HOME_GLOBS) == wanted


def test_falls_back_to_an_unnamed_env_with_torch(box, monkeypatch):
    env_py = _stub_python(
        os.path.join(box["home"], ".conda/envs/someenv/bin/python"), ok=True)
    assert _run(box, monkeypatch, env_globs=HOME_GLOBS) == env_py


def test_cache_hit_skips_rescanning(box, monkeypatch):
    conda = _stub_python(
        os.path.join(box["home"], ".conda/envs/livetalking/bin/python"), ok=True)
    with open(os.path.join(box["home"], "livetalking/.deploy/python"), "w") as fh:
        fh.write(conda + "\n")
    assert _run(box, monkeypatch) == conda


def test_glob_never_leaks_a_literal_star(box, monkeypatch):
    """An unmatched glob must not come back as a path."""
    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert "*" not in got, f"literal glob leaked into the interpreter path: {got}"


def test_env_with_torch_but_no_grpc_is_still_chosen(box, monkeypatch, capsys):
    """REGRESSION: on a fresh box the env has torch but not grpc yet.

    Gating on `import torch, grpc` sent the deploy to the torch-less system
    python and told the user to install deps into THAT interpreter -- which
    never fixes the env the worker actually needs.
    """
    env_py = os.path.join(box["home"], ".conda/envs/livetalking/bin/python")
    os.makedirs(os.path.dirname(env_py), exist_ok=True)
    with open(env_py, "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\n"
                 'case "$*" in *grpc*) exit 1;; esac\n'
                 "exit 0\n")
    os.chmod(env_py, 0o755)
    sysbin = os.path.join(box["home"], "sysbin")
    _stub_python(os.path.join(sysbin, "python3"), ok=False)
    _shadow_path(box, sysbin)

    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert got == env_py, f"picked {got} instead of the env that has torch"
    out = capsys.readouterr().out
    # It must SAY grpc is missing, in the env it picked.
    assert "grpc is not installed in that env" in out
    assert env_py in out


def test_interpreter_glob_is_expanded_not_passed_literally(box, monkeypatch):
    """REGRESSION: a glob stored in a shell variable is not re-expanded.

    Storing the pattern in $g and doing `for p in $g` handed the literal
    "$HOME/.conda/envs/*/bin/python" to the -x test, matched nothing, and the
    resolver silently fell through to python3.
    """
    env_py = _stub_python(
        os.path.join(box["home"], ".conda/envs/livetalking/bin/python"), ok=True)
    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert got == env_py
    assert "$HOME" not in got
    assert "*" not in got


def test_stale_cached_python3_is_ignored(box, monkeypatch, capsys):
    """REGRESSION: your box cached `python3`, which cannot import torch.

    The old resolver trusted any cache entry, so a first run that fell back to
    the system interpreter locked the box into that answer for every later
    deploy -- and told the user to pip-install into an interpreter the worker
    never uses.
    """
    cache = os.path.join(box["home"], "livetalking/.deploy/python")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write("python3\n")
    env_py = _stub_python(
        os.path.join(box["home"], ".conda/envs/livetalking/bin/python"), ok=True)
    sysbin = os.path.join(box["home"], "sysbin")
    _stub_python(os.path.join(sysbin, "python3"), ok=False)
    _shadow_path(box, sysbin)

    got = _run(box, monkeypatch, env_globs=HOME_GLOBS)
    assert got == env_py, f"stale cache won: {got}"
    out = capsys.readouterr().out
    assert "ignoring stale cached interpreter python3" in out
    # And the cache must now hold the real env, not python3.
    assert ".conda/envs/livetalking" in open(cache).read()


def test_fallback_interpreter_is_never_cached(box, monkeypatch):
    """REGRESSION: caching a torch-less python3 poisoned every later deploy."""
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "ENV_GLOB_SH", ['"/nonexistent/*/bin/python"'])
    monkeypatch.setattr(dz, "PYTHON_CANDIDATES", ["python3"])
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    monkeypatch.delenv("LT_REMOTE_PYTHON", raising=False)

    # Provide exactly one bare interpreter so the fallback path is taken.
    sysbin = os.path.join(box["home"], "onlysys")
    _stub_python(os.path.join(sysbin, "python3"), ok=False)
    _shadow_path(box, sysbin)

    dz.resolve_python({"username": "u", "host": "h", "port": 22}, "~/livetalking")
    cache = os.path.join(box["home"], "livetalking/.deploy/python")
    assert not os.path.exists(cache), "a fallback interpreter must not be cached"


def test_no_python_at_all_fails_loudly(box, monkeypatch):
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "ENV_GLOB_SH", ['"/nonexistent/*/bin/python"'])
    monkeypatch.setattr(dz, "PYTHON_CANDIDATES",
                        ["/nonexistent/py1", "/nonexistent/py2"])
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    monkeypatch.delenv("LT_REMOTE_PYTHON", raising=False)
    with pytest.raises(SystemExit) as exc:
        dz.resolve_python({"username": "u", "host": "h", "port": 22},
                          "~/livetalking")
    assert exc.value.code == 1
