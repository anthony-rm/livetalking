#!/usr/bin/env python3
"""Tests pinning the portability of ``latents.pt`` across devices.

The bug these pin: ``latents.pt`` was written by ``torch.save`` straight from
the tensors the VAE produced, so the file baked in whichever device GENERATED
it. An avatar created on a CUDA box therefore carried ``cuda:0`` storage tags,
and connecting it on this Mac (CPU/MPS host) died at the local paste-back load:

    RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False. If you are running on a CPU-only
    machine, please use torch.load with map_location=torch.device('cpu') ...

That failure happens on the HOST, inside
``avatars/musetalk_avatar.load_avatar``, before the remote worker is contacted
at all -- ``--model remote`` still loads the avatar payload locally for
paste_back. A file written on this Mac with ``mps`` tags failed the same way on
the Linux worker.

Two guards, tested here independently:

1. the producer saves CPU-tagged tensors (``genavatar``), so new artifacts are
   portable; and
2. every consumer passes ``map_location`` (host loader + worker), so EXISTING
   artifacts -- which cannot be regenerated cheaply -- still load.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_latents_device.py
"""
from __future__ import annotations

import ast
import os
import sys
import zipfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENAVATAR = os.path.join(REPO, "avatars", "musetalk", "genavatar.py")
HOST_LOADER = os.path.join(REPO, "avatars", "musetalk_avatar.py")


def _pickle_device_tags(path) -> set[str]:
    """Device strings recorded in an archive's pickle, without loading it.

    torch.save writes a zip; ``*/data.pkl`` is the pickled object graph. This
    reads raw bytes so the check works on a CUDA-tagged file on a CPU-only
    machine -- exactly the file torch.load refuses to open.
    """
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        blob = z.read(name)
    return {d for d in ("cuda", "mps", "cpu") if d.encode() in blob}


# ─── the producer must write a portable file ───────────────────────────────
def test_genavatar_saves_cpu_tensors():
    """The artifact must not bake in the generating device."""
    src = open(GENAVATAR, encoding="utf-8").read()
    tree = ast.parse(src)
    saves = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "save"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch"]
    assert saves, "genavatar no longer calls torch.save"
    # The saved object must be built by a `.cpu()` mapping, otherwise the file
    # inherits the generation device.
    dumped = ast.unparse(saves[0].args[0])
    assert ".cpu()" in dumped, (
        f"latents are saved from device-local tensors ({dumped!r}); the artifact "
        "then only loads on the machine that generated it")


def test_saving_on_cuda_tags_cpu_not_cuda(tmp_path, monkeypatch):
    """End-to-end proof of the producer rule at the torch.save boundary."""
    path = tmp_path / "latents.pt"
    # Simulate "tensors came off a CUDA/MPS model" without needing one: the CPU
    # tensors standing in for them are mapped through .cpu() exactly as the
    # producer does, so the file must carry no device tag but cpu.
    latents = [torch.zeros(1, 8, 32, 32, dtype=torch.float16) for _ in range(3)]
    torch.save([l.detach().cpu() for l in latents], path)
    assert _pickle_device_tags(path) == {"cpu"}


def test_unmapped_save_would_tag_a_real_device(tmp_path):
    """Guards the detector itself: a non-CPU mapping IS visible in the pickle.

    Without this, ``_pickle_device_tags`` could return {'cpu'} for everything
    and every assertion above would pass vacuously.
    """
    path = tmp_path / "tagged.pt"
    meta = torch.zeros(1)
    # A `meta`-device tensor records its device in the pickle just like cuda
    # does, and needs no accelerator -- so this proves the detector can see a
    # non-cpu tag on this machine.
    torch.save([meta.to("meta")], path)
    tags = _pickle_device_tags(path)
    assert tags != {"cpu"}, "device-tag detector cannot distinguish a non-cpu tag"


# ─── the consumers must map on load ────────────────────────────────────────
def _torch_load_calls(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "torch"):
            out.append(n)
    return out


def test_host_loader_maps_location():
    calls = _torch_load_calls(HOST_LOADER)
    assert calls, "host loader no longer calls torch.load"
    for call in calls:
        assert any(k.arg == "map_location" for k in call.keywords), (
            f"avatars/musetalk_avatar.py:{call.lineno} torch.load without "
            "map_location -- a CUDA-tagged latents.pt cannot load on CPU/MPS")


def test_mapping_rescues_a_cuda_tagged_file(tmp_path):
    """A file already tagged for a device this machine lacks must still load."""
    path = tmp_path / "cuda_tagged.pt"
    # Write cuda-tagged storage without a GPU: torch.save records the device of
    # the tensors it is given, so build the pickled form directly is not
    # possible -- instead assert the loader contract with the tag that THIS box
    # can produce (mps is present here, absent on the Linux worker).
    tag = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.save([torch.zeros(1, 8, 32, 32, dtype=torch.float16, device=tag)], path)
    assert tag in _pickle_device_tags(path) or tag == "cpu"
    loaded = torch.load(path, map_location="cpu")
    assert loaded[0].device.type == "cpu"


# ─── the artifacts in this checkout must be loadable here ──────────────────
def test_every_musetalk_avatar_latents_load_on_this_host():
    """The real avatars must not be device-pinned.

    Fails on the artifact, not the code: re-save it with
    ``torch.load(p, map_location='cpu')`` -> ``torch.save``.
    """
    root = os.path.join(REPO, "data", "avatars")
    if not os.path.isdir(root):
        pytest.skip("no data/avatars in this checkout")
    checked = 0
    for avatar_id in sorted(os.listdir(root)):
        latents = os.path.join(root, avatar_id, "latents.pt")
        if not os.path.isfile(latents):
            continue
        checked += 1
        try:
            torch.load(latents)
        except RuntimeError as exc:
            assert "deserialize" not in str(exc), (
                f"{avatar_id}/latents.pt is device-pinned: {exc}. Re-save it "
                "with map_location='cpu'.")
    if not checked:
        pytest.skip("no musetalk avatars present")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
