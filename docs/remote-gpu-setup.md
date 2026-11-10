# Remote GPU box — installation (SSH-only, no exposed ports)

Target box in this deployment: **`ecpkn_bac@ecpkn-server-System-Product-Name`**
(`10.130.105.203`), Ubuntu/Debian-family Linux, **NVIDIA GeForce RTX 3060 Ti**,
CUDA driver supporting **torch cu124**. The host is the macOS machine running the
LiveTalking server; the box only runs `remote.worker`.

Already verified on the box (2026-11-09), so these steps are done:

| Item | State |
|------|-------|
| `torch` with CUDA 12.4 | present, `torch.cuda.is_available() → True` |
| `git curl rsync tar gzip gcc g++ python3` | all present |
| conda env `livetalking` | exists at **`~/.conda/envs/livetalking`** |

Two notes that follow from what you reported:

- The env is under **`~/.conda/envs/`**, not `~/miniconda3/envs/`. The deploy
  globs for `<prefix>/envs/*/bin/python`, so it finds it; if you ever recreate the
  env elsewhere, pass `LT_REMOTE_PYTHON=/abs/path/to/envs/livetalking/bin/python`.
- `torch.version` printed a module path, not a version — that is because the
  command asked for `torch.version` (the module) instead of
  `torch.__version__`. Harmless; `torch.version.cuda` came back `12.4` and
  `is_available()` was `True`, which is what matters.

**Security model — read this first.** The worker's gRPC protocol is plaintext and
unauthenticated. It therefore listens on **`127.0.0.1` only**; the host reaches it
through an `ssh -L` local forward, which the server opens for you. Consequences:

- **No inbound firewall rule is needed or wanted.** Do not open 50051.
- SSH is the only way in. If SSH works, the GPU is reachable; if it does not,
  nothing is.
- The only account needed is your existing SSH account. No service user, no
  tokens, no TLS certificates.

Run the steps in order, on the box. **No step needs `sudo`.** This is written for
a shared GPU server where you are a normal user: everything lives under your
`$HOME`, nothing touches system config, and the deploy falls back to `tar` when
`rsync` is not installed.

Two things genuinely cannot be done without an administrator, and both are
checked early so you find out now rather than after installing 4 GB of weights:

| Thing | If it is not already right |
|-------|----------------------------|
| `AllowTcpForwarding` in sshd | **You cannot change it.** Ask the admin for `AllowTcpForwarding local`. Without it the tunnel cannot open and this design cannot work. `./scripts/check_ssh_forwarding.sh` tells you in one step. |
| A C++ toolchain | Not needed on your box (`gcc`/`g++` are present). Only matters when a package has no prebuilt wheel. |

**Key facts about your box:**

- worker binds only `127.0.0.1:50051` → no firewall rule, not reachable externally
- the host tunnels in with `ssh -L` → **SSH is the only way in**
- your env: `~/.conda/envs/livetalking` (torch cu124, RTX 3060 Ti)

Nothing else needs root: Python, CUDA torch, the deps, the weights and the
worker all live in your home directory.

---

## 0. Check the GPU and the driver (no sudo)

```bash
nvidia-smi
```

You need a driver new enough for the CUDA wheel you will install: **`cu124`
requires driver ≥ 550**, `cu121` requires ≥ 525. Note the driver version and the
GPU name — the deploy verifies `device == cuda` later.

---

## 1. Check what the box already provides (no sudo)

Do not install anything yet — find out what is missing, because you cannot
install system packages.

```bash
for b in git curl rsync tar gzip gcc g++ python3; do
  printf '%-8s ' "$b"; command -v "$b" >/dev/null && echo ok || echo MISSING
done
```

How to read the result:

- **`tar` + `gzip` missing** — very unlikely (they ship with every Linux). If
  they really are missing, the deploy cannot copy files at all; ask the admin.
- **`rsync` missing** — fine. The deploy detects it and streams a tarball over
  SSH instead. You just lose incremental transfers, which barely matters since
  each release is a fresh directory.
- **`git` missing** — fine, you never clone on the box; the deploy ships the
  tree.
- **`gcc`/`g++` missing** — fine unless a `pip install` later tries to build from
  source. Watch step 8; the error is unmistakable (`error: command 'gcc' failed`).
- **`curl` missing** — needed only for the MuseTalk UNet downloader; you can copy
  weights from the host instead (step 9, option B).

If you have `sudo` after all, the one command that makes life easier is
`sudo apt-get install -y rsync` (or `dnf`). Optional, not required.

---

## 2. Miniconda (no sudo, user-local)

```bash
cd ~
curl -fsSL -o /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
rm -f /tmp/miniconda.sh
"$HOME/miniconda3/bin/conda" init bash
```

Then open a **new shell** (or `source ~/.bashrc`) so `conda` is on PATH.

```bash
# What your env's interpreter actually is -- record this, the deploy needs it.
conda run -n livetalking python -c "import sys; print(sys.executable)"
```

The deploy finds it automatically: it globs `<prefix>/envs/*/bin/python`, which
covers BOTH a user-level `~/.conda/envs/livetalking` (what `conda create` gives
you when your base is elsewhere) and `~/miniconda3/envs/livetalking`. It prefers
an env literally named `livetalking`, and only accepts one that can
`import torch, grpc`. If yours lives somewhere unusual, set it explicitly:

```bash
LT_REMOTE_PYTHON=/absolute/path/to/envs/livetalking/bin/python ./deploy.sh
```

Whatever it picks is cached in `~/livetalking/.deploy/python` and reused, so it
resolves once, not on every deploy.

---

## 3. The `livetalking` env with CUDA torch (no sudo)

```bash
conda create -y -n livetalking python=3.12
conda activate livetalking
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify before going further.** This is the step that silently goes wrong —
a bare `pip install torch` installs the CPU build and everything downstream is
wasted:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: a `+cu124` version, a CUDA version string, `True`, and your GPU name.
If `is_available()` is `False`: you are in the wrong env, or the driver is older
than the wheel, or you installed the CPU wheel. Fix it now.

Leave the env **deactivated** afterwards (`conda deactivate`). The deploy runs
the worker with the env's absolute interpreter path, which works whether or not
the env is active — and a non-interactive SSH session never sees `conda activate`
from `.bashrc` anyway.

---

## 4. The install root

```bash
mkdir -p ~/livetalking
```

That is all. **Do not clone the repo here.** The deploy creates
`~/livetalking/releases/<rev>-<stamp>/` and points a `current` symlink at the
newest one. Weights, avatar data and logs live at the root and are shared by
every release, so a deploy is ~6 MB and a rollback cannot lose a checkpoint.

---

## 5. SSH key for the host (no sudo)

The host needs to log in without a password so the deploy and the tunnel work
unattended. You have two ways; **try the automatic one first**, it needs the
password you already use for SSH.

```bash
# ON THE HOST (this Mac), once:
./deploy.sh --setup-key
```

If that fails (for example the server forbids password auth, or you would rather
not put the password in the keychain), use the no-password path — it prints the
key and the exact commands:

```bash
./deploy.sh --print-key
```

Then paste the printed block **on the box**, logged in as yourself. This needs no
privileges at all:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
grep -qxF 'ssh-ed25519 AAAA... user@mac' ~/.ssh/authorized_keys 2>/dev/null || \
  echo 'ssh-ed25519 AAAA... user@mac' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Check it from the host — it must not ask for a password:

```bash
ssh -o BatchMode=yes ecpkn_bac@10.130.105.203 true && echo SSH_OK
```

If this prints `Permission denied (publickey)`, the key did not land: confirm you
appended to the right `~/.ssh/authorized_keys`, that `~` is `/home/<you>` (not
root's), and that `chmod` values are 700 for `.ssh` and 600 for the file. Note
that `chmod` on a home directory owned by you is allowed; if `~` itself is
group-writable, sshd may still refuse — `chmod g-w ~` fixes that without sudo.

---

## 6. Verify SSH port forwarding (no sudo — read only)

The tunnel needs TCP forwarding on the server side. You cannot change this
setting, only check it and escalate to the admin if it is off.

Do **not** test this with a single `ssh -L` + `curl` against the worker: if the
worker is not running yet (it isn't, at this stage), you get `000` and cannot tell
"forwarding forbidden" from "nothing listening". Use the diagnostic, which
separates the four causes:

```bash
# on the HOST (this Mac)
./scripts/check_ssh_forwarding.sh
```

It runs four tests in order, each naming its own cause:

| Test | What it proves |
|------|----------------|
| A. key login | auth works at all |
| B. forward to the box's own sshd on :22 | **forwarding is permitted** (sshd is always listening, so this isolates the SSH permission from the worker) |
| C. is anything listening on the worker port | the worker is running, and whether it is bound to loopback or `0.0.0.0` |
| D. gRPC Health through the tunnel | end-to-end, the real thing the app does |

Typical output when only step B matters right now:

```
A. SSH login to ecpkn_bac@10.130.105.203
  ✓ key-based login works

B. Is TCP forwarding permitted?  (forwards to the box's own sshd on :22)
  ✓ forwarding works: reached the remote sshd (SSH-2.0-OpenSSH_9.6)
```

If B fails with `channel_setup_fwd_listener_tcpip: administratively prohibited`,
forwarding is disabled on the server and **only the administrator can enable it**:

> Ask your admin to set `AllowTcpForwarding local` (or `yes`) in
> `/etc/ssh/sshd_config` and reload sshd. No port needs to be opened; this is a
> permission to forward, not a firewall rule.

If they refuse, the fallback is direct mode (`LT_SSH_TUNNEL=0`), which requires
the worker to bind `0.0.0.0` — do **not** do that on a shared server; it exposes
an unauthenticated GPU port to everyone on the network. In that situation, the
honest answer is that this box cannot be used safely from outside.

`sshd -T` (used in earlier drafts) requires root and usually fails with
`Missing privilege separation directory` for a normal user, so it is not a
reliable check — test the capability instead.

---

## 7. First deploy, from the host

```bash
# on this Mac, in the repo
./deploy.sh
```

That copies the source, links the shared dirs, starts the worker on
`127.0.0.1:50051`, opens the tunnel and asserts that the worker reports the same
revision and `device == cuda`.

The first run will stop at the dependency install, because the code only exists
on the box after the copy. That is expected. You will see:

```
[deploy] NOTE: grpc is not installed in that env yet, so the worker cannot start...
[deploy] current -> releases/...
[deploy] worker NOT restarted: its dependencies are missing in that interpreter
```

That is a clean stop, not a failure: the source is deployed and nothing was left
half-running. Do step 8, then start the worker without re-copying anything:

```bash
./deploy.sh --restart-only
```

---

## 8. Worker dependencies (once, after the first deploy)

The worker needs the Python deps from `requirements.txt`, in the `livetalking`
env. It does **not** need `dlib`/`face_recognition` — those are only used when
*generating* an avatar, which happens on the host.

First pin down which interpreter to use. Use the one the deploy cached (it is the
one it will actually run), falling back to conda's answer:

```bash
PY=$(cat ~/livetalking/.deploy/python 2>/dev/null) \
  || PY=$(conda run -n livetalking python -c "import sys; print(sys.executable)")
echo "using: $PY"
"$PY" -c "import sys, torch; print(sys.executable, torch.__version__)"
```

`using:` must be **inside your env** (`.../envs/livetalking/bin/python`). If it
says `/usr/bin/python3`, the deploy picked the system interpreter — delete the
stale cache and re-run:

```bash
rm -f ~/livetalking/.deploy/python
# or force it: LT_REMOTE_PYTHON=/abs/path/envs/livetalking/bin/python ./deploy.sh
```

A bare `python3` cannot be imported into, so `pip install` there would not help
the worker at all.

Then install and patch, from the deployed release:

```bash
cd ~/livetalking/current
"$PY" -m pip install -r requirements.txt
"$PY" scripts/patch_basicsr.py
```

`patch_basicsr.py` is not optional: torchvision ≥ 0.17 removed
`torchvision.transforms.functional_tensor`, which basicsr 1.4.2 imports, so
`realesrgan`, `gfpgan` and `basicsr` all fail to import without it.

`sentencepiece`, `tqdm` and similar usually arrive as wheels. If `pip` starts
compiling and fails with `error: command 'gcc' failed`, that one package needs a
build toolchain — your box has `gcc`/`g++`, so it should not happen.

Then sanity-check the env end to end:

```bash
cd ~/livetalking/current
"$PY" -m remote.worker --check
```

Expected one line, e.g.
`revision=<rev> device=cuda backend=lazy cwd=/home/.../livetalking/current`
plus `gpu=NVIDIA GeForce RTX 3060 Ti`. If it says `device=cpu`, stop and re-read
step 3.

If it fails with **`No module named 'google'`**, protobuf is missing. `grpcio`
does not install it, and the generated stubs import `google.protobuf` directly:

```bash
"$PY" -m pip install "protobuf==7.35.1"    # version the stubs validate against
```

---

## 9. Weights

The worker needs the model weights, which are **not** shipped by the deploy
(they are gigabytes). Two options.

**A — download on the box** (needs outbound HTTPS):

```bash
cd ~/livetalking/current
PY=$(cat ~/livetalking/.deploy/python 2>/dev/null) \
  || PY=$(conda run -n livetalking python -c "import sys; print(sys.executable)")

# MuseTalk: VAE + V1.5 UNet + whisper + face-parse + dwpose + syncnet
"$PY" scripts/download_musetalk_weights.py
bash   scripts/download_musetalk_unet.sh      # 3.4 GB, resumable curl

# Enhance pipeline: Real-ESRGAN + GFPGAN
"$PY" scripts/download_enhance_weights.py
```

**B — copy from the host** (if the box cannot reach the internet). From this Mac,
using tar so it works even when the box has no rsync:

```bash
tar -C . -czf - --exclude='./models/whisper' --exclude='*.pth.bak' models \
  | ssh ecpkn_bac@10.130.105.203 'mkdir -p ~/livetalking && tar -xzf - -C ~/livetalking'
```

`models/whisper/` is deliberately excluded: whisper is the host's ASR feature
extractor and stays on this machine. Skip `models/wav2lip.pth` too unless you
stream a Wav2Lip avatar.

(If both machines have rsync, `rsync -az --stats --exclude 'whisper/' models/
ecpkn_bac@10.130.105.203:~/livetalking/models/` is faster — it resumes.)

Then confirm what the worker will look for:

```bash
cd ~/livetalking
ls -la models/sd-vae models/musetalkV15 models/RealESRGAN_x4plus.pth \
       models/GFPGANv1.4.pth 2>&1
```

---

## 10. facexlib weights (avoid a 195 MB download mid-job)

The enhance stack resolves its face-detection weights **relative to the working
directory**. Either let the first enhance job download them, or copy them:

```bash
# on the host
mkdir -p gfpgan/weights
tar -C . -czf - gfpgan/weights \
  | ssh ecpkn_bac@10.130.105.203 'mkdir -p ~/livetalking/gfpgan && tar -xzf - -C ~/livetalking/gfpgan'
```

The deploy symlinks `~/livetalking/gfpgan/weights` into every release, so this
survives updates.

---

## 11. Avatars

Avatar artifacts are model-specific and 70–360 MB each, and they are **not**
shipped by a normal deploy. Copy the ones you stream:

```bash
# on the host
./deploy.sh --avatars anthony007
```

That lands them in the shared `~/livetalking/data/avatars/`, visible to every
release.

Not sure which avatar the host is configured for? `grep avatar_id config.yaml` on
the Mac.

---

## 12. Keep the worker running (optional — check this is even allowed)

The deploy starts the worker with `nohup` and a pidfile. That survives an SSH
disconnect but **not** a reboot. On a shared server, a systemd *user* instance is
often disabled, so check before trying:

```bash
systemctl --user status >/dev/null 2>&1 \
  && echo "user services available" \
  || echo "no systemd user bus — stick with nohup/tmux (below)"
```

**If user services work**, this is the cleanest option:

```bash
mkdir -p ~/.config/systemd/user
PY=$(cat ~/livetalking/.deploy/python)
cat > ~/.config/systemd/user/livetalking-worker.service <<EOF
[Unit]
Description=LiveTalking remote GPU worker
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/livetalking/current
ExecStart=$PY -m remote.worker --host 127.0.0.1 --port 50051
Restart=on-failure
RestartSec=3
StandardOutput=append:%h/livetalking/logs/worker.log
StandardError=append:%h/livetalking/logs/worker.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now livetalking-worker
loginctl enable-linger "$USER"      # keep it running after you log out
```

The deploy detects this unit and uses `systemctl --user restart` instead of
`nohup`, so the two do not fight. Check it:

```bash
systemctl --user status livetalking-worker --no-pager
ss -ltnp | grep 50051     # must show 127.0.0.1:50051, never 0.0.0.0
```

**If user services are unavailable** (common on a headless shared box), the
nohup path is already fine for anything except a reboot — and a worker does not
need to survive reboots. If you want it to, `tmux` is the no-privilege answer:

```bash
PY=$(cat ~/livetalking/.deploy/python)
tmux new -d -s lt-worker \
  "cd ~/livetalking/current && exec $PY -m remote.worker --host 127.0.0.1 --port 50051 >> ~/livetalking/logs/worker.log 2>&1"
tmux ls                                  # lt-worker: 1 windows
```

Do not add this to the deploy's search path: the deploy already kills a stray
worker and launches its own, and a tmux session cannot be managed that way. Use
one mechanism or the other, not both.

---

## 13. Confirm nothing is exposed

From the box:

```bash
ss -ltnp | grep 50051
```

Must be `127.0.0.1:50051` (loopback). If it says `0.0.0.0:50051` or `*:50051`,
the worker was started with the old `--host 0.0.0.0`; stop it — the deploy
starts it correctly from now on.

From **another machine** on the same network, this must fail (timeout or
refused):

```bash
nc -vz -w 3 10.130.105.203 50051
```

Failure is the desired result. SSH (port 22) is the only port that should answer.

---

## Final verification, from the host

```bash
./deploy.sh              # copies, restarts, opens the tunnel, verifies revision + cuda
./deploy.sh --rollback   # optional: prove rollback works before you need it
```

Expected tail:

```
[deploy] current -> releases/<rev>-<stamp>
[deploy] worker RESTARTED=nohup PID=...
[deploy] opening SSH tunnel for verification
[deploy] verifying worker at 127.0.0.1:50051 ...
[deploy] OK: revision <rev> on cuda (NVIDIA ...)
[deploy] deployed <rev>. Roll back with: ./deploy.sh --rollback
```

Then start the server on the Mac and watch the first seconds of its log:

```bash
./run.sh
# expect:
# [tunnel] forwarding 127.0.0.1:50051 -> box:127.0.0.1:50051
# [remote] worker 127.0.0.1:50051 ready: NVIDIA ..., revision <rev>
```

And in the browser: `/admin.html` → **Remote GPU Worker** → **Connect** (or
**Test** for the login only) should show all five steps green, with the GPU name
and a matching revision.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ssh: connect to host ... port 22: Connection refused` | sshd not running, or the box is off the network. Check on the box: `systemctl status ssh`. |
| `Permission denied (publickey,password)` | Key not installed. Run `./deploy.sh --setup-key` on the host. |
| `channel_setup_fwd_listener_tcpip: cannot listen to port: 50051` | Something on **the host** already holds 50051 (a previous tunnel). `pkill -f 'ssh -N -L'`, then retry. |
| `administratively prohibited: open failed` | `AllowTcpForwarding` is off on the box — step 6. |
| tunnel is up but `Health` times out | The worker is not listening, or is listening on a different port. On the box: `ss -ltnp \| grep 50051` and `tail ~/livetalking/logs/worker.log`. |
| `remote interpreter: ... (WARNING: torch/grpc not importable)` | Deps missing in that env — step 8. |
| `worker answered but reports no revision` | Old code running. `./deploy.sh` restarts it. |
| `worker exited immediately after restart` | Read `~/livetalking/logs/worker.log`; usually a missing import (step 8) or a wrong interpreter. |
| `ModuleNotFoundError: No module named 'utils.logger'` | The worker was started outside the repo root. Start it from `~/livetalking/current` as the deploy does. |
| `device=cpu` in `--check` | CPU torch wheel installed. Redo step 3 **inside the env**. |
| `CUDA out of memory` | Another process holds the GPU. `nvidia-smi`, then `POST /api/model/unload` from the host, or stop the other job. |
| Slow (below 25 `inferfps`) | Check the tunnel is not the bottleneck: `ssh -o Compression=no` is already off; test on the LAN first, then over WAN. |

Logs: `~/livetalking/logs/worker.log` (worker), `data/tunnel.log` on the host
(tunnel), and `livetalking.log` on the host (server).
