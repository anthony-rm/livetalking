# Remote GPU worker (Option B) — local host, remote CUDA box over SSH

Keep the LiveTalking server wherever it runs best (Mac/PC/VM: HTTP API, TTS, ASR
feature extraction, paste-back, WebRTC/RTMP/virtualcam output, recording). Move
the two heavy GPU stages to a remote Linux box with an NVIDIA GPU:

  1. **avatar inference** — MuseTalk UNet+VAE / Wav2Lip convnet (`InferBatch`)
  2. **base-video enhance** — RVM matting + Real-ESRGAN + GFPGAN (`EnhanceVideo`)

The transport is gRPC. Tensors cross the wire as `(dtype, shape, raw bytes)` —
no pickle. See `remote/proto/inference.proto` for the full contract.

```
 Host (any OS, MPS/CPU)                       GPU server (Linux, CUDA)
 ┌──────────────────────────────────┐         ┌───────────────────────────┐
 │ app.py  (aiohttp + WebRTC)       │         │ remote.worker             │
 │ TTS ─► ASR features (whisper)    │         │  ├─ LoadModel(backend)    │
 │   │                              │  gRPC   │  ├─ LoadAvatar(id)        │
 │   └─► infer_batch ───────────────┼────────►│  ├─ InferBatch  (GPU)     │
 │         ▲ pred crops             │◄────────┼──┘                        │
 │         │                        │         │  └─ EnhanceVideo (GPU)    │
 │   paste_back (cv2, local) ─► out │         │     RVM+ESRGAN+GFPGAN     │
 └──────────────────────────────────┘         └───────────────────────────┘
```

## Setup — the GPU server

**Full step-by-step install: [`docs/remote-gpu-setup.md`](remote-gpu-setup.md).**
It assumes **no root on the box** — everything lives under `$HOME`, and the only
things that may need an administrator are `AllowTcpForwarding` in sshd and,
rarely, a C++ toolchain.

Any Linux machine with an NVIDIA GPU. Driver only — you do **not** need the CUDA
toolkit installed, the pip wheels carry their own runtime. Check the driver is
new enough for the wheel you pick (`cu124` needs driver ≥ 550; `cu121` ≥ 525):

```bash
nvidia-smi
```

1. Create the environment and install CUDA-enabled torch:

   ```bash
   conda create -y -n livetalking python=3.12
   conda activate livetalking
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   Expect `True` and your GPU name. If `is_available()` is `False`, the driver is
   older than the wheel or you installed the CPU build — fix this before
   continuing, everything below is wasted effort otherwise.

2. Install the dependencies ONCE at the install root. You don't need to clone the
   repo here — `./deploy.sh` from the host creates `~/livetalking/releases/...`
   and the `current` symlink. Create the root and install there:

   ```bash
   mkdir -p ~/livetalking && cd ~/livetalking
   # the deploy puts source in releases/ + current/; install deps from any copy
   pip install -r requirements.txt
   python scripts/patch_basicsr.py     # required: torchvision >=0.17 breaks basicsr
   ```

   Run that AFTER the first `./deploy.sh` (it ships `scripts/patch_basicsr.py`),
   or from a temporary clone. Put the install root's path in `PYTHONPATH` if you
   prefer a single `pip install` target for all releases:

   ```bash
   echo 'export PYTHONPATH=$HOME/livetalking/current' >> ~/.bashrc
   ```

   `paramiko`/`keyring` in `requirements.txt` are for the *host* (SSH login +
   credential storage); harmless here. dlib/`face_recognition` are NOT needed —
   they are only used when *generating* an avatar, which stays on the host.

3. Fetch the model weights. Either run the downloaders, or copy from the host:

   ```bash
   python scripts/download_musetalk_weights.py
   bash   scripts/download_musetalk_unet.sh      # 3.4 GB, resumable curl
   python scripts/download_enhance_weights.py
   ```

   The worker only needs:
   - `models/musetalkV15/` + `models/sd-vae/` (MuseTalk)
   - `models/wav2lip.pth` (Wav2Lip)
   - `models/RealESRGAN_x4plus.pth`, `models/GFPGANv1.4.pth`, and
     `gfpgan/weights/` (enhancer; facexlib resolves those **cwd-relative**, so
     copy them or let the first enhance job download ~195 MB)
   - `data/avatars/<id>/` for avatars you stream

   Not needed: `models/whisper/` (ASR stays on the host).

4. Start the worker **from the repo root** — `avatars/musetalk/utils/__init__.py`
   does a `sys.path.append` that can shadow the project's `utils` package if the
   working directory is wrong, breaking `import utils.logger`:

   ```bash
   cd ~/livetalking/current          # the release the deploy activated
   python -m remote.worker --backend musetalk
   ```

   You normally do NOT start it by hand: `./deploy.sh` on the host restarts it.
   Use this only for a first smoke test, and remember a hand-started worker is
   killed and replaced by the next deploy.

   The worker binds **`127.0.0.1:50051` by default** — loopback only. That is
   deliberate: the protocol is plaintext and unauthenticated, so it must not be
   exposed. The host reaches it through an `ssh -L` forward, which the server
   opens for you (see *SSH-only transport* below). Pass
   `--host 0.0.0.0` only on a trusted LAN where you accept the exposure.
   Preload a backend with `--backend musetalk|wav2lip`, or omit it to load lazily
   on the first `LoadModel`. `python -m remote.worker --check` prints the
   revision/device/backend it would serve and exits — the fastest way to prove
   the box sees CUDA.

5. Reachability. **No firewall rule is needed**: the port stays closed. Health
   check from the host goes through the tunnel:

   ```bash
   ./deploy.sh --dry-run                  # local checks, touches nothing
   python -m remote.tunnel start          # open the ssh -L forward
   python -c "from remote.client import probe; print(probe('127.0.0.1:50051'))"
   ```

   If the tunnel cannot open:

   ```bash
   # on the box -- forwarding must be allowed
   sudo sshd -T | grep -i allowtcpforwarding     # want: yes / local
   ```

   A hand-opened tunnel works too, and the server reuses a port that is already
   listening:

   ```bash
   ssh -N -L 50051:127.0.0.1:50051 user@gpu-host
   ```

## Setup — the host

**This machine runs everything by default.** `config.yaml` ships
`worker_mode: local`, so plain `./run.sh` keeps avatar inference, avatar
generation and video enhance on this machine. Nothing is sent to the box —
having an address configured is not the same as using it.

To send GPU work to the box, flip the toggle: **`/admin.html` → Remote GPU
Worker → "Remote GPU worker"**. That switch covers every GPU task (inference,
avatar creation, enhance) and is saved in `data/remote_worker.json`, so it
survives a restart and wins over both the config file and the CLI.

The same choice is available without the UI:

```bash
./run.sh --worker_mode remote              # this run uses the box
LT_WORKER_MODE=remote ./run.sh             # same, from the environment
./run.sh --worker_mode local               # this run stays local
```

`--model remote` is still accepted as legacy shorthand for "worker_mode =
remote". `--remote_worker` / `LT_REMOTE_WORKER` only set the address; they do
not turn the worker on.

At startup the host logs the mode, and in remote mode the worker address, its
source, the local revision and whether the box answers on CUDA — so a dead or
stale worker is visible in the first seconds rather than on the first browser
connect.

### Configuring from the admin console

`/admin.html` → **Remote GPU Worker** card. A two-button mode switch (This
machine / Remote GPU worker), then four fields — host/IP, username, password,
SSH port — and three buttons.

The mode switch applies to the next connect, avatar task or enhance job; no
restart. If a model is already resident from the other mode the card says so —
unload it (**Model & System Control**) and the next connect loads it in the new
mode. `POST /api/remote/mode {"mode": "local"|"remote"}` is the API behind it.

**Connect** does the whole job in one step and reports each stage separately:

1. **SSH login** — a real paramiko login (runs `hostname`), so it proves the
   credentials, not just that port 22 is open.
2. **Authorize this host's key** — appends this machine's `~/.ssh/id_*.pub` to
   the box's `authorized_keys` (idempotent, no sudo). A password is only good
   for logging in; the `ssh -L` forward that carries the worker traffic runs
   `BatchMode=yes` and cannot answer a prompt, so the key has to be there.
3. **Key-based login** — proves that non-interactive login now works, with the
   same options the forward uses.
4. **SSH tunnel** — opens `ssh -N -L 127.0.0.1:50051:127.0.0.1:50051 user@box`.
5. **Worker health** — a gRPC `Health` call through the tunnel: device, GPU name,
   and whether the box runs the same revision as this checkout.

Credentials are only written once step 1 succeeds, so a typo'd host cannot
replace a working configuration. The password goes to the OS keychain (Keychain
Access on macOS, SecretService/KWallet on Linux, Credential Manager on Windows);
username and host go to `data/remote_worker.json`. Nothing sensitive is written
to the repo, `.env`, or logs, and the API never returns the password. Host keys
are trust-on-first-use: the first successful login pins the fingerprint, and a
later change is refused rather than silently accepted.

**Test** checks only the password login and pins the host key on first contact —
use it when you want to know whether the credentials work before granting the key.

**Disconnect** closes the forward and keeps it closed (the background supervisor
respects it). The saved login is kept; **Connect** re-opens it. The app can only
close a forward it started itself — a hand-run `ssh -N -L` in a terminal, or a
forward left over from an earlier server run, has to be closed there, and the
card says which case you are in.

The worker address is `host:worker_port` (50051 by default), derived from the
same host field; no separate address box exists. Resolution order is unchanged:

1. host saved from the admin console → `data/remote_worker.json`
2. `--remote_worker` CLI flag / `LT_REMOTE_WORKER` env var
3. unset → no worker to dial (toggling remote then fails the session loudly)

API behind the card: `GET /api/remote/status`, `POST /api/remote/mode`,
`POST /api/remote/connect`, `POST /api/remote/disconnect`,
`POST /api/remote/ssh-test` — all the usual `{code,msg,data}` shape.
`GET /api/remote/status` reports `mode` (`local`/`remote`) and
`remote_active`. `POST /api/remote/save`, `POST /api/remote/test`,
`POST /api/remote/credentials` and `POST /api/remote/credentials/clear` still
exist for scripting and for `./deploy.sh`, but the card no longer uses them.

Enhancement routes automatically with the toggle: with the mode set to
`remote`, enhance jobs (web flow and `/api/enhance/task`) run on the worker, and
the enhanced mp4 plus matte/person sidecars are zipped back into
`data/enhance/<id>/`, so the existing status/preview/recompose/avatar routes
work unchanged. Recompose (background swap) stays local — cheap CPU work.

## Updating the box: `./deploy.sh`

The box runs code from a release directory, not from a git checkout you edit by
hand. One command ships your working tree, swaps it in, restarts the worker and
proves the restart took:

```bash
./deploy.sh --dry-run      # local checks only; touches nothing
./deploy.sh --setup-key    # ONCE, if key-based SSH is not set up yet
./deploy.sh                # ship + restart + verify
./deploy.sh --rollback     # back to the previous release
```

**First-time box setup is a separate document**: `docs/remote-gpu-setup.md` has
the full install sequence (CUDA torch, deps, weights, avatars, sshd forwarding,
optional systemd unit) with the exact commands to run on the box.

What it does, in order:

1. **Local preflight** — resolves the revision (`remote/revision.py`: `LT_REVISION`
   > `.deploy/revision` > `git rev-parse`, with a `+` suffix when the tree has
   uncommitted edits) and refuses to run if `inference.proto` is newer than its
   generated stub. A stale stub is the classic silent host/worker mismatch.
2. **Remote preflight** — SSH in and check the install dir and the available
   tools. Picks `rsync` when present, otherwise falls back to streaming a tarball
   over the same SSH connection; **neither needs root on the box**. Clear error if
   the SSH key is missing.
3. **Copy into a new release** — into
   `~/livetalking/releases/<rev>-<stamp>/`, source files only: `models/`, `data/`,
   `gfpgan/weights/`, `.git/`, `.env`, caches and logs are excluded by one shared
   predicate, so both transports ship exactly the same set. That is ~6 MB / ~230
   files. Vendored `avatars/*/models/*.py` sources **are** included — they are
   code the worker imports, not weights.
4. **Share the big stuff** — the release symlinks `models`, `data`,
   `gfpgan/weights` (and `.env` if present) back to the install root, so weights
   and avatars are copied once ever and a rollback cannot lose them.
5. **Stamp the revision** — writes `.deploy/revision` next to the code it
   describes, which is what `Health` reports back.
6. **Swap `current`** — atomically repoints the `current` symlink. `mv -T` makes
   this replace the link instead of nesting inside the directory it points to.
7. **Restart** — prefers a `livetalking-worker` systemd unit (user, then system);
   otherwise stops the pidfile process, kills any worker started by hand, and
   launches `python -m remote.worker` under `nohup` with the log at
   `logs/worker.log`. It refuses to restart while `Health.busy` is true, i.e.
   while a session is mid-inference or mid-enhance, unless you pass `--force`.
8. **Verify** — polls gRPC `Health` for up to 90 s and fails the deploy unless
   the worker reports the same revision AND `device == "cuda"`. A deploy that
   copied files but did not restart cannot report success.
9. **Prune** — keeps the last 3 releases (override with `--keep N`).

Avatars are shipped separately because they are 70–360 MB each and mostly
already on the box:

```bash
./deploy.sh --avatars anthony007 my_new_avatar
```

Files land in the shared `~/livetalking/data/avatars/`, so every release sees
them and a rollback keeps them.

Environment knobs: `LT_REMOTE_DIR` (default `~/livetalking`),
`LT_REMOTE_TARGET` (`user@host`), `LT_REMOTE_PORT` (50051),
`LT_REMOTE_KEEP_RELEASES` (3). Credentials come from `data/remote_worker.json`
plus the OS keychain — the same values the admin console manages.

### Why ship the working tree, not `git pull`

The box may have no GitHub access, and you usually want the edit you just made —
including uncommitted work — to be what the box runs. Git would silently ship the
last commit and the deploy would look successful while running stale code. The
`+` dirty marker plus the post-restart revision check make the difference
visible instead of invisible.

### Why a symlinked release dir

Copying over a live checkout risks serving half-written files, and a failed
deploy leaves no way back. Releases + one atomic symlink mean a failure mid-copy
never reaches service and `--rollback` is a link flip. `prune` keeps 3 by
default so a bad deploy is recoverable without re-uploading.

### Version identity in the API

`Health` now returns two extra fields (`remote/proto/inference.proto`):

| Field | Meaning |
|-------|---------|
| `revision` | code revision the worker process was STARTED from (cached per process, so it cannot lie after a file swap) |
| `busy` | true while a GPU call holds the mutex |

The admin console's **Connect** shows revision drift, and the host logs
it at startup: with the mode set to `remote` it announces the worker address,
revision and CUDA state in the first seconds instead of letting the first
browser connect discover a dead or stale worker. `GET /api/remote/status` and
`POST /api/remote/test` also return `host_revision` / `revision_match`.

### Local tests

`pytest tests/test_deploy_remote.py tests/test_deploy_e2e.py` covers the deploy
logic and its quoting/symlink/exclude behaviour against a **fake** box
(`tests/fake_gpu_box.py` shims `ssh`/`rsync` to a temp dir) — no network, no
GPU, nothing on the real server. The restart step is stubbed there on purpose.

## What stays local vs remote

| Stage                              | Where | Why |
|------------------------------------|-------|-----|
| TTS                                | host  | low latency, streaming |
| ASR feature extraction (whisper/mel/hubert) | host | small model, feeds features |
| avatar inference (`inference_batch`) | GPU  | the heavy UNet/VAE/convnet |
| paste_back (crop → full frame)     | host  | cheap cv2; keeps the small crops on the wire |
| enhance (matte + SR + face restore) | GPU  | RVM/Real-ESRGAN/GFPGAN |
| enhance recompose (bg swap)        | host  | pure compositing, no GPU models |
| WebRTC/RTMP/virtualcam, recording  | host  | output devices live on the host |

## Bandwidth & latency

Per inference batch only the small pred crops cross the wire (e.g. MuseTalk
`256×256×3` uint8 per frame × batch ≈ 0.8 MB for batch 16), plus the tiny audio
features in. On a LAN that is a few ms; the worker returns crops, not full
frames, so paste-back stays local. Wired gigabit is strongly preferred for
real-time; check the timing gate (`inferfps`/`finalfps` ≥ 25) in the server logs.

## SSH-only transport (default)

The gRPC protocol is plaintext and unauthenticated, so the supported deployment
never exposes it. Instead:

```
 box                                   host (this Mac)
 127.0.0.1:50051  ◄──── ssh -L ────   127.0.0.1:50051
 (worker, loopback)     encrypted      (what the app dials)
```

- The worker binds **`127.0.0.1`** on the box (`remote/worker.py` default). No
  firewall rule is needed or wanted, and `nc box 50051` from anywhere else fails
  by design.
- The host opens the forward itself: `remote/tunnel.py` runs
  `ssh -N -L 127.0.0.1:50051:127.0.0.1:50051 -p <ssh port> user@box`
  with `BatchMode=yes` (never prompt) and `ExitOnForwardFailure=yes` (fail fast
  instead of sitting connected-but-useless), and supervises it in a daemon
  thread so a dropped link is repaired rather than silently breaking sessions.
- `remote.settings.effective_address()` returns `127.0.0.1:50051` in tunnel mode,
  so the avatar plugin, the enhance dispatcher, the deploy verifier and the admin
  card all agree on the address. Source shows as `ssh-tunnel`.
- Credentials are the SSH ones already managed by the admin console
  (`data/remote_worker.json` + OS keychain). The SSH port is `ssh_port` in that
  file, default 22.

Controls:

```bash
python -m remote.tunnel status        # enabled / running / listening / address
python -m remote.tunnel start|stop    # manual control
```

or the admin card (**Connect** / **Disconnect**), or
`POST /api/remote/tunnel {"action": "start"|"stop"|"mode", "enabled": bool}`.
Set `LT_SSH_TUNNEL=0` (or the card's mode action) for direct RPC on a trusted
LAN; that path dials the saved address as-is.

Requirements on the box: SSH key auth (the deploy's `--setup-key` does it) and
`AllowTcpForwarding` enabled (`yes` or `local`); otherwise the tunnel fails with
`administratively prohibited`.

Bandwidth note: SSH adds a little overhead but no meaningful latency on a
tunneled loopback port. The measured gate is still `inferfps`/`finalfps` ≥ 25
in the server log.

## Worker behaviour

- One backend + one avatar resident at a time (the app is single-model by
  design). `LoadAvatar` replaces the resident avatar; `RemoteAvatar` re-asserts
  residency on session creation, so switching back to a host-cached avatar
  re-ships it if the worker moved on.
- GPU work (inference, model load, enhance) is serialised by one mutex — VRAM
  use is not reentrant.
- `Unload` (via `POST /api/model/unload` on the host) frees worker VRAM.
- Falls back to MPS/CPU when CUDA is unavailable, which is useful for testing the
  transport locally but is slow.

## Regenerating the gRPC stubs

After editing `remote/proto/inference.proto`, run `scripts/gen_remote_stubs.sh`
(it also rewrites protoc's absolute `import inference_pb2` into a
package-relative import, which the generator does not do).

## Troubleshooting

- **`No remote worker address`** — set it in the admin console, or pass
  `--remote_worker` / export `LT_REMOTE_WORKER`. (That only sets the address;
  the mode toggle still has to be on `remote`.)
- **worker unreachable** — in the default SSH-only mode the port is loopback on
  the box, so a firewall is almost never the cause. Check `python -m
  remote.tunnel status` and `data/tunnel.log`. A tunnel that cannot open is
  usually key auth (`./deploy.sh --setup-key`) or `AllowTcpForwarding` off
  (`administratively prohibited`). In direct mode, `connection refused` means
  nothing is listening; a hang means dropped packets (firewall).
- **reachable but `cuda: false`** — CUDA torch is not installed in the worker's
  environment. Re-check step 1 *inside that env*.
- **`no model loaded; call LoadModel first`** — the host connected but the worker
  has no weights; confirm `models/` and that `--remote_backend` matches.
- **`ModuleNotFoundError: No module named 'utils.logger'`** — the worker was
  started outside the repo root, or as a file path instead of `-m remote.worker`.
- **slow first batch** — model load + `warm_up` happen on the first session;
  later batches reuse the resident model.
- **`SSH login failed (no usable key)`** — run `./deploy.sh --setup-key` once.
  It installs your public key using the keychain password via paramiko; nothing
  is put on a command line.
- **`revision mismatch: box serves X, local is Y`** — the files copied but the
  process did not restart, or it restarted from the wrong directory. Check
  `~/livetalking/logs/worker.log` on the box and that no second worker is
  running (`pgrep -af remote.worker`). Re-run `./deploy.sh`; it kills strays.
- **`worker answered but reports no revision`** — the box runs a build older
  than the revision field. Restart it once by hand, then deploy normally.
- **deploy refuses to restart, "worker stayed busy"** — a session is live
  (inference or enhance holds the GPU mutex). Stop it, or `./deploy.sh --force`.
- **`remote/proto/inference.proto is newer than inference_pb2.py`** — run
  `scripts/gen_remote_stubs.sh`; the deploy refuses to ship a stub that does not
  match the .proto.
- **`No module named 'google'`** — protobuf is missing. grpcio does **not** bundle
  it, and the generated `inference_pb2` imports `google.protobuf` directly, so the
  worker dies during import even though `pip install grpcio` succeeded. Install the
  version the stubs demand (they call `ValidateProtobufRuntimeVersion`):
  `python -m pip install "protobuf==7.35.1"`. If the stubs are regenerated, bump
  this pin to match.
- **`restart failed on the box (5)` / `PREFLIGHT=FAILED`** — best case. The
  dependency preflight failed *before* any process started, so nothing was left
  running and every missing module is named. Install them, then use
  `./deploy.sh --restart-only` (no re-copy).
- **`restart failed on the box (4)` / `RESTARTED=DEAD`** — the worker started but
  died. The log excerpt is this run only (`tail -c +LOG_MARK`), because the log
  is append-only and a plain `tail -30` can show a *previous* worker's output.
- **`restart failed on the box (6)`** — `~/livetalking/current` is missing or not
  a symlink. Re-run a normal deploy.
- **`SSH connection failed ... the command never ran`** — genuine transport
  failure (key, host key, forwarding). The deploy prints what `ssh` actually said;
  copy that line, it distinguishes them. A *remote* command exiting non-zero is
  reported as `remote command failed (N)` instead, so the two are never confused.
- **`ABORTING due to invalid path from sender`** — the macOS/openrsync file-list
  quirk. Fixed by passing the list in a temp file instead of `--files-from=-`;
  if it reappears, the argv regressed (`tests/test_deploy_e2e.py::test_rsync_file_list_is_passed_via_a_file_not_a_pipe`).
- **`rsync: unrecognized option`** — the host's rsync is openrsync (macOS). The
  deploy passes an explicit file list, which both rsyncs understand; if you edit
  the flags, keep them portable.
- **`tar stream interrupted`** — the far end died. Usually a full disk on the box
  or `tar`/`gzip` missing; check `df -h ~` and `command -v tar gzip` there.
- **`ModuleNotFoundError: No module named 'avatars.musetalk.models'` after a
  deploy** — the transfer dropped vendored sources. Should be impossible now
  (`should_ship()` keeps them); if it happens, run
  `python3 -m pytest tests/test_deploy_e2e.py -k vendored` locally to see which
  exclusion broke.
