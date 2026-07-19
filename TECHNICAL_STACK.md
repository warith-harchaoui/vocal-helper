# Technical stack — GPU server deployment

Reference guide for running the **entire AI Helpers suite** on a Linux
NVIDIA GPU box (bare metal or cloud). Written for the operator who
will provision the server, install the code, and keep it healthy.

The suite is a chain of narrow libraries — each layer sits on the one
below and adds one concern. On a GPU server every layer benefits from
CUDA in a different way ; this doc lays out the whole picture so you
don't wire up whisper-with-CUDA and then discover pyannote silently
fell back to CPU.

## Suite topology

| Package | Concern | Producer / consumer of |
|---|---|---|
| `os-helper` | Cross-platform primitives (paths, subprocess, filesystem) | Foundation |
| `audio-helper` | ffmpeg wrappers, load / mix / silence, Demucs stem separation | Consumes `os-helper` |
| `capture-helper` | Microphone input via `sounddevice` | Consumes `os-helper` ; **skip on server** |
| `podcast-helper` | Universal audio streaming (URL → 16 kHz mono PCM) via ffmpeg + yt-dlp | Consumes `os-helper` |
| `youtube-helper` | Video frames, captions, metadata, comments, engagement | Consumes `os-helper` |
| `vocal-helper` | Speech pipeline : VAD → diar → ASR → LLM analyst | Consumes `podcast-helper`, `audio-helper` |
| `music-helper` | Music transcription pipeline : stems → notes → score | Consumes `podcast-helper`, `audio-helper` |

The GPU is exercised by four workloads :

1. **whisper.cpp** (ASR) — in `vocal-helper`, via `pywhispercpp`.
2. **pyannote 3.1** (VAD-free segmentation + speaker diarization) — in `vocal-helper`, via `pyannote.audio`.
3. **Ollama** (LLM analyst, semantic EOT gating) — external service.
4. **Demucs HDEMUCS_HIGH_MUSDB_PLUS** (music stem separation) — in `music-helper` via `audio-helper.separate_sources`.
5. **Basic Pitch** (polyphonic note transcription) — in `music-helper` via `basic-pitch`.

Silero VAD (`vocal-helper.SileroVADStage`) is ONNX-runtime CPU only and
stays on CPU — 200× real-time on one core, no GPU wanted.

## Hardware

### Minimum viable

| Component | Target | Why |
|---|---|---|
| GPU | NVIDIA ≥ 16 GB VRAM | Whisper-turbo ~1.5 GB + pyannote 3.1 ~500 MB + Demucs ~1 GB + gemma4:e4b Q4 ~5 GB + KV cache ~4 GB ~ 12 GB working set, 16 GB gives margin |
| CUDA | ≥ 12.1 | PyTorch 2.5 wheels, whisper.cpp CUDA backend |
| NVIDIA driver | ≥ 550 | CUDA 12.4 compatibility |
| System RAM | 16–32 GB | ffmpeg PCM buffers + HF Hub download stream + Python |
| Disk | 100 GB SSD | Models cache (~ 10 GB) + HF cache + Ollama models + logs |
| OS | Ubuntu 22.04 LTS / Debian 12 | PyTorch wheels ship for these ; NVIDIA driver packaging clean |

### Concrete GPU picks by budget

| GPU | VRAM | Cost | Notes |
|---|---|---|---|
| RTX 4090 | 24 GB | ~ 1500 € on-prem or ~ $0.4/h spot (Runpod, Vast) | Best price / performance ; 5-6× parallel streams |
| RTX 4080 | 16 GB | ~ 1100 € | Tight but workable with Q4 Ollama models |
| A10G | 24 GB | ~ $1/h (AWS `g5.xlarge`) | Cloud sweet spot ; enterprise support |
| L4 | 24 GB | ~ $0.7/h (GCP `g2-standard-4`, AWS `g6.xlarge`) | Newer, better $/perf than A10G |
| RTX 6000 Ada | 48 GB | ~ 7000 € | Multi-tenant, hosts several LLMs |
| H100 80 GB | 80 GB | ~ $3/h | Overkill except for real batch loads |
| T4 | 16 GB | ~ $0.35/h (AWS `g4dn.xlarge`) | Entry-level ; Gemma 4 e4b Q4 tight |

Recommendation for a first prod deploy : **RTX 4090 on Runpod** (spot)
or **AWS `g5.xlarge` (A10G 24 GB)** for AWS-committed shops.

## OS setup

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config git curl \
    python3.11 python3.11-venv python3.11-dev \
    ffmpeg libsndfile1 sox

# Persistent volumes — mount here in cloud
sudo mkdir -p /data/{hf-cache,ollama-models,whisper-models,repos,logs}
sudo chown -R $USER:$USER /data

# yt-dlp (moves fast — keep it fresh)
pipx install yt-dlp
```

Add a nightly cron so yt-dlp keeps up with YouTube's breaking changes :

```cron
0 3 * * * pipx upgrade yt-dlp >> /data/logs/yt-dlp-upgrade.log 2>&1
```

## Python environment

```bash
python3.11 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip wheel setuptools
```

Pin Python 3.11. Reasons :
- `pywhispercpp` wheels are strongest here.
- `basic-pitch` has a hard TensorFlow-macos pin that clashes with 3.13.
- `pyannote.audio` and `music21` are all 3.11-tested.

## Layer 1 — PyTorch + CUDA

```bash
# Match CUDA 12.1 to your driver ; keep everything on one CUDA release.
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

Sanity check :

```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

`vocal_helper.diar._auto_torch_device` will pick `"cuda"` automatically —
no need to pass `device=` anywhere unless you want to pin a specific GPU.

## Layer 2 — Helpers (foundation)

Install the four helpers from git — they're not on PyPI :

```bash
pip install \
    'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@main' \
    'audio-helper[demucs] @ git+https://github.com/warith-harchaoui/audio-helper.git@main' \
    'podcast-helper @ git+https://github.com/warith-harchaoui/podcast-helper.git@main' \
    'youtube-helper @ git+https://github.com/warith-harchaoui/youtube-helper.git@main'
```

`audio-helper[demucs]` pulls torch + torchaudio (already installed with
CUDA above), plus `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS` at
first use. First call downloads ~ 80 MB. On CUDA this runs at ~ 0.2×
real-time for 4-stem separation.

## Layer 3 — vocal-helper

### Install

```bash
# Note : no [mic] extra on server — no capture-helper wanted.
pip install \
    'vocal-helper[pyannote,llm,stream] @ git+https://github.com/warith-harchaoui/vocal-helper.git@main'
```

### Extras summary

| Extra | Pulls | Needed for |
|---|---|---|
| `pyannote` | `pyannote.audio>=3.3` | Diarization (online + offline) |
| `llm` | `ollama` client | Gemma analyst + semantic EOT gating |
| `stream` | `podcast-helper` (via git URL) | `from_url` (YouTube, RSS, HLS…) |
| `nemo` | `torch`, `torchaudio`, `nemo-toolkit[asr]` | Alternative TitaNet diar backend (`backend="nemo"`) — **now the default per the 2026-06-30 sweep** |
| `sherpa` | `sherpa-onnx>=1.13` | Torch-free TitaNet diar backend (`backend="sherpa"`) — same embedder via onnxruntime, embeddable anywhere |
| `mic` | `capture-helper` | Skip on server |
| `all` | Everything except `mic` and `nemo` | One-line install for prod |

### Whisper — the trickiest piece

`pywhispercpp` ships only CPU wheels on PyPI. To get whisper.cpp's
CUDA backend you must **build from source with `GGML_CUDA=on`** :

```bash
sudo apt install -y nvidia-cuda-toolkit  # only if not already there
CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install pywhispercpp --no-binary pywhispercpp --force-reinstall
```

Build takes 2-3 minutes. Verify at runtime :

```python
from pywhispercpp.model import Model
m = Model('large-v3-turbo-q5_0', n_threads=6)
# On import you should see :
#   whisper_backend_init_gpu: using CUDA backend
```

**Alternative — `faster-whisper` backend**. On modern NVIDIA GPUs
CTranslate2 is often **2× faster than whisper.cpp CUDA**. Study
already exists at `studies/stt_faster_whisper_vs_pywhispercpp.py`.
If numbers hold, swap `WhisperStage` to a faster-whisper backend
before shipping.

```bash
pip install faster-whisper ctranslate2
```

### Diarization backend choice

`OnlineDiarStage` defaults to `backend="nemo"` (TitaNet) since the
2026-06-30 study (76 % better separability than pyannote/embedding on
AMI). If you skip the `nemo` extra to save the ~ 5 GB install
footprint, force `backend="pyannote"` :

```python
diar={"backend": "pyannote"}
```

A third option, `backend="sherpa"`, runs the same TitaNet-large through
onnxruntime (the `sherpa` extra) — **no PyTorch**, so it installs light and
embeds on any platform. Same quality as `nemo`, torch-free.

`OfflineDiarStage` uses `pyannote/speaker-diarization-3.1` and will run
on CUDA automatically thanks to `_auto_torch_device`. The wrapper
handles the pyannote 3.x `DiarizeOutput` API change transparently.

### Backend router — the *aiguilleur*

You don't pick by hand. `vocal_helper.router` (`voh.select_diarization`) turns the
measured quality×speed trade-off into one explicit, tested decision — the CLIs
delegate to it and it reports both **DER** (quality, lower better) and **RTF**
(speed, `< 1` faster than real time). Numbers re-validated on-machine
(`studies/router_profile_validation.py`, `pyannote.metrics` collar 0.25, median
DER + RTF) against bagarre (30 short mixes) + AMI dev-slice; `sherpa` from ADR
0002 :

| Mode | Scenario | Backend | DER | RTF | Why |
|---|---|---|---|---|---|
| offline | short ≤ 300 s, ≤ 4 speakers | **`nemo`** | **0.142** | 0.051 | End-to-end slot attribution, ~2.3× better than pyannote on short dense turns (0.330). |
| offline | long / unknown / > 4 speakers | **`pyannote`** | **0.122** | 0.067 | Robust default; NeMo hangs past ~25 min, caps at 4 speakers. |
| offline | torch-free | **`sherpa`** | 0.174 | 0.58 | ONNX TitaNet-large, beats NeMo Sortformer 0.267, FR+EN (ADR 0002). |
| online | any live stream | **`nemo`** | 0.586 | 0.030 | Best online embedder at every length (beats online pyannote 0.590/0.844). |
| online | torch-free | **`sherpa`** | 0.174 | 0.58 | Periodic offline re-diarization (ADR 0002). |

Offline has a real length crossover (nemo short ↔ pyannote long) so it needs the
router; online has none — the streaming clusterer is a latency-bound
approximation where nemo wins at every length, so streaming always routes to
nemo. `select_diarization(...)` returns a `BackendPlan(mode, backend,
expected_der, expected_rtf, reason)`; see the [README router
section](https://github.com/warith-harchaoui/vocal-helper#backend-router--the-aiguilleur)
for the authoritative narrative.

## Layer 4 — music-helper

```bash
pip install \
    'music-helper[transcribe,stems] @ git+https://github.com/warith-harchaoui/music-helper.git@main'
```

### Extras summary

| Extra | Pulls | Needed for |
|---|---|---|
| `stems` | `audio-helper[demucs]` | Demucs 4-stem separation |
| `transcribe` | `basic-pitch`, `librosa` | Polyphonic note transcription + tempo / beat / key |
| `dev` | pytest, ruff, soundfile | CI / development |

### Basic Pitch caveat

`basic-pitch` transitively pins `tensorflow-macos<2.15.1`. On Linux
CUDA this pin is a no-op (the macOS wheel is skipped) and you get
regular `tensorflow-cpu` + `tensorflow[and-cuda]`. Verify :

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # should list your NVIDIA card
```

If it doesn't, install the GPU-enabled TF explicitly :

```bash
pip install 'tensorflow[and-cuda]==2.15.*'
```

## Layer 5 — Ollama (LLM service)

Ollama runs as a system service ; `vocal-helper` and `music-helper`
talk to it over HTTP.

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama

# Move the model cache off the root disk
sudo systemctl edit ollama
```

Add :

```ini
[Service]
Environment="OLLAMA_MODELS=/data/ollama-models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
```

Then :

```bash
sudo systemctl restart ollama
ollama pull gemma4:e4b        # ~ 5 GB Q4
```

Ollama's llama.cpp is compiled with CUDA by default on Linux — it will
grab the GPU automatically. Verify with `nvidia-smi` while Ollama is
generating.

**Model choice** :
- `gemma4:e4b` — the vocal-helper default. ~ 5 GB, ~ 50 tok/s on RTX 4090.
- `qwen3:8b` — 15 % better summaries in the 2026-06-30 sweep, but ~ 8 GB.
- `qwen3:0.6b` — for cheap-and-fast EOT gating (Warith's WIP `SemanticEOTStage`).

## Configuration

### Configuration

```bash
sudo install -d -m 0700 -o vocalhelper /data/vocal-helper
cat > /data/vocal-helper/settings.yaml <<'YAML'
# The self-hosted model bundle — the only config the project needs.
# No HuggingFace token required.
engines:
  diarization_url: https://deraison.ai/diarization-engines.zip
YAML
sudo chmod 0600 /data/vocal-helper/settings.yaml
```

All model weights (offline pyannote 3.1, NeMo Sortformer, online
`pyannote/embedding`, SpeechBrain VoxLingua107) ship in this bundle, so
**no HuggingFace account or token is needed** and no gated licences must
be accepted. TitaNet loads from NVIDIA NGC (also HF-free).

### Environment variables

```bash
# In /etc/environment or the service unit
export HF_HUB_CACHE=/data/hf-cache
export VOCAL_HELPER_SETTINGS=/data/vocal-helper/settings.yaml
export MUSIC_HELPER_SETTINGS=/data/music-helper/settings.yaml
export OLLAMA_HOST=http://localhost:11434
export TOKENIZERS_PARALLELISM=false        # silences a HF warning
```

Persistent caches on `/data` mean model downloads survive container
rebuilds and disk snapshots.

## Deployment patterns

### Pattern A — systemd (simplest, single-tenant)

```ini
# /etc/systemd/system/vocal-helper@.service
[Unit]
Description=vocal-helper worker (%i)
After=ollama.service
Requires=ollama.service

[Service]
Type=simple
User=vocalhelper
Group=vocalhelper
WorkingDirectory=/data/vocal-helper
Environment=HF_HUB_CACHE=/data/hf-cache
Environment=VOCAL_HELPER_SETTINGS=/data/vocal-helper/settings.yaml
Environment=OLLAMA_HOST=http://127.0.0.1:11434
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/venv/bin/vocal-helper file /data/vocal-helper/queue/%i.wav \
    --offline --llm --jsonl
Restart=on-failure
RestartSec=5s
StandardOutput=append:/data/logs/vocal-helper-%i.jsonl
StandardError=append:/data/logs/vocal-helper-%i.err

[Install]
WantedBy=multi-user.target
```

Usage : `systemctl start vocal-helper@meeting-2026-07-02.service`.

### Pattern B — Docker Compose (portable, reproducible)

> `vocal-helper` ships **no Dockerfile** — the toolbox is library + CLIs + API +
> MCP, nothing container-specific. The Compose and image recipes below are
> self-contained *examples you author yourself*; they install `vocal-helper`
> straight from git, so nothing in the repo needs to change to use them.

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    volumes:
      - /data/ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  vocal-helper:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    depends_on: [ollama]
    environment:
      OLLAMA_HOST: http://ollama:11434
      HF_HUB_CACHE: /cache/hf
      VOCAL_HELPER_SETTINGS: /secrets/settings.yaml
    volumes:
      - /data/hf-cache:/cache/hf
      - /data/vocal-helper/settings.yaml:/secrets/settings.yaml:ro
      - /data/vocal-helper/queue:/queue
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Base image :

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      build-essential cmake pkg-config git curl \
      ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools && \
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121

# whisper.cpp with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install \
    pywhispercpp --no-binary pywhispercpp

RUN pip install \
    'vocal-helper[pyannote,llm,stream] @ git+https://github.com/warith-harchaoui/vocal-helper.git@main'

# Pre-warm the diarization-engines bundle at build time (optional; caches
# the weights into the image so first run is instant). No HuggingFace, no
# secret — just the self-hosted bundle URL.
ARG VH_DIARIZATION_ENGINES=https://deraison.ai/diarization-engines.zip
RUN VH_DIARIZATION_ENGINES="$VH_DIARIZATION_ENGINES" \
    python -c "from vocal_helper.diar import resolve_diarization_engines as r; assert r()"

ENTRYPOINT ["vocal-helper"]
CMD ["--help"]
```

### Pattern C — Kubernetes (multi-tenant, autoscaling)

Only worth it above ~ 10 parallel streams. Sketch :

- One `Deployment` per pipeline flavour (streaming, offline, music).
- `nvidia.com/gpu: 1` resource request per pod.
- Horizontal Pod Autoscaler on queue depth (Kafka / Redis-Stream lag).
- Ollama in a `StatefulSet` with a shared cache PV — or one Ollama per pod for isolation.
- HF Hub cache in a `ReadOnlyMany` PVC pre-populated by an init container.

Skip this until the systemd or Compose deploy is production-hardened.

## Expected performance

Real-time factor (RTF < 1.0 = faster than real-time).

| GPU | pyannote 3.1 offline | whisper-turbo | Basic Pitch (music-helper) | Demucs (music-helper) | Ollama gemma4:e4b |
|---|---|---|---|---|---|
| RTX 4090 24 GB | 0.05× | 0.03× | 0.10× | 0.20× | 50 tok/s |
| A10G 24 GB | 0.15× | 0.08× | 0.30× | 0.50× | 25 tok/s |
| L4 24 GB | 0.20× | 0.10× | 0.35× | 0.60× | 20 tok/s |
| T4 16 GB | 0.35× | 0.20× | 0.60× | 0.90× | 10 tok/s |
| M2 Max (reference) | 0.9× (MPS) | 0.5× (Metal) | 1.5× | 2.0× | 15 tok/s |

**One live stream** (real-time = 1.0×) leaves large headroom on any GPU
≥ A10G — you can host **5-8 parallel streams on one RTX 4090** if your
memory budget accepts it.

## Production hardening

### 1. Model pre-warming

At container / systemd start, run a synthetic 5-second inference to
force lazy model loads. Otherwise the first user request pays the ~ 30 s
whisper + pyannote init tax.

```python
# /opt/warmup.py
import asyncio, numpy as np, vocal_helper as voh
async def main():
    pcm = np.zeros(16000 * 5, dtype=np.float32)
    p = voh.OfflinePipeline(
        source=lambda: voh.sources.from_numpy_array(pcm),
        config=voh.OfflinePipelineConfig(diar={"backend": "pyannote"}),
    )
    async for _ in p.run():
        pass
asyncio.run(main())
```

Wire it in the systemd unit's `ExecStartPre=`.

### 2. Health check endpoint

If you expose the pipeline over HTTP, `/health` should :

- Confirm CUDA is available.
- Confirm the pyannote pipeline is loaded (cheap attribute check).
- Ping Ollama's `/api/tags` (< 200 ms).

Return 200 only when all three pass. Ties in with Kubernetes / load
balancer readiness probes.

### 3. Observability

The pipeline logs a WARNING through the `vocal_helper.pipeline`
logger whenever a stage or subscriber crashes, with full traceback and
the offending stage / callback name. Route this into your log store :

```python
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
    stream=sys.stdout,
)
```

For metrics, wrap each stage's `run()` with a Prometheus histogram
timing the wall-clock per utterance. Alert if diar or ASR RTF
degrades > 2× baseline.

### 4. Configuration

`settings.yaml` holds only the diarization-engines bundle URL — no
secrets, no HuggingFace token — so it can be shipped in the clear (git
config, a ConfigMap, or baked into the image). There is nothing to
rotate. Point it at your own mirror of `diarization-engines.zip` for
air-gapped deploys.

### 5. YouTube rate-limits

Heavy yt-dlp usage triggers YT's 429. Two mitigations :

- **Proxy pool** — Bright Data, Oxylabs, or a rotating residential
  proxy. Pass via `cookies_from_browser=` and `headers=` on
  `from_url`.
- **Backoff & queue** — process URLs sequentially per source, with a
  60 s pause between requests to the same host. `youtube-helper`
  and `podcast-helper` don't ship this today ; add it in your job
  scheduler.

## Full install checklist

```bash
# 1. OS
sudo apt update && sudo apt install -y build-essential cmake pkg-config git curl \
    python3.11 python3.11-venv python3.11-dev ffmpeg libsndfile1

# 2. NVIDIA driver + CUDA runtime (assumed present on cloud GPU images)
nvidia-smi

# 3. Python env
python3.11 -m venv /opt/venv && source /opt/venv/bin/activate
pip install --upgrade pip wheel setuptools

# 4. PyTorch CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Helpers
pip install \
    'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@main' \
    'audio-helper[demucs] @ git+https://github.com/warith-harchaoui/audio-helper.git@main' \
    'podcast-helper @ git+https://github.com/warith-harchaoui/podcast-helper.git@main' \
    'youtube-helper @ git+https://github.com/warith-harchaoui/youtube-helper.git@main'

# 6. whisper.cpp with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install pywhispercpp --no-binary pywhispercpp

# 7. vocal-helper + music-helper
pip install \
    'vocal-helper[pyannote,llm,stream] @ git+https://github.com/warith-harchaoui/vocal-helper.git@main' \
    'music-helper[transcribe,stems] @ git+https://github.com/warith-harchaoui/music-helper.git@main'

# 8. Ollama + models
curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /data/ollama-models && sudo chown ollama:ollama /data/ollama-models
sudo systemctl enable --now ollama
ollama pull gemma4:e4b

# 9. Configuration (no HuggingFace token — just the bundle URL)
sudo install -d -m 0700 -o vocalhelper /data/vocal-helper
sudo -u vocalhelper tee /data/vocal-helper/settings.yaml >/dev/null <<'YAML'
engines:
  diarization_url: https://deraison.ai/diarization-engines.zip
YAML
sudo chmod 0600 /data/vocal-helper/settings.yaml

# 10. Verify
export VOCAL_HELPER_SETTINGS=/data/vocal-helper/settings.yaml
python -c "
import torch, vocal_helper as voh, music_helper as mh
assert torch.cuda.is_available(), 'CUDA not visible'
print('torch :', torch.cuda.get_device_name(0))
print('vocal-helper :', voh.__version__)
print('music-helper :', mh.__version__ if hasattr(mh, '__version__') else 'ok')
"
```

## Quick reference — what runs where

| Workload | CPU | GPU (CUDA) | GPU (Metal, dev only) |
|---|---|---|---|
| Silero VAD | ✅ default | — | — |
| pyannote embedding | ok | ✅ auto-picked | ok (fallback CPU on unsupported ops) |
| pyannote 3.1 speaker-diarization | slow (~ 15× RT) | ✅ auto-picked | ok |
| whisper.cpp large-v3-turbo | ok (~ 1× RT) | ✅ built with `GGML_CUDA=on` | ✅ Metal (default on macOS) |
| Ollama `gemma4:e4b` | ok (~ 10 tok/s) | ✅ auto | ✅ Metal |
| Demucs HDEMUCS_HIGH_MUSDB_PLUS | slow (~ 5× RT) | ✅ auto | ✅ MPS |
| Basic Pitch (TensorFlow) | ok (~ 0.5× RT) | ✅ `tensorflow[and-cuda]` | ✅ tensorflow-macos |
| music21 score assembly | ✅ | — | — |

## Bill of materials — reproducible install manifest

Pin versions in your requirements / container image once you've tested
them end-to-end. As of the July 2026 sweep :

```
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
pyannote.audio==3.3.2
pywhispercpp==1.2.0                # built with GGML_CUDA=on
ollama==0.4.4                      # Python client, server is 0.11+
tensorflow==2.15.1                 # for basic-pitch on Linux CUDA
basic-pitch==0.4.0
music21==10.5.0
librosa==0.10.2
numpy>=1.24,<2                     # pyannote is not 2.0-ready yet
```

Keep this file in sync when you upgrade — a matrix that ran green
yesterday can bit-rot on the next torch bump.
