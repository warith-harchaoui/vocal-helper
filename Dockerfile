# syntax=docker/dockerfile:1.6
#
# vocal-helper — reproducible container image.
#
# Two-stage build: the base stage pulls system deps (ffmpeg + libsndfile
# are mandatory — ffmpeg for URL / mp3 / m4a intake, libsndfile for the
# soundfile-based WAV path) and installs the package with the [api,mcp]
# extras so the container can serve the HTTP + MCP surfaces out of the
# box. Heavy optional extras ([pyannote], [nemo], [mic], [stream]) are
# not installed by default — they either drag in ~2 GB of torch or need
# device-specific runtimes (yt-dlp, PortAudio). Enable them with the
# corresponding --build-arg flags below.
#
# Build:
#   docker build -t vocal-helper .
#   docker build --build-arg WITH_PYANNOTE=1 -t vocal-helper:pyannote .
#   docker build --build-arg WITH_STREAM=1   -t vocal-helper:stream .
#
# Run (HTTP + MCP on 0.0.0.0:8000):
#   docker run --rm -p 8000:8000 vocal-helper
#
# Run CLI one-shot:
#   docker run --rm -v $PWD:/data vocal-helper \
#     vocal-helper transcribe /data/clip.wav --language en

# --- base -------------------------------------------------------------------
FROM python:3.11-slim AS base

# System deps: ffmpeg for every non-WAV format, libsndfile for soundfile,
# tini for signal handling. No compilers — we install from wheels only.
RUN apt-get update && apt-get install --no-install-recommends -y \
        ffmpeg \
        libsndfile1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; the app never needs root at runtime.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# --- deps -------------------------------------------------------------------
# Copy the package first so pip picks up pyproject.toml before we invalidate
# the layer with source changes.
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app vocal_helper ./vocal_helper

# Build-arg switches: opt-in the heavy extras. Default = light image.
ARG WITH_PYANNOTE=0
ARG WITH_STREAM=0
ARG WITH_LLM=0
RUN pip install --no-cache-dir --upgrade pip \
 && EXTRAS="api,mcp,cli" \
 && if [ "$WITH_PYANNOTE" = "1" ] ; then EXTRAS="$EXTRAS,pyannote" ; fi \
 && if [ "$WITH_STREAM" = "1" ]   ; then EXTRAS="$EXTRAS,stream" ; fi \
 && if [ "$WITH_LLM" = "1" ]      ; then EXTRAS="$EXTRAS,llm" ; fi \
 && pip install --no-cache-dir ".[$EXTRAS]"

# --- runtime ----------------------------------------------------------------
USER app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    VOCAL_HELPER_HOST=0.0.0.0 \
    VOCAL_HELPER_PORT=8000

# tini reaps orphan children (ffmpeg subprocesses) cleanly on SIGTERM.
ENTRYPOINT ["/usr/bin/tini", "--"]
# Default: serve FastAPI + MCP. Override for one-shot CLI usage.
CMD ["vocal-helper-mcp"]
