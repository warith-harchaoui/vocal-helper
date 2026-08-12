# Security policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security-sensitive
reports. Instead, email:

- **warith@deraison.ai**

with subject `SECURITY: vocal-helper <one-line summary>`. Include:

- Affected version(s) or commit SHA
- Steps to reproduce
- Impact assessment (data exposure, RCE, credential leak, DoS…)
- A proposed fix if you have one

You should receive an acknowledgement within **72 hours**. Coordinated
disclosure timeline is 90 days from acknowledgement unless the issue
is already being actively exploited.

## Scope

The following are in scope:

- **Credential and settings handling**: `settings.yaml` parsing
  (`_settings.py`'s hand-rolled parser), the `engines.diarization_url` /
  `$VH_DIARIZATION_ENGINES` resolution order, the LLM engine descriptor
  resolved by `best-engine-ai-helper` into the gitignored
  `vocal_helper/llm.engine.yaml`, and accidental logging of secrets.
- **URL ingestion**: `vocal_helper.sources.from_url` and its
  `podcast-helper` backend. Anything that lets a URL escape the
  ffmpeg + yt-dlp sandbox, cause command injection, or read arbitrary
  files.
- **Model bundle download and integrity**: the self-hosted
  "diarization-engines" ZIP fetched from `engines.diarization_url`,
  cached under `~/.cache/vocal-helper`, and verified against its
  `manifest.json` sha256 hashes. Anything that lets a tampered or
  substituted bundle execute code at load time (pickle deserialisation,
  custom code paths).
- **CLI argument parsing**: `vocal-helper file …` and
  `vocal-helper mic …`. Path traversal, shell injection.
- **Deserialisation**: WAV / VTT parsing, the settings parser in
  `_settings.py`.

## Out of scope

- Model quality issues, ASR / diar accuracy, transcription errors.
- Denial-of-service via legitimately large inputs (long audio,
  large batch): those are performance concerns tracked as issues.
- Attacks that require write access to the local disk / shell
  where the pipeline runs.
- Third-party service outages (the self-hosted bundle host, Ollama,
  vLLM, YouTube).

## Supported versions

Only the latest release on `main` receives security fixes. Backports
to older tags are best-effort.

## Attribution

Reporters who follow this policy are credited in the CHANGELOG
under "Security" for the fix release, unless anonymity is requested.

## Known-safe deployment notes

- Run the service as a **non-root user** with `chmod 0600` on
  `settings.yaml`; the `SECURITY` section of
  [`TECHNICAL_STACK.md`](TECHNICAL_STACK.md) covers systemd
  hardening.
- The stack needs no HuggingFace account or token: model weights come
  from the self-hosted diarization-engines bundle (`engines.diarization_url`)
  plus NVIDIA NGC for TitaNet, and `HF_HUB_OFFLINE=1` is safe to set.
- Point `$VH_DIARIZATION_ENGINES` at a directory on a volume you
  control for air-gapped deployments, instead of the default
  `~/.cache/vocal-helper`.
- Do **not** enable `trust_remote_code=True` on any model load ; the
  current codebase never does.
