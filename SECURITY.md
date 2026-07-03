# Security policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security-sensitive
reports. Instead, email :

- **warith@deraison.ai**

with subject `SECURITY: vocal-helper <one-line summary>`. Include :

- Affected version(s) or commit SHA
- Steps to reproduce
- Impact assessment (data exposure, RCE, credential leak, DoS…)
- A proposed fix if you have one

You should receive an acknowledgement within **72 hours**. Coordinated
disclosure timeline is 90 days from acknowledgement unless the issue
is already being actively exploited.

## Scope

The following are in scope :

- **Credential handling** — `settings.yaml` parsing, `HF_TOKEN`
  resolution, `resolve_hf_token()` edge cases, accidental logging of
  secrets.
- **URL ingestion** — `vocal_helper.sources.from_url` and its
  `podcast-helper` backend. Anything that lets a URL escape the
  ffmpeg + yt-dlp sandbox, cause command injection, or read arbitrary
  files.
- **Model download** — HuggingFace Hub interaction. Anything that
  lets a malicious model checkpoint execute code at load time
  (pickle deserialisation, custom code repos, `trust_remote_code=True`
  paths).
- **CLI argument parsing** — `vocal-helper file …` and
  `vocal-helper mic …`. Path traversal, shell injection.
- **Deserialisation** — WAV / VTT parsing, `_parse_minimal_yaml`
  in `_settings.py`.

## Out of scope

- Model quality issues, ASR / diar accuracy, transcription errors.
- Denial-of-service via legitimately large inputs (long audio,
  large batch) — those are performance concerns tracked as issues.
- Attacks that require write access to the local disk / shell
  where the pipeline runs.
- Third-party service outages (HuggingFace Hub, Ollama, YouTube).

## Supported versions

Only the latest release on `main` receives security fixes. Backports
to older tags are best-effort.

## Attribution

Reporters who follow this policy are credited in the CHANGELOG
under "Security" for the fix release, unless anonymity is requested.

## Known-safe deployment notes

- Run the service as a **non-root user** with `chmod 0600` on
  `settings.yaml` — the `SECURITY` section of
  [`TECHNICAL_STACK.md`](TECHNICAL_STACK.md) covers systemd
  hardening.
- Set `HF_HUB_CACHE=/data/hf-cache` on a volume you control ; the
  default `~/.cache/huggingface` is fine for local dev but shared on
  CI runners.
- Do **not** enable `trust_remote_code=True` on any HF model load ;
  the current codebase never does.
- Rotate the HuggingFace token every 6 months.
