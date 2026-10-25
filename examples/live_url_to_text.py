"""Live URL (YouTube / Twitch / RSS / direct audio) → diarized transcript.

Streams any ``yt-dlp``-supported URL through the same online pipeline
as ``live_mic_to_text.py`` — VAD → online diarization → Whisper turbo →
(optionally) Gemma 4 e4b rolling summary.

Install :

    pip install 'vocal-helper[all]'

(``[all]`` brings the ``stream`` extra, which pulls ``podcast-helper`` —
that one wraps ``yt-dlp`` + ``ffmpeg`` for URL ingestion.)

Then run a YouTube VOD at real-time pace :

    python examples/live_url_to_text.py \\
        "https://www.youtube.com/watch?v=YE7VzlLtp-4" --llm

…or burst-transcribe a VOD at 2× speed (offline benchmark) :

    python examples/live_url_to_text.py \\
        "https://www.youtube.com/watch?v=YE7VzlLtp-4" \\
        --no-realtime --speed 2.0

Twitch live works the same way (``--no-realtime`` and ``--speed`` are
both ignored on live streams ; the source paces itself).

Stop with Ctrl-C ; the pipeline cancels every running task cleanly.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import asyncio
import sys

import click

import vocal_helper as voh


async def amain(
    *,
    url: str,
    whisper_model: str,
    language: str,
    threads: int,
    diar_backend: str,
    hf_token: str | None,
    llm: bool,
    llm_model: str,
    realtime: bool,
    speed: float,
    cookies_from_browser: str | None,
    record_to: str | None,
) -> None:
    """Stream one URL through the pipeline and print every event.

    Parameters
    ----------
    url : str
        YouTube / Twitch / RSS / direct audio URL handed to ``yt-dlp``.
    whisper_model : str
        pywhispercpp model tag handed to the ASR stage.
    language : str
        ISO-639-1 code, or ``"auto"`` for language identification.
    threads : int
        whisper.cpp CPU threads.
    diar_backend : str
        ``"pyannote"`` or ``"nemo"`` online embedding backend.
    hf_token : str or None
        HuggingFace token for the gated pyannote weights; ``None`` delegates
        resolution to the stage constructor.
    llm : bool
        ``True`` to enable the rolling-summary analyst stage.
    llm_model : str
        Ollama model tag used by the analyst when ``llm`` is set.
    realtime : bool
        Pace a VOD at wall-clock speed (the live-like default) or drain it
        as fast as the pipeline can chew.
    speed : float
        VOD playback rate (``atempo``); raises on live streams.
    cookies_from_browser : str or None
        Browser name whose cookie jar ``yt-dlp`` should reuse.
    record_to : str or None
        Optional path for a parallel compressed archive.

    Returns
    -------
    None
        Everything is printed as it arrives; nothing is returned.
    """
    # Same shape as live_mic_to_text — only the source factory changes.
    config = voh.PipelineConfig(
        diar={
            "backend": diar_backend,
            "hf_token": hf_token,
        },
        asr={
            "model": whisper_model,
            "language": language,
            "threads": threads,
        },
        llm=({"model": llm_model, "recent_window_s": 60.0} if llm else None),
    )
    pipeline = voh.Pipeline(
        source=lambda: voh.sources.from_url(
            url,
            realtime=realtime,
            speed=speed,
            cookies_from_browser=cookies_from_browser,
            record_to=record_to,
        ),
        config=config,
    )
    print(f"vocal-helper streaming {url} — Ctrl-C to stop", file=sys.stderr)
    try:
        async for ev in pipeline.run():
            if "text" in ev:
                print(f"[{ev['t0']:7.2f}-{ev['t1']:7.2f} {ev['speaker']}] {ev['text']}")
            elif "summary" in ev:
                print(f"\n=== rolling summary @ {ev['t0']:.1f}s ===")
                print(ev["summary"])
                print("=== recent ===")
                print(ev["recent"])
                print("===\n")
    except KeyboardInterrupt:
        print("\nstopping…", file=sys.stderr)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("url")
@click.option("--whisper-model", default="large-v3-turbo-q5_0", show_default=True)
@click.option("--language", default="auto", show_default=True)
@click.option("--threads", type=int, default=6, show_default=True)
@click.option(
    "--diar-backend",
    type=click.Choice(["pyannote", "nemo"]),
    default="pyannote",
    show_default=True,
)
@click.option(
    "--hf-token",
    default=None,
    help="HuggingFace token (falls back to $HF_TOKEN, then settings.yaml secrets.hf_token).",
)
@click.option("--llm", is_flag=True, default=False)
@click.option("--llm-model", default="gemma4:e4b", show_default=True)
# Realtime is the live-like default. --no-realtime is for VOD batches. The
# paired "--realtime/--no-realtime" spelling is click's equivalent of the two
# argparse store_true / store_false actions that used to share one dest.
@click.option("--realtime/--no-realtime", default=True, show_default=True)
@click.option(
    "--speed",
    type=float,
    default=1.0,
    show_default=True,
    help="VOD playback rate (atempo). Raises on live streams.",
)
@click.option(
    "--cookies-from-browser",
    default=None,
    help="firefox / chrome / safari — used by yt-dlp for age-gated or private content.",
)
@click.option(
    "--record-to",
    default=None,
    help="Optional path for a parallel compressed archive "
    "(podcast-helper's record_to passthrough).",
)
def cli(
    url: str,
    whisper_model: str,
    language: str,
    threads: int,
    diar_backend: str,
    hf_token: str | None,
    llm: bool,
    llm_model: str,
    realtime: bool,
    speed: float,
    cookies_from_browser: str | None,
    record_to: str | None,
) -> None:
    """Transcribe URL (YouTube / Twitch / RSS / direct audio), one line per utterance.

    \f

    Parameters
    ----------
    url, whisper_model, language, threads, diar_backend, hf_token, llm, llm_model, realtime, speed, cookies_from_browser, record_to
        Passed straight through to :func:`amain`; see its docstring.

    Returns
    -------
    None
    """
    # The ``\f`` ending the summary above is click's help-truncation marker:
    # click renders the *whole* docstring into ``--help``, so without it the
    # numpydoc "Parameters" / "Returns" plumbing below would be printed to
    # users as if it were part of the usage text.
    asyncio.run(
        amain(
            url=url,
            whisper_model=whisper_model,
            language=language,
            threads=threads,
            diar_backend=diar_backend,
            hf_token=hf_token,
            llm=llm,
            llm_model=llm_model,
            realtime=realtime,
            speed=speed,
            cookies_from_browser=cookies_from_browser,
            record_to=record_to,
        )
    )


if __name__ == "__main__":
    cli()
