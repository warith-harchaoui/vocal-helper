"""Live microphone → diarized transcript (+ optional Gemma summary).

Run with the default microphone, pyannote diarization, whisper.cpp
turbo and the Gemma 4 e4b analyst :

    pip install 'vocal-helper[all]'

Then supply the pyannote HuggingFace token by any of these means
(highest priority first) :

* pass ``--hf-token hf_…`` on the CLI ;
* export ``HF_TOKEN=hf_…`` in the shell ;
* copy ``settings.yaml.example`` to ``settings.yaml`` and fill in
  ``secrets.hf_token``.

Then run :

    python examples/live_mic_to_text.py --llm

Stop with Ctrl-C ; the pipeline cancels every running task cleanly.
"""

from __future__ import annotations

import asyncio
import sys

import click

import vocal_helper as voh


async def amain(
    *,
    device: str | None,
    whisper_model: str,
    language: str,
    threads: int,
    diar_backend: str,
    hf_token: str | None,
    llm: bool,
    llm_model: str,
) -> None:
    """Stream the microphone through the pipeline and print every event.

    Parameters
    ----------
    device : str or None
        Substring of the microphone name, or ``None`` for the default input.
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

    Returns
    -------
    None
        Everything is printed as it arrives; nothing is returned.
    """
    # The ``hf_token=None`` here delegates resolution to the stage
    # constructor, which calls ``_settings.resolve_hf_token`` and walks
    # the documented order : env var, then settings.yaml.
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
        source=lambda: voh.sources.from_microphone(device_name=device),
        config=config,
    )
    print("vocal-helper live — Ctrl-C to stop", file=sys.stderr)
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
@click.option("--device", default=None, help="substring of the microphone name")
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
@click.option("--llm-model", default="gemma3:4b", show_default=True)
def cli(
    device: str | None,
    whisper_model: str,
    language: str,
    threads: int,
    diar_backend: str,
    hf_token: str | None,
    llm: bool,
    llm_model: str,
) -> None:
    """Transcribe the live microphone, one diarized utterance per line.

    \f

    Parameters
    ----------
    device, whisper_model, language, threads, diar_backend, hf_token, llm, llm_model
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
            device=device,
            whisper_model=whisper_model,
            language=language,
            threads=threads,
            diar_backend=diar_backend,
            hf_token=hf_token,
            llm=llm,
            llm_model=llm_model,
        )
    )


if __name__ == "__main__":
    cli()
