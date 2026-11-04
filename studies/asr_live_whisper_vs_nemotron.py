"""En direct : whisper par segment contre Nemotron 3.5, par segment et en flux.

Pourquoi cette mesure
---------------------
Le moteur ``auto`` garde whisper en direct parce que Nemotron perd du contexte
en flux (``asr_nemotron_online.py``). Mais cette étude-là compare Nemotron en
flux à Nemotron hors ligne, pas à ce que la pipeline fait aujourd'hui en
direct : Silero découpe des segments de parole, et whisper transcrit chacun,
langue devinée. C'est cette comparaison qui décide du moteur en direct.

Moteurs, sur les mêmes fichiers et les mêmes coupes que l'étude en flux
-----------------------------------------------------------------------
- ``whisper-segment`` : Silero (réglages par défaut de ``SileroVADStage``) puis
  whisper turbo sur chaque segment, ``language="auto"`` — le direct actuel ;
- ``nemotron-segment`` : les mêmes segments, transcrits par le graphe Nemotron
  (ce que fait ``--asr-engine nemotron`` en direct aujourd'hui) ;
- ``nemotron-stream`` : fenêtres glissantes de 2 s, 16 s de contexte, 0,48 s
  d'anticipation, sans Silero.

Délai : pour un segment, un mot attend en moyenne la moitié du segment, puis
le silence qui le ferme (300 ms) et le calcul ; pour le flux, celui que rend
``streaming_frames``. Le calcul est chronométré sur CPU.

Usage
-----
    .venv/bin/python studies/asr_live_whisper_vs_nemotron.py \\
        --graph /Volumes/orange-dev/nemotron-study/onnx/nemotron-asr-ctc.int8.onnx
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import click
import numpy as np

from vocal_helper.bench import wer_report

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_nemotron_online import _cropped_reference, _text  # noqa: E402
from asr_nemotron_vs_whisper_offline import _load, corpus  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "asr_live_whisper_vs_nemotron.json"
SR = 16000
#: Le silence qui ferme un segment Silero (``min_silence_ms`` par défaut).
CLOSE_S = 0.3


def segments(audio: np.ndarray) -> list[dict]:
    """Les segments de parole que Silero émettrait en direct, trame de 32 ms par trame."""
    from vocal_helper.vad import SileroVADStage

    stage = SileroVADStage()
    stage._ensure_model()
    out: list[dict] = []
    step = 512
    for i in range(0, len(audio) - step + 1, step):
        out += stage._consume_frame({"t0": i / SR, "sample_rate": SR, "pcm": audio[i : i + step]})
    if stage._run is not None:
        last = stage._emit_run(stage._run, end_t=stage._absolute_sample)
        if last is not None:
            out.append(last)
    return [{**s, "speaker": "S0"} for s in out]


def per_segment(transcribe, segs: list[dict]) -> tuple[str, float, float]:
    """Texte, délai moyen par mot et facteur temps réel du calcul."""
    texts, delays, weights, spent = [], [], [], 0.0
    for seg in segs:
        t0 = time.perf_counter()
        text = transcribe(seg)
        elapsed = time.perf_counter() - t0
        spent += elapsed
        if text:
            texts.append(text)
            n = len(text.split())
            delays.append((seg["t1"] - seg["t0"]) / 2 + CLOSE_S + elapsed)
            weights.append(n)
    seconds = sum(s["t1"] - s["t0"] for s in segs) or 1.0
    delay = float(np.average(delays, weights=weights)) if delays else 0.0
    return " ".join(texts), delay, spent / seconds


@click.command()
@click.option("--graph", type=click.Path(path_type=pathlib.Path, exists=True), required=True)
@click.option("--corpora", default="bagarre-rich,cv-fr,tv-fr,ami", show_default=True)
@click.option("--limit", default=12, show_default=True)
@click.option("--crop-s", default=300.0, show_default=True, help="Durée maximale par fichier.")
def main(graph: pathlib.Path, corpora: str, limit: int, crop_s: float) -> None:
    """Chaque moteur sur chaque fichier ; JSON reprenable."""
    from vocal_helper.asr import WhisperStage
    from vocal_helper.nemotron_asr import NemotronAsrStage, NemotronOnnxAsr

    whisper = WhisperStage(word_timestamps=False)
    whisper._ensure_model()
    asr = NemotronOnnxAsr(graph, providers=["CPUExecutionProvider"])
    stage = NemotronAsrStage(graph)
    stage._asr = asr

    def whisper_one(seg: dict) -> str:
        utt = whisper._transcribe_blocking(seg)
        return utt["text"] if utt else ""

    def nemotron_one(seg: dict) -> str:
        return " ".join(u["text"] for u in stage.transcribe_chunk([seg]) if u.get("text"))

    # Préchauffer : le premier appel charge les poids, ce n'est pas du direct.
    warm = {"t0": 0.0, "t1": 1.0, "sample_rate": SR, "speaker": "S0"}
    warm["pcm"] = np.random.default_rng(0).normal(0, 0.01, SR).astype(np.float32)
    whisper_one(warm)
    nemotron_one(warm)

    results = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    done = {(r["corpus"], r["id"], r["engine"]) for r in results["rows"]}
    for name in corpora.split(","):
        for ident, path, reference, *crop in corpus(name, limit):
            try:
                audio = _load(path)
            except (OSError, RuntimeError):
                print(name, ident, "illisible, sauté", flush=True)
                continue
            audio = audio[: int(min(crop[0] if crop else crop_s, crop_s) * SR)]
            if name in ("ami", "tv-fr"):
                reference = _cropped_reference(name, path, len(audio) / SR)
            segs = None
            for engine in ("whisper-segment", "nemotron-segment", "nemotron-stream"):
                if (name, ident, engine) in done:
                    continue
                if engine == "nemotron-stream":
                    t0 = time.perf_counter()
                    ids, best, _, delay = asr.streaming_frames(
                        audio, SR, hop_s=2.0, left_s=16.0, right_s=0.48
                    )
                    rtf = (time.perf_counter() - t0) / (len(audio) / SR)
                    text, delay_s = _text(asr, ids, best), float(delay.mean())
                else:
                    segs = segs if segs is not None else segments(audio)
                    run = whisper_one if engine == "whisper-segment" else nemotron_one
                    text, delay_s, rtf = per_segment(run, segs)
                row = {
                    "corpus": name,
                    "id": ident,
                    "engine": engine,
                    "seconds": round(len(audio) / SR, 2),
                    "wer": wer_report(reference, text).get("wer"),
                    "delay_mean_s": round(delay_s, 3),
                    "rtf": round(rtf, 4),
                    "ref_words": len(reference.split()),
                }
                results["rows"].append(row)
                OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
                print(name, ident, engine, row["wer"], row["delay_mean_s"], row["rtf"], flush=True)


if __name__ == "__main__":
    main()
