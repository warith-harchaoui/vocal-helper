"""Nemotron entend-il la parole, ou son seuil la coupe-t-il ?

La décomposition du taux d'erreur, sur une réunion d'AMI au micro d'ambiance,
ne laisse pas beaucoup de place au doute :

================================  =========  ==============  ===========
condition                           manqué    fausse alarme    confusion
================================  =========  ==============  ===========
pyannote community-1, différé         0,095           0,061        0,036
Nemotron, flux ``low_latency``        0,461           0,024        0,003
================================  =========  ==============  ===========

Sa **confusion de locuteurs est douze fois plus basse** que celle du moteur que
nous utilisons aujourd'hui — l'attribution des voix est donc quasi parfaite —
et toute son erreur est de la parole qu'il n'a pas émise : 616 secondes sur 815
annotées. La perte est uniforme sur toute la séance, entre 0,59 et 0,82 par
tranche de deux minutes, donc ce n'est pas un trou localisé ni une boucle de
flux qui perdrait de l'audio.

Reste le seuil. ``extract_speaker_dict`` déclare une trame parlante au-dessus
d'une probabilité de 0,5 par défaut, et une réunion lointaine et réverbérée est
exactement le matériel où des probabilités honnêtes passent sous une barre
calibrée ailleurs. Cette étude le balaie sur un seul passage du modèle : les
logits sont gardés, chaque seuil n'est qu'une extraction.

Si l'optimum est franchement sous 0,5, le modèle n'a pas de défaut de détection
— c'est son réglage de sortie qui ne convient pas à cette matière, et c'est une
constante à poser, pas un modèle à écarter.

Usage
-----
    python studies/diar_nemotron_threshold.py --root <dev-slice> --limit 1
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import click
import numpy as np
import soundfile as sf

from vocal_helper.consistency import speaker_consistency

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diar_incremental_vs_greedy import reference_from_rttm  # noqa: E402
from diar_nemotron_streaming import chunks_of, hf_token, load  # noqa: E402

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def components(reference, hypothesis) -> dict[str, float]:
    """Les trois parts du taux d'erreur, pour savoir où il part."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def annotate(rows):
        out = Annotation()
        for t0, t1, speaker in rows:
            if t1 > t0:
                out[Segment(t0, t1)] = speaker
        return out

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    total_der = float(metric(annotate(reference), annotate(hypothesis)))
    parts = metric[:]
    total = parts["total"] or 1.0
    return {
        "der": total_der,
        "missed": parts["missed detection"] / total,
        "false_alarm": parts["false alarm"] / total,
        "confusion": parts["confusion"] / total,
    }


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_nemotron_threshold.json")
@click.option(
    "--mode",
    type=click.Choice(["low_latency", "very_low_latency", "ultra_low_latency"]),
    default="low_latency",
)
@click.option("--limit", default=1, help="Nombre de réunions.")
def main(root: str, out_path: str, mode: str, limit: int) -> None:
    """Balayer le seuil d'extraction sur un seul passage du modèle."""
    import torch

    meetings = sorted(p for p in pathlib.Path(root).iterdir() if (p / "mix.wav").exists())
    processor, model = load(hf_token())
    processor.set_streaming_mode(mode)
    sampling_rate = processor.feature_extractor.sampling_rate
    relevé: dict[str, dict] = {}

    for meeting in meetings[:limit]:
        truth = reference_from_rttm(meeting / "words.rttm")
        n_truth = len({s for _, _, s in truth})
        audio, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        duration = len(audio) / sr

        started = time.perf_counter()
        cache, pieces = None, []
        with torch.inference_mode():
            for inputs in chunks_of(processor, audio, sampling_rate):
                inputs = inputs.to(model.device, dtype=model.dtype)
                outputs = model(**inputs, speaker_cache=cache)
                pieces.append(outputs.logits)
                cache = outputs.speaker_cache
        logits = torch.cat(pieces, dim=1)
        rtf = (time.perf_counter() - started) / duration
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {n_truth} locuteurs"
            f" · {logits.shape[1]} trames · RTF {rtf:.3f}"
        )
        print(
            f"   {'seuil':>6} {'DER':>6} {'manqué':>7} {'f. alarme':>10} "
            f"{'confusion':>10} {'étiq.':>6} {'≥300s':>7}"
        )

        here: dict[str, dict] = {"truth": n_truth, "rtf": rtf, "mode": mode}
        for threshold in THRESHOLDS:
            rows = processor.extract_speaker_dict(logits, threshold=threshold)[0]
            hypothesis = sorted(
                (float(r["Start"]), float(r["End"]), f"S{int(r['Speaker'])}")
                for r in rows
                if float(r["End"]) > float(r["Start"])
            )
            if not hypothesis:
                print(f"   {threshold:6.2f} — aucune parole émise")
                continue
            row = components(truth, hypothesis)
            report = speaker_consistency(truth, hypothesis)
            row["labels"] = len({s for _, _, s in hypothesis})
            row["long_range_300"] = report.long_range_agreement(gap_s=300)
            row["emitted_s"] = sum(t1 - t0 for t0, t1, _ in hypothesis)
            here[f"{threshold:g}"] = row
            print(
                f"   {threshold:6.2f} {row['der']:6.3f} {row['missed']:7.3f} "
                f"{row['false_alarm']:10.3f} {row['confusion']:10.3f} "
                f"{row['labels']:6} {row['long_range_300']:7.3f}",
                flush=True,
            )
        relevé[meeting.name] = here
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    print("\n   pour mémoire, même tranche et même protocole :")
    print("   pyannote différé 0,176 · fenêtre croissante 0,176 (en direct)")
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
