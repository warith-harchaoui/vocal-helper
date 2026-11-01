"""Nemotron 3 Diarization contre notre chaîne, sur la même tranche d'AMI.

Pourquoi cette mesure
---------------------
Toute la journée a été passée à faire monter la diarisation en direct gratuite
par l'orchestration : regroupement incrémental sous contrainte, représentation
par exemplaires, transport d'identité par recouvrement temporel, fenêtre
croissante. Le meilleur résultat obtenu est la **qualité du différé en direct**
— taux d'erreur médian 0,176 sur les huit réunions d'AMI dev-slice, à RTF 0,10
à 0,16 — mais il reste deux limites que l'orchestration ne peut pas lever.

La première est la fraîcheur : la fenêtre croissante nomme définitivement un
passage à la passe suivante, soit plusieurs minutes plus tard, et ce retard
grandit avec la durée de la séance. La seconde est le plafond du moteur : sur
ES2011a, pyannote community-1 fusionne deux locuteurs sur quatre **en voyant le
fichier entier**, et aucune orchestration ne répare un modèle qui ne sait pas
séparer deux voix.

Un modèle entraîné de bout en bout pour le flux n'a ni l'une ni l'autre. Celui
mesuré ici tient un cache de locuteurs par ordre d'arrivée, donc il nomme à 80
ms et ne révise rien. Cette étude le confronte à nos chiffres, sous **notre**
protocole — appariement optimal, collier de 0,25 s — parce que les 14,72 %
qu'il annonce sont mesurés sans collier sur un autre banc et ne sont donc pas
comparables tels quels.

Usage
-----
    python studies/diar_nemotron_streaming.py --root <dev-slice> --mode low_latency
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
from diar_incremental_vs_greedy import der, reference_from_rttm  # noqa: E402

MODEL_ID = "nvidia/Nemotron-3-Diarization"


def load(token: str | None):
    """Charger le processeur et le modèle, du cache local si possible."""
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        MODEL_ID, device_map="auto", token=token
    )
    model.eval()
    return processor, model


def hf_token() -> str | None:
    """Le jeton du fichier privé, s'il existe. Jamais affiché."""
    import yaml

    path = pathlib.Path.home() / "notes-helper" / ".private" / "hf.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    return str(data["hf"]).strip() if isinstance(data, dict) and data.get("hf") else None


def chunks_of(processor, audio: np.ndarray, sampling_rate: int):
    """Découper l'audio comme le processeur l'attend en flux.

    Reprend la boucle que la documentation du modèle publie : un premier
    morceau de taille particulière, puis des morceaux réguliers, puis un dernier
    marqué comme tel pour que le tampon soit vidé.
    """
    yield processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
    )
    mel_frame_idx = processor.num_mel_frames_per_step
    start = processor.audio_chunk_start(mel_frame_idx)
    while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield processor(
            audio[start:end],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
        )
        mel_frame_idx += processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel_frame_idx)
    yield processor(
        audio[start:],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=False,
        is_last_audio_chunk=True,
    )


def to_segments(processor, logits) -> list[tuple[float, float, str]]:
    """Traduire les sorties du modèle en ``(t0, t1, locuteur)``."""
    rows = processor.extract_speaker_dict(logits)[0]
    return sorted(
        (float(r["Start"]), float(r["End"]), f"S{int(r['Speaker'])}")
        for r in rows
        if float(r["End"]) > float(r["Start"])
    )


@click.command()
@click.option("--root", required=True, help="Le dossier dev-slice d'AMI.")
@click.option("--out", "out_path", default="studies/diar_nemotron_streaming.json")
@click.option(
    "--mode",
    type=click.Choice(["low_latency", "very_low_latency", "ultra_low_latency"]),
    default="low_latency",
)
@click.option("--limit", default=2, help="Nombre de réunions.")
def main(root: str, out_path: str, mode: str, limit: int) -> None:
    """Mesurer le modèle en flux, sous le protocole du dépôt."""
    import torch

    meetings = sorted(p for p in pathlib.Path(root).iterdir() if (p / "mix.wav").exists())
    processor, model = load(hf_token())
    processor.set_streaming_mode(mode)
    latency_ms = getattr(processor, "streaming_latency_ms", float("nan"))
    print(
        f"modèle chargé · mode {mode} · latence annoncée {latency_ms} ms"
        f" · appareil {next(model.parameters()).device}"
    )
    sampling_rate = processor.feature_extractor.sampling_rate
    relevé: dict[str, dict] = {}

    for meeting in meetings[:limit]:
        truth = reference_from_rttm(meeting / "words.rttm")
        n_truth = len({s for _, _, s in truth})
        audio, sr = sf.read(str(meeting / "mix.wav"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != sampling_rate:
            raise SystemExit(f"{meeting.name} est à {sr} Hz, le modèle veut {sampling_rate}")
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
        hypothesis = to_segments(processor, logits)

        report = speaker_consistency(truth, hypothesis)
        row = {
            "truth": n_truth,
            "der": der(truth, hypothesis),
            "drift": report.worst_drift,
            "misattributed": report.misattributed_share,
            "labels": len({s for _, _, s in hypothesis}),
            "long_range_300": report.long_range_agreement(gap_s=300),
            "rtf": rtf,
            "mode": mode,
            "latency_ms": latency_ms,
        }
        relevé[meeting.name] = row
        print(
            f"\n── {meeting.name} · {duration / 60:.1f} min · {n_truth} locuteurs"
            f"\n   DER {row['der']:.3f} · dérive {row['drift']:.3f}"
            f" · mal attr. {row['misattributed']:.3f} · {row['labels']} étiquettes"
            f" · accord ≥300 s {row['long_range_300']:.3f} · RTF {rtf:.3f}",
            flush=True,
        )
        pathlib.Path(out_path).write_text(json.dumps(relevé, indent=2, ensure_ascii=False))

    if relevé:
        rows = list(relevé.values())
        print("\n── médianes")
        for key in ("der", "labels", "long_range_300", "rtf"):
            print(f"   {key:16} {float(np.median([r[key] for r in rows])):.3f}")
        print("\n   pour mémoire, même tranche et même protocole :")
        print("   différé pleine fenêtre 0,176 · fenêtre croissante 0,176 (en direct)")
        print("   incrémental 0,254 · en ligne glouton 0,369")
    print(f"\nrelevé écrit : {out_path}")


if __name__ == "__main__":
    main()
