"""Transcription en ligne par l'encodeur Nemotron 3.5 : WER contre retard.

Pourquoi cette mesure
---------------------
L'encodeur a appris des fenêtres de 30 s vues d'un bloc. En flux, on ne
connaît ni la suite ni la moyenne de l'enregistrement : on transcrit des
fenêtres glissantes, normalisées sur elles-mêmes, et on n'arrête que les
trames assez loin du bord droit. Trois réglages décident :

- le **pas** ``hop_s`` : la fréquence des appels, donc le coût et une part du
  retard (une trame attend jusqu'à un pas) ;
- le **contexte gauche** ``left_s`` : ce que l'encodeur revoit du passé ;
- l'**anticipation** ``right_s`` : ce qu'il voit après la trame avant de
  l'arrêter, payé tel quel en retard.

On mesure, par réglage, le WER contre la vérité et l'écart au même moteur
hors ligne, et le retard réel (moyen et 95e centile) entre la fin de l'audio
d'une trame et son écriture, calcul compris.

Usage
-----
    .venv/bin/python studies/asr_nemotron_online.py \\
        --graph /Volumes/orange-dev/nemotron-study/onnx/nemotron-asr-ctc.int8.onnx
"""

from __future__ import annotations

import json
import pathlib
import sys

import click
import numpy as np

from vocal_helper.bench import wer_report
from vocal_helper.nemotron_asr import NemotronOnnxAsr, greedy_ctc, tokens_to_words

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_nemotron_vs_whisper_offline import _load, corpus  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "asr_nemotron_online.json"

#: ``(hop_s, left_s, right_s, normalize)`` : un réglage à la fois autour de (2, 8, 0,48, cumulée).
GRID = [
    (0.48, 8.0, 0.48, "running"),
    (1.0, 8.0, 0.48, "running"),
    (2.0, 8.0, 0.48, "running"),
    (4.0, 8.0, 0.48, "running"),
    (2.0, 4.0, 0.48, "running"),
    (2.0, 16.0, 0.48, "running"),
    (2.0, 24.0, 0.48, "running"),
    (2.0, 8.0, 0.0, "running"),
    (2.0, 8.0, 1.0, "running"),
    (2.0, 8.0, 2.0, "running"),
    (2.0, 8.0, 0.48, "window"),
]


def _text(asr: NemotronOnnxAsr, ids, best) -> str:
    s = asr.settings
    words, _ = tokens_to_words(greedy_ctc(ids, best, s.blank_id), s.vocab, s.frame_s)
    return " ".join(w.text for w in words)


@click.command()
@click.option("--graph", type=click.Path(path_type=pathlib.Path, exists=True), required=True)
@click.option("--corpora", default="bagarre-rich,cv-fr,tv-fr,ami", show_default=True)
@click.option("--limit", default=12, show_default=True)
@click.option("--crop-s", default=300.0, show_default=True, help="Durée maximale par fichier.")
def main(graph: pathlib.Path, corpora: str, limit: int, crop_s: float) -> None:
    """Balayer la grille et écrire le JSON."""
    asr = NemotronOnnxAsr(graph, providers=["CPUExecutionProvider"])
    results = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    done = {(r["corpus"], r["id"], r["config"]) for r in results["rows"]}
    for name in corpora.split(","):
        for ident, path, reference, *crop in corpus(name, limit):
            audio = _load(path)[: int(min(crop[0] if crop else crop_s, crop_s) * 16000)]
            if name in ("ami", "tv-fr"):
                # La vérité tronquée doit suivre l'audio tronqué : on la refait.
                reference = _cropped_reference(name, path, len(audio) / 16000)
            offline = _text(asr, *asr.frames(audio, 16000)[:2])
            configs = [("offline", None)] + [
                (f"{h}/{lft}/{r}/{n}", (h, lft, r, n)) for h, lft, r, n in GRID
            ]
            for label, cfg in configs:
                if (name, ident, label) in done:
                    continue
                if cfg is None:
                    text, delay = offline, np.zeros(1)
                else:
                    h, lft, r, n = cfg
                    ids, best, _, delay = asr.streaming_frames(
                        audio, 16000, hop_s=h, left_s=lft, right_s=r, normalize=n
                    )
                    text = _text(asr, ids, best)
                row = {
                    "corpus": name,
                    "id": ident,
                    "config": label,
                    "seconds": round(len(audio) / 16000, 2),
                    "wer": wer_report(reference, text).get("wer"),
                    "vs_offline": wer_report(offline, text).get("wer"),
                    "delay_mean_s": round(float(delay.mean()), 3),
                    "delay_p95_s": round(float(np.percentile(delay, 95)), 3),
                    "ref_words": len(reference.split()),
                }
                results["rows"].append(row)
                OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
                print(
                    name,
                    ident,
                    label,
                    row["wer"],
                    row["vs_offline"],
                    row["delay_mean_s"],
                    flush=True,
                )
    results["summary"] = summarize(results["rows"])
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps(results["summary"], indent=1))


def _cropped_reference(name: str, path: pathlib.Path, seconds: float) -> str:
    if name == "ami":
        rows = []
        for line in (path.parent / "words.rttm").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 10 and float(parts[3]) + float(parts[4]) <= seconds:
                rows.append((float(parts[3]), " ".join(parts[9:])))
        return " ".join(w for _, w in sorted(rows))
    utterances = json.loads((path.parent / "transcript.json").read_text())["utterances"]
    return " ".join(u["text"] for u in utterances if u["t1"] <= seconds)


def summarize(rows: list[dict]) -> dict:
    """Par corpus et réglage : WER pondéré, écart au hors-ligne, retards médians."""
    groups: dict = {}
    for row in rows:
        if row["wer"] is None:
            continue
        groups.setdefault((row["corpus"], row["config"]), []).append(row)
    out = {}
    for (name, config), rs in sorted(groups.items()):
        words = sum(r["ref_words"] for r in rs)
        out[f"{name}/{config}"] = {
            "files": len(rs),
            "wer": round(sum(r["wer"] * r["ref_words"] for r in rs) / max(words, 1), 4),
            "vs_offline": round(float(np.mean([r["vs_offline"] or 0 for r in rs])), 4),
            "delay_mean_s": round(float(np.median([r["delay_mean_s"] for r in rs])), 3),
            "delay_p95_s": round(float(np.median([r["delay_p95_s"] for r in rs])), 3),
        }
    return out


if __name__ == "__main__":
    main()
