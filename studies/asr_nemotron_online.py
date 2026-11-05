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

Verdict :
---------
2026-11-04 ; 12 fichiers coupés à 300 s pour bagarre-rich, Common Voice
français et la télévision, 5 réunions AMI (étude arrêtée là, les tendances
étant nettes). Taux d'erreur sur les mots pondéré ; délai moyen par mot.

- **Le réglage retenu est 2 s de pas, 16 s de contexte, 0,48 s
  d'anticipation, normalisation cumulée** : 0,372 / 0,303 / 0,482 / 0,192
  (bagarre-rich, CV-fr, TV, AMI) pour 1,7 s de délai, contre 0,318 / 0,258 /
  0,481 / 0,180 hors ligne. Le flux perd 0 à 0,05 ; à la télévision, rien.
- **L'anticipation est le levier qui compte** : sans elle (0 s), l'erreur
  monte de 0,05 à 0,11 partout ; au-delà de 0,48 s, plus rien ne se gagne.
- **Le contexte passé aide les mixages** (8 → 16 s : 0,413 → 0,372) puis
  plafonne ; 24 s n'apporte presque rien de plus et allonge le calcul.
- Un pas plus court ne coûte que du calcul et un peu de justesse (0,48 s :
  0,43 sur bagarre-rich pour 0,9 s de délai) ; normaliser par fenêtre plutôt
  que cumulativement ne change rien.
- Ce flux n'est pas encore le défaut en direct : ``asr_live_whisper_vs_nemotron.py``
  montre que Nemotron par segment fait aussi bien pour dix fois moins de calcul.

Usage
-----
    .venv/bin/python studies/asr_nemotron_online.py \\
        --graph "$VH_NEMOTRON_STUDY/onnx/nemotron-asr-ctc.int8.onnx"
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
            try:
                audio = _load(path)
            except (OSError, RuntimeError):
                # Corpus TV vivant : un fichier peut disparaître pendant l'étude.
                print(name, ident, "illisible, sauté", flush=True)
                continue
            audio = audio[: int(min(crop[0] if crop else crop_s, crop_s) * 16000)]
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
