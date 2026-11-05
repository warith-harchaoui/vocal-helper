"""Détection de parole : Silero contre les deux sous-produits de Nemotron.

Pourquoi cette mesure
---------------------
Le dépôt détecte la parole avec Silero v5, un réseau dédié de 2 Mo. Or les
deux graphes Nemotron en donnent déjà une, sans coût supplémentaire :

- le **diariseur** rend une probabilité d'activité par locuteur toutes les
  10 ms ; leur maximum dit si quelqu'un parle ;
- la **tête CTC** de l'ASR rend la log-probabilité du blanc toutes les 40 ms ;
  ``1 − exp(blanc)`` dit si un jeton est en train d'être émis.

Si l'un des deux égale Silero, on peut retirer un modèle du chemin.

Protocole
---------
Vérité : l'union des intervalles de mots de la RTTM, les trous de moins de
300 ms comblés (le ``min_silence_ms`` de ``SileroVADStage``), sur une grille de
10 ms. On rapporte l'aire sous la courbe ROC (indépendante du seuil), puis, au
seuil 0,5 et au meilleur seuil, la F-mesure, le taux de parole manquée et le
taux de fausse alarme.

Hors ligne et en ligne : Silero et le diariseur sont causaux par construction,
leur score est donc le même ; la tête CTC est mesurée brute et lissée (maximum
sur 0,6 s, ses pics étant séparés de blancs même en pleine parole), sur le
fichier entier (hors ligne) et en flux simulé à 2 s de pas, 16 s de contexte et 0,48 s
d'anticipation (en ligne).

Verdict :
---------
Silero reste le détecteur de parole par défaut (2026-11-04 ; 8 réunions AMI
coupées à 600 s, 20 mixages bagarre et 20 bagarre-rich ; médianes par fichier).

- **Silero est le meilleur sur les vraies réunions** : aire sous la courbe ROC
  (AUC) 0,964 sur AMI, contre 0,929 pour le diariseur et 0,923 pour la tête CTC
  lissée.
- **Le diariseur Nemotron fait jeu égal sur les mixages** : AUC 0,894 et 0,900
  contre 0,889 et 0,910 pour Silero, F-mesure 0,974 et 0,944 contre 0,968 et
  0,948. Quand il tourne de toute façon, son activité vaut une détection de
  parole gratuite ; elle ne justifie pas de retirer Silero du chemin en direct,
  qui en a besoin avant tout diariseur.
- **La tête CTC brute ne détecte pas la parole** (AUC 0,60 à 0,73, 74 à 80 %
  de parole manquée au seuil 0,5) : ses pics sont séparés de blancs même en
  pleine parole. Lissée sur 0,6 s, elle rattrape Silero sur AMI (F 0,918) mais
  crie à la parole partout sur les mixages (75 à 82 % de fausses alarmes).
- Hors ligne et en flux (2 s de pas, 16 s de contexte) donnent la même chose
  pour la tête CTC, à 0,01 d'AUC près.

Usage
-----
    .venv/bin/python studies/vad_nemotron_vs_silero.py \\
        --asr-graph "$VH_NEMOTRON_STUDY/onnx/nemotron-asr-ctc.int8.onnx"
"""

from __future__ import annotations

import json
import pathlib
import sys

import click
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_nemotron_vs_whisper_offline import DATA, _load  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "vad_nemotron_vs_silero.json"
STEP = 0.01
FILL_GAP_S = 0.3


def reference_mask(rttm: pathlib.Path, seconds: float) -> np.ndarray:
    """La vérité de parole sur la grille de 10 ms."""
    spans = []
    for line in rttm.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "SPEAKER":
            t0, dur = float(parts[3]), float(parts[4])
            if t0 < seconds:
                spans.append((t0, min(t0 + dur, seconds)))
    spans.sort()
    merged: list[list[float]] = []
    for t0, t1 in spans:
        if merged and t0 - merged[-1][1] < FILL_GAP_S:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    mask = np.zeros(int(round(seconds / STEP)), dtype=bool)
    for t0, t1 in merged:
        mask[int(t0 / STEP) : int(np.ceil(t1 / STEP))] = True
    return mask


def to_grid(scores: np.ndarray, frame_s: float, n: int) -> np.ndarray:
    """Ramener un score par trame de ``frame_s`` sur la grille de 10 ms."""
    index = np.minimum((np.arange(n) * STEP / frame_s).astype(int), len(scores) - 1)
    return scores[index] if len(scores) else np.zeros(n)


def silero_scores(audio: np.ndarray) -> np.ndarray:
    """Silero v5, fenêtres consécutives de 512 échantillons (32 ms)."""
    from vocal_helper.vad import SileroVADStage

    stage = SileroVADStage()
    stage._ensure_model()
    if hasattr(stage._model, "reset_states"):
        stage._model.reset_states()
    n = len(audio) // 512
    return np.array([stage._score_window(audio[i * 512 : (i + 1) * 512]) for i in range(n)])


#: Une tête CTC émet des pics : entre deux jetons d'un même mot, le blanc domine.
#: Le maximum sur 0,6 s (15 trames de 40 ms) rend la parole continue.
CTC_SMOOTH_FRAMES = 15


def ctc_detectors(asr, audio: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """La parole selon la tête CTC, brute et lissée, hors ligne et en ligne."""
    from scipy.ndimage import maximum_filter1d

    offline = 1 - np.exp(asr.frames(audio, 16000)[2])
    online = 1 - np.exp(asr.streaming_frames(audio, 16000, hop_s=2.0, left_s=16.0, right_s=0.48)[2])
    out = {}
    for name, speech in (("ctc-offline", offline), ("ctc-online", online)):
        out[name] = to_grid(speech, 0.04, n)
        smooth = maximum_filter1d(speech, size=CTC_SMOOTH_FRAMES, mode="nearest")
        out[f"{name}-smooth"] = to_grid(smooth, 0.04, n)
    return out


def metrics(scores: np.ndarray, truth: np.ndarray) -> dict:
    """AUC, puis F, manqué et fausse alarme au seuil 0,5 et au meilleur seuil."""

    def at(threshold: float) -> dict:
        pred = scores >= threshold
        tp = float((pred & truth).sum())
        fp = float((pred & ~truth).sum())
        fn = float((~pred & truth).sum())
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        return {
            "threshold": round(threshold, 3),
            "f1": round(f, 4),
            "miss": round(fn / max(truth.sum(), 1), 4),
            "false_alarm": round(fp / max((~truth).sum(), 1), 4),
        }

    order = np.argsort(-scores, kind="stable")
    hits = truth[order]
    tpr = np.concatenate([[0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    auc = float(np.trapezoid(tpr, fpr))
    grid = [at(t) for t in np.linspace(0.05, 0.95, 19)]
    return {"auc": round(auc, 4), "at_0.5": at(0.5), "best": max(grid, key=lambda m: m["f1"])}


def files(limit: int) -> list[tuple[str, str, pathlib.Path, pathlib.Path]]:
    """``(corpus, id, audio, rttm)``."""
    out = []
    for d in sorted((DATA / "ami/dev-slice").iterdir()):
        if (d / "mix.wav").exists():
            out.append(("ami", d.name, d / "mix.wav", d / "words.rttm"))
    for name in ("bagarre", "bagarre-rich"):
        for w in sorted((DATA / name).glob("mix_*.wav"))[:limit]:
            out.append((name, w.stem, w, w.with_suffix(".rttm")))
    return out


@click.command()
@click.option("--asr-graph", type=click.Path(path_type=pathlib.Path, exists=True), required=True)
@click.option("--limit", default=20, show_default=True, help="Mixages par corpus bagarre.")
@click.option("--crop-s", default=600.0, show_default=True, help="Durée maximale par fichier.")
def main(asr_graph: pathlib.Path, limit: int, crop_s: float) -> None:
    """Scorer chaque détecteur sur chaque fichier, puis agréger par corpus."""
    from vocal_helper.nemotron import NemotronOnnxDiar
    from vocal_helper.nemotron_asr import NemotronOnnxAsr

    asr = NemotronOnnxAsr(asr_graph, providers=["CPUExecutionProvider"])
    diar = NemotronOnnxDiar()
    results = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    done = {(r["corpus"], r["id"]) for r in results["rows"]}
    for corpus, ident, wav, rttm in files(limit):
        if (corpus, ident) in done:
            continue
        audio = _load(wav)[: int(crop_s * 16000)]
        seconds = len(audio) / 16000
        truth = reference_mask(rttm, seconds)
        n = len(truth)
        detectors = {
            "silero": to_grid(silero_scores(audio), 0.032, n),
            "nemotron-diar": to_grid(diar.probabilities(audio, 16000).max(axis=1), 0.01, n),
            **ctc_detectors(asr, audio, n),
        }
        row = {"corpus": corpus, "id": ident, "seconds": round(seconds, 1)}
        for name, scores in detectors.items():
            row[name] = metrics(scores, truth)
        results["rows"].append(row)
        OUT.write_text(json.dumps(results, indent=1))
        print(corpus, ident, {k: row[k]["auc"] for k in detectors}, flush=True)
    results["summary"] = summarize(results["rows"])
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results["summary"], indent=1))


def summarize(rows: list[dict]) -> dict:
    """Médianes par corpus et détecteur."""
    out: dict = {}
    for row in rows:
        for name, m in row.items():
            if not isinstance(m, dict):
                continue
            s = out.setdefault(
                f"{row['corpus']}/{name}",
                {"auc": [], "f1_0.5": [], "f1_best": [], "miss_0.5": [], "fa_0.5": []},
            )
            s["auc"].append(m["auc"])
            s["f1_0.5"].append(m["at_0.5"]["f1"])
            s["f1_best"].append(m["best"]["f1"])
            s["miss_0.5"].append(m["at_0.5"]["miss"])
            s["fa_0.5"].append(m["at_0.5"]["false_alarm"])
    return {
        key: {"files": len(v["auc"]), **{k: round(float(np.median(x)), 4) for k, x in v.items()}}
        for key, v in sorted(out.items())
    }


if __name__ == "__main__":
    main()
