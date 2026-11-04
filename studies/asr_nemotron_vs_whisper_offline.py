"""Transcription hors ligne : whisper.cpp turbo contre l'encodeur Nemotron 3.5 en ONNX.

Pourquoi cette mesure
---------------------
Le défaut actuel du dépôt est whisper.cpp ``large-v3-turbo-q5_0``. L'encodeur
de Nemotron 3.5 Transcribe, exporté par ``studies/nemotron_asr_export.py``, est
un candidat sérieux : une tête CTC ne peut pas halluciner une phrase entière
comme le décodeur autorégressif de whisper le fait dans les silences, elle
horodate chaque jeton à 40 ms, et le même graphe tourne sur tout ce
qu'onnxruntime couvre. Reste à savoir ce qu'elle coûte en erreurs.

Corpus
------
- ``ami`` : les huit réunions d'AMI dev-slice (anglais, micro de salle mixé,
  vérité au mot, fichiers entiers de 15 à 30 minutes) ;
- ``bagarre-rich`` : mixages de 30 s en anglais et français mêlés, vérité
  exacte par construction ;
- ``cv-fr`` et ``cv-en`` : phrases courtes de Common Voice, un locuteur ;
- ``tv-fr`` : les dix premières minutes d'émissions de télévision françaises,
  contre la transcription du chemin payant (Gladia) retouchée par LLM. Ce n'est
  pas une vérité : le chiffre est un **désaccord avec le payant**, pas un WER.

Chaque moteur reçoit le fichier entier et découpe comme il l'entend ; le WER
est calculé sur le texte normalisé de ``vocal_helper.bench`` (casse,
ponctuation), en deux versions : stricte, et tolérante aux hésitations.

Usage
-----
    .venv/bin/python studies/asr_nemotron_vs_whisper_offline.py \\
        --onnx-dir /Volumes/orange-dev/nemotron-study/onnx
"""

from __future__ import annotations

import json
import pathlib
import time

import click
import numpy as np
import soundfile as sf

from vocal_helper.bench import wer_report

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path.home() / "pasdebonneoudemauvaisesituation/data"
OUT = HERE / "asr_nemotron_vs_whisper_offline.json"
#: La langue de chaque corpus, connue d'avance seulement pour ``whisper-oracle``.
LANGUAGE = {"ami": "en", "cv-fr": "fr", "cv-en": "en", "tv-fr": "fr"}
#: Les émissions TV sont tronquées à leurs dix premières minutes.
TV_CROP_S = 600.0


def _rttm_text(path: pathlib.Path) -> str:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[0] == "SPEAKER":
            rows.append((float(parts[3]), " ".join(parts[9:])))
    return " ".join(word for _, word in sorted(rows))


def _load(path: pathlib.Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    return audio


def corpus(name: str, limit: int) -> list[tuple[str, pathlib.Path, str]]:
    """``(identifiant, audio, référence)`` pour un corpus."""
    if name == "ami":
        root = DATA / "ami/dev-slice"
        return [
            (d.name, d / "mix.wav", _rttm_text(d / "words.rttm"))
            for d in sorted(root.iterdir())
            if (d / "mix.wav").exists()
        ][:limit]
    if name == "bagarre-rich":
        wavs = sorted((DATA / "bagarre-rich").glob("mix_*.wav"))[:limit]
        return [(w.stem, w, _rttm_text(w.with_suffix(".rttm"))) for w in wavs]
    if name.startswith("cv-"):
        lang = name[3:]
        items = []
        for meta in sorted((DATA / "common-voice-subset" / lang).rglob("common_voice_*.json")):
            if meta.name.endswith(".consistency.json"):
                continue
            info = json.loads(meta.read_text())
            items.append((meta.stem, meta.with_name(info["audio"]), info["sentence"]))
        # Un clip par locuteur d'abord, pour ne pas mesurer cinq fois la même voix.
        seen, picked = set(), []
        for item in items:
            speaker = item[1].parent.name
            if speaker not in seen:
                seen.add(speaker)
                picked.append(item)
        return picked[:limit]
    if name == "tv-fr":
        root = pathlib.Path.home() / "parole-publique/output"
        out = []
        for transcript in sorted(root.glob("*/transcript.json")):
            # ``-free`` : la sortie du chemin gratuit, pas une référence. Un
            # dossier peut aussi disparaître en cours d'étude (renommé ailleurs).
            if (
                transcript.parent.name.endswith("-free")
                or not (transcript.parent / "audio.mp3").exists()
            ):
                continue
            utterances = json.loads(transcript.read_text())["utterances"]
            kept = [u for u in utterances if u["t1"] <= TV_CROP_S]
            if kept:
                text = " ".join(u["text"] for u in kept)
                out.append(
                    (transcript.parent.name, transcript.parent / "audio.mp3", text, kept[-1]["t1"])
                )
        return out[:limit]
    raise click.BadParameter(f"corpus inconnu {name}")


def engines(onnx_dir: pathlib.Path, wanted: list[str]):
    """Les moteurs demandés, chacun ``audio -> texte``."""
    out = {}
    if "whisper" in wanted:
        from vocal_helper.asr import transcribe_pcm

        out["whisper-turbo"] = lambda a, _lang: transcribe_pcm(a, 16000)
    if "whisper-oracle" in wanted:
        from vocal_helper.asr import transcribe_pcm

        # Borne haute de whisper : on lui donne la langue, que la pipeline ne
        # connaît pas d'avance. Sans objet sur les mixages bilingues.
        out["whisper-oracle"] = lambda a, lang: (
            transcribe_pcm(a, 16000, language=lang) if lang else None
        )
    if "whisper-lid" in wanted:
        from vocal_helper.asr import transcribe_pcm
        from vocal_helper.lid import detect_language

        # Ce que la pipeline peut faire sans rien savoir : découvrir la langue
        # sur 30 s prises au milieu du fichier (loin des génériques), puis la
        # donner à whisper. Le temps de la détection compte dans le RTF.
        def whisper_lid(a, _lang):
            mid = max(0, len(a) // 2 - 15 * 16000)
            lang, _p = detect_language(a[mid : mid + 30 * 16000])
            return transcribe_pcm(a, 16000, language=lang)

        out["whisper-lid"] = whisper_lid
    from vocal_helper.nemotron_asr import NemotronOnnxAsr

    for precision in ("fp32", "int8"):
        if f"nemotron-{precision}" in wanted:
            asr = NemotronOnnxAsr(onnx_dir / f"nemotron-asr-ctc.{precision}.onnx")
            out[f"nemotron-{precision}"] = lambda a, _lang, asr=asr: asr.transcribe(a, 16000).text
    return out


@click.command()
@click.option("--onnx-dir", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--corpora", default="cv-fr,cv-en,bagarre-rich,ami", show_default=True)
@click.option(
    "--engines", "engine_names", default="nemotron-int8,nemotron-fp32,whisper,whisper-oracle"
)
@click.option("--limit", default=40, show_default=True, help="Fichiers par corpus.")
def main(onnx_dir: pathlib.Path, corpora: str, engine_names: str, limit: int) -> None:
    """Mesurer WER, CER et facteur temps réel, fichier par fichier."""
    results = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    done = {(r["corpus"], r["id"], r["engine"]) for r in results["rows"]}
    runners = engines(onnx_dir, engine_names.split(","))
    for name in corpora.split(","):
        for ident, path, reference, *crop in corpus(name, limit):
            audio = _load(path)
            if crop:
                audio = audio[: int(crop[0] * 16000)]
            seconds = len(audio) / 16000
            for engine, run in runners.items():
                if (name, ident, engine) in done:
                    continue
                t0 = time.perf_counter()
                hypothesis = run(audio, LANGUAGE.get(name))
                if hypothesis is None:
                    continue
                elapsed = time.perf_counter() - t0
                row = {
                    "corpus": name,
                    "id": ident,
                    "engine": engine,
                    "seconds": round(seconds, 2),
                    "rtf": round(elapsed / seconds, 4),
                    "strict": wer_report(reference, hypothesis),
                    "tolerant": wer_report(reference, hypothesis, style_tolerant=True),
                    "hypothesis": hypothesis[:2000],
                }
                results["rows"].append(row)
                OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
                print(name, ident, engine, row["strict"].get("wer"), row["rtf"], flush=True)
    results["summary"] = summarize(results["rows"])
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps(results["summary"], indent=1))


def summarize(rows: list[dict]) -> dict:
    """WER pondéré par le nombre de mots de référence, et RTF médian."""
    summary: dict = {}
    for row in rows:
        key = f"{row['corpus']}/{row['engine']}"
        s = summary.setdefault(
            key, {"files": 0, "ref_words": 0, "errors": 0.0, "cer": [], "rtf": []}
        )
        strict = row["strict"]
        if strict.get("wer") is None:
            continue
        s["files"] += 1
        s["ref_words"] += strict["ref_words"]
        s["errors"] += strict["wer"] * strict["ref_words"]
        s["cer"].append(strict["cer"])
        s["rtf"].append(row["rtf"])
    return {
        key: {
            "files": s["files"],
            "wer": round(s["errors"] / max(s["ref_words"], 1), 4),
            "cer_median": round(float(np.median(s["cer"])), 4) if s["cer"] else None,
            "rtf_median": round(float(np.median(s["rtf"])), 4) if s["rtf"] else None,
        }
        for key, s in sorted(summary.items())
    }


if __name__ == "__main__":
    main()
