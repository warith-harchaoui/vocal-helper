# Vocal Helper

[🇫🇷](LISEZMOI.md) · [🇬🇧](README.md)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#)

`Vocal Helper` fait partie de la collection `AI Helpers` — des bibliothèques Python pensées pour bâtir des outils d'intelligence artificielle.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

Vocal Helper est un **pipeline producteur/consommateur asynchrone** qui transforme un flux audio PCM en direct en énoncés diarizés et transcrits — et, en option, en résumé glissant produit par un LLM.

## Pipeline

```
[Source]   →  [VAD]   →  [Diarisation en ligne]  →  [STT]  →  [Analyste LLM (optionnel)]
  PCM         segments  segments étiquetés         texte         résumé glissant
  20 ms       voisés    par locuteur
```

Toutes les frontières entre étages sont des `asyncio.Queue` bornées ; chaque étage est sa propre coroutine.

| Étage | Modèle | Notes |
|---|---|---|
| **VAD** | Silero v5 ONNX (CPU) | Fenêtre 32 ms, `activity_threshold=0.5`, `min_silence_ms=300` par défaut. |
| **Diarisation (online)** | `pyannote/embedding` (défaut) ou `nvidia/titanet_large` (NeMo) | Embedding par segment + clustering moyenne-mobile par distance cosinus, `join_threshold=0.30`. Calibré sur AMI dev-slice N=8 (2026-06-30). |
| **STT** | [`pywhispercpp`](https://github.com/abdeladim-s/pywhispercpp) turbo | `large-v3-turbo-q5_0` par défaut, timestamps mots activés. Exécution en thread pool pour ne jamais bloquer la boucle async. |
| **Analyste LLM** *(optionnel)* | Gemma 4 e4b servi par Ollama (`gemma4:e4b`) | Résumé glissant de tout ce qui est **plus vieux que 60 s**. La fenêtre récente de 60 s reste verbatim. La variante `-mlx` est auto-sélectionnée par Ollama sur Apple-Silicon. |

## Démarrage rapide

### Installation

```bash
pip install 'vocal-helper[all]'
```

L'extra `[all]` installe la source micro, pyannote et Ollama. À la carte si tout n'est pas nécessaire :

| Extra | Apporte | Requis si |
|---|---|---|
| (aucun) | `pywhispercpp`, `silero-vad`, `audio-helper` | Sources fichier / numpy, sans diarisation |
| `[mic]` | `capture-helper` | Entrée microphone live |
| `[pyannote]` | `pyannote.audio` | `diar={'backend': 'pyannote'}` (défaut) |
| `[nemo]` | `torch`, `nemo-toolkit[asr]` | `diar={'backend': 'nemo'}` |
| `[llm]` | `ollama` | `llm={'model': 'gemma4:e4b'}` |
| `[all]` | Tout ce qui précède | Installation en une ligne |

[Ollama](https://ollama.com) doit également tourner en local si l'analyste LLM est activé :

```bash
ollama pull gemma4:e4b
ollama serve
```

### Micro live → terminal

```bash
export HF_TOKEN=hf_yourtoken    # nécessaire pour télécharger pyannote/embedding
vocal-helper mic --llm
```

### API Python

```python
import asyncio
import vocal_helper as vh

async def main():
    pipeline = vh.Pipeline(
        source=lambda: vh.sources.from_microphone(),
        config=vh.PipelineConfig(
            diar={"backend": "pyannote"},
            asr={"model": "large-v3-turbo-q5_0", "language": "fr"},
            llm={"model": "gemma4:e4b"},   # retirer pour désactiver
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f} {ev['speaker']}] {ev['text']}")
        elif "summary" in ev:
            print(f"--- résumé glissant ---\n{ev['summary']}")

asyncio.run(main())
```

### Rejouer un WAV à travers le pipeline

```bash
vocal-helper file chemin/vers/conversation.wav --llm
```

La source fichier respecte le tempo réel par défaut ; `--no-real-time` accélère le traitement (mode batch).

## Abonnés — fan-out sans posséder la boucle

Chaque étage peut être observé sans consommer le flux fusionné :

```python
async def on_voiced(seg): print("VAD :", seg["t0"], seg["t1"])
async def on_diar(seg):   print(" → ", seg["speaker"], seg["t0"], seg["t1"])

pipeline.subscribe_voiced(on_voiced)
pipeline.subscribe_diarized(on_diar)

async for ev in pipeline.run():
    ...
```

Pratique pour des relais WebSocket / SSE, du rendu UI live, ou une persistance JSONL.

## Choix de la diarisation — pourquoi le **clustering cosinus en ligne**

L'étude `pdbms` (2026-06-29, N=2089 par système) classe les diariseurs en streaming :

| Mode | Recommandé | DER (clean) |
|---|---|---|
| Streaming ≤ 300 s | `hungarian_nemo` (w=20 s) | 0.13 – 0.20 |
| Streaming > 300 s | `hungarian_pyannote` (w=30 s) | 0.30 – 0.45 |

Vocal Helper spécialise cette décision : puisque le VAD isole déjà chaque segment voisé, la machinerie à fenêtre glissante se réduit à un embedding par segment + clustering par moyenne mobile sur distance cosinus. Le `join_threshold=0.30` par défaut est la valeur sélectionnée sur AMI dev-slice N=8 dans le sweep `pyannote_stitch_threshold_sweep` du 2026-06-30.

## Auteur

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui) — `warith@deraison.ai`
