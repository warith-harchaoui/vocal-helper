# Exemples

[🇫🇷 Français](https://github.com/warith-harchaoui/vocal-helper/blob/main/EXEMPLES.md) · [🇬🇧 EXAMPLES.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/EXAMPLES.md)

Des recettes choisies pour les formes dont on a réellement besoin : un micro en
direct, un fichier enregistré, un relais vers votre propre code puis la mesure
qui dit si tout cela a marché.

Chaque extrait ci-dessous a été vérifié contre le code source. Ceux qui ne
demandent qu'un processeur ont été **exécutés** : la sortie affichée est celle
qu'ils ont imprimée, collée telle quelle plutôt que décrite. Ceux qui exigent
des poids de modèle, un micro ou un modèle de langue servi sont signalés : ils
ont été vérifiés contre le code sans être lancés.

## Table des matières

- [Mise en place](#mise-en-place)
- [1. Micro en direct → transcript dans le terminal](#1-micro-en-direct--transcript-dans-le-terminal)
- [2. Rejouer un WAV en rafale (lot hors ligne)](#2-rejouer-un-wav-en-rafale-lot-hors-ligne)
- [3. Un résumé glissant sur une réunion](#3-un-résumé-glissant-sur-une-réunion)
- [4. Relais WebSocket personnalisé via l'API d'abonnement](#4-relais-websocket-personnalisé-via-lapi-dabonnement)
- [5. Diariser sans PyTorch puis régler le regroupement](#5-diariser-sans-pytorch-puis-régler-le-regroupement)
- [6. Transcription en un seul appel, sans chaîne](#6-transcription-en-un-seul-appel-sans-chaîne)
- [7. Noter une transcription : pourquoi le cpWER plutôt que le WER](#7-noter-une-transcription--pourquoi-le-cpwer-plutôt-que-le-wer)
- [8. Demander au routeur quel backend prendre](#8-demander-au-routeur-quel-backend-prendre)
- [Ligne de commande](#ligne-de-commande)
- [Dépannage](#dépannage)

## Mise en place

```bash
pip install 'vocal-helper[all]'
cp settings.yaml.example settings.yaml   # porte l'URL des diarization-engines
```

L'analyste LLM (l'exemple 3 ainsi que le drapeau `--llm`) demande en plus un
backend de service démarré. `best-engine-ai-helper` choisit le modèle adapté à
votre machine depuis le brief versionné `vocal_helper/llm.brief.yaml` ; le
fichier d'engine résolu nomme le `ollama pull` exact à lancer :

```bash
ollama serve
```

Aucun jeton HuggingFace nulle part. Tous les poids de modèle viennent du lot
diarization-engines auto-hébergé nommé dans `settings.yaml`
(`engines.diarization_url`) — voir
[le LISEZMOI](LISEZMOI.md#poids-des-modèles--aucun-huggingface-requis).

Tous les extraits ci-dessous supposent ces trois imports, rien de plus :

```python
import asyncio

import numpy as np

import vocal_helper as voh
```

`voh` est l'alias canonique de ce paquet dans la suite — celui-là même
qu'emploient les docstrings et le README, si bien que du code recopié de l'un à
l'autre continue de fonctionner. Un extrait qui a besoin d'autre chose que ces
trois noms l'importe sur place, là où il s'en sert.

## 1. Micro en direct → transcript dans le terminal

Demande l'extra `[mic]` plus un micro : celui-là n'a pas été exécuté.

```bash
vocal-helper mic \
  --initial-prompt "Réunion d'équipe : design, marketing, planning, livrables"
```

`--initial-prompt` est **fortement recommandé**, ce n'est pas un ornement : le
balayage du 2026-06-30 sur AMI dev-slice
(`studies/whisper_prompt_lang_lock.py`) a mesuré qu'un prompt de biais aligné
sur le domaine fait chuter le WER de 15 à 25 points de pourcentage et économise
jusqu'à 39 % de RTF. Nommez votre domaine conversationnel ainsi qu'une poignée
de noms propres ou de termes techniques que vous attendez.

La même chose en Python (la démo livrée est `examples/live_mic_to_text.py`) :

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.Pipeline(
        source=lambda: voh.sources.from_microphone(),
        config=voh.PipelineConfig(
            asr={
                "language": "auto",  # découverte depuis l'audio, aucun défaut
                "initial_prompt": "Réunion d'équipe : design, marketing, planning, livrables",
            },
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f}s {ev['speaker']}] {ev['text']}")


asyncio.run(main())
```

## 2. Rejouer un WAV en rafale (lot hors ligne)

```bash
vocal-helper file ./conversation.wav --no-real-time --jsonl > out.jsonl
```

`--no-real-time` supprime le cadencement à l'horloge : les trames partent aussi
vite qu'elles se décodent. Le drapeau bascule aussi le run sur le diariseur
**hors ligne** à buffer entier quand il y en a un d'installé, le chemin fiable
pour le traitement par lots. `--jsonl` émet un événement JSON par ligne, ce
qu'il vous faut pour alimenter `jq` ou un stockage.

Le routeur écrit sa décision sur stderr avant le premier énoncé, donc une sortie
standard redirigée reste du JSON propre :

```
vocal-helper: router → offline pyannote (quality DER 0.122, speed RTF 0.067; …). Pass --diar-backend to override.
```

## 3. Un résumé glissant sur une réunion

Demande un modèle de langue servi : celui-là n'a pas été exécuté.

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.OfflinePipeline(
        source=lambda: voh.sources.from_wav_file("./reunion.wav", real_time=False),
        config=voh.OfflinePipelineConfig(
            asr={"language": "auto"},  # découverte depuis l'audio, aucun défaut
            llm={
                "engine": voh.resolve_engine(),  # best-engine-ai-helper choisit le modèle
                "recent_window_s": 60.0,  # les 60 dernières secondes restent verbatim
                "flush_every_s": 60.0,  # rafraîchit par 60 s de parole évincée
            },
        ),
    )
    async for ev in pipeline.run():
        if "summary" in ev:
            print("\n--- résumé glissant ---")
            print(ev["summary"])


asyncio.run(main())
```

Le dictionnaire `llm` est déversé tel quel dans `GemmaAnalystStage` : ses clés
sont les noms de paramètres de ce constructeur. Il n'y a **aucune clé `model`** :
depuis `2.0.0` le modèle ne se nomme plus ici — `voh.resolve_engine()` rend le
descripteur que `best-engine-ai-helper` a résolu pour cette machine ; passer
un tag de modèle lève `TypeError`. `flush_every_s` prime sur la cadence par
compte `flush_every_n`, qui ne s'applique que si `flush_every_s=None`.

## 4. Relais WebSocket personnalisé via l'API d'abonnement

`subscribe_utterances` observe l'étage sans consommer le flux d'événements
fusionné : la boucle `async for` ci-dessous reste libre de faire autre chose.
Demande `aiohttp` plus un micro, donc celui-là n'a pas été exécuté.

```python
import asyncio
import json

from aiohttp import web

import vocal_helper as voh

clients: set[web.WebSocketResponse] = set()


async def on_utterance(u):
    payload = json.dumps(
        {
            "t0": u["t0"],
            "t1": u["t1"],
            "speaker": u["speaker"],
            "text": u["text"],
        }
    )
    for ws in list(clients):
        try:
            await ws.send_str(payload)
        except Exception:
            clients.discard(ws)


async def ws_handler(req):
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    clients.add(ws)
    async for _ in ws:
        pass
    clients.discard(ws)
    return ws


async def run_pipeline():
    pipeline = voh.Pipeline(source=lambda: voh.sources.from_microphone())
    pipeline.subscribe_utterances(on_utterance)
    async for _ in pipeline.run():
        pass


async def main():
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8765)
    await site.start()
    await run_pipeline()


asyncio.run(main())
```

`subscribe_voiced` puis `subscribe_diarized` branchent de la même façon les deux
étages antérieurs : c'est ainsi qu'un sous-titre arrive à l'écran avant que la
transcription existe.

## 5. Diariser sans PyTorch puis régler le regroupement

Le backend `sherpa` est le même TitaNet-large par onnxruntime : pas de torch,
pas de CUDA, embarquable sur iOS comme sur Android. Demande les modèles sherpa
du lot, donc celui-là n'a pas été exécuté.

```python
import asyncio

import vocal_helper as voh


async def main():
    pipeline = voh.OfflinePipeline(
        source=lambda: voh.sources.from_wav_file("./appel.wav", real_time=False),
        config=voh.OfflinePipelineConfig(
            diar={
                "backend": "sherpa",
                "sherpa_num_clusters": 2,  # appel à deux connu : on saute l'auto-détection
            },
            asr={"language": "auto"},
        ),
    )
    async for ev in pipeline.run():
        if "text" in ev:
            print(f"[{ev['t0']:7.2f}s {ev['speaker']}] {ev['text']}")


asyncio.run(main())
```

`sherpa_num_clusters=2` n'est pas un détail. Le seuil de l'auto-détecteur a été
réglé sur de l'audio de réunion propre et il sur-segmente méchamment la
téléphonie bruitée : le cas mesuré laissait une trentaine de locuteurs sur un
appel à deux. Quand vous connaissez le compte, dites-le.

Du côté streaming, le levier équivalent est `join_threshold`, la distance
cosinus en dessous de laquelle un segment rejoint un locuteur existant :

```python
config = voh.PipelineConfig(
    diar={"backend": "nemo", "join_threshold": 0.60},  # plus serré que le défaut 0.50
    asr={"language": "auto"},
)
```

`nemo` est déjà l'embedder online par défaut : ce que l'extrait change, c'est
uniquement le seuil. Le monter fusionne plus volontiers, ce qu'il faut quand une
même personne se fait découper en plusieurs ; le baisser sépare plus volontiers,
pour le cas inverse. Le 0,50 livré est lui-même le résultat d'une montée depuis
0,30 : sur 8 réunions AMI à 4 locuteurs, 0,30 inventait 87 locuteurs là où 0,50
en trouve 28 ; 0,60 approche le vrai compte en forçant des personnes réellement
différentes à se confondre — la direction dont on ne revient pas.

## 6. Transcription en un seul appel, sans chaîne

**Exécuté.** Un buffer PCM en entrée, du texte en sortie : ni VAD, ni
diarisation, ni résumé. La langue est découverte depuis l'audio par défaut
(`language="auto"`) : il n'y a ni langue imposée ni appariement de langues.

```python
import numpy as np

import vocal_helper as voh

pcm = np.zeros(16_000 * 5, dtype=np.float32)  # cinq secondes de silence numérique
text = voh.transcribe_pcm(pcm, sr=16_000)  # langue découverte automatiquement
print(repr(text))
```

```
'Thank you.'
```

Cette sortie n'est pas une coquille et c'est la chose la plus utile que ce
fichier puisse vous montrer : devant du silence, whisper invente une courte
phrase plausible au lieu de ne rien rendre. Ne lui donnez pas de l'audio non
filtré en faisant confiance au résultat — c'est précisément le rôle de l'étage
VAD placé devant la chaîne.

Besoin de la langue sur laquelle whisper s'est arrêté ? L'assistant jumeau la
rend à côté du texte :

```python
from vocal_helper.asr import transcribe_pcm_with_language

text, language = transcribe_pcm_with_language(pcm, sr=16_000)
print(repr(language), repr(text))
```

```
None 'Thank you.'
```

`None` est la réponse honnête ici : il n'y avait aucune parole à identifier. Sur
de la vraie parole, c'est un code ISO-639-1 comme `'fr'`.

## 7. Noter une transcription : pourquoi le cpWER plutôt que le WER

**Exécuté.** Deux locuteurs, chaque mot transcrit correctement — et le diariseur
les a fondus en un seul. Le WER simple ne voit pas l'erreur ; le cpWER facture
ensemble les mots et le locuteur.

```python
import vocal_helper as voh

reference = [
    {"speaker": "A", "text": "on se voit mardi pour la revue de design"},
    {"speaker": "B", "text": "parfait je prepare les maquettes"},
]
# Les mêmes mots, mais le diariseur a fondu les deux voix en un seul locuteur.
merged = [
    {"speaker": "S0", "text": "on se voit mardi pour la revue de design"},
    {"speaker": "S0", "text": "parfait je prepare les maquettes"},
]

flat = lambda turns: " ".join(t["text"] for t in turns)
print(voh.bench.wer_report(flat(reference), flat(merged))["wer"])
print(voh.bench.cpwer(reference, merged))
```

```
0.0
{'cpwer': 0.7143, 'errors': 10, 'ref_words': 14, 'n_ref_speakers': 2, 'n_hyp_speakers': 1, 'mapping': {'S0': 'A'}, 'style_tolerant': False}
```

Un 0.0 parfait à côté d'un 0.71 : cet écart est tout l'argument du module.
`n_ref_speakers` contre `n_hyp_speakers` dit d'un coup d'œil que les dégâts sont
dans l'attribution et non dans la transcription, donc quel étage aller réparer.

La métrique compagne répond à « **quels** mots ont été ratés », ce qu'un taux ne
fait jamais :

```python
print(voh.bench.term_error_rate("le Figaro publie demain", "le Fegaro publie demain", ["Figaro"]))
```

```
{'present': 1, 'found': 0, 'missed': ['Figaro'], 'error_rate': 1.0}
```

Une lettre fausse est une erreur d'arrondi en WER et un enregistrement perdu
dans n'importe quelle recherche à l'échelle du corpus.

## 8. Demander au routeur quel backend prendre

**Exécuté.** Du Python pur, aucun modèle chargé : l'*aiguilleur* répond depuis le
profil mesuré et il rapporte la qualité ainsi que la vitesse, pas seulement un
nom.

```python
import vocal_helper as voh

plan = voh.select_diarization(live=False, duration_s=45.0, max_speakers=3)
print(plan.backend, plan.expected_der, plan.expected_rtf)
print(plan.reason)
print(voh.select_diarization(live=False, duration_s=1800.0).backend)
print(voh.select_diarization(live=True, duration_s=None).backend)
print(voh.select_diarization(live=False, duration_s=600.0, torch_free=True).backend)
```

```
nemo 0.142 0.051
≤300s, ≤4 speakers → nemo Sortformer (end-to-end, confusion ~0; DER 0.142, RTF 0.051)
pyannote
nemo
sherpa
```

Court et dense part sur `nemo`, la forme longue sur `pyannote`, un flux en
direct sur `nemo` et une machine sans PyTorch sur `sherpa`. Le CLI puis la
surface HTTP appellent cette même fonction : la décision n'a jamais à être
refaite à la main — et lire `plan.reason` est la façon de la vérifier au lieu de
lui faire confiance.

## Ligne de commande

Un point d'entrée, quatre sous-commandes. **Exécuté** — voici le vrai `--help` :

```
$ vocal-helper --help
Usage: vocal-helper [OPTIONS] COMMAND [ARGS]...

  Turn audio into diarized, transcribed utterances — live or from a file.

  Each subcommand feeds one source (microphone, WAV, URL) through the VAD →
  diarization → ASR pipeline, with an optional rolling LLM summary. The
  "transcribe" command is the one-shot escape hatch: no VAD, no diarization.

Options:
  -V, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  file        Replay a 16 kHz mono WAV through the pipeline.
  mic         Live microphone input (needs the [mic] extra).
  transcribe  One-shot transcription of a WAV file (skip VAD / diarization).
  url         Stream from any URL yt-dlp can reach (needs the [stream]...
```

```bash
vocal-helper -V                                   # vocal-helper, version 3.3.0

vocal-helper transcribe clip.wav                  # texte seul, langue découverte
vocal-helper transcribe clip.wav --jsonl          # {"path": …, "text": …} sur stdout

vocal-helper file reunion.wav --offline --llm     # diarisation buffer entier + résumé
vocal-helper file reunion.wav --no-real-time --jsonl > out.jsonl

vocal-helper mic --llm --eot                      # en direct, avec le détecteur de tour
vocal-helper url "https://youtu.be/…" --language fr

vocal-helper file reunion.wav --diar-backend sherpa   # forcer, contre le routeur
```

`--whisper-model`, `--language`, `--threads`, `--initial-prompt`,
`--diar-backend`, `--join-threshold`, `--llm`, `--llm-recent-window-s`,
`--endpoint`, `--eot` puis `--jsonl` sont partagés par `mic`, `file` et `url` ;
`vocal-helper <sous-commande> --help` imprime la surface complète. `--language`
vaut `auto` par défaut : ne le passez que pour *forcer* une langue plutôt que
pour en découvrir une.

## Dépannage

**`vocal-helper: no offline diarizer installed — using the online diarizer with
the refine pass.`** Le chemin par lots voulait le diariseur à buffer entier sans
en trouver un. Installez `vocal-helper[pyannote]` puis copiez
`settings.yaml.example` vers `settings.yaml` pour que le lot puisse être
récupéré. Le run aboutit quand même, avec un DER moins bon.

**`StageCrashed`** au sortir de `run()`. Depuis `3.0.0`, une chaîne qui a perdu
un étage lève au lieu de rendre une transcription vide dont la raison dort dans
un log. Relevez les exceptions sur la chaîne :

```python
from vocal_helper.pipeline import StageCrashed

try:
    async for ev in pipeline.run():
        ...
except StageCrashed:
    for stage, err in pipeline.stage_failures:
        print(stage, type(err).__name__, err)
```

`voh.Pipeline(source=…, strict=False)` redonne l'indulgence d'avant ;
`stage_failures` est rempli dans les deux cas, donc même le chemin indulgent
peut être interrogé sur ce qui a cassé.

**Une transcription apparaît là où il n'y avait que du silence.** Voir
l'exemple 6 : whisper hallucine sur le silence numérique. Gardez l'étage VAD
devant l'ASR puis traitez `transcribe_pcm` sur de l'audio non filtré comme un
test de fumée plutôt que comme une mesure.

**Des dizaines de locuteurs sur un appel téléphonique à deux.**
L'auto-détecteur de `sherpa` a été réglé sur de l'audio de réunion propre.
Passez `sherpa_num_clusters=2` ; à défaut, montez
`sherpa_cluster_threshold` au-dessus de son défaut `0.5`.

**L'analyste LLM bloque ou plante au premier énoncé.** `resolve_engine()` ne
fait que résoudre le descripteur : il ne démarre pas le serveur. Lancez
`ollama serve` (ou pointez `--endpoint` vers un hôte distant) puis vérifiez que
le `vocal_helper/llm.engine.yaml` généré nomme un modèle que vous avez tiré.

**`ModuleNotFoundError` sur une source.** `from_microphone` demande l'extra
`[mic]`, `from_url` demande `[stream]` et les surfaces HTTP / MCP demandent
`[api,mcp]`. L'installation de base ne porte volontairement que les sources
fichier et numpy.
