# Exemples

Recettes choisies pour les cas d'usage courants. Tous les exemples supposent :

```bash
pip install 'vocal-helper[all]'
cp settings.yaml.example settings.yaml   # porte l'URL des diarization-engines
ollama serve                             # l'analyste LLM
```

Aucun jeton HuggingFace n'est nécessaire : tous les poids de modèle viennent du
lot diarization-engines auto-hébergé, configuré dans `settings.yaml`
(`engines.diarization_url`). Voir le
[README](README.md#model-weights--no-huggingface-needed) pour les détails.

---

## 1. Micro en direct → transcript dans le terminal

```bash
vocal-helper mic \
  --initial-prompt "Réunion d'équipe : design, marketing, planning, livrables"
```

L'argument `--initial-prompt` est **fortement recommandé** : le balayage du
2026-06-30 sur AMI dev-slice (`studies/whisper_prompt_lang_lock.py`) a montré
qu'un prompt de biais aligné sur le domaine fait chuter le WER de 15 à 25
points de pourcentage et économise jusqu'à 39 % de RTF : nommez votre domaine
conversationnel et une poignée de noms propres ou de termes techniques
attendus.

Ou en Python (la démo dans `examples/live_mic_to_text.py`) :

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_microphone(),
        config=voh.PipelineConfig(
            asr={
                "language": "auto",  # découverte à partir de l'audio, aucun défaut
                "initial_prompt": "Réunion d'équipe : design, marketing, planning, livrables",
            },
        ),
    )
    async for ev in p.run():
        if "text" in ev:
            print(f"[{ev['t0']:.1f}s {ev['speaker']}] {ev['text']}")

asyncio.run(main())
```

---

## 2. Rejouer un WAV en rafale (traitement par lots hors ligne)

```bash
vocal-helper file ./conversation.wav --no-real-time --jsonl > out.jsonl
```

`--jsonl` émet un événement par ligne, idéal pour un pipe vers `jq` ou un
stockage en aval.

---

## 3. Réunion à deux locuteurs avec résumé Gemma glissant

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_wav_file("./meeting.wav"),
        config=voh.PipelineConfig(
            asr={"language": "auto"},  # découverte à partir de l'audio, aucun défaut
            llm={
                "model": "gemma4:e4b",
                "recent_window_s": 60.0,   # fenêtre verbatim de 60 s
                "flush_every_n": 5,        # résume toutes les 5 énonciations évincées
            },
        ),
    )
    async for ev in p.run():
        if "summary" in ev:
            print("\n--- résumé glissant ---")
            print(ev["summary"])

asyncio.run(main())
```

---

## 4. Relais WebSocket personnalisé via l'API d'abonnement

```python
import asyncio, json, vocal_helper as voh
from aiohttp import web

clients: set[web.WebSocketResponse] = set()

async def on_utterance(u):
    payload = json.dumps({
        "t0": u["t0"], "t1": u["t1"],
        "speaker": u["speaker"], "text": u["text"],
    })
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
    p = voh.Pipeline(source=lambda: voh.sources.from_microphone())
    p.subscribe_utterances(on_utterance)
    async for _ in p.run():
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

---

## 5. Diarisation NeMo TitaNet sur un mixage bruité

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_wav_file("./noisy.wav"),
        config=voh.PipelineConfig(
            diar={
                "backend": "nemo",
                "join_threshold": 0.35,   # la distribution de TitaNet est plus large
            },
            asr={"language": "auto"},     # découverte à partir de l'audio, aucun défaut
        ),
    )
    async for ev in p.run():
        if "text" in ev:
            print(ev)

asyncio.run(main())
```

---

## 6. Transcription synchrone en un seul appel (sans pipeline)

Pour quand vous avez un seul buffer PCM et voulez juste récupérer le texte. La
langue est **découverte** à partir de l'audio par défaut (`language="auto"`),
sans valeur imposée ni appariement :

```python
import numpy as np, vocal_helper as voh

pcm = np.zeros(16_000 * 5, dtype=np.float32)  # cinq secondes de silence
text = voh.transcribe_pcm(pcm, sr=16_000)     # langue découverte automatiquement
print(text)
```

Besoin de la langue que whisper a réellement détectée ? Utilisez l'assistant
jumeau qui la renvoie aux côtés du texte :

```python
from vocal_helper.asr import transcribe_pcm_with_language

text, language = transcribe_pcm_with_language(pcm, sr=16_000)
print(language, text)   # ex. 'fr Bonjour tout le monde'
```
