# Examples

Hand-picked recipes for the common shapes. All examples assume :

```bash
pip install 'vocal-helper[all]'
cp settings.yaml.example settings.yaml   # carries the diarization-engines URL
ollama serve                             # LLM analyst
```

No HuggingFace token needed — all model weights come from the self-hosted
diarization-engines bundle configured in `settings.yaml`
(`engines.diarization_url`). See the
[README](README.md#model-weights--no-huggingface-needed) for details.

---

## 1. Live mic → terminal transcript

```bash
vocal-helper mic \
  --initial-prompt "Réunion d'équipe : design, marketing, planning, livrables"
```

The `--initial-prompt` arg is **strongly recommended** : the 2026-06-30 sweep on AMI dev-slice (`studies/whisper_prompt_lang_lock.py`) showed a domain-aligned bias prompt drops WER by 15-25 percentage points and saves up to 39 % RTF — name your conversational domain and a handful of expected proper nouns or technical terms.

Or in Python (the demo at `examples/live_mic_to_text.py`) :

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_microphone(),
        config=voh.PipelineConfig(
            asr={
                "language": "auto",  # discovered from the audio — no default
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

## 2. Replay a WAV at burst speed (offline batch)

```bash
vocal-helper file ./conversation.wav --no-real-time --jsonl > out.jsonl
```

`--jsonl` emits one event per line, ideal for piping into `jq` or downstream stores.

---

## 3. Two-speaker meeting with rolling Gemma summary

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_wav_file("./meeting.wav"),
        config=voh.PipelineConfig(
            asr={"language": "auto"},  # discovered from the audio — no default
            llm={
                "model": "gemma4:e4b",
                "recent_window_s": 60.0,   # 60 s verbatim window
                "flush_every_n": 5,        # summarise every 5 evicted utterances
            },
        ),
    )
    async for ev in p.run():
        if "summary" in ev:
            print("\n--- rolling summary ---")
            print(ev["summary"])

asyncio.run(main())
```

---

## 4. Custom WebSocket relay via the subscriber API

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

## 5. NeMo TitaNet diarization on a noisy mix

```python
import asyncio, vocal_helper as voh

async def main():
    p = voh.Pipeline(
        source=lambda: voh.sources.from_wav_file("./noisy.wav"),
        config=voh.PipelineConfig(
            diar={
                "backend": "nemo",
                "join_threshold": 0.35,   # TitaNet's distribution is wider
            },
            asr={"language": "auto"},     # discovered from the audio — no default
        ),
    )
    async for ev in p.run():
        if "text" in ev:
            print(ev)

asyncio.run(main())
```

---

## 6. Synchronous one-shot transcription (no pipeline)

For when you have a single PCM buffer and just want text back. The language is
**discovered** from the audio by default (`language="auto"`) — no default, no
pairing :

```python
import numpy as np, vocal_helper as voh

pcm = np.zeros(16_000 * 5, dtype=np.float32)  # five seconds of silence
text = voh.transcribe_pcm(pcm, sr=16_000)     # language auto-discovered
print(text)
```

Need the language whisper actually detected? Use the sibling helper that
returns it alongside the text :

```python
from vocal_helper.asr import transcribe_pcm_with_language

text, language = transcribe_pcm_with_language(pcm, sr=16_000)
print(language, text)   # e.g. 'fr Bonjour tout le monde'
```
