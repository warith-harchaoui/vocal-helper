# Examples

Hand-picked recipes for the common shapes. All examples assume :

```bash
pip install 'vocal-helper[all]'
export HF_TOKEN=hf_yourtoken    # pyannote backend
ollama serve                    # LLM analyst
```

---

## 1. Live mic → terminal transcript

```bash
vocal-helper mic --language fr
```

Or in Python (the demo at `examples/live_mic_to_text.py`) :

```python
import asyncio, vocal_helper as vh

async def main():
    p = vh.Pipeline(
        source=lambda: vh.sources.from_microphone(),
        config=vh.PipelineConfig(
            asr={"language": "fr"},
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
import asyncio, vocal_helper as vh

async def main():
    p = vh.Pipeline(
        source=lambda: vh.sources.from_wav_file("./meeting.wav"),
        config=vh.PipelineConfig(
            asr={"language": "en"},
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
import asyncio, json, vocal_helper as vh
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
    p = vh.Pipeline(source=lambda: vh.sources.from_microphone())
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
import asyncio, vocal_helper as vh

async def main():
    p = vh.Pipeline(
        source=lambda: vh.sources.from_wav_file("./noisy.wav"),
        config=vh.PipelineConfig(
            diar={
                "backend": "nemo",
                "join_threshold": 0.35,   # TitaNet's distribution is wider
            },
            asr={"language": "en"},
        ),
    )
    async for ev in p.run():
        if "text" in ev:
            print(ev)

asyncio.run(main())
```

---

## 6. Synchronous one-shot transcription (no pipeline)

For when you have a single PCM buffer and just want text back :

```python
import numpy as np, vocal_helper as vh

pcm = np.zeros(16_000 * 5, dtype=np.float32)  # five seconds of silence
text = vh.transcribe_pcm(pcm, sr=16_000, language="en")
print(text)
```
