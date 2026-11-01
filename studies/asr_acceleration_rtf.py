"""Mesurer le RTF réel des moteurs de transcription locaux disponibles ici.

RTF = temps de calcul / durée audio. Sous 1, le moteur va plus vite que le temps réel ;
au-dessus, un direct décroche définitivement. Un coup de chauffe est jeté, puis on garde
la médiane de deux passages chronométrés.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

AUDIO = sys.argv[1]
DURATION = 60.0
MODELS = "/Users/warithharchaoui/notes-helper/apps/Models"
N = 2


def timed(fn, label):
    try:
        fn()  # chauffe, jetée
    except Exception as exc:
        return {"engine": label, "error": f"{type(exc).__name__}: {exc}"[:160]}
    runs = []
    for _ in range(N):
        t = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            return {"engine": label, "error": f"{type(exc).__name__}: {exc}"[:160]}
        runs.append(time.perf_counter() - t)
    wall = statistics.median(runs)
    return {"engine": label, "wall_s": round(wall, 2), "rtf": round(wall / DURATION, 4)}


results = []

# 1 & 2. whisper.cpp via pywhispercpp — GPU (Metal) puis CPU seul
for use_gpu in (True, False):
    label = f"whisper.cpp large-v3-turbo-q5 ({'Metal' if use_gpu else 'CPU seul'})"

    def run(use_gpu=use_gpu):
        from pywhispercpp.model import Model

        m = Model(
            f"{MODELS}/ggml-large-v3-turbo-q5_0.bin",
            n_threads=6, print_realtime=False, print_progress=False,
            token_timestamps=True, language="fr",
            **({} if use_gpu else {"context_params": {"use_gpu": False}}),
        )
        m.transcribe(AUDIO)

    results.append(timed(run, label))
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)

# 3. mlx-whisper (GPU Apple via MLX)
def run_mlx():
    import mlx_whisper

    mlx_whisper.transcribe(
        AUDIO, path_or_hf_repo=f"{MODELS}/mlx-whisper-large-v3-turbo",
        language="fr", word_timestamps=True, verbose=None,
    )

results.append(timed(run_mlx, "mlx-whisper large-v3-turbo (MLX / GPU)"))
print(json.dumps(results[-1], ensure_ascii=False), flush=True)

# 4. faster-whisper (CTranslate2, CPU int8 sur Apple Silicon)
def run_fw():
    from faster_whisper import WhisperModel

    m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=6)
    list(m.transcribe(AUDIO, language="fr", word_timestamps=True)[0])

results.append(timed(run_fw, "faster-whisper large-v3-turbo (CT2 int8 CPU)"))
print(json.dumps(results[-1], ensure_ascii=False), flush=True)

print("\n=== RÉSULTATS ===")
for r in results:
    if "error" in r:
        print(f"  {r['engine']:48} → ERREUR {r['error']}")
    else:
        print(f"  {r['engine']:48} → {r['wall_s']:6.2f} s  RTF {r['rtf']:.3f}")
with open(os.path.join(os.path.dirname(AUDIO), "rtf_results.json"), "w") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)
