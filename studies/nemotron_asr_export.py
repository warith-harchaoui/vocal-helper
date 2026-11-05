"""Exporter l'encodeur ASR de Nemotron 3.5 Transcribe et sa tête CTC en ONNX.

Pourquoi cet export
-------------------
Nemotron 3.5 Transcribe est un LLM de parole de 30 milliards de paramètres,
officiellement réservé aux GPU NVIDIA. Sa partie audio, elle, est un encodeur
transformer de 600 millions de paramètres suivi d'une tête CTC qui transcrit
seule, sans le LLM, avec des horodatages à 40 ms. C'est elle qui nous
intéresse : un graphe ONNX tourne partout où tourne onnxruntime — Linux,
Windows, macOS, CPU, CUDA, CoreML — et se charge aussi depuis Rust.

Aucun export ONNX de cet encodeur n'est publié. Celui-ci part des poids
extraits du checkpoint officiel (``perception.*`` et ``ctc_timestamp.*``,
1,67 Go) et des modules NeMo, l'attention FlexAttention remplacée par une
attention matricielle à biais additif, exacte en mode ``full``.

Le graphe
---------
Entrées : ``mel`` ``(1, 128, T)`` float32, log-mel normalisé par bande, ``T``
multiple de 8 ; ``enc_len`` ``(1,)`` int64, le nombre de trames de 80 ms
valides. Sorties, une trame toutes les 40 ms : ``token_ids`` (argmax),
``token_logprob`` et ``blank_logprob``. Le dernier donne une détection de
parole gratuite. L'axe ``T`` est dynamique.

Les métadonnées (préfixe ``vocal_helper.asr.``) portent tout ce dont l'hôte a
besoin : le banc de filtres et la fenêtre du front-end, le vocabulaire, l'id
du blanc. Le graphe se suffit ainsi à lui-même, comme celui du diariseur.

Usage
-----
    <venv d'étude>/bin/python studies/nemotron_asr_export.py \\
        --lite "$VH_NEMOTRON_STUDY/lite" \\
        --out "$VH_NEMOTRON_STUDY/onnx"
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import time

import click
import numpy as np

META_PREFIX = "vocal_helper.asr."
CONTRACT = "nemotron-asr-ctc-1"
STACK = 8
N_MELS = 128


def build(lite_dir: pathlib.Path):
    """Le modèle PyTorch de référence et l'enveloppe à exporter."""
    import torch
    from torch import nn

    sys.path.insert(0, str(lite_dir / "scripts"))
    from lite_model import LiteASR

    lite = LiteASR(device="cpu")

    class LiteCTC(nn.Module):
        """Encodeur puis tête CTC, l'empilement de trames réécrit en ``reshape``."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = lite.encoder
            self.proj = lite.encoder.pre_encode.proj
            self.ctc = lite.ctc

        def forward(self, mel, enc_len):
            x = mel.transpose(1, 2).reshape(1, -1, N_MELS * STACK)
            states, _ = self.encoder(
                audio_signal=self.proj(x), length=enc_len, bypass_pre_encode=True
            )
            ctc = self.ctc
            up = ctc.upsample(states)
            up_len = enc_len * ctc.upsampling_factor
            bridged, _ = ctc.transformer(
                audio_signal=up.transpose(1, 2), length=up_len, bypass_pre_encode=True
            )
            h = up + ctc.residual_scale.to(up.dtype) * bridged
            logp = torch.log_softmax(ctc.decoder_layers(h).transpose(1, 2).float(), dim=-1)
            best, ids = logp.max(dim=-1)
            return ids.to(torch.int32), best, logp[..., -1]

    return lite, LiteCTC().eval()


def host_mel(lite, audio: np.ndarray):
    """Le front-end de référence (NeMo), normalisé, complété à un multiple de 8."""
    import torch
    from nemo.collections.asr.parts.preprocessing.features import normalize_batch

    sig = torch.from_numpy(audio)[None]
    mel, mel_len = lite.preprocessor(input_signal=sig, length=torch.tensor([sig.shape[1]]))
    mel, _, _ = normalize_batch(mel, mel_len, normalize_type=lite.normalize_type)
    n = int(mel_len[0])
    padded = -(-n // STACK) * STACK
    out = torch.zeros(1, N_MELS, padded)
    out[:, :, :n] = mel[:, :, :n]
    return out, torch.tensor([padded // STACK], dtype=torch.int64), n


def metadata(lite) -> dict[str, str]:
    """Ce que l'hôte doit savoir, rangé dans le graphe."""
    feat = lite.preprocessor.featurizer
    fb = feat.fb[0].float().numpy().astype("<f4")
    window = feat.window.float().numpy().astype("<f4")
    vocab = list(lite.vocab)
    vocab_json = json.dumps(vocab, ensure_ascii=False)
    return {
        "contract": CONTRACT,
        "sample_rate": "16000",
        "n_fft": "512",
        "hop_length": "160",
        "win_length": str(window.size),
        "preemphasis": "0.97",
        "log_guard": repr(2.0**-24),
        "n_mels": str(N_MELS),
        "normalize": lite.normalize_type,
        "stack": str(STACK),
        "frame_s": "0.04",
        "chunk_s": str(lite.chunk_seconds),
        "blank_id": str(lite.blank_id),
        "vocab": vocab_json,
        "vocab_sha256": hashlib.sha256(vocab_json.encode()).hexdigest(),
        "mel_fb": base64.b64encode(fb.tobytes()).decode(),
        "mel_window": base64.b64encode(window.tobytes()).decode(),
    }


def clips() -> list[tuple[str, np.ndarray]]:
    """Trois extraits réels de longueurs différentes, FR et EN, pour la parité."""
    import soundfile as sf

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _paths import data_root

    data = data_root()  # $VH_STUDY_DATA_ROOT
    out = []
    cv = sorted((data / "common-voice-subset/fr").rglob("*.mp3"))[0]
    audio, sr = sf.read(cv, dtype="float32")
    if sr != 16000:  # Common Voice est en 48 kHz
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    out.append((cv.name, audio))
    mix, sr = sf.read(data / "ami/dev-slice/ES2011a/mix.wav", dtype="float32")
    assert sr == 16000
    out.append(("ES2011a[60:77s]", mix[60 * sr : 77 * sr]))
    out.append(("ES2011a[120:150s]", mix[120 * sr : 150 * sr]))
    return out


def parity(lite, session, clips_: list) -> list[dict]:
    """Accord d'argmax ONNX contre PyTorch sur les trames valides."""
    import torch

    rows = []
    for name, audio in clips_:
        mel, enc_len, _ = host_mel(lite, audio)
        t0 = time.perf_counter()
        ids, best, blank = session.run(None, {"mel": mel.numpy(), "enc_len": enc_len.numpy()})
        dt = time.perf_counter() - t0
        with torch.no_grad():
            ref = lite.log_probs(torch.from_numpy(audio))
        n = min(ref.shape[0], ids.shape[1])
        agree = float((ids[0, :n] == ref.argmax(-1)[:n].numpy()).mean())
        ref_best = ref.max(-1).values[:n].numpy()
        rows.append(
            {
                "clip": name,
                "seconds": round(len(audio) / 16000, 2),
                "frames": n,
                "argmax_agreement": round(agree, 5),
                "max_abs_logprob_err": float(np.abs(best[0, :n] - ref_best).max()),
                "rtf_cpu": round(dt / (len(audio) / 16000), 4),
                "text": lite.tokenizer.ids_to_text(lite.greedy(torch.from_numpy(ref.numpy()))),
            }
        )
        print(rows[-1])
    return rows


@click.command()
@click.option("--lite", "lite_dir", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--out", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--opset", default=17, show_default=True)
@click.option("--int8/--no-int8", default=True, show_default=True)
def main(lite_dir: pathlib.Path, out: pathlib.Path, opset: int, int8: bool) -> None:
    """Exporter, vérifier la parité, puis quantifier en int8 dynamique."""
    import onnx
    import onnxruntime as ort
    import torch

    out.mkdir(parents=True, exist_ok=True)
    lite, model = build(lite_dir)
    samples = clips()
    mel, enc_len, _ = host_mel(lite, samples[1][1])

    fp32 = out / "nemotron-asr-ctc.fp32.onnx"
    if not fp32.exists():
        t0 = time.perf_counter()
        # Au-delà de 2 Go, l'export écrit un fichier par tenseur à côté du
        # graphe : on le fait dans un dossier jetable, puis on regroupe.
        scratch = pathlib.Path(tempfile.mkdtemp(dir=out))
        with torch.no_grad():
            torch.onnx.export(
                model,
                (mel, enc_len),
                str(scratch / fp32.name),
                input_names=["mel", "enc_len"],
                output_names=["token_ids", "token_logprob", "blank_logprob"],
                dynamic_axes={
                    "mel": {2: "T"},
                    "token_ids": {1: "F"},
                    "token_logprob": {1: "F"},
                    "blank_logprob": {1: "F"},
                },
                opset_version=opset,
                dynamo=False,
            )
        graph = onnx.load(str(scratch / fp32.name), load_external_data=True)
        for key, value in metadata(lite).items():
            graph.metadata_props.add(key=META_PREFIX + key, value=value)
        shutil.rmtree(scratch)
        onnx.save(
            graph,
            str(fp32),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=fp32.name + ".data",
        )
        print(f"export fp32 : {time.perf_counter() - t0:.0f} s")

    report: dict = {"opset": opset}
    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    report["fp32"] = parity(lite, session, samples)

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q8 = out / "nemotron-asr-ctc.int8.onnx"
        if not q8.exists():
            quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8, per_channel=True)
        session = ort.InferenceSession(str(q8), providers=["CPUExecutionProvider"])
        report["int8"] = parity(lite, session, samples)

    (out / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "opset"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
