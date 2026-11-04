"""Transcription par l'encodeur de Nemotron 3.5 Transcribe, sans torch : onnxruntime et numpy.

Pourquoi ce module
------------------
Nemotron 3.5 Transcribe est un LLM de parole de 30 milliards de paramètres,
réservé aux GPU NVIDIA. Sous le LLM, son encodeur audio de 600 millions de
paramètres porte une tête CTC qui transcrit seule, dans une quarantaine de
langues, avec un horodatage par trame de 40 ms. C'est cette partie, exportée
en ONNX par ``studies/nemotron_asr_export.py``, que ce module fait tourner :
le même fichier marche sous Linux, Windows et macOS, sur CPU comme sur CUDA, et
se charge depuis Rust par la caisse ``ort``.

Ce que l'hôte fait autour du graphe
-----------------------------------
Le graphe prend un log-mel normalisé et rend, toutes les 40 ms, le jeton le
plus probable, sa log-probabilité et celle du blanc. Le reste est ici, en numpy
pur, donc vérifiable sans modèle :

- le front-end de NeMo, avec la fenêtre et le banc de filtres **du point de
  contrôle**, que le graphe porte dans ses métadonnées ;
- la normalisation par bande, sur l'enregistrement entier ;
- le découpage en fenêtres de 30 s, la longueur que l'encodeur a apprise ;
- le décodage CTC glouton, puis le regroupement des pièces SentencePiece en
  mots horodatés, sans dépendre de ``sentencepiece``.

Le vocabulaire contient aussi des balises : changement de locuteur
(``<spk:N>``), événement (``<ev:laughter>``…), début et fin de style
(``<sty:stt:…>``). Elles sont rendues à part, jamais mêlées aux mots.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vocal_helper.nemotron import _graph_metadata, _preferred_providers, _requested_providers

__all__ = [
    "AsrSettings",
    "CtcToken",
    "Word",
    "Transcript",
    "log_mel",
    "normalize_per_feature",
    "greedy_ctc",
    "tokens_to_words",
    "NemotronOnnxAsr",
    "NemotronAsrStage",
]

#: Préfixe des métadonnées du graphe ASR.
META_PREFIX = "vocal_helper.asr."

#: Le contrat que cet hôte sait lire.
CONTRACT = "nemotron-asr-ctc-1"

#: La variable d'environnement qui désigne le graphe.
GRAPH_ENV = "VH_NEMOTRON_ASR_GRAPH"

#: Le séparateur de mots de SentencePiece.
WORD_START = "▁"

#: Le plancher d'écart-type de NeMo (``features.CONSTANT``).
STD_FLOOR = 1e-5


@dataclass(frozen=True)
class AsrSettings:
    """Le front-end et le vocabulaire, tels que le graphe les porte.

    Attributes
    ----------
    fb : NDArray[np.float32]
        Banc de filtres mel ``(n_mels, n_fft // 2 + 1)`` du point de contrôle.
    window : NDArray[np.float32]
        Fenêtre d'analyse ``(win_length,)``, centrée dans la transformée.
    vocab : tuple of str
        Les pièces SentencePiece, le blanc exclu.
    blank_id : int
        L'indice du blanc CTC, juste après le vocabulaire.
    """

    fb: NDArray[np.float32]
    window: NDArray[np.float32]
    vocab: tuple[str, ...]
    blank_id: int
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    preemphasis: float = 0.97
    log_guard: float = 2.0**-24
    stack: int = 8
    frame_s: float = 0.04
    chunk_s: float = 30.0

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> AsrSettings:
        """Lire les réglages des métadonnées du graphe.

        Raises
        ------
        ValueError
            Si le graphe ne suit pas le contrat attendu.
        """

        def get(key: str) -> str:
            value = metadata.get(META_PREFIX + key)
            if value is None:
                raise ValueError(f"métadonnée {META_PREFIX + key} absente du graphe")
            return value

        if get("contract") != CONTRACT:
            raise ValueError(f"contrat {get('contract')!r}, cet hôte lit {CONTRACT!r}")
        n_mels, n_fft = int(get("n_mels")), int(get("n_fft"))
        fb = np.frombuffer(base64.b64decode(get("mel_fb")), dtype="<f4").astype(np.float32)
        window = np.frombuffer(base64.b64decode(get("mel_window")), dtype="<f4").astype(np.float32)
        return cls(
            fb=fb.reshape(n_mels, n_fft // 2 + 1),
            window=window,
            vocab=tuple(json.loads(get("vocab"))),
            blank_id=int(get("blank_id")),
            sample_rate=int(get("sample_rate")),
            n_fft=n_fft,
            hop_length=int(get("hop_length")),
            preemphasis=float(get("preemphasis")),
            log_guard=float(get("log_guard")),
            stack=int(get("stack")),
            frame_s=float(get("frame_s")),
            chunk_s=float(get("chunk_s")),
        )


@dataclass(frozen=True)
class CtcToken:
    """Un jeton émis par le décodage glouton, avec ses trames."""

    id: int
    first: int
    last: int
    logprob: float


@dataclass(frozen=True)
class Word:
    """Un mot horodaté ; ``confidence`` est la moyenne des probabilités de ses pièces."""

    text: str
    start_s: float
    end_s: float
    confidence: float


@dataclass
class Transcript:
    """Les mots, et à part les balises que la tête CTC émet."""

    words: list[Word] = field(default_factory=list)
    tags: list[tuple[float, str]] = field(default_factory=list)
    speech: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))

    @property
    def text(self) -> str:
        """Le texte, mots séparés par une espace."""
        return " ".join(w.text for w in self.words)


def log_mel(audio: NDArray[np.float32], settings: AsrSettings) -> tuple[NDArray[np.float32], int]:
    """Le log-mel de NeMo, ``(n_mels, T)``, et le nombre de trames valides.

    La recette de ``FilterbankFeatures`` : pré-accentuation, transformée centrée
    sur un remplissage **constant** (et non par réflexion, à la différence du
    front-end du diariseur), fenêtre de 400 points centrée dans 512, puissance,
    banc de filtres, ``log(x + 2**-24)``. Les trames au-delà de
    ``len(audio) // hop`` sont mises à zéro, comme NeMo le fait.

    Examples
    --------
    >>> import numpy as np
    >>> s = AsrSettings(np.ones((128, 257), np.float32), np.hanning(400).astype(np.float32), (), 0)
    >>> mel, valid = log_mel(np.zeros(16000, np.float32), s)
    >>> mel.shape, valid
    ((128, 101), 100)
    """
    x = np.asarray(audio, dtype=np.float32).ravel()
    valid = x.size // settings.hop_length
    if x.size == 0:
        return np.zeros((settings.fb.shape[0], 1), np.float32), 0
    x = np.concatenate([x[:1], x[1:] - np.float32(settings.preemphasis) * x[:-1]])
    n_fft, hop = settings.n_fft, settings.hop_length
    window = np.zeros(n_fft, np.float32)
    offset = (n_fft - settings.window.size) // 2
    window[offset : offset + settings.window.size] = settings.window
    padded = np.pad(x, (n_fft // 2, n_fft // 2), mode="constant")
    n_frames = 1 + (padded.size - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spectrum = np.fft.rfft(padded[index] * window, axis=1)
    power = (spectrum.real.astype(np.float64) ** 2 + spectrum.imag.astype(np.float64) ** 2).astype(
        np.float32
    )
    mel = np.log(settings.fb @ power.T + np.float32(settings.log_guard)).astype(np.float32)
    mel[:, valid:] = 0.0
    return mel, valid


def normalize_per_feature(mel: NDArray[np.float32], valid: int) -> NDArray[np.float32]:
    """Centrer et réduire chaque bande sur les trames valides, comme NeMo.

    Écart-type sans biais (division par ``n - 1``), plancher de ``1e-5``
    ajouté après, trames invalides remises à zéro.

    Examples
    --------
    >>> import numpy as np
    >>> out = normalize_per_feature(np.array([[1.0, 3.0, 9.0]], np.float32), 2)
    >>> [round(float(v), 4) for v in out[0]]
    [-0.7071, 0.7071, 0.0]
    """
    x = mel.astype(np.float64)
    core = x[:, :valid]
    mean = core.mean(axis=1, keepdims=True) if valid else np.zeros((x.shape[0], 1))
    std = (
        np.sqrt(((core - mean) ** 2).sum(axis=1, keepdims=True) / (valid - 1)) if valid > 1 else 0.0
    )
    out = (x - mean) / (std + STD_FLOOR)
    out[:, valid:] = 0.0
    return out.astype(np.float32)


def greedy_ctc(
    ids: NDArray[np.integer], logprob: NDArray[np.floating], blank_id: int
) -> list[CtcToken]:
    """Fusionner les répétitions et retirer les blancs, en gardant les trames.

    Examples
    --------
    >>> import numpy as np
    >>> toks = greedy_ctc(np.array([9, 5, 5, 9, 5, 7]), np.zeros(6), blank_id=9)
    >>> [(t.id, t.first, t.last) for t in toks]
    [(5, 1, 2), (5, 4, 4), (7, 5, 5)]
    """
    tokens: list[CtcToken] = []
    prev = blank_id
    for frame, token in enumerate(np.asarray(ids).tolist()):
        if token != blank_id:
            if token == prev and tokens:
                last = tokens[-1]
                tokens[-1] = CtcToken(
                    last.id, last.first, frame, max(last.logprob, float(logprob[frame]))
                )
            else:
                tokens.append(CtcToken(int(token), frame, frame, float(logprob[frame])))
        prev = token
    return tokens


def _is_tag(piece: str) -> bool:
    return piece.startswith("<") and piece.endswith(">") and len(piece) > 2


def tokens_to_words(
    tokens: list[CtcToken], vocab: tuple[str, ...], frame_s: float
) -> tuple[list[Word], list[tuple[float, str]]]:
    """Regrouper les pièces en mots horodatés, et sortir les balises à part.

    Une pièce qui commence par ``▁`` ouvre un mot ; les autres le prolongent.
    Une balise (``<spk:1>``, ``<ev:laughter>``…) clôt le mot en cours.

    Examples
    --------
    >>> vocab = ("<spk:1>", "▁bon", "jour", "▁à", "▁tous")
    >>> toks = [CtcToken(1, 0, 1, 0.0), CtcToken(2, 3, 3, 0.0), CtcToken(0, 5, 5, 0.0),
    ...         CtcToken(3, 6, 6, 0.0), CtcToken(4, 8, 9, 0.0)]
    >>> words, tags = tokens_to_words(toks, vocab, 0.04)
    >>> [(w.text, w.start_s, w.end_s) for w in words]
    [('bonjour', 0.0, 0.16), ('à', 0.24, 0.28), ('tous', 0.32, 0.4)]
    >>> tags
    [(0.2, '<spk:1>')]
    """
    words: list[Word] = []
    tags: list[tuple[float, str]] = []
    text, first, last, probs = "", 0, 0, []

    def close() -> None:
        if text:
            words.append(
                Word(
                    text,
                    round(first * frame_s, 3),
                    round((last + 1) * frame_s, 3),
                    float(np.mean(np.exp(probs))),
                )
            )

    for token in tokens:
        piece = vocab[token.id] if 0 <= token.id < len(vocab) else ""
        if _is_tag(piece):
            close()
            text, probs = "", []
            if piece != "<unk>":
                tags.append((round(token.first * frame_s, 3), piece))
            continue
        if piece.startswith(WORD_START) or not text:
            close()
            text, first, probs = piece.lstrip(WORD_START), token.first, []
            if not text:  # un ``▁`` seul n'ouvre qu'un mot vide
                continue
        else:
            text += piece
        last = token.last
        probs.append(token.logprob)
    close()
    return words, tags


def _mel_per_frame(s: AsrSettings) -> int:
    return int(round(s.frame_s * s.sample_rate / s.hop_length))


def _concat(
    parts: list[tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]],
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    if not parts:
        empty = np.zeros(0, np.float32)
        return np.zeros(0, np.int32), empty, empty
    ids, best, blank = zip(*parts, strict=True)
    return np.concatenate(ids), np.concatenate(best), np.concatenate(blank)


class NemotronOnnxAsr:
    """Transcrire un fichier avec le graphe ONNX de l'encodeur Nemotron.

    Parameters
    ----------
    graph : str or Path, optional
        Le graphe. Par défaut ``$VH_NEMOTRON_ASR_GRAPH``.
    providers : list of str, optional
        Les moteurs d'exécution ; même ordre d'autorité que le diariseur :
        l'argument, puis ``$VOCAL_HELPER_PROVIDER``, puis l'ordre mesuré.
    threads : int, optional
        Fils du moteur CPU ; ``None`` laisse onnxruntime choisir.
    """

    def __init__(
        self,
        graph: str | Path | None = None,
        *,
        providers: list[str] | None = None,
        threads: int | None = None,
    ) -> None:
        """Repérer le graphe ; la session se crée au premier appel."""
        self._graph = Path(graph) if graph is not None else None
        self._providers = providers
        self._threads = threads
        self._settings: AsrSettings | None = None
        self._session: Any = None

    def locate(self) -> Path:
        """Le chemin du graphe.

        Raises
        ------
        FileNotFoundError
            Si aucun graphe n'est désigné ou s'il n'existe pas.
        """
        path = self._graph or (Path(os.environ[GRAPH_ENV]) if os.environ.get(GRAPH_ENV) else None)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"graphe Nemotron ASR introuvable ({path or 'non désigné'}) ; "
                f"le passer en argument ou par ${GRAPH_ENV}"
            )
        return path

    @property
    def settings(self) -> AsrSettings:
        """Les réglages du graphe, lus sans charger ses poids."""
        if self._settings is None:
            self._settings = AsrSettings.from_metadata(_graph_metadata(self.locate()))
        return self._settings

    def _load(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            if self._threads:
                options.intra_op_num_threads = int(self._threads)
            providers = self._providers or _requested_providers() or _preferred_providers()
            self._session = ort.InferenceSession(str(self.locate()), options, providers=providers)
        return self._session

    def frames(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
        """Jetons, log-probabilités et log-probabilités du blanc, toutes les 40 ms.

        Normalisation sur l'enregistrement entier, puis fenêtres indépendantes
        de ``chunk_s`` secondes : l'encodeur de référence découpe ainsi.

        Raises
        ------
        ValueError
            Si la fréquence n'est pas celle du modèle.
        """
        s = self.settings
        if sample_rate != s.sample_rate:
            raise ValueError(f"le modèle veut {s.sample_rate} Hz, reçu {sample_rate} Hz")
        mel, valid = log_mel(audio, s)
        mel = normalize_per_feature(mel, valid)
        chunk = int(round(s.chunk_s * s.sample_rate / s.hop_length))
        parts = [
            self._run(mel[:, start : min(start + chunk, valid)]) for start in range(0, valid, chunk)
        ]
        return _concat(parts)

    def _run(
        self, mel: NDArray[np.float32]
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
        """Une fenêtre de log-mel normalisé ``(n_mels, n)`` → ses trames de 40 ms."""
        s = self.settings
        n = mel.shape[1]
        padded = -(-n // s.stack) * s.stack
        window = np.zeros((1, mel.shape[0], padded), np.float32)
        window[0, :, :n] = mel
        ids, best, blank = self._load().run(
            None, {"mel": window, "enc_len": np.array([padded // s.stack], np.int64)}
        )
        keep = -(-n // _mel_per_frame(s))
        return ids[0, :keep], best[0, :keep], blank[0, :keep]

    def streaming_frames(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        *,
        hop_s: float = 2.0,
        left_s: float = 8.0,
        right_s: float = 0.0,
        normalize: str = "running",
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Les trames d'un flux simulé, et le retard de chacune.

        Toutes les ``hop_s`` secondes, on transcrit les ``left_s + hop_s +
        right_s`` dernières secondes et on **arrête** les trames de
        ``[t - right_s - hop_s, t - right_s)``. Une trame attend donc entre
        ``right_s`` et ``right_s + hop_s`` avant d'être écrite, calcul en sus.
        La concaténation des trames arrêtées se décode comme un fichier.

        Un flux ne connaît pas la suite, donc pas la moyenne de l'enregistrement
        que la normalisation hors ligne emploie. ``normalize="running"`` prend
        la moyenne et l'écart-type par bande de **tout l'audio déjà vu**, ce
        qui reste causal ; ``"window"`` les prend sur la fenêtre seule.

        Returns
        -------
        tuple
            ``ids``, ``logprob``, ``blank_logprob`` et ``delay_s``, le retard
            de chaque trame entre la fin de son audio et son écriture, calcul
            compris.

        Raises
        ------
        ValueError
            Si la fréquence n'est pas celle du modèle, si la fenêtre dépasse
            celle que l'encodeur a apprise, ou si ``normalize`` est inconnu.
        """
        import time

        if normalize not in ("running", "window"):
            raise ValueError(f"normalisation {normalize!r} inconnue : 'running' ou 'window'")
        s = self.settings
        if sample_rate != s.sample_rate:
            raise ValueError(f"le modèle veut {s.sample_rate} Hz, reçu {sample_rate} Hz")
        if left_s + hop_s + right_s > s.chunk_s:
            raise ValueError(f"fenêtre de {left_s + hop_s + right_s} s pour {s.chunk_s} s apprises")
        per_frame = int(round(s.frame_s * s.sample_rate))  # échantillons par trame CTC
        hop = max(1, int(round(hop_s / s.frame_s))) * per_frame
        left = int(round(left_s / s.frame_s)) * per_frame
        right = int(round(right_s / s.frame_s)) * per_frame
        parts, delays = [], []
        n_mels = s.fb.shape[0]
        seen_sum, seen_sq, seen = np.zeros((n_mels, 1)), np.zeros((n_mels, 1)), 0
        committed = end = 0  # ``committed`` reste un multiple de ``per_frame``
        total = len(audio)
        while committed < total:
            previous, end = end, min(end + hop, total)
            target = total if end == total else max(committed, end - right)
            if target == committed:  # pas encore assez d'audio au-delà de l'anticipation
                continue
            start = max(0, committed - left)
            t0 = time.perf_counter()
            mel, valid = log_mel(audio[start:end], s)
            if normalize == "window":
                mel = normalize_per_feature(mel, valid)
            else:
                fresh = mel[:, (previous - start) // s.hop_length : valid].astype(np.float64)
                seen_sum += fresh.sum(axis=1, keepdims=True)
                seen_sq += (fresh**2).sum(axis=1, keepdims=True)
                seen += fresh.shape[1]
                mean = seen_sum / max(seen, 1)
                var = (seen_sq - seen * mean**2) / max(seen - 1, 1)
                std = np.sqrt(np.clip(var, 0.0, None))
                mel = ((mel - mean) / (std + STD_FLOOR)).astype(np.float32)
            ids, best, blank = self._run(mel[:, :valid])
            elapsed = time.perf_counter() - t0
            lo = (committed - start) // per_frame
            hi = lo + -(-(target - committed) // per_frame)
            parts.append((ids[lo:hi], best[lo:hi], blank[lo:hi]))
            frame_end = np.arange(committed // per_frame, committed // per_frame + hi - lo) + 1
            delays.append((end / s.sample_rate - frame_end * s.frame_s).clip(0) + elapsed)
            committed = target
        ids, best, blank = _concat(parts)
        delay = np.concatenate(delays).astype(np.float32) if delays else np.zeros(0, np.float32)
        return ids, best, blank, delay

    def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> Transcript:
        """Les mots horodatés, les balises, et la probabilité de parole par trame."""
        s = self.settings
        ids, best, blank = self.frames(audio, sample_rate)
        words, tags = tokens_to_words(greedy_ctc(ids, best, s.blank_id), s.vocab, s.frame_s)
        speech = (1.0 - np.exp(blank)).astype(np.float32)
        return Transcript(words=words, tags=tags, speech=speech)


class NemotronAsrStage:
    """L'étage de transcription de la pipeline, par le graphe Nemotron.

    Même contrat que :class:`vocal_helper.asr.WhisperStage` : il consomme des
    :class:`~vocal_helper.types.DiarizedSegment` et rend une
    :class:`~vocal_helper.types.Utterance` par segment, dans l'ordre, avec le
    locuteur du diariseur. On le choisit par ``asr={"engine": "nemotron"}``.

    Les segments consécutifs sont mis bout à bout dans des fenêtres d'au plus
    ``max_chunk_s`` secondes, un appel au graphe par fenêtre ; chaque mot
    revient au segment qui contient son milieu. Hors ligne c'est le chemin à
    plein régime ; en direct (``batch=False``) chaque segment part seul, dès
    qu'il arrive.

    Le modèle ne dit pas la langue qu'il entend : ``language`` reste ``None``.

    Parameters
    ----------
    graph : str or Path, optional
        Le graphe ; par défaut ``$VH_NEMOTRON_ASR_GRAPH``.
    batch : bool
        Regrouper les segments (hors ligne) ou les traiter un par un (direct).
    max_chunk_s : float
        Longueur maximale d'une fenêtre ; 30 s, celle que l'encodeur a apprise.
    providers, threads
        Comme :class:`NemotronOnnxAsr`.
    """

    def __init__(
        self,
        graph: str | Path | None = None,
        *,
        batch: bool = False,
        max_chunk_s: float = 30.0,
        providers: list[str] | None = None,
        threads: int | None = None,
        **_ignored: Any,
    ) -> None:
        """Préparer l'étage ; les options propres à whisper sont ignorées."""
        self.batch = batch
        self.max_chunk_s = float(max_chunk_s)
        self._asr = NemotronOnnxAsr(graph, providers=providers, threads=threads)

    def transcribe_chunk(self, chunk: list[dict]) -> list[dict]:
        """Transcrire des segments mis bout à bout ; une énonciation par segment."""
        from vocal_helper.asr import _BATCH_PAD_S, _assign_window

        sr = chunk[0]["sample_rate"]
        pad = np.zeros(int(_BATCH_PAD_S * sr), dtype=np.float32)
        pieces: list[NDArray[np.float32]] = []
        windows: list[tuple[float, float]] = []
        cursor = 0.0
        for i, seg in enumerate(chunk):
            if i:
                pieces.append(pad)
                cursor += pad.shape[0] / sr
            duration = seg["pcm"].shape[0] / sr
            windows.append((cursor, cursor + duration))
            pieces.append(np.asarray(seg["pcm"], dtype=np.float32))
            cursor += duration
        transcript = self._asr.transcribe(np.concatenate(pieces), sr)

        words: list[list[tuple[float, float, str]]] = [[] for _ in chunk]
        for word in transcript.words:
            index = _assign_window(windows, (word.start_s + word.end_s) / 2.0)
            local, seg = windows[index][0], chunk[index]
            words[index].append(
                (
                    seg["t0"] + max(0.0, word.start_s - local),
                    seg["t0"] + max(0.0, word.end_s - local),
                    word.text,
                )
            )
        return [
            {
                "t0": seg["t0"],
                "t1": seg["t1"],
                "speaker": seg["speaker"],
                "text": " ".join(w[2] for w in ws),
                "words": ws,
                "language": None,
            }
            for seg, ws in zip(chunk, words, strict=True)
        ]

    async def run(self, inbox: Any, outbox: Any) -> None:
        """Consommer les segments de ``inbox``, pousser les énonciations dans ``outbox``."""
        import asyncio

        from vocal_helper.asr import _pack_segments

        if not self.batch:
            while (item := await inbox.get()) is not None:
                for utterance in await asyncio.to_thread(self.transcribe_chunk, [item]):
                    await outbox.put(utterance)
            await outbox.put(None)
            return
        buffered = []
        while (item := await inbox.get()) is not None:
            buffered.append(item)
        for chunk in _pack_segments(buffered, self.max_chunk_s):
            for utterance in await asyncio.to_thread(self.transcribe_chunk, chunk):
                await outbox.put(utterance)
        await outbox.put(None)
