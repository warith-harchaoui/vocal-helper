"""Diarisation en flux par Nemotron 3, sans torch : onnxruntime et numpy.

Pourquoi ce module
------------------
Tout ce que ce dépôt a construit pour la diarisation en direct gratuite repose
sur un regroupement d'empreintes orchestré autour d'un moteur différé. Le
meilleur résultat obtenu est la qualité du différé atteinte en direct — taux
d'erreur médian 0,176 sur AMI dev-slice — et deux limites qu'aucune
orchestration ne lève : la fraîcheur, puisqu'une étiquette définitive arrive à
la passe suivante, et le plafond du moteur lui-même.

Nemotron 3 Diarization n'a ni l'une ni l'autre. C'est un modèle de bout en bout
qui tient son propre cache de locuteurs par ordre d'arrivée, donc il nomme à
l'instant et ne révise rien. Mesuré ici sous le protocole du dépôt, sur un
mixage dont la vérité est exacte par construction, il donne **0,090 de taux
d'erreur avec une confusion de locuteurs de 0,000 et quatre étiquettes pour
quatre locuteurs**, là où le différé pyannote fait 0,299 et le meilleur étage
en flux maison 0,171.

Pourquoi par ONNX plutôt que par la bibliothèque d'origine
----------------------------------------------------------
La voie officielle passe par ``transformers`` et torch, et appelle un dépôt
distant à l'exécution. Ici le graphe est un fichier du bundle de poids que le
projet sert déjà, et il tourne sur ``onnxruntime``, qui est **déjà une
dépendance de base** par sherpa. Trois conséquences : aucun jeton ni appel
réseau, pas de torch sur le chemin de décision, et le même fichier tourne
depuis Rust par la caisse ``ort`` — ce que torch ne permet pas.

Une précision utile : onnxruntime n'a pas de moteur Metal. Sur Apple Silicon
l'accélération passerait par **CoreML**, pas par MPS — mais la mesure la
déconseille, CoreML ne prenant que 1193 nœuds sur 2827 et les découpant en 316
partitions, ce qui le rend 2,4 fois plus lent que le CPU seul et le place
au-dessus du temps réel. Le défaut est donc le CPU, qui tient 1,9 fois le temps
réel sur un portable.

Ce que ce module doit écrire lui-même
-------------------------------------
Le graphe est **un pas de flux**, pas un diariseur de fichier : il reçoit
``[cache | file | morceau]`` et rend des probabilités. La tenue du cache et de
la file — quelles trames y entrent, lesquelles en sortent, comment le cache se
comprime quand il déborde — vit autour du graphe, et c'est le portage de
``SortformerModules.streaming_update`` de NeMo. Tout ce portage est du numpy
pur, donc il se vérifie sans modèle et sans audio.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MelSettings",
    "NemotronSchedule",
    "StreamingState",
    "log_mel",
    "log_pred_scores",
    "disable_low_scores",
    "boost_top_scores",
    "compress_cache",
    "streaming_update",
    "probabilities_to_segments",
    "NemotronOnnxDiar",
]

#: Nom du dossier du moteur dans le bundle de poids.
BUNDLE_DIR = "nemotron3"

#: Le fichier du graphe, dans ce dossier.
GRAPH_NAME = "nemotron-3-diarization.onnx"

#: Préfixe des métadonnées que l'export inscrit dans le graphe.
META_PREFIX = "vernacula.diar."

#: Calendrier par défaut, et pourquoi ce n'est **pas** celui du graphe.
#:
#: Le graphe porte dans ses métadonnées le calendrier hors ligne — 340 trames
#: de morceau plus 40 de contexte, soit 30,4 s de tampon — et son export
#: affirme que c'est le seul vérifié de bout en bout. Mesuré ici, il **tombe**
#: dès que le cache de locuteurs se remplit :
#:
#:     Reshape '/pos_enc/Reshape_3' : Input shape {5000,64},
#:     requested shape {1,1,5144,64}
#:
#: L'encodage positionnel est tracé pour 5000 positions et la séquence
#: ``[cache | file | morceau]`` en demande 5144 au régime établi. Le défaut
#: échouait donc sur tout audio dépassant quelques minutes — exactement le cas
#: pour lequel un mode hors ligne existe — et passait inaperçu sur un extrait
#: court, le cache n'étant pas encore plein.
#:
#: ``"low_latency"`` tient largement dans la longueur tracée et se trouve être
#: le mieux mesuré : 0,090 de taux d'erreur avec une confusion de locuteurs
#: nulle, contre 0,209 pour le meilleur autre moteur gratuit local.
DEFAULT_MODE = "low_latency"


# --------------------------------------------------------------------------- #
#  Le front-end log-mel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MelSettings:
    """Les paramètres du front-end, tels que la configuration du modèle les donne.

    Attributes
    ----------
    sample_rate : int
        16 kHz, imposé par le modèle.
    n_fft : int
        Taille de la transformée.
    win_length : int
        Longueur de la fenêtre d'analyse, **plus courte** que la transformée.
    hop_length : int
        Avance entre deux trames : 160 échantillons, soit 10 ms.
    n_mels : int
        Nombre de bandes.
    preemphasis : float
        Pré-accentuation appliquée avant tout découpage.
    log_guard : float
        Plancher ajouté avant le logarithme. ``2**-24`` est la valeur que
        l'extracteur officiel emploie, retrouvée en comparant les sorties.
    """

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    preemphasis: float = 0.97
    log_guard: float = 2.0**-24


def _mel_filterbank(settings: MelSettings) -> NDArray[np.float64]:
    """Le banc de filtres mel, convention Slaney, de 0 à Nyquist.

    Construit à la main plutôt que par une dépendance de plus : c'est une
    matrice triangulaire, et la convention qui compte — l'échelle Slaney et sa
    normalisation par la largeur de bande — tient en quelques lignes.
    """
    sr, n_fft, n_mels = settings.sample_rate, settings.n_fft, settings.n_mels

    def to_mel(hz: NDArray[np.float64]) -> NDArray[np.float64]:
        # Slaney : linéaire sous 1 kHz, logarithmique au-dessus.
        lin_hz, lin_mel, step = 1000.0, 15.0, 200.0 / 3.0
        out = hz / step
        above = hz >= lin_hz
        out[above] = lin_mel + np.log(hz[above] / lin_hz) / (np.log(6.4) / 27.0)
        return out

    def to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
        lin_hz, lin_mel, step = 1000.0, 15.0, 200.0 / 3.0
        out = mel * step
        above = mel >= lin_mel
        out[above] = lin_hz * np.exp((mel[above] - lin_mel) * (np.log(6.4) / 27.0))
        return out

    edges = to_hz(
        np.linspace(to_mel(np.array([0.0]))[0], to_mel(np.array([sr / 2.0]))[0], n_mels + 2)
    )
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    weights = np.zeros((n_mels, len(bins)), dtype=np.float64)
    for i in range(n_mels):
        low, centre, high = edges[i], edges[i + 1], edges[i + 2]
        rising = (bins - low) / max(centre - low, 1e-12)
        falling = (high - bins) / max(high - centre, 1e-12)
        weights[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    # Normalisation Slaney : chaque filtre intègre à surface constante, sans
    # quoi les bandes hautes, plus larges, domineraient.
    widths = edges[2 : n_mels + 2] - edges[:n_mels]
    weights *= (2.0 / np.where(widths > 0, widths, 1.0))[:, None]
    return weights


def log_mel(
    audio: NDArray[np.float32], sample_rate: int, settings: MelSettings | None = None
) -> NDArray[np.float32]:
    """Les trames log-mel que le graphe attend, ``(T, n_mels)``.

    Deux conventions ont été retrouvées en comparant à l'extracteur officiel
    plutôt que devinées, et elles comptent toutes les deux.

    La fenêtre d'analyse de 400 points est **centrée** dans la transformée de
    512, et non collée au début. Mettre la fenêtre au début multiplie par
    seize l'écart aux sorties officielles — 0,530 au lieu de 0,033 — parce que
    le décalage de phase que cela introduit change le spectre.

    Le plancher du logarithme est ``2**-24``, retrouvé en lisant le minimum des
    sorties officielles : −16,36, c'est-à-dire exactement son logarithme.

    Au régime établi l'écart moyen aux sorties officielles est de 0,006 pour
    des valeurs d'écart-type 1,3, soit un demi pour cent. Les trames de bord
    diffèrent : l'extracteur officiel remplit sa dernière trame d'une valeur
    sentinelle, ce qu'il n'y a pas lieu de reproduire.

    Parameters
    ----------
    audio : NDArray[np.float32]
        Signal mono.
    sample_rate : int
        Sa fréquence, qui doit être celle du modèle.
    settings : MelSettings, optional
        Les paramètres, ceux du modèle par défaut.

    Returns
    -------
    NDArray[np.float32]
        ``(T, n_mels)``, une trame toutes les 10 ms.

    Raises
    ------
    ValueError
        Si la fréquence d'échantillonnage n'est pas celle du modèle.

    Examples
    --------
    >>> import numpy as np
    >>> frames = log_mel(np.zeros(16000, dtype=np.float32), 16000)
    >>> frames.shape
    (101, 128)

    Le silence tombe sur le plancher du logarithme, partout pareil.

    >>> bool(np.allclose(frames, np.log(2.0**-24)))
    True
    """
    settings = settings or MelSettings()
    if sample_rate != settings.sample_rate:
        raise ValueError(f"le modèle veut {settings.sample_rate} Hz, reçu {sample_rate} Hz")
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros((0, settings.n_mels), dtype=np.float32)
    x = np.concatenate([x[:1], x[1:] - settings.preemphasis * x[:-1]])
    pad = settings.n_fft // 2
    x = np.pad(x, pad, mode="reflect" if x.size > pad else "constant")

    window = np.zeros(settings.n_fft, dtype=np.float64)
    offset = (settings.n_fft - settings.win_length) // 2
    # Hann périodique, comme ``torch.hann_window`` par défaut.
    window[offset : offset + settings.win_length] = np.hanning(settings.win_length + 1)[
        : settings.win_length
    ]

    count = 1 + max(0, (x.size - settings.n_fft) // settings.hop_length)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(count, settings.n_fft),
        strides=(x.strides[0] * settings.hop_length, x.strides[0]),
    )
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = power @ _mel_filterbank(settings).T
    return np.log(mel + settings.log_guard).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Le calendrier de flux, lu dans le graphe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NemotronSchedule:
    """Le calendrier de flux, en trames de 80 ms.

    Les valeurs vivent dans les métadonnées du graphe, donc elles ne sont pas
    recopiées ici : un moteur mis à jour apporte les siennes, et un écart entre
    le code et le fichier est impossible par construction.

    Attributes
    ----------
    num_speakers : int
        Nombre maximal de locuteurs que le modèle peut distinguer.
    emb_dim : int
        Dimension des empreintes du cache et de la file.
    n_mels : int
        Bandes attendues en entrée.
    subsampling : int
        Rapport entre les trames de 10 ms et celles de 80 ms.
    upsample_factor : int
        Rapport inverse, pour les probabilités à 10 ms.
    spkcache_len, fifo_len, chunk_len, chunk_right_context : int
        Longueurs du cache, de la file, du morceau et de son contexte droit.
    spkcache_update_period : int
        Fréquence de rafraîchissement du cache.
    sil_threshold, pred_score_threshold : float
        Seuils de silence et de score.
    strong_boost_rate, weak_boost_rate, min_pos_scores_rate : float
        Les taux qui équilibrent le cache entre locuteurs.
    scores_boost_latest : float
        Prime donnée aux trames les plus récentes.
    spkcache_sil_frames_per_spk : int
        Créneaux de silence réservés par locuteur dans le cache.
    """

    num_speakers: int = 8
    emb_dim: int = 512
    n_mels: int = 128
    subsampling: int = 8
    upsample_factor: int = 8
    spkcache_len: int = 264
    fifo_len: int = 40
    chunk_len: int = 340
    chunk_right_context: int = 40
    spkcache_update_period: int = 300
    sil_threshold: float = 0.2
    pred_score_threshold: float = 0.25
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_pos_scores_rate: float = 0.5
    scores_boost_latest: float = 0.05
    spkcache_sil_frames_per_spk: int = 1

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> NemotronSchedule:
        """Lire le calendrier des métadonnées du graphe.

        Les clés inconnues sont ignorées et les absentes gardent leur défaut,
        pour qu'un export enrichi n'empêche pas le chargement.

        Examples
        --------
        >>> NemotronSchedule.from_metadata(
        ...     {"vernacula.diar.fifo_len": "64", "vernacula.diar.inconnu": "x"}
        ... ).fifo_len
        64
        """
        # ``from __future__ import annotations`` rend les annotations sous forme
        # de CHAÎNES, donc comparer à ``int`` échoue en silence et tout
        # devenait flottant — un ``fifo_len`` de 64.0 là où le graphe veut 64.
        known = {f.name: str(f.type).strip("'\"") for f in cls.__dataclass_fields__.values()}
        values: dict[str, Any] = {}
        for key, raw in metadata.items():
            name = key[len(META_PREFIX) :] if key.startswith(META_PREFIX) else key
            if name not in known:
                continue
            values[name] = int(raw) if known[name] == "int" else float(raw)
        return cls(**values)

    @property
    def latency_s(self) -> float:
        """Audio à attendre avant de traiter un morceau, en secondes.

        Examples
        --------
        >>> round(NemotronSchedule(chunk_len=9, chunk_right_context=4).latency_s, 3)
        1.04
        """
        return (self.chunk_len + self.chunk_right_context) * 0.08


#: Les trois calendriers que la carte du modèle recommande, en trames de 80 ms,
#: sous la forme (chunk_len, chunk_right_context). Les latences correspondantes
#: sont 1,04 s, 0,64 s et 0,32 s.
STREAMING_MODES: dict[str, tuple[int, int]] = {
    "low_latency": (9, 4),
    "very_low_latency": (6, 2),
    "ultra_low_latency": (3, 1),
}


# --------------------------------------------------------------------------- #
#  La tenue du cache — portage numpy de SortformerModules.streaming_update
# --------------------------------------------------------------------------- #


def log_pred_scores(predictions: NDArray[np.float32], *, floor: float) -> NDArray[np.float64]:
    """Le score d'une trame pour chaque locuteur, à partir des probabilités.

    Le score est conçu pour être élevé quand **un seul** locuteur parle avec
    confiance : il récompense la probabilité du locuteur considéré et pénalise
    celle des autres. C'est ce qui fait qu'un cache rempli par ces scores
    contient de la parole propre et non du recouvrement.

    Examples
    --------
    Un locuteur seul et sûr obtient un score positif ; deux locuteurs
    simultanés, des scores plus faibles.

    >>> import numpy as np
    >>> alone = log_pred_scores(np.array([[0.9, 0.05]], dtype=np.float32), floor=0.25)
    >>> both = log_pred_scores(np.array([[0.9, 0.9]], dtype=np.float32), floor=0.25)
    >>> bool(alone[0, 0] > both[0, 0])
    True
    """
    probabilities = np.asarray(predictions, dtype=np.float64)
    log_p = np.log(np.clip(probabilities, floor, None))
    log_1p = np.log(np.clip(1.0 - probabilities, floor, None))
    return log_p - log_1p + log_1p.sum(axis=1, keepdims=True) - math.log(0.5)


def disable_low_scores(
    predictions: NDArray[np.float32],
    scores: NDArray[np.float64],
    *,
    min_positive: int,
) -> NDArray[np.float64]:
    """Écarter du cache ce qui n'est pas de la parole propre.

    Deux coups de balai. Ce qui n'est pas de la parole pour ce locuteur est
    écarté définitivement. Et ce qui a un score non positif — en général du
    recouvrement — l'est aussi, **à condition** que le locuteur dispose déjà
    d'assez de trames franches : sans cette condition, un locuteur qui ne parle
    jamais seul disparaîtrait du cache.

    Examples
    --------
    >>> import numpy as np
    >>> preds = np.array([[0.9, 0.1], [0.4, 0.1]], dtype=np.float32)
    >>> scores = np.zeros((2, 2))
    >>> out = disable_low_scores(preds, scores, min_positive=99)
    >>> bool(np.isneginf(out[1, 0])), bool(np.isneginf(out[0, 1]))
    (True, True)
    """
    probabilities = np.asarray(predictions, dtype=np.float64)
    out = np.array(scores, dtype=np.float64, copy=True)
    is_speech = probabilities > 0.5
    out = np.where(is_speech, out, -np.inf)
    is_positive = out > 0
    enough = is_positive.sum(axis=0, keepdims=True) >= min_positive
    out = np.where((~is_positive) & is_speech & enough, -np.inf, out)
    return out


def boost_top_scores(
    scores: NDArray[np.float64], count: int, *, scale: float, offset: float = 0.5
) -> NDArray[np.float64]:
    """Relever les ``count`` meilleurs scores de chaque locuteur.

    Appliqué deux fois par la compression, avec deux intensités : une forte,
    pour garantir à chaque locuteur un minimum de place dans le cache, et une
    faible, pour empêcher un bavard d'occuper tout l'espace. Les trames déjà
    écartées restent à moins l'infini.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([[0.0], [1.0], [-np.inf]])
    >>> boost_top_scores(scores, 1, scale=1.0)[:, 0].round(3).tolist()
    [0.0, 1.693, -inf]
    """
    out = np.array(scores, dtype=np.float64, copy=True)
    if count <= 0:
        return out
    frames, speakers = out.shape
    count = min(count, frames)
    bonus = -scale * math.log(offset)
    for speaker in range(speakers):
        column = out[:, speaker]
        chosen = np.argpartition(-column, count - 1)[:count]
        finite = chosen[np.isfinite(column[chosen])]
        column[finite] += bonus
    return out


def _top_slots(
    scores: NDArray[np.float64], *, keep: int, real_frames: int, sentinel: int = 99999
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Les ``keep`` meilleures trames, toutes voix confondues, remises en ordre.

    Les scores sont mis à plat **par locuteur** avant le tri, si bien qu'une
    même trame peut être retenue au titre de deux locuteurs ; le modulo rend
    ensuite l'indice de trame. L'ordre chronologique est rétabli, parce que le
    cache est lu comme une séquence.
    """
    flat = scores.T.reshape(-1)
    keep = min(keep, flat.size)
    chosen = np.argpartition(-flat, keep - 1)[:keep]
    chosen = np.where(np.isneginf(flat[chosen]), sentinel, chosen)
    chosen = np.sort(chosen)
    disabled = chosen == sentinel
    frames = np.where(disabled, 0, chosen % max(scores.shape[0], 1))
    disabled = disabled | (frames >= real_frames)
    frames = np.where(disabled, 0, frames)
    return frames.astype(np.int64), disabled


def compress_cache(
    embeddings: NDArray[np.float32],
    predictions: NDArray[np.float32],
    schedule: NemotronSchedule,
    silence: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Ramener un cache débordant à sa taille, en gardant ce qui compte.

    On garde les trames dont les scores disent qu'elles portent de la parole
    propre, en réservant à chaque locuteur sa part et en laissant quelques
    créneaux de silence par locuteur. Les créneaux vides reçoivent l'empreinte
    de silence **apprise** que l'export transporte, et non la moyenne des
    trames silencieuses — c'est l'un des deux écarts de ce point de contrôle
    par rapport à Streaming Sortformer 2.x.

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        ``(n, emb_dim)``, plus long que le cache.
    predictions : NDArray[np.float32]
        ``(n, num_speakers)``, les probabilités correspondantes.
    schedule : NemotronSchedule
        Les longueurs et les taux.
    silence : NDArray[np.float32]
        L'empreinte de silence apprise, ``(emb_dim,)``.

    Returns
    -------
    tuple
        Le cache et ses probabilités, à la longueur voulue.

    Examples
    --------
    Un cache de quatre trames ramené à deux garde les deux voix, une chacune,
    plutôt que les deux trames du locuteur le plus bavard.

    >>> import numpy as np
    >>> schedule = NemotronSchedule(num_speakers=2, emb_dim=3, spkcache_len=2,
    ...                             spkcache_sil_frames_per_spk=0)
    >>> embs = np.array([[1., 0, 0], [1., 0, 0], [1., 0, 0], [0, 1., 0]], np.float32)
    >>> preds = np.array([[0.9, 0.0], [0.9, 0.0], [0.9, 0.0], [0.0, 0.9]], np.float32)
    >>> cache, kept = compress_cache(embs, preds, schedule, np.zeros(3, np.float32))
    >>> cache.shape, [int(np.argmax(row)) for row in kept]
    ((2, 3), [0, 1])
    """
    frames, speakers = predictions.shape
    per_speaker = max(1, schedule.spkcache_len // speakers - schedule.spkcache_sil_frames_per_spk)
    scores = log_pred_scores(predictions, floor=schedule.pred_score_threshold)
    scores = disable_low_scores(
        predictions, scores, min_positive=math.floor(per_speaker * schedule.min_pos_scores_rate)
    )
    if schedule.scores_boost_latest > 0 and frames > schedule.spkcache_len:
        scores[schedule.spkcache_len :, :] += schedule.scores_boost_latest
    scores = boost_top_scores(
        scores, math.floor(per_speaker * schedule.strong_boost_rate), scale=2.0
    )
    scores = boost_top_scores(scores, math.floor(per_speaker * schedule.weak_boost_rate), scale=1.0)

    real_frames = frames
    if schedule.spkcache_sil_frames_per_spk > 0:
        # Des créneaux toujours retenus, qui recevront l'empreinte de silence.
        pad = np.full((schedule.spkcache_sil_frames_per_spk, speakers), np.inf)
        scores = np.concatenate([scores, pad], axis=0)

    slots, disabled = _top_slots(scores, keep=schedule.spkcache_len, real_frames=real_frames)
    cache = embeddings[slots].astype(np.float32, copy=True)
    kept = predictions[slots].astype(np.float32, copy=True)
    cache[disabled] = silence
    kept[disabled] = 0.0
    return cache, kept


@dataclass
class StreamingState:
    """Le cache de locuteurs et la file, entre deux pas de flux.

    Attributes
    ----------
    spkcache : NDArray[np.float32]
        ``(T_cache, emb_dim)`` — ce que le flux a retenu des locuteurs.
    spkcache_preds : NDArray[np.float32] or None
        Les probabilités du cache, ``None`` tant qu'il n'a jamais été rempli.
    fifo : NDArray[np.float32]
        ``(T_fifo, emb_dim)`` — le contexte récent.
    fifo_preds : NDArray[np.float32]
        Ses probabilités.
    """

    spkcache: NDArray[np.float32]
    fifo: NDArray[np.float32]
    spkcache_preds: NDArray[np.float32] | None = None
    fifo_preds: NDArray[np.float32] = field(default=None)  # type: ignore[assignment]

    @classmethod
    def empty(cls, schedule: NemotronSchedule) -> StreamingState:
        """Un état de départ : tout vide, rien de retenu.

        Examples
        --------
        >>> state = StreamingState.empty(NemotronSchedule())
        >>> state.spkcache.shape, state.fifo.shape
        ((0, 512), (0, 512))
        """
        return cls(
            spkcache=np.zeros((0, schedule.emb_dim), dtype=np.float32),
            fifo=np.zeros((0, schedule.emb_dim), dtype=np.float32),
            fifo_preds=np.zeros((0, schedule.num_speakers), dtype=np.float32),
        )


def streaming_update(
    state: StreamingState,
    chunk_embeddings: NDArray[np.float32],
    predictions: NDArray[np.float32],
    schedule: NemotronSchedule,
    silence: NDArray[np.float32],
) -> tuple[StreamingState, NDArray[np.float32]]:
    """Faire entrer un morceau dans la file, et la file dans le cache.

    Le portage de ``SortformerModules.streaming_update``, pour un seul flux.
    Les probabilités reçues couvrent ``[cache | file | morceau]`` : on en
    découpe la part de la file et celle du morceau, le morceau rejoint la file,
    et ce que la file pousse dehors rejoint le cache — qu'on comprime s'il
    déborde.

    Parameters
    ----------
    state : StreamingState
        L'état précédent.
    chunk_embeddings : NDArray[np.float32]
        ``(T_chunk, emb_dim)`` rendues par le graphe pour ce morceau, contexte
        droit déjà retiré.
    predictions : NDArray[np.float32]
        ``(T_cache + T_fifo + T_chunk, num_speakers)``.
    schedule : NemotronSchedule
        Les longueurs.
    silence : NDArray[np.float32]
        L'empreinte de silence apprise.

    Returns
    -------
    tuple
        Le nouvel état, et les probabilités du seul morceau.

    Examples
    --------
    La file se remplit d'abord sans rien pousser vers le cache.

    >>> import numpy as np
    >>> schedule = NemotronSchedule(num_speakers=2, emb_dim=2, spkcache_len=4, fifo_len=4)
    >>> state = StreamingState.empty(schedule)
    >>> embs = np.ones((2, 2), dtype=np.float32)
    >>> preds = np.full((2, 2), 0.9, dtype=np.float32)
    >>> state, chunk_preds = streaming_update(state, embs, preds, schedule,
    ...                                       np.zeros(2, np.float32))
    >>> state.fifo.shape, state.spkcache.shape, chunk_preds.shape
    ((2, 2), (0, 2), (2, 2))
    """
    cache_len, fifo_len = len(state.spkcache), len(state.fifo)
    chunk_len = len(chunk_embeddings)
    fifo_preds = np.asarray(predictions[cache_len : cache_len + fifo_len], dtype=np.float32)
    chunk_preds = np.asarray(
        predictions[cache_len + fifo_len : cache_len + fifo_len + chunk_len], dtype=np.float32
    )

    fifo = np.concatenate([state.fifo, chunk_embeddings], axis=0)
    fifo_preds = np.concatenate([fifo_preds, chunk_preds], axis=0)
    spkcache, spkcache_preds = state.spkcache, state.spkcache_preds

    if fifo_len + chunk_len > schedule.fifo_len:
        pop = min(max(1, chunk_len), schedule.fifo_len, fifo_len + chunk_len)
        popped, popped_preds = fifo[:pop], fifo_preds[:pop]
        fifo, fifo_preds = fifo[pop:], fifo_preds[pop:]
        spkcache = np.concatenate([spkcache, popped], axis=0)
        if spkcache_preds is not None:
            spkcache_preds = np.concatenate([spkcache_preds, popped_preds], axis=0)
        elif cache_len:
            # Premier remplissage : les probabilités du cache viennent de ce
            # que le graphe vient d'en dire.
            spkcache_preds = np.concatenate(
                [np.asarray(predictions[:cache_len], dtype=np.float32), popped_preds], axis=0
            )
        else:
            spkcache_preds = popped_preds
        if len(spkcache) > schedule.spkcache_len:
            spkcache, spkcache_preds = compress_cache(spkcache, spkcache_preds, schedule, silence)

    return (
        StreamingState(
            spkcache=spkcache,
            fifo=fifo,
            spkcache_preds=spkcache_preds,
            fifo_preds=fifo_preds,
        ),
        chunk_preds,
    )


# --------------------------------------------------------------------------- #
#  Des probabilités aux segments
# --------------------------------------------------------------------------- #


def probabilities_to_segments(
    probabilities: NDArray[np.float32],
    *,
    frame_s: float = 0.01,
    threshold: float = 0.5,
    bridge_s: float = 0.5,
) -> list[tuple[float, float, str]]:
    """Seuiller les probabilités par trame et recoller les miettes.

    Deux locuteurs simultanés donnent deux segments qui se recouvrent, ce qui
    est voulu : le modèle sait le dire, et l'écraser perdrait l'information.
    Les segments d'un même locuteur séparés de moins de ``bridge_s`` sont
    réunis, parce qu'un silence de respiration n'est pas un changement de tour
    et qu'une transcription n'a rien à faire de fragments de trois dixièmes.

    Parameters
    ----------
    probabilities : NDArray[np.float32]
        ``(T, num_speakers)`` à une trame par ``frame_s``.
    frame_s : float
        Durée d'une trame. 10 ms pour la sortie haute résolution du graphe.
    threshold : float
        Probabilité au-dessus de laquelle la trame compte comme parole.
    bridge_s : float
        Silence en deçà duquel deux segments d'un même locuteur sont réunis.

    Returns
    -------
    list of tuple
        ``[(t0, t1, locuteur), …]`` trié par début.

    Examples
    --------
    >>> import numpy as np
    >>> probs = np.array([[0.9], [0.9], [0.1], [0.9]], dtype=np.float32)
    >>> probabilities_to_segments(probs, bridge_s=0.0)
    [(0.0, 0.02, 'S0'), (0.03, 0.04, 'S0')]

    Le même signal, en recollant les silences courts :

    >>> probabilities_to_segments(probs, bridge_s=0.5)
    [(0.0, 0.04, 'S0')]
    """
    out: list[tuple[float, float, str]] = []
    active = np.asarray(probabilities) > threshold
    for speaker in range(active.shape[1]):
        column = active[:, speaker]
        if not column.any():
            continue
        edges = np.diff(np.concatenate([[False], column, [False]]).astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        spans: list[list[float]] = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            t0, t1 = start * frame_s, end * frame_s
            if spans and t0 - spans[-1][1] <= bridge_s:
                spans[-1][1] = t1
            else:
                spans.append([t0, t1])
        out.extend((round(a, 6), round(b, 6), f"S{speaker}") for a, b in spans)
    return sorted(out)


# --------------------------------------------------------------------------- #
#  L'étage
# --------------------------------------------------------------------------- #


class NemotronOnnxDiar:
    """Diarisation en flux par le graphe Nemotron, sans torch.

    Parameters
    ----------
    graph : str or Path, optional
        Le fichier ``.onnx``. ``None`` le cherche dans le bundle de poids que
        le projet sert déjà, donc sans jeton ni appel réseau.
    mode : str, optional
        Un des calendriers de :data:`STREAMING_MODES`. ``"low_latency"`` par
        défaut, et **non** celui que le graphe porte dans ses métadonnées : ce
        dernier est le calendrier hors ligne, et il dépasse l'encodage
        positionnel que l'export a tracé. Voir :data:`DEFAULT_MODE`.
    providers : list of str, optional
        Les moteurs d'exécution, par ordre de préférence. ``None`` prend
        CoreML s'il est disponible — c'est la voie du GPU et du moteur neuronal
        sur Apple Silicon, onnxruntime n'ayant pas de moteur Metal — puis le
        CPU, qui suffit de toute façon.
    threads : int, optional
        Fils pour le moteur CPU. ``None`` — le défaut — laisse onnxruntime
        choisir, et c'est le meilleur choix : forcer huit fils sur une machine à
        huit cœurs performants mesure 0,953 de facteur temps réel contre 0,528
        en laissant faire, soit presque deux fois pire. L'argument existe pour
        brider la consommation, pas pour accélérer.
    threshold : float
        Seuil de parole appliqué aux probabilités à 10 ms.
    """

    def __init__(
        self,
        graph: str | Path | None = None,
        *,
        mode: str | None = DEFAULT_MODE,
        providers: list[str] | None = None,
        threads: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Repérer le graphe ; la session se crée au premier appel."""
        if mode is not None and mode not in STREAMING_MODES:
            raise ValueError(f"calendrier inconnu {mode!r} ; au choix {sorted(STREAMING_MODES)}")
        self._graph = Path(graph) if graph is not None else None
        self._mode = mode
        self._providers = providers
        self._threads = threads
        self.threshold = float(threshold)
        self._session: Any = None
        self._schedule: NemotronSchedule | None = None
        self._silence: NDArray[np.float32] | None = None

    # ----- chargement -------------------------------------------------

    def locate(self) -> Path:
        """Le chemin du graphe, dans le bundle si on ne l'a pas précisé.

        Raises
        ------
        FileNotFoundError
            Si le bundle ne contient pas le moteur.
        """
        if self._graph is not None:
            return self._graph
        from vocal_helper.diar import resolve_diarization_engines

        bundle = resolve_diarization_engines()
        candidate = (bundle / BUNDLE_DIR / GRAPH_NAME) if bundle else None
        if candidate is None or not candidate.exists():
            raise FileNotFoundError(
                f"{GRAPH_NAME} absent du bundle de diarisation. Le moteur est "
                f"distribué dans ``{BUNDLE_DIR}/`` ; vider "
                f"``~/.cache/vocal-helper/diarization-engines/`` force une "
                f"reprise du bundle."
            )
        return candidate

    def _load(self) -> tuple[Any, NemotronSchedule, NDArray[np.float32]]:
        """Ouvrir la session, lire le calendrier et l'empreinte de silence."""
        if self._session is not None and self._schedule is not None:
            assert self._silence is not None
            return self._session, self._schedule, self._silence

        import onnxruntime as ort

        path = self.locate()
        metadata = _graph_metadata(path)
        schedule = NemotronSchedule.from_metadata(metadata)
        if self._mode is not None:
            chunk_len, right = STREAMING_MODES[self._mode]
            schedule = NemotronSchedule(
                **{
                    **{f: getattr(schedule, f) for f in schedule.__dataclass_fields__},
                    "chunk_len": chunk_len,
                    "chunk_right_context": right,
                }
            )

        encoded = metadata.get(f"{META_PREFIX}learnable_sil_emb_f32le_b64")
        if not encoded:
            raise ValueError(
                f"{path.name} ne porte pas l'empreinte de silence apprise ; "
                "ce point de contrôle la demande."
            )
        silence = np.frombuffer(base64.b64decode(encoded), dtype="<f4").astype(np.float32)
        if silence.size != schedule.emb_dim:
            raise ValueError(
                f"empreinte de silence de {silence.size} valeurs pour {schedule.emb_dim} attendues"
            )

        options = ort.SessionOptions()
        if self._threads:
            options.intra_op_num_threads = int(self._threads)
        providers = self._providers or _preferred_providers()
        self._session = ort.InferenceSession(str(path), options, providers=providers)
        self._schedule, self._silence = schedule, silence
        return self._session, schedule, silence

    @property
    def schedule(self) -> NemotronSchedule:
        """Le calendrier effectif, celui du graphe ou celui demandé."""
        return self._load()[1]

    @property
    def providers(self) -> list[str]:
        """Les moteurs d'exécution que la session a réellement retenus."""
        return list(self._load()[0].get_providers())

    # ----- la boucle --------------------------------------------------

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """Diariser un tampon, en le faisant défiler comme un flux.

        Les trois arguments de compte sont acceptés pour la parité d'interface
        avec les autres étages du paquet, et ignorés : ce modèle ordonne ses
        huit voies par ordre d'arrivée et n'a pas de compte à recevoir.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Tampon mono.
        sr : int
            Sa fréquence, qui doit être 16 kHz.

        Returns
        -------
        list of tuple
            ``[(t0, t1, locuteur), …]`` trié par début, la même forme que les
            autres étages de diarisation.
        """
        session, schedule, silence = self._load()
        frames = log_mel(pcm, sr, MelSettings(n_mels=schedule.n_mels))
        if not len(frames):
            return []

        step = schedule.chunk_len * schedule.subsampling
        right = schedule.chunk_right_context * schedule.subsampling
        state = StreamingState.empty(schedule)
        collected: list[NDArray[np.float32]] = []

        for start in range(0, len(frames), step):
            window = frames[start : start + step + right]
            if not len(window):
                break
            # Le morceau ne doit pas être complété au-delà du multiple de huit
            # suivant : la tête à 10 ms est une convolution de rayon un, donc
            # la dernière trame réelle lirait l'état d'une trame de garnissage.
            usable = (len(window) // schedule.subsampling) * schedule.subsampling
            if usable < schedule.subsampling:
                break
            window = window[:usable]
            try:
                outputs = _run_step(session, window, usable, state)
            except Exception as problem:  # noqa: BLE001 - on explique plutôt que de relayer
                raise RuntimeError(
                    f"le graphe a refusé une séquence de "
                    f"{schedule.spkcache_len + schedule.fifo_len + usable // schedule.subsampling} "
                    f"trames ({schedule.spkcache_len} de cache, {schedule.fifo_len} de "
                    f"file, {usable // schedule.subsampling} de morceau). Son encodage "
                    f"positionnel est tracé pour une longueur finie, et le calendrier "
                    f"hors ligne la dépasse dès que le cache se remplit — d'où un échec "
                    f"qui n'apparaît pas sur un extrait court. Demander un calendrier "
                    f"de flux : {sorted(STREAMING_MODES)}."
                ) from problem
            named = dict(zip([o.name for o in session.get_outputs()], outputs, strict=True))
            coarse = np.asarray(named["spkcache_fifo_chunk_preds"][0], dtype=np.float32)
            fine = np.asarray(named["spkcache_fifo_chunk_preds_hr"][0], dtype=np.float32)
            embeddings = np.asarray(named["chunk_pre_encode_embs"][0], dtype=np.float32)

            # Retirer le contexte droit : il a servi à mieux décider du morceau,
            # il ne fait pas partie du morceau.
            kept = max(0, usable // schedule.subsampling - schedule.chunk_right_context)
            if start + step >= len(frames):
                kept = usable // schedule.subsampling  # dernier morceau : rien à retirer
            state, _ = streaming_update(state, embeddings[:kept], coarse, schedule, silence)
            # Les probabilités à 10 ms couvrent ``[cache | file | morceau]`` ;
            # la part du morceau commence donc là où finit le reste, et cette
            # longueur se lit sur les sorties du graphe sans toucher à l'état.
            head = (len(coarse) - embeddings.shape[0]) * schedule.upsample_factor
            collected.append(fine[head : head + kept * schedule.upsample_factor])

        if not collected:
            return []
        return probabilities_to_segments(
            np.concatenate(collected, axis=0), threshold=self.threshold
        )


def _run_step(
    session: Any,
    window: NDArray[np.float32],
    usable: int,
    state: StreamingState,
) -> list[Any]:
    """Un pas de flux : ``[cache | file | morceau]`` dans le graphe."""
    return session.run(
        None,
        {
            "chunk": window[None, :, :].astype(np.float32),
            "chunk_lengths": np.array([usable], dtype=np.int64),
            "spkcache": state.spkcache[None, :, :],
            "spkcache_lengths": np.array([len(state.spkcache)], dtype=np.int64),
            "fifo": state.fifo[None, :, :],
            "fifo_lengths": np.array([len(state.fifo)], dtype=np.int64),
        },
    )


def _graph_metadata(path: Path) -> dict[str, str]:
    """Les métadonnées du graphe, sans charger ses poids."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def _preferred_providers() -> list[str]:
    """Les moteurs d'exécution par ordre de préférence, mesuré et non supposé.

    **Le CPU passe devant CoreML**, ce qui est contre-intuitif et pourtant
    mesuré : sur deux minutes d'un mixage à quatre locuteurs, CoreML en tête
    donne un facteur temps réel de 1,257 — donc **plus lent que le direct**,
    inutilisable — contre 0,528 pour le CPU seul, soit 2,4 fois mieux. La cause
    se lit dans le journal de la session : CoreML ne prend que 1193 nœuds sur
    2827 et les découpe en **316 partitions**, donc autant d'allers-retours
    entre les deux moteurs à chaque pas de flux. Un transformeur à axes
    dynamiques est précisément le cas où le déport partiel coûte plus qu'il ne
    rapporte. CoreML reste accessible en le demandant explicitement, pour une
    machine ou une version où le découpage serait meilleur.

    CUDA passe devant, lui, parce que sa couverture d'opérateurs est complète
    et qu'il n'y a donc pas de découpage à payer — mais ce n'est pas mesuré
    ici, faute de machine NVIDIA sous la main.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    order = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
        "CoreMLExecutionProvider",
    ]
    return [name for name in order if name in available] or ["CPUExecutionProvider"]
