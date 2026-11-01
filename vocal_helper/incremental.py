"""Regroupement incrémental sous contrainte des locuteurs d'un flux.

Pourquoi ce module existe
-------------------------
:class:`~vocal_helper.diar.OnlineDiarStage` rattache chaque segment voisé au
centroïde le plus proche sous un seuil unique, puis met ce centroïde à jour en
moyenne glissante. Ce schéma glouton a trois faiblesses, mesurées sur un mixage
synthétique de 4 locuteurs (``studies/rediar_speaker_consistency.py``), où il
s'effondre sur **une seule étiquette** pour les quatre voix :

1. il n'a aucune notion de simultanéité, donc rien n'interdit à deux voix qui
   parlent en même temps de tomber sur le même locuteur ;
2. il embarque un segment recouvert comme un tout, ce qui produit une empreinte
   *mélangée* qui tombe entre deux centroïdes et les rapproche ;
3. toute jointure met le centroïde à jour, même douteuse, et la moyenne
   glissante propage la contamination jusqu'à la fusion générale.

La réponse implémentée ici s'inspire de diart. L'algorithme est **écrit sur
place et non importé** : diart est une dépendance lourde avec son propre
environnement, alors que le principe tient en quelques dizaines de lignes de
numpy et de scipy — testables sans audio, sans modèle et sans boucle
d'événements, et transposables en Rust.

Les trois garde-fous
--------------------
``contrainte d'injectivité``
    L'affectation locaux → globaux passe par un algorithme hongrois
    (:func:`scipy.optimize.linear_sum_assignment`), donc deux locuteurs actifs
    dans la même fenêtre ne peuvent pas recevoir la même étiquette globale.
    C'est ce qui rend l'effondrement impossible par construction.
``rho_update``
    Un locuteur local ne met son centroïde à jour que si sa part de parole
    *propre* dans la fenêtre atteint :attr:`IncrementalPolicy.rho_update_s`.
    En dessous, il est rattaché mais n'écrit rien.
``delta_new``
    Si le centroïde le plus proche reste au-delà de
    :attr:`IncrementalPolicy.delta_new` en distance cosinus, on crée un nouveau
    locuteur global plutôt que de forcer une mauvaise jointure. C'est ainsi que
    le nombre de locuteurs se **devine** au fil du flux.

Écart assumé
------------
L'inspiration calcule une empreinte pondérée par le masque d'activité du
locuteur, ce que ses encodeurs acceptent. Les nôtres prennent un simple ``(pcm, sr)``. On
obtient l'équivalent utile en n'embarquant que les trames où le locuteur local
est actif **et seul** : une empreinte propre plutôt qu'une empreinte pondérée.
La durée de cette part propre est précisément ce que ``rho_update`` filtre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["IncrementalPolicy", "IncrementalClusterer", "cosine_distances"]


def cosine_distances(
    locals_: NDArray[np.float32], centroids: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Matrice des distances cosinus entre empreintes locales et centroïdes.

    Les deux entrées sont supposées normalisées L2, ce que
    :meth:`IncrementalClusterer.assign` garantit.

    Parameters
    ----------
    locals_ : NDArray[np.float32]
        ``(k, d)`` — une ligne par locuteur local de la fenêtre.
    centroids : NDArray[np.float32]
        ``(m, d)`` — une ligne par locuteur global déjà connu.

    Returns
    -------
    NDArray[np.float32]
        ``(k, m)`` de distances dans ``[0, 2]``.

    Examples
    --------
    >>> import numpy as np
    >>> e0 = np.array([[1.0, 0.0]], dtype=np.float32)
    >>> e1 = np.array([[0.0, 1.0]], dtype=np.float32)
    >>> cosine_distances(e0, np.concatenate([e0, e1]))
    array([[0., 1.]], dtype=float32)
    """
    return (1.0 - locals_ @ centroids.T).astype(np.float32)


@dataclass
class IncrementalPolicy:
    """Les réglages du regroupement incrémental.

    Les valeurs par défaut reprennent celles de la méthode dont ce module
    s'inspire, hors ``max_speakers``, qui est un ajout.

    Attributes
    ----------
    tau_active : float
        Seuil d'activité de la segmentation locale. Sous ce seuil, un locuteur
        local n'est pas considéré comme parlant dans la fenêtre.
    rho_update_s : float
        Part de parole *propre* minimale, en secondes, pour qu'un locuteur local
        soit autorisé à mettre son centroïde à jour.
    delta_new : float
        Distance cosinus au-delà de laquelle on crée un locuteur global plutôt
        que de forcer une jointure.
    max_speakers : int, optional
        Plafond dur du nombre de locuteurs globaux. ``None`` (défaut) laisse le
        compte libre — c'est le mode « devine ». Un plafond est la garde la plus
        efficace contre la prolifération d'étiquettes : la mesure du
        2026-11-01 montre qu'imposer un plafond au moteur différé vaut exactement
        le compte exact (DER 0,242 contre 0,299 en découverte libre).
    ema_alpha : float
        Coefficient de la moyenne glissante des centroïdes. ``None`` prend la
        moyenne courante exacte, ce que fait diart.
    """

    tau_active: float = 0.5
    rho_update_s: float = 0.4
    delta_new: float = 0.6
    max_speakers: int | None = None
    ema_alpha: float | None = None


@dataclass
class _Global:
    """Un locuteur global et sa statistique courante."""

    speaker_id: str
    vector: NDArray[np.float32]
    n_updates: int = 1


@dataclass
class IncrementalClusterer:
    """Rattache les locuteurs locaux d'une fenêtre aux locuteurs globaux.

    L'objet est volontairement sans audio et sans modèle : il reçoit des
    empreintes et des durées, il rend des étiquettes. Tout le comportement
    décrit dans l'en-tête du module se teste donc directement.

    Examples
    --------
    Deux voix orthogonales ouvrent le flux ; chacune reçoit son étiquette.

    >>> import numpy as np
    >>> e0 = np.array([1.0, 0.0], dtype=np.float32)
    >>> e1 = np.array([0.0, 1.0], dtype=np.float32)
    >>> c = IncrementalClusterer()
    >>> c.assign(np.stack([e0, e1]), [1.0, 1.0])
    ['S0', 'S1']

    La fenêtre suivante les présente dans l'autre sens. La contrainte
    d'injectivité et l'algorithme hongrois les suivent sans les confondre —
    c'est ce qu'un plus-proche-centroïde glouton ne garantit pas.

    >>> c.assign(np.stack([e1, e0]), [1.0, 1.0])
    ['S1', 'S0']

    Une empreinte mélangée, à mi-chemin des deux voix, reste sous
    ``delta_new`` : elle est rattachée au plus proche. Mais sa part propre est
    trop courte, donc elle n'écrit dans aucun centroïde.

    >>> blend = np.array([0.7071, 0.7071], dtype=np.float32)
    >>> c.assign(blend[None, :], [0.1])
    ['S0']
    >>> c.updates
    4

    Une troisième voix, elle, est trop loin des deux premières : le flux
    découvre un locuteur de plus plutôt que de forcer la jointure.

    >>> e2 = np.array([-1.0, 0.0], dtype=np.float32)
    >>> c.assign(e2[None, :], [1.0])
    ['S2']
    >>> c.speaker_count
    3

    Sous plafond, le même flux ne crée rien et se rabat sur le plus proche.

    >>> capped = IncrementalClusterer(policy=IncrementalPolicy(max_speakers=2))
    >>> capped.assign(np.stack([e0, e1]), [1.0, 1.0])
    ['S0', 'S1']
    >>> capped.assign(e2[None, :], [1.0])
    ['S1']
    >>> capped.speaker_count
    2
    """

    policy: IncrementalPolicy = field(default_factory=IncrementalPolicy)
    _globals: list[_Global] = field(default_factory=list, init=False)
    _updates: int = field(default=0, init=False)

    @property
    def speaker_count(self) -> int:
        """Nombre de locuteurs globaux découverts jusqu'ici."""
        return len(self._globals)

    @property
    def speaker_labels(self) -> list[str]:
        """Les étiquettes globales, dans l'ordre de leur découverte."""
        return [g.speaker_id for g in self._globals]

    @property
    def updates(self) -> int:
        """Nombre de mises à jour de centroïde acceptées par ``rho_update``."""
        return self._updates

    def centroids(self) -> NDArray[np.float32]:
        """Les centroïdes courants, ``(m, d)`` normalisés L2."""
        if not self._globals:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([g.vector for g in self._globals])

    def assign(self, embeddings: NDArray[np.float32], clean_durations: list[float]) -> list[str]:
        """Étiqueter les locuteurs locaux d'une fenêtre.

        Parameters
        ----------
        embeddings : NDArray[np.float32]
            ``(k, d)`` — une empreinte par locuteur local actif. Normalisée ici,
            l'appelant n'a pas à s'en soucier.
        clean_durations : list of float
            ``k`` durées, en secondes, de la part de parole *propre* de chaque
            locuteur local — les trames où il est actif et seul. C'est ce que
            ``rho_update`` filtre.

        Returns
        -------
        list of str
            ``k`` étiquettes globales, dans l'ordre des lignes reçues.

        Raises
        ------
        ValueError
            Si les deux entrées n'ont pas la même longueur.
        """
        from scipy.optimize import linear_sum_assignment

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings doit être (k, d), reçu {embeddings.shape}")
        if len(clean_durations) != len(embeddings):
            raise ValueError(f"{len(embeddings)} empreintes pour {len(clean_durations)} durées")
        if not len(embeddings):
            return []

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        locals_ = (embeddings / np.where(norms > 0, norms, 1.0)).astype(np.float32)

        labels: list[str | None] = [None] * len(locals_)
        taken: set[int] = set()
        created: set[int] = set()

        if self._globals:
            cost = cosine_distances(locals_, self.centroids())
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
                if cost[i, j] <= self.policy.delta_new:
                    labels[i] = self._globals[j].speaker_id
                    taken.add(j)

        # Les locaux restants : soit on découvre, soit — sous plafond — on se
        # rabat sur le centroïde libre le plus proche, pour que la contrainte
        # d'injectivité tienne même quand la création est interdite.
        for i, label in enumerate(labels):
            if label is not None:
                continue
            cap = self.policy.max_speakers
            if cap is None or len(self._globals) < cap:
                new = _Global(speaker_id=f"S{len(self._globals)}", vector=locals_[i])
                self._globals.append(new)
                labels[i] = new.speaker_id
                taken.add(len(self._globals) - 1)
                created.add(len(self._globals) - 1)
                self._updates += 1
                continue
            free = [j for j in range(len(self._globals)) if j not in taken]
            if not free:  # plus de locuteur disponible : la fenêtre est saturée
                labels[i] = self._globals[
                    int(np.argmin(cosine_distances(locals_[i : i + 1], self.centroids())))
                ].speaker_id
                continue
            distances = cosine_distances(locals_[i : i + 1], self.centroids()[free])[0]
            j = free[int(np.argmin(distances))]
            labels[i] = self._globals[j].speaker_id
            taken.add(j)

        # rho_update : seules les parts propres assez longues écrivent.
        index = {g.speaker_id: j for j, g in enumerate(self._globals)}
        for i, label in enumerate(labels):
            if clean_durations[i] < self.policy.rho_update_s:
                continue
            j = index[str(label)]
            if j in created:
                continue  # tout juste créé depuis cette empreinte
            g = self._globals[j]
            alpha = (
                self.policy.ema_alpha
                if self.policy.ema_alpha is not None
                else 1.0 / (g.n_updates + 1)
            )
            merged = (1.0 - alpha) * g.vector + alpha * locals_[i]
            norm = float(np.linalg.norm(merged))
            g.vector = (merged / (norm or 1.0)).astype(np.float32)
            g.n_updates += 1
            self._updates += 1

        return [str(label) for label in labels]


@dataclass
class _Window:
    """Une fenêtre glissante et ce que la segmentation locale y a vu."""

    t0: float
    t1: float
    local: list[tuple[float, float, str]]


class IncrementalDiarStage:
    """Diarisation en ligne par fenêtre glissante, segmenteur au choix.

    L'étage fait glisser une fenêtre sur le flux, fait segmenter *localement*
    chaque fenêtre par le backend qu'on lui donne, embarque la part propre de
    chaque locuteur local, puis confie le rattachement à
    :class:`IncrementalClusterer`. Ni le segmenteur ni l'encodeur ne sont
    imposés : c'est tout l'intérêt d'avoir porté l'algorithme plutôt que de
    dépendre de diart, dont le segmenteur local est câblé sur pyannote.

    Parameters
    ----------
    segmenter : str
        Backend de segmentation locale, passé à
        :class:`~vocal_helper.diar.OfflineDiarStage`. ``"auto"`` (défaut) prend
        le meilleur backend installé ; ``"sherpa"`` donne un chemin sans torch.
    embedder : str
        Backend d'empreinte vocale, passé à
        :class:`~vocal_helper.diar.OnlineDiarStage`. ``"auto"`` par défaut.
    policy : IncrementalPolicy, optional
        Les réglages du regroupement. Défauts de diart.
    window_s : float
        Longueur de la fenêtre glissante. 5 s, la valeur de référence.
    step_s : float
        Pas de la fenêtre. La méthode d'origine avance de 0,5 s parce que son
        segmenteur local est un petit modèle dédié ; les nôtres sont des
        diariseurs complets,
        donc le défaut est ici 2,5 s — un recouvrement de moitié, qui garde le
        vote de recouvrement sans rendre le RTF absurde.
    local_max_speakers : int
        Plafond passé au segmenteur local. 3, la limite usuelle d'un modèle de
        segmentation locale, et un plafond est ce qui empêche la prolifération
        d'étiquettes.
    min_embed_s : float
        Part propre minimale pour tenter une empreinte. Sous ce seuil, le
        locuteur local est ignoré : mieux vaut ne rien dire que de verser une
        empreinte de bruit dans le regroupement.
    """

    def __init__(
        self,
        *,
        segmenter: str = "auto",
        embedder: str = "auto",
        policy: IncrementalPolicy | None = None,
        window_s: float = 5.0,
        step_s: float = 2.5,
        local_max_speakers: int = 3,
        min_embed_s: float = 0.5,
    ) -> None:
        """Configurer l'étage ; les modèles se chargent au premier appel."""
        if window_s <= 0 or step_s <= 0:
            raise ValueError("window_s et step_s doivent être strictement positifs")
        if step_s > window_s:
            raise ValueError(f"step_s={step_s} dépasse window_s={window_s}")
        self.policy = policy or IncrementalPolicy()
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self.local_max_speakers = int(local_max_speakers)
        self.min_embed_s = float(min_embed_s)
        self._segmenter_backend = segmenter
        self._embedder_backend = embedder
        self._segmenter: object | None = None
        self._embedder: object | None = None
        self.clusterer = IncrementalClusterer(policy=self.policy)

    @property
    def discovered_speakers(self) -> int:
        """Nombre de locuteurs que le flux a découverts."""
        return self.clusterer.speaker_count

    @property
    def speaker_labels(self) -> list[str]:
        """Les étiquettes globales, dans l'ordre de découverte."""
        return self.clusterer.speaker_labels

    def _load(self) -> tuple[object, object]:
        """Charger segmenteur et encodeur une fois pour tout le flux."""
        if self._segmenter is None or self._embedder is None:
            from vocal_helper.diar import OfflineDiarStage, OnlineDiarStage

            self._segmenter = OfflineDiarStage(backend=self._segmenter_backend)
            online = OnlineDiarStage(backend=self._embedder_backend)
            online._ensure_embedder()
            self._embedder = online._embedder
        return self._segmenter, self._embedder

    def windows(self, duration_s: float) -> list[tuple[float, float]]:
        """Les bornes des fenêtres couvrant ``duration_s``.

        La dernière fenêtre est ramenée à la fin du flux plutôt que tronquée,
        pour que la fin de l'audio soit vue autant que le reste.

        Examples
        --------
        >>> stage = IncrementalDiarStage(window_s=5.0, step_s=2.5)
        >>> stage.windows(10.0)
        [(0.0, 5.0), (2.5, 7.5), (5.0, 10.0)]
        >>> stage.windows(6.0)
        [(0.0, 5.0), (1.0, 6.0)]
        >>> stage.windows(3.0)
        [(0.0, 3.0)]
        """
        if duration_s <= self.window_s:
            return [(0.0, float(duration_s))]
        starts, t = [], 0.0
        while t + self.window_s < duration_s - 1e-9:
            starts.append(round(t, 6))
            t += self.step_s
        starts.append(round(duration_s - self.window_s, 6))
        return [(s, round(s + self.window_s, 6)) for s in starts]

    def _clean_spans(
        self, local: list[tuple[float, float, str]], speaker: str
    ) -> list[tuple[float, float]]:
        """Les intervalles où ``speaker`` parle **seul** dans la fenêtre.

        C'est le remplaçant de l'empreinte pondérée par masque : nos
        encodeurs prennent un ``(pcm, sr)`` nu, donc on leur donne de l'audio
        propre au lieu d'une pondération.

        Examples
        --------
        >>> stage = IncrementalDiarStage()
        >>> local = [(0.0, 2.0, "A"), (1.5, 3.0, "B")]
        >>> stage._clean_spans(local, "A")
        [(0.0, 1.5)]
        >>> stage._clean_spans(local, "B")
        [(2.0, 3.0)]
        """
        others = [(a, b) for a, b, s in local if s != speaker]
        clean: list[tuple[float, float]] = []
        for a, b, s in local:
            if s != speaker:
                continue
            pieces = [(a, b)]
            for oa, ob in others:
                nxt = []
                for pa, pb in pieces:
                    if ob <= pa or oa >= pb:
                        nxt.append((pa, pb))
                        continue
                    if pa < oa:
                        nxt.append((pa, min(oa, pb)))
                    if pb > ob:
                        nxt.append((max(ob, pa), pb))
                pieces = nxt
            clean.extend(p for p in pieces if p[1] - p[0] > 1e-6)
        return sorted(clean)

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """Diariser un tampon entier par fenêtres glissantes.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Tampon mono.
        sr : int
            Sa fréquence d'échantillonnage.
        num_speakers, min_speakers : int, optional
            Acceptés pour la parité d'interface avec les autres étages. Le
            regroupement incrémental **devine** le compte par ``delta_new`` ;
            un compte exact n'a pas de sens au fil de l'eau, puisque les
            locuteurs arrivent les uns après les autres.
        max_speakers : int, optional
            Plafond dur, qui l'emporte sur celui de la politique. C'est le
            réglage utile : la mesure du 2026-11-01 montre qu'un plafond vaut
            le compte exact sur le moteur différé.

        Returns
        -------
        list of tuple
            ``[(t0, t1, locuteur), …]`` trié par début, la même forme que tous
            les autres étages de diarisation du paquet.
        """
        if max_speakers is not None:
            self.policy.max_speakers = int(max_speakers)
        segmenter, embedder = self._load()
        duration = len(pcm) / float(sr)
        grid_hz = 100.0
        n_frames = int(round(duration * grid_hz))
        votes: dict[str, NDArray[np.float32]] = {}

        for t0, t1 in self.windows(duration):
            chunk = pcm[int(t0 * sr) : int(t1 * sr)]
            if not len(chunk):
                continue
            local = segmenter.diarize(chunk, sr, max_speakers=self.local_max_speakers)
            if not local:
                continue
            vectors, durations, speakers = [], [], []
            for speaker in sorted({s for _, _, s in local}):
                spans = self._clean_spans(local, speaker)
                clean_s = sum(b - a for a, b in spans)
                if clean_s < self.min_embed_s:
                    continue
                audio = np.concatenate([chunk[int(a * sr) : int(b * sr)] for a, b in spans])
                vectors.append(np.asarray(embedder.embed(audio, sr), dtype=np.float32))
                durations.append(clean_s)
                speakers.append(speaker)
            if not vectors:
                continue
            global_ids = self.clusterer.assign(np.stack(vectors), durations)
            mapping = dict(zip(speakers, global_ids, strict=True))
            for a, b, s in local:
                gid = mapping.get(s)
                if gid is None:
                    continue
                row = votes.setdefault(gid, np.zeros(n_frames, dtype=np.float32))
                lo = max(0, int(round((t0 + a) * grid_hz)))
                hi = min(n_frames, int(round((t0 + b) * grid_hz)))
                if hi > lo:
                    row[lo:hi] += 1.0

        return _votes_to_segments(votes, grid_hz)


def _votes_to_segments(
    votes: dict[str, NDArray[np.float32]], grid_hz: float
) -> list[tuple[float, float, str]]:
    """Réduire les votes des fenêtres recouvrantes en segments.

    Chaque fenêtre vote pour un locuteur global sur les trames qu'elle couvre ;
    une trame revient au locuteur le plus voté, et le silence à personne. C'est
    ce vote qui absorbe les désaccords de bord entre fenêtres voisines.

    Examples
    --------
    >>> import numpy as np
    >>> votes = {"S0": np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32),
    ...          "S1": np.array([1.0, 0.0, 0.0, 3.0], dtype=np.float32)}
    >>> _votes_to_segments(votes, grid_hz=10.0)
    [(0.0, 0.2, 'S0'), (0.3, 0.4, 'S1')]
    """
    if not votes:
        return []
    labels = sorted(votes)
    grid = np.stack([votes[k] for k in labels])
    best = grid.argmax(axis=0)
    silent = grid.max(axis=0) <= 0.0
    segments: list[tuple[float, float, str]] = []
    start, current = 0, None
    for i in range(len(best) + 1):
        here = None if i == len(best) or silent[i] else labels[int(best[i])]
        if here != current:
            if current is not None:
                segments.append((start / grid_hz, i / grid_hz, current))
            start, current = i, here
    return segments
