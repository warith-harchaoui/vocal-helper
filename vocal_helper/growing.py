"""Diariser le tampon entier, périodiquement : la qualité du différé, en direct.

Le raisonnement
---------------
Deux mesures se combinent mieux qu'il n'y paraît. D'une part l'identité voyage
très bien par **recouvrement temporel** entre deux diarisations qui partagent
de l'audio. D'autre part le diariseur différé coûte **RTF 0,039** : une réunion
d'une heure se diarise en deux minutes.

Un découpage en morceaux a besoin d'un amorçage synthétique pour transporter
l'identité, parce qu'un locuteur peut être absent de la zone commune aux deux
diarisations. Si la seconde passe porte sur **tout l'audio depuis le début**, ce
cas disparaît : quiconque a parlé est dans la zone commune, qui couvre alors la
quasi-totalité du signal. L'appariement devient presque exact, aucun son
synthétique n'est nécessaire, et surtout **la dernière passe est une
diarisation du fichier entier** — donc la qualité du différé, atteinte en
direct.

Le coût se calcule et se mesure. Rafraîchir aux instants ``t``, ``r·t``,
``r²·t``… diarise au total un petit multiple de la durée, donc le coût reste
proportionnel à la séance et non à son carré : 2,6 à 3,3 fois l'audio mesuré,
pour un facteur temps réel de 0,10 à 0,16.

Ce que ça donne
---------------
Sur les huit réunions d'AMI dev-slice, le résultat final est **exactement**
celui du différé pleine fenêtre, à la troisième décimale — taux d'erreur médian
0,176, le bon nombre de locuteurs, et le meilleur accord longue portée mesuré.
Pour comparaison sur le même corpus : l'étage en ligne glouton donne 0,369 avec
soixante-cinq étiquettes pour quatre locuteurs, et le regroupement incrémental
0,254 avec vingt-quatre.

Ce que ça coûte
---------------
Pas la qualité, la **fraîcheur** : la parole postérieure à la dernière passe
n'a pas encore d'étiquette faisant autorité. Un espacement géométrique rend ce
retard proportionnel à la séance — 0,375 fois sa durée à ``ratio=1.6`` — ce qui
va pour vingt minutes et non pour deux heures ; ``max_period_s`` le borne au
prix d'un facteur temps réel plus élevé. Et les étiquettes se **révisent** :
c'est le prix de l'exactitude, et :class:`~vocal_helper.LabelRevision` le dit à
l'appelant plutôt que de le cacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vocal_helper.rediar import LabelRevision

__all__ = ["GrowingBufferPolicy", "GrowingBufferDiarStage", "map_by_overlap", "refresh_times"]


def map_by_overlap(
    left: list[tuple[float, float, str]], right: list[tuple[float, float, str]]
) -> dict[str, str]:
    """La correspondance un-pour-un entre deux diarisations, par temps partagé.

    Les noms de grappes sont arbitraires et interchangeables ; ce qui doit être
    conservé d'une passe à l'autre, c'est **qui est qui**. C'est un appariement
    bijectif, donc résolu par algorithme hongrois et non par un argmax par
    étiquette : un argmax peut envoyer deux étiquettes de gauche sur la même à
    droite, ce qui fusionnerait deux locuteurs sans le dire.

    Parameters
    ----------
    left, right : list of tuple
        Deux diarisations ``(t0, t1, étiquette)`` sur la même horloge.

    Returns
    -------
    dict of str
        ``{étiquette de gauche: étiquette de droite}``, seulement pour les
        paires qui partagent effectivement du temps.

    Examples
    --------
    >>> map_by_overlap([(0.0, 2.0, "a"), (2.0, 4.0, "b")],
    ...                [(0.0, 2.0, "y"), (2.0, 4.0, "x")])
    {'a': 'y', 'b': 'x'}

    Rien de partagé, rien d'apparié — plutôt qu'un rapprochement inventé.

    >>> map_by_overlap([(0.0, 1.0, "a")], [(9.0, 10.0, "z")])
    {}
    """
    from scipy.optimize import linear_sum_assignment

    rows = sorted({label for _, _, label in left})
    cols = sorted({label for _, _, label in right})
    if not rows or not cols:
        return {}
    shared = np.zeros((len(rows), len(cols)), dtype=np.float64)
    row_at = {label: i for i, label in enumerate(rows)}
    col_at = {label: j for j, label in enumerate(cols)}
    for a0, a1, a in left:
        for b0, b1, b in right:
            overlap = min(a1, b1) - max(a0, b0)
            if overlap > 0:
                shared[row_at[a], col_at[b]] += overlap
    out: dict[str, str] = {}
    for i, j in zip(*linear_sum_assignment(-shared), strict=True):
        if shared[i, j] > 0:
            out[rows[int(i)]] = cols[int(j)]
    return out


def refresh_times(
    duration_s: float, *, first_s: float, ratio: float, max_period_s: float | None = None
) -> list[float]:
    """Les instants de rafraîchissement, espacés géométriquement.

    Un pas constant ferait croître le coût comme le **carré** de la durée,
    puisque chaque passe rediarise tout depuis le début. Un pas géométrique le
    garde proportionnel : la somme des préfixes reste un petit multiple de la
    durée totale. ``max_period_s`` borne l'attente au prix d'un coût plus élevé,
    pour une séance longue où un retard proportionnel serait inacceptable.

    Examples
    --------
    >>> refresh_times(100.0, first_s=10.0, ratio=2.0)
    [10.0, 20.0, 40.0, 80.0, 100.0]

    Plus court que la première passe : une seule, à la fin.

    >>> refresh_times(8.0, first_s=10.0, ratio=2.0)
    [8.0]

    Avec un plafond, l'attente ne grandit plus indéfiniment.

    >>> refresh_times(100.0, first_s=10.0, ratio=2.0, max_period_s=25.0)
    [10.0, 20.0, 40.0, 65.0, 90.0, 100.0]
    """
    if duration_s <= 0:
        return []
    out: list[float] = []
    t = first_s
    while t < duration_s:
        out.append(round(t, 6))
        step = t * (ratio - 1.0)
        if max_period_s is not None:
            step = min(step, max_period_s)
        t += max(step, 1e-6)
    out.append(round(duration_s, 6))
    return out


@dataclass
class GrowingBufferPolicy:
    """Quand rafraîchir, et jusqu'où laisser le retard grandir.

    Attributes
    ----------
    first_s : float
        Instant de la première passe. En dessous, le diariseur n'a pas de quoi
        travailler et rendrait un seul locuteur.
    ratio : float
        Facteur d'espacement. 1,6 mesure un coût de 2,6 à 3,3 fois l'audio pour
        un retard valant 0,375 fois la séance ; descendre vers 1,2 raccourcit le
        retard et monte le coût.
    max_period_s : float, optional
        Attente maximale entre deux passes. ``None`` laisse le retard croître
        avec la séance, ce qui va pour une réunion et non pour une journée.
    min_speaker_s : float
        Temps de parole minimal pour qu'une étiquette soit publiée. Écarte les
        locuteurs d'une seconde que le diariseur invente sur un bruit.
    gap_s : float
        Silence considéré comme une séparation de tours, pour récolter le bruit
        de fond qui reconstitue les creux entre segments voisés.
    """

    first_s: float = 60.0
    ratio: float = 1.6
    max_period_s: float | None = None
    min_speaker_s: float = 0.0
    gap_s: float = 0.8

    def times(self, duration_s: float) -> list[float]:
        """Les instants de rafraîchissement pour cette durée."""
        return refresh_times(
            duration_s,
            first_s=self.first_s,
            ratio=self.ratio,
            max_period_s=self.max_period_s,
        )

    def worst_staleness_s(self, duration_s: float) -> float:
        """Le plus long délai avant qu'un passage reçoive son étiquette définitive.

        Examples
        --------
        >>> GrowingBufferPolicy(first_s=60.0, ratio=1.6).worst_staleness_s(1200.0)
        377.48736
        """
        times = self.times(duration_s)
        if not times:
            return 0.0
        gaps = [b - a for a, b in zip([0.0, *times], times, strict=False)]
        return round(max(gaps), 6)


@dataclass
class GrowingBufferDiarStage:
    """Diarisation en direct par re-diarisation périodique du tampon entier.

    L'étage accumule l'audio et, aux instants que la politique fixe, le fait
    diariser **en entier** par un moteur différé. Les étiquettes de chaque passe
    sont rattachées à celles de la précédente par recouvrement temporel, de
    sorte que le lecteur voie des identités stables ; les changements sont
    rendus comme des révisions explicites.

    Parameters
    ----------
    backend : str
        Moteur différé, passé à :class:`~vocal_helper.OfflineDiarStage`.
        ``"auto"`` prend le meilleur installé.
    policy : GrowingBufferPolicy, optional
        Le calendrier des passes.
    offline : object, optional
        Un diariseur déjà construit, qui doit exposer
        ``diarize(pcm, sr) -> [(t0, t1, étiquette), …]``. Sert aux tests et
        permet de brancher un autre moteur sans toucher à cette classe.
    online : object, optional
        L'étage en ligne qui donne l'étiquette **provisoire**, en une demi-
        seconde, le temps que la première passe arrive. ``None`` ne publie rien
        avant la première passe.
    on_revision : callable, optional
        Appelée pour chaque étiquette démentie par une passe. Sans elle, les
        corrections sont calculées et perdues.

    Examples
    --------
    Un diariseur de pacotille suffit à montrer le mécanisme : les étiquettes
    qu'il rend changent de nom à chaque passe, et l'étage les stabilise.

    >>> import numpy as np
    >>> class Flighty:
    ...     def __init__(self): self.calls = 0
    ...     def diarize(self, pcm, sr, **kw):
    ...         self.calls += 1
    ...         tag = f"local{self.calls}"
    ...         return [(0.0, len(pcm) / sr, tag)]
    >>> stage = GrowingBufferDiarStage(
    ...     offline=Flighty(), policy=GrowingBufferPolicy(first_s=1.0, ratio=2.0)
    ... )
    >>> segments = stage.diarize(np.zeros(16000 * 4, dtype=np.float32), 16000)
    >>> [label for _, _, label in segments]
    ['P0']
    >>> stage.speaker_labels
    ['P0']

    Trois passes ont eu lieu, et le locuteur n'a été découvert qu'une fois.

    >>> stage.passes
    3
    """

    backend: str = "auto"
    policy: GrowingBufferPolicy = field(default_factory=GrowingBufferPolicy)
    offline: Any = None
    online: Any = None
    on_revision: Any = None
    _labels: list[str] = field(default_factory=list, init=False)
    _published: list[tuple[float, float, str]] = field(default_factory=list, init=False)
    _passes: int = field(default=0, init=False)
    _minted: int = field(default=0, init=False)

    # ----- ce que le flux a découvert ---------------------------------

    @property
    def speaker_labels(self) -> list[str]:
        """Les locuteurs publiés, dans l'ordre de découverte."""
        return list(self._labels)

    @property
    def discovered_speakers(self) -> int:
        """Combien de locuteurs ont été découverts."""
        return len(self._labels)

    @property
    def passes(self) -> int:
        """Combien de passes de re-diarisation ont eu lieu."""
        return self._passes

    def segments(self) -> list[tuple[float, float, str]]:
        """La diarisation publiée, telle que la dernière passe l'a laissée."""
        return list(self._published)

    def reset(self) -> None:
        """Oublier le flux, en gardant la configuration."""
        self._labels, self._published = [], []
        self._passes = self._minted = 0

    # ----- le moteur --------------------------------------------------

    def _engine(self) -> Any:
        """Le diariseur différé, construit une fois pour tout le flux."""
        if self.offline is None:
            from vocal_helper.diar import OfflineDiarStage

            self.offline = OfflineDiarStage(backend=self.backend)
        return self.offline

    # ----- une passe --------------------------------------------------

    def refresh(self, pcm: NDArray[np.float32], sr: int, **hints: Any) -> list[LabelRevision]:
        """Rediariser tout le tampon, et rendre ce que ça change.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            **Tout** l'audio depuis le début, pas seulement le nouveau.
        sr : int
            Sa fréquence.
        **hints
            Transmis au moteur différé, par exemple ``max_speakers``.

        Returns
        -------
        list of LabelRevision
            Les passages dont l'étiquette change par rapport à ce qui avait été
            publié. Vide à la première passe, puisque rien n'avait été dit.
        """
        raw = self._engine().diarize(pcm, sr, **hints)
        self._passes += 1
        if not raw:
            return []

        # Rattacher les étiquettes de cette passe à celles déjà publiées, sur
        # leur préfixe commun — qui est ici presque tout l'audio, d'où un
        # appariement quasi exact.
        bridge = map_by_overlap(raw, self._published) if self._published else {}
        mapping: dict[str, str] = {}
        for label in sorted({name for _, _, name in raw}):
            carried = bridge.get(label)
            if carried is not None:
                mapping[label] = carried
            else:
                mapping[label] = f"P{self._minted}"
                self._minted += 1

        renamed = sorted((t0, t1, mapping[name]) for t0, t1, name in raw)
        if self.policy.min_speaker_s > 0:
            spoken: dict[str, float] = {}
            for t0, t1, name in renamed:
                spoken[name] = spoken.get(name, 0.0) + (t1 - t0)
            renamed = [row for row in renamed if spoken[row[2]] >= self.policy.min_speaker_s]

        revisions = self._revisions(renamed)
        self._published = renamed
        self._labels = list(dict.fromkeys(label for _, _, label in renamed))
        return revisions

    def _revisions(self, fresh: list[tuple[float, float, str]]) -> list[LabelRevision]:
        """Ce que cette passe contredit de ce qui avait été publié."""
        out: list[LabelRevision] = []
        for t0, t1, was in self._published:
            best, share = None, 0.0
            for b0, b1, now in fresh:
                overlap = min(t1, b1) - max(t0, b0)
                if overlap > share:
                    best, share = now, overlap
            if best is not None and best != was:
                out.append(LabelRevision(t0, t1, was, best))
        return out

    # ----- le flux entier ---------------------------------------------

    def diarize(
        self,
        pcm: NDArray[np.float32],
        sr: int,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """Dérouler un tampon comme un direct, et rendre l'état final.

        Utile pour mesurer : la dernière passe voit tout, donc le résultat est
        celui du différé pleine fenêtre, obtenu en ayant payé le prix du
        direct. Pour un vrai flux, appeler :meth:`refresh` aux instants que
        :meth:`GrowingBufferPolicy.times` indique et lire les révisions.

        Parameters
        ----------
        pcm : NDArray[np.float32]
            Tampon mono.
        sr : int
            Sa fréquence.
        num_speakers, min_speakers, max_speakers : int, optional
            Transmis au moteur différé à chaque passe.

        Returns
        -------
        list of tuple
            ``[(t0, t1, locuteur), …]``, la même forme que les autres étages.
        """
        hints = {
            name: value
            for name, value in (
                ("num_speakers", num_speakers),
                ("min_speakers", min_speakers),
                ("max_speakers", max_speakers),
            )
            if value is not None
        }
        duration = len(pcm) / float(sr)
        for upto in self.policy.times(duration):
            self.refresh(pcm[: int(upto * sr)], sr, **hints)
        return self.segments()

    # ----- la surface de flux -----------------------------------------

    async def run(self, inbox, outbox) -> None:  # noqa: ANN001 - file d'attente, typée par usage
        """Consommer des segments voisés, publier aussitôt, corriger ensuite.

        Même contrat que l'étage en ligne, donc c'est un remplacement direct :
        un consommateur qui lit ``outbox`` n'a rien à changer. Ce qui est
        nouveau, c'est qu'une étiquette déjà publiée peut être démentie, et
        ``on_revision`` le dit — un affichage qui ignore les révisions montre
        des étiquettes dont on lui a signalé qu'elles sont fausses.

        L'étage garde **tout** l'audio voisé, puisque chaque passe rediarise
        depuis le début : à 16 kHz en flottant simple cela fait 64 ko par
        seconde de parole, soit environ 230 Mo pour une heure de réunion. C'est
        le prix de l'exactitude, et il est borné par la durée de la séance.

        Parameters
        ----------
        inbox, outbox
            Files d'attente de segments voisés puis diarisés, comme pour les
            autres étages.
        """
        from vocal_helper.rediar import harvest_room_tone, reconstruct_window

        voiced: list[tuple[float, float, NDArray[np.float32]]] = []
        sample_rate = 16000
        room: NDArray[np.float32] | None = None
        done: set[float] = set()

        def rebuild(upto: float) -> NDArray[np.float32]:
            """Le tampon depuis le début, silence reconstitué entre les tours."""
            nonlocal room
            kept = [v for v in voiced if v[0] < upto]
            if room is None and kept:
                room = harvest_room_tone(kept[0][2], sample_rate, self.policy.gap_s)
            return reconstruct_window(
                kept,
                room if room is not None else np.zeros(1, dtype=np.float32),
                sample_rate,
                window_start_s=0.0,
            )

        while True:
            item = await inbox.get()
            if item is None:
                # Une dernière passe avant de fermer : la fin de la séance n'a
                # jamais été rediarisée, et c'est ce que le lecteur regarde.
                if voiced:
                    self._emit(self.refresh(rebuild(voiced[-1][1]), sample_rate))
                await outbox.put(None)
                return
            sample_rate = int(item["sample_rate"])
            voiced.append((item["t0"], item["t1"], item["pcm"]))
            if self.online is not None:
                labelled = self.online._label(item)
                if labelled is not None:
                    await outbox.put(labelled)
            for upto in self.policy.times(item["t1"]):
                if upto <= item["t1"] and upto not in done:
                    done.add(upto)
                    self._emit(self.refresh(rebuild(upto), sample_rate))

    def _emit(self, revisions: list[LabelRevision]) -> None:
        """Signaler les révisions, si quelqu'un écoute."""
        if self.on_revision is None:
            return
        for revision in revisions:
            self.on_revision(revision)
