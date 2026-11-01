"""Regroupement incrémental **sans centroïde** : des exemplaires, et le meilleur point.

Pourquoi abandonner le centroïde
--------------------------------
:class:`~vocal_helper.incremental.IncrementalClusterer` tient un centroïde par
locuteur, c'est-à-dire une moyenne mobile de ses empreintes. Une moyenne est un
**point**, et un point ne peut pas représenter un étalement. Or c'est l'étalement
qui pose problème sur du matériel réel : la même voix embarquée près du micro et
loin du micro, de face et de profil, dans le silence et sous le brouhaha, occupe
une région — parfois deux régions franchement séparées. Le centroïde se pose
alors entre les deux modes, là où le locuteur ne se trouve jamais, et il est à
la fois trop loin de ses propres observations et trop près de celles du voisin.

La mesure dit exactement cela. Sur AMI, avec le seuil qui minimise le taux
d'erreur, le regroupement à centroïdes laisse **vingt étiquettes pour quatre
locuteurs**, et aucun réglage de seuil ne le corrige — quatre paramétrages ont
été essayés et réfutés (voir le commentaire en tête de
:func:`~vocal_helper.incremental.separating_threshold`). Pendant ce temps,
l'espace d'empreintes lui-même est sain : la variation d'une même voix tourne
autour de 0,12 quand la distance entre voix différentes tourne autour de 0,83,
soit un facteur sept. Ce n'est donc pas l'empreinte qui limite, c'est sa
réduction à un point.

Ce que fait ce module
---------------------
Chaque locuteur n'est plus un point mais un **ensemble borné d'exemplaires**,
chacun gardé avec sa confiance. La ressemblance à un locuteur est la distance à
son **meilleur point**, pas à sa moyenne — un locuteur bimodal est alors
reconnu par celui de ses deux modes qui correspond, au lieu d'être manqué par
une moyenne qui n'en est aucun.

Quatre choses y gagnent par rapport au schéma à centroïdes :

1. **Pas de moyenne.** Une voix à deux régimes est représentée par les deux,
   pas par leur milieu.
2. **La fusion globale devient définissable.** Comparer deux ensembles
   d'exemplaires a un sens ; comparer deux moyennes mobiles n'en avait pas, car
   une moyenne a déjà perdu de quoi savoir si deux groupes se recouvrent. C'est
   ce qui permet de rattraper la sur-segmentation après coup, sans rien
   recalculer — le seul levier que le diagnostic laissait ouvert.
3. **La confiance remplace la moyenne mobile aveugle.** Un centroïde était mis
   à jour au même taux par une empreinte tirée de huit secondes de parole
   propre et par une tirée de quatre dixièmes. Ici la durée propre est la
   confiance, elle décide de l'admission et elle survit à l'éviction.
4. **Le chaînage est réglable.** Le meilleur point pur est un lien simple, donc
   sensible à un exemplaire aberrant qui ferait pont entre deux locuteurs.
   ``neighbours`` permet de demander le deuxième ou le troisième plus proche au
   lieu du premier, ce qui coupe le pont sans perdre la multimodalité.

Ce qui est conservé du schéma précédent, parce que c'est ce qui marchait : la
contrainte d'injectivité par algorithme hongrois, qui interdit à deux voix
simultanées de recevoir la même étiquette ; le plancher de confiance avant
d'écrire quoi que ce soit ; et la création explicite d'un locuteur au-delà d'un
seuil, qui est ce qui fait **deviner** le compte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["Exemplar", "ExemplarPolicy", "ExemplarClusterer"]


def _unit(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normaliser des lignes en L2, sans diviser par zéro."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms > 0, norms, 1.0)).astype(np.float32)


@dataclass
class Exemplar:
    """Une observation gardée, et ce qu'elle vaut.

    Attributes
    ----------
    vector : NDArray[np.float32]
        L'empreinte, normalisée L2.
    confidence : float
        Les secondes de parole **propre** dont elle a été tirée. C'est la seule
        mesure de qualité disponible sans vérité terrain, et elle est juste :
        une empreinte calculée sur huit secondes où le locuteur est seul vaut
        mieux qu'une calculée sur quatre dixièmes de seconde.
    """

    vector: NDArray[np.float32]
    confidence: float


@dataclass
class ExemplarPolicy:
    """Les réglages du regroupement par exemplaires.

    Attributes
    ----------
    rho_update_s : float
        Confiance minimale, en secondes de parole propre, pour qu'une
        observation soit gardée. En dessous, elle est quand même étiquetée —
        on ne perd personne — mais elle n'entre pas dans la mémoire du
        locuteur.
    delta_new : float
        Distance au meilleur point au-delà de laquelle on crée un locuteur
        plutôt que de forcer une jointure. Reste à 0,6 : trois réunions d'AMI
        donnent 0,750 / 0,600 / 0,600 comme optimum, et aucune normalisation
        n'a réduit cette dispersion.
    neighbours : int
        Combien de voisins regarder. 1 est le meilleur point pur, donc un lien
        simple, sensible à un exemplaire aberrant faisant pont entre deux
        locuteurs. 2 ou 3 coupent le pont tout en gardant la multimodalité.
    max_exemplars : int
        Taille de la mémoire par locuteur. C'est ce qui garde le coût borné en
        direct : la distance à un locuteur coûte au plus ``max_exemplars``
        produits scalaires.
    diversity_margin : float
        En dessous de cette distance à un exemplaire déjà gardé, une
        observation est un quasi-doublon : elle renforce la confiance de
        l'existant au lieu de prendre une place. C'est ce qui empêche la
        mémoire de se remplir de dix copies du même instant.
    max_speakers : int, optional
        Plafond dur du nombre de locuteurs. ``None`` laisse le compte libre,
        c'est-à-dire deviné.
    merge : bool
        Faire ou non la reprise globale. ``True`` par défaut, parce que c'est
        l'apport principal de la représentation par exemplaires. Le mettre à
        ``False`` sert à isoler les deux effets dans une mesure : l'absence de
        moyenne d'une part, la reprise globale de l'autre.
    merge_threshold : float, optional
        Distance entre deux ensembles d'exemplaires en dessous de laquelle
        :meth:`ExemplarClusterer.merge` les réunit. ``None`` reprend
        ``delta_new``, ce qui est cohérent : deux locuteurs plus proches que le
        seuil de création n'auraient jamais dû être deux.
    """

    rho_update_s: float = 0.4
    delta_new: float = 0.6
    neighbours: int = 1
    max_exemplars: int = 12
    diversity_margin: float = 0.10
    max_speakers: int | None = None
    merge: bool = True
    merge_threshold: float | None = None


@dataclass
class ExemplarClusterer:
    """Rattache les locuteurs locaux d'une fenêtre, sans jamais moyenner.

    L'interface est celle de
    :class:`~vocal_helper.incremental.IncrementalClusterer` — ``assign`` prend
    des empreintes et des durées propres, rend des étiquettes — donc les deux
    sont interchangeables dans
    :class:`~vocal_helper.incremental.IncrementalDiarStage`, et comparables à
    extraction identique.

    Examples
    --------
    Deux voix orthogonales ouvrent le flux, chacune reçoit son étiquette, et la
    correspondance tient quand elles se présentent dans l'autre sens.

    >>> import numpy as np
    >>> e0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    >>> e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    >>> c = ExemplarClusterer()
    >>> c.assign(np.stack([e0, e1]), [5.0, 5.0])
    ['S0', 'S1']
    >>> c.assign(np.stack([e1, e0]), [5.0, 5.0])
    ['S1', 'S0']

    Voici ce qu'un centroïde ne peut pas faire. La même voix revient dans un
    autre régime, assez loin pour être gardée comme second exemplaire mais
    assez près pour être reconnue :

    >>> other = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    >>> c.assign(other[None, :], [5.0])
    ['S0']
    >>> c.exemplar_counts
    [2, 1]

    Ce second régime est désormais reconnu **exactement**, parce que la mesure
    est la distance au meilleur point et non à une moyenne qui serait tombée
    entre les deux :

    >>> float(round(c.distances(other[None, :])[0, 0], 6))
    0.0

    Un quasi-doublon ne prend pas de place, il renforce la confiance :

    >>> c.assign((other + 0.01 * e0)[None, :], [5.0])
    ['S0']
    >>> c.exemplar_counts
    [2, 1]

    Et la fusion globale, que l'absence de moyenne rend définissable. Deux voix
    quasi identiques vues **en même temps** sont forcément séparées par la
    contrainte d'injectivité — c'est le mécanisme même de la sur-segmentation :

    >>> d = ExemplarClusterer()
    >>> near = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    >>> d.assign(np.stack([e0, near]), [5.0, 5.0])
    ['S0', 'S1']
    >>> d.merge()
    {'S1': 'S0'}
    >>> d.speaker_count
    1
    """

    policy: ExemplarPolicy = field(default_factory=ExemplarPolicy)
    _labels: list[str] = field(default_factory=list, init=False)
    _sets: list[list[Exemplar]] = field(default_factory=list, init=False)
    _minted: int = field(default=0, init=False)

    # ----- ce que le flux a découvert ---------------------------------

    @property
    def speaker_count(self) -> int:
        """Nombre de locuteurs découverts."""
        return len(self._labels)

    @property
    def speaker_labels(self) -> list[str]:
        """Les étiquettes, dans l'ordre de découverte."""
        return list(self._labels)

    @property
    def exemplar_counts(self) -> list[int]:
        """Combien d'exemplaires chaque locuteur garde."""
        return [len(s) for s in self._sets]

    # ----- la mesure --------------------------------------------------

    def distances(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Distance de chaque empreinte au **meilleur point** de chaque locuteur.

        Avec ``neighbours > 1``, c'est le k-ième plus proche et non le plus
        proche : un seul exemplaire aberrant ne suffit alors plus à rapprocher
        deux locuteurs.

        Parameters
        ----------
        embeddings : NDArray[np.float32]
            ``(k, d)``, pas nécessairement normalisées.

        Returns
        -------
        NDArray[np.float32]
            ``(k, m)`` de distances cosinus, ``m`` étant le nombre de locuteurs.
        """
        unit = _unit(np.atleast_2d(embeddings))
        if not self._sets:
            return np.zeros((len(unit), 0), dtype=np.float32)
        out = np.empty((len(unit), len(self._sets)), dtype=np.float32)
        for j, exemplars in enumerate(self._sets):
            stored = np.stack([e.vector for e in exemplars])
            every = 1.0 - unit @ stored.T  # (k, n_exemplaires)
            rank = min(self.policy.neighbours, every.shape[1]) - 1
            out[:, j] = np.partition(every, rank, axis=1)[:, rank]
        return out

    # ----- la mémoire -------------------------------------------------

    def _admit(self, index: int, vector: NDArray[np.float32], confidence: float) -> None:
        """Garder une observation chez un locuteur, si elle apporte quelque chose."""
        exemplars = self._sets[index]
        stored = np.stack([e.vector for e in exemplars])
        gaps = 1.0 - stored @ vector
        nearest = int(np.argmin(gaps))
        if float(gaps[nearest]) < self.policy.diversity_margin:
            # Quasi-doublon : il ne dit rien de nouveau sur l'étalement de cette
            # voix, mais il confirme ce qu'on en sait. On garde la meilleure
            # confiance des deux plutôt qu'une place de mémoire.
            exemplars[nearest].confidence = max(exemplars[nearest].confidence, float(confidence))
            return
        exemplars.append(Exemplar(vector=vector, confidence=float(confidence)))
        if len(exemplars) <= self.policy.max_exemplars:
            return
        # Mémoire pleine : évincer le plus redondant — celui dont le voisin le
        # plus proche dans l'ensemble est le plus proche — et départager par la
        # confiance la plus faible. Évincer le plus ancien jetterait au hasard,
        # et évincer le plus isolé détruirait précisément l'étalement qu'on
        # cherche à garder.
        vectors = np.stack([e.vector for e in exemplars])
        pairwise = 1.0 - vectors @ vectors.T
        np.fill_diagonal(pairwise, np.inf)
        closeness = pairwise.min(axis=1)
        worst = min(
            range(len(exemplars)),
            key=lambda i: (float(closeness[i]), exemplars[i].confidence),
        )
        exemplars.pop(worst)

    # ----- le rattachement --------------------------------------------

    def assign(self, embeddings: NDArray[np.float32], clean_durations: list[float]) -> list[str]:
        """Étiqueter les locuteurs locaux d'une fenêtre.

        Parameters
        ----------
        embeddings : NDArray[np.float32]
            ``(k, d)`` — une empreinte par locuteur local actif.
        clean_durations : list of float
            ``k`` durées de parole propre, qui servent de confiance.

        Returns
        -------
        list of str
            ``k`` étiquettes, dans l'ordre des lignes reçues.

        Raises
        ------
        ValueError
            Si les deux entrées n'ont pas la même longueur, ou si les
            empreintes ne sont pas une matrice.
        """
        from scipy.optimize import linear_sum_assignment

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings doit être (k, d), reçu {embeddings.shape}")
        if len(clean_durations) != len(embeddings):
            raise ValueError(f"{len(embeddings)} empreintes pour {len(clean_durations)} durées")
        if not len(embeddings):
            return []

        unit = _unit(embeddings)
        labels: list[str | None] = [None] * len(unit)
        taken: set[int] = set()
        fresh: set[int] = set()

        if self._sets:
            cost = self.distances(unit)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
                if cost[i, j] <= self.policy.delta_new:
                    labels[i] = self._labels[j]
                    taken.add(j)

        for i, label in enumerate(labels):
            if label is not None:
                continue
            cap = self.policy.max_speakers
            if cap is None or len(self._labels) < cap:
                name = f"S{self._minted}"
                self._minted += 1
                self._labels.append(name)
                self._sets.append([Exemplar(vector=unit[i], confidence=float(clean_durations[i]))])
                labels[i] = name
                taken.add(len(self._labels) - 1)
                fresh.add(len(self._labels) - 1)
                continue
            # Sous plafond, se rabattre sur le locuteur libre le plus proche :
            # la contrainte d'injectivité doit tenir même quand la création est
            # interdite.
            free = [j for j in range(len(self._labels)) if j not in taken]
            row = self.distances(unit[i : i + 1])[0]
            if not free:  # fenêtre saturée : personne n'est perdu pour autant
                labels[i] = self._labels[int(np.argmin(row))]
                continue
            j = free[int(np.argmin(row[free]))]
            labels[i] = self._labels[j]
            taken.add(j)

        where = {name: j for j, name in enumerate(self._labels)}
        for i, label in enumerate(labels):
            if clean_durations[i] < self.policy.rho_update_s:
                continue
            j = where[str(label)]
            if j in fresh:
                continue  # l'exemplaire vient d'être créé depuis cette empreinte
            self._admit(j, unit[i], clean_durations[i])

        return [str(label) for label in labels]

    # ----- la reprise globale -----------------------------------------

    def merge(self) -> dict[str, str]:
        """Réunir les locuteurs que la mémoire révèle identiques.

        C'est ce que l'absence de moyenne rend possible. Deux ensembles
        d'exemplaires se comparent — on regarde si leurs observations se
        recouvrent — là où deux moyennes mobiles avaient déjà perdu de quoi le
        savoir. La passe est exacte, ne recalcule aucune empreinte, et répare
        après coup la sur-segmentation que la décision locale ne pouvait pas
        éviter : la contrainte d'injectivité **doit** séparer deux locuteurs
        locaux simultanés, même quand la segmentation s'est trompée et qu'il
        n'y en avait qu'un.

        Returns
        -------
        dict
            ``{ancienne étiquette: étiquette survivante}`` pour les locuteurs
            absorbés. Vide si rien ne fusionne. À l'appelant de réécrire ses
            propres sorties avec cette table.
        """
        if not self.policy.merge:
            return {}
        threshold = (
            self.policy.merge_threshold
            if self.policy.merge_threshold is not None
            else self.policy.delta_new
        )
        parent = list(range(len(self._labels)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(len(self._sets)):
            for b in range(a + 1, len(self._sets)):
                left = np.stack([e.vector for e in self._sets[a]])
                right = np.stack([e.vector for e in self._sets[b]])
                across = (1.0 - left @ right.T).ravel()
                rank = min(self.policy.neighbours, across.size) - 1
                if float(np.partition(across, rank)[rank]) <= threshold:
                    ra, rb = root(a), root(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

        mapping: dict[str, str] = {}
        survivors: list[str] = []
        sets: list[list[Exemplar]] = []
        moved: dict[int, int] = {}
        for i, name in enumerate(self._labels):
            r = root(i)
            if r == i:
                moved[i] = len(survivors)
                survivors.append(name)
                sets.append(list(self._sets[i]))
        for i, name in enumerate(self._labels):
            r = root(i)
            if r != i:
                mapping[name] = self._labels[r]
                sets[moved[r]].extend(self._sets[i])

        if not mapping:
            return {}

        self._labels, self._sets = survivors, sets
        # Les ensembles réunis peuvent dépasser la mémoire : les ramener à la
        # taille en évinçant les plus redondants, pas les plus récents.
        for j, exemplars in enumerate(self._sets):
            while len(exemplars) > self.policy.max_exemplars:
                vectors = np.stack([e.vector for e in exemplars])
                pairwise = 1.0 - vectors @ vectors.T
                np.fill_diagonal(pairwise, np.inf)
                closeness = pairwise.min(axis=1)
                worst = min(
                    range(len(exemplars)),
                    key=lambda i, c=closeness, ex=exemplars: (float(c[i]), ex[i].confidence),
                )
                exemplars.pop(worst)
            self._sets[j] = exemplars
        return mapping
