# Diarisation : ce qu'on a mesuré, et ce qu'on livre

*2026-11-02. Pendant de [`diarization-roadmap.md`](diarization-roadmap.md), qui posait le problème
le 2026-10-20. Celui-ci y répond par des chiffres, y compris les réponses qui se sont révélées
fausses.*

## D'où on part, où on en est

| | gratuit local, direct | gratuit local, différé |
|---|---:|---:|
| **avant** | 0,673 d'erreur, **1 étiquette** pour 4 locuteurs | 0,299 d'erreur, 5 étiquettes |
| **après** | **0,090**, 4 étiquettes, 0,000 de confusion | **0,090**, 4 étiquettes |

Le même moteur dans les deux modes : **Nemotron 3 Diarization**, par son graphe ONNX, sans torch
sur le chemin de décision, sans Hugging Face à l'exécution et sans jeton. Les poids voyagent dans
le bundle de diarisation que le projet sert déjà.

Contre une référence **payante** — des émissions de télévision françaises déjà diarisées par
pyannoteAI, lues dans le cache du portefeuille — sur les trois seules émissions où ce service
trouve huit locuteurs ou moins :

| moteur | émission 1 *(17 min, 8 locuteurs payés)* | émission 2 *(20 min, 7)* |
|---|---:|---:|
| **nemotron** | **0,017** *(8 étiquettes)* | **0,072** *(6)* |
| pyannote community-1 | 0,059 *(7)* | 0,155 *(7)* |
| nemo sortformer | 0,128 *(10)* | 0,108 *(13)* |
| sherpa | 0,447 *(29)* | 0,551 *(**102**)* |

Ce chiffre doit être lu pour ce qu'il est : un **désaccord avec un service payant**, et non un taux
d'erreur absolu. pyannoteAI n'est pas la vérité — son propre taux est de l'ordre de 0,10 à 0,20, et
deux moteurs peuvent s'en écarter en ayant tous les deux raison. C'est en revanche la question que
se pose quelqu'un avant de décider de payer.

## Le diagnostic de la feuille de route était juste : compter, pas le DER

Elle disait que le problème n'était pas un taux d'erreur trop élevé mais que **rien ne savait
compter les locuteurs**, et qu'une métrique pondérée par la durée cache exactement cela. Tout ce
qui a été mesuré depuis le confirme, et la colonne du nombre d'étiquettes a été la plus
informative de tous les tableaux. Deux illustrations :

- Le regroupement glouton livré comme défaut du direct se trompe dans les **deux** sens selon la
  matière : une étiquette pour quatre locuteurs sur du synthétique propre, quarante-quatre à
  soixante-huit sur des réunions réelles. Aucun seuil ne répare ça, puisque la valeur qui empêche
  l'explosion provoque l'effondrement.
- sherpa, le chemin sans torch, émet **102 étiquettes pour 7 locuteurs** sur de l'audio de
  télévision réel. Il est dernier de l'ordre de préférence, mais sa documentation le présentait
  comme « portable et un peu moins bon », ce qui est faux : sur du réel il est inutilisable.

## Quatre paramétrages de seuil, quatre réfutations

Avant de changer de modèle, on a cherché à faire tenir l'identité au regroupement existant. Chaque
tentative était une façon de décider *quand deux voix sont la même voix* :

| paramétrage | pourquoi ça devait marcher | ce que la mesure dit |
|---|---|---|
| seuil cosinus absolu | le plus simple | 0,8 est optimal sur une réunion et fusionne deux vrais locuteurs sur une autre |
| fraction de l'inter-locuteurs médian | la salle change l'échelle des distances | les deux réunions ont la **même** médiane, 0,834 et 0,847 |
| fraction d'un quantile bas | c'est la paire confusable qui décide | les sépare correctement (0,447 contre 0,287) mais sur-corrige du double |
| position entre l'intra et l'inter | plus aucune constante choisie à la main | dispersion de 0,213 contre 0,150 pour les seuils bruts |

Sur trois réunions, les seuils optimaux valent 0,750 / 0,600 / 0,600 et les positions normalisées
correspondantes 0,879 / 0,666 / 0,687. **Normaliser élargit la dispersion.** Avec le recul, c'était
l'indice : on réglait une décision qu'un modèle devrait prendre. Nemotron n'a pas de seuil — il
tient un cache de locuteurs par ordre d'arrivée et **apprend** l'identité.

Les relevés sont dans `studies/diar_incremental_delta_sweep.py` et
`studies/diar_incremental_scale_calibration.py`, et la réfutation est inscrite dans le module pour
que personne ne la retente à l'aveugle.

## Le regroupement sans centroïde : une branche fermée

Le centroïde d'un locuteur est une moyenne, donc un **point**, et un point ne peut pas représenter
un étalement — or c'est bien ce qu'est une voix près et loin d'un micro. On l'a donc remplacé par
un ensemble borné d'exemplaires, la ressemblance étant mesurée au **meilleur point**. Quatre règles
de référence et cinq seuils de fusion plus tard, le centroïde gagne sur toutes les métriques, pire
cas compris.

La moyenne joue en fait un rôle de **régularisateur** qu'on sous-estimait, et le meilleur point est
un estimateur *optimiste* : un minimum sur un ensemble diminue mécaniquement quand l'ensemble
grandit, si bien qu'un seul exemplaire bruité suffit à capturer un locuteur. Demander le k-ième
plus proche au lieu du plus proche récupère l'essentiel de l'écart (0,529 → 0,363) sans le combler.
`vocal_helper/exemplar.py` garde l'implémentation et le verdict ; ce n'est pas le défaut et ça ne
doit pas le devenir sans élément nouveau.

## L'identité peut voyager par le temps seul

Un mécanisme à consigner parce qu'il fonctionne et parce qu'il a produit deux corrections de bugs
réels. Le morceau courant est diarisé **deux fois** : une fois seul, une fois précédé d'un amorçage
synthétique où chaque locuteur connu occupe un créneau de position *connue*. Relire quelle grappe
couvre quel créneau nomme les grappes ; le recouvrement temporel transporte ensuite ces noms vers la
première diarisation. Aucune géométrie d'empreintes nulle part.

Deux mesures l'ont cadré. Découper coûte peu — 0,025 à 0,051 d'erreur — donc travailler morceau par
morceau est légitime. Mais le nommage coûtait 0,111, remarquablement stable d'une réunion à
l'autre, donc d'une seule cause. Elle était grossière : **les références étaient gelées à la
première vue**, si bien que les premières secondes jamais entendues d'un locuteur le définissaient
pour toute la séance, y compris récoltées sur une grappe déjà fusionnée. Dé-gelées, la perte de
nommage tombe à **0,003** — le dispositif atteint son propre plafond.

Et la relecture de l'amorçage faisait un argmax glouton là où c'est une **bijection**. Dès que le
moteur fusionnait deux locuteurs de l'amorçage, une étiquette gagnait deux créneaux et la seconde
écriture **écrasait** la première : un locuteur connu perdait son étiquette en silence et revenait
comme nouveau venu. Le dispositif inventait des locuteurs précisément là où il devait en
reconnaître. C'est une affectation optimale désormais, et `read_back_identities` explique pourquoi.

## La qualité du différé, en direct : la fenêtre croissante

Poussée au bout, cette logique rend l'amorçage inutile. Il n'existe que pour un cas : un locuteur
absent de la zone que les deux diarisations partagent. Si la seconde passe porte sur **tout l'audio
depuis le début**, ce cas disparaît — quiconque a parlé est dans la zone commune, qui couvre alors
la quasi-totalité du signal.

La dernière passe est donc une diarisation du fichier entier, et la réponse finale **est** celle du
différé. Mesuré sur les huit réunions d'AMI dev-slice : 0,176 en médiane, le bon nombre de
locuteurs, à la troisième décimale. L'espacement géométrique garde le coût proportionnel à la
séance et non à son carré — 2,6 à 3,3 fois l'audio, facteur temps réel de 0,10 à 0,16.

Ce que ça coûte n'est pas la qualité mais la **fraîcheur** : la parole postérieure à la dernière
passe n'a pas encore d'étiquette faisant autorité, et ce retard vaut 0,375 fois la séance au réglage
par défaut — bon pour vingt minutes, pas pour deux heures, où `max_period_s` le borne.
`voh.GrowingBufferDiarStage` le livre, et il reste le meilleur chemin sans modèle supplémentaire,
ainsi que le seul mesuré sur un enregistrement réel avec une référence humaine.

## AMI ne peut pas arbitrer, et on l'a prouvé au lieu de le supposer

Nemotron arrive **dernier** sur AMI — 0,487 contre 0,192 pour pyannote — tout en affichant une
confusion de locuteurs de 0,003 contre 0,036. Ça ne se tient pas, et tout l'écart est de la parole
qu'il n'émet pas : 616 secondes sur les 815 annotées. Quatre explications ont été testées :

| hypothèse | test | verdict |
|---|---|---|
| seuil d'extraction | balayé jusqu'à 0,05 | le manqué ne tombe qu'à 0,359 et la fausse alarme triple |
| niveau du signal | moyenne portée ×2,8 | le manqué **empire** légèrement |
| réverbération | 0,3 s et 0,6 s de RT60 | le manqué reste plat : 0,090 / 0,093 / 0,101 |
| **l'annotation** | énergie du signal là où il rate | **21,8 dB sous** la parole émise, 5,4 dB au-dessus du silence annoté |

La parole manquée est au plancher de bruit du micro de salle. La référence d'AMI est transcrite sur
les micros-**casques** alors qu'on donne au moteur le mixage de salle : elle marque donc des mots
que le signal ne porte pas — et pyannote a AMI dans ses données d'entraînement. Les deux biais
jouent dans le même sens et aucun ne se sépare sur ce corpus. Le corollaire est moins flatteur pour
pyannote : il émet jusque dans ce plancher de bruit, ce que la référence biaisée récompense et
qu'une référence exacte punit — sur le mixage réverbéré son manqué est de 0,296 contre 0,101.

`studies/diar_far_field.py` et l'étude contre le payant portent ces mesures.

## Le réglage de latence, et un défaut d'abord mal choisi

La latence est un réglage, en secondes ; le calendrier s'en déduit. Le premier défaut prenait le
plus long contexte viable — 6,4 s — parce que plus de contexte est **moins cher** à qualité égale :
facteur temps réel de 0,166 contre 0,258, pour 0,092 d'erreur contre 0,090.

Cette égalité n'existe qu'en salle **anéchoïque**, et une salle de réunion ne l'est pas. Réverbérée
à 0,6 s de RT60, les deux se séparent nettement :

| latence | erreur | confusion | accord ≥300 s |
|---|---:|---:|---:|
| 6,40 s | 0,350 | 0,190 | 0,372 |
| **1,04 s** | **0,252** | **0,151** | **0,668** |

L'accord longue portée est presque doublé. Le contexte long dilue la décision dès que la salle
smeare le signal : plus de matière, mais une matière plus confuse. Le défaut est donc le calendrier
que la provenance publie et a validé — ce qui aurait dû être l'hypothèse de départ.

`discussion-helper` garde `diar_latency_s=1.04` ; un enregistrement propre et de près peut demander
`latency_s=None` et économiser un tiers du calcul.

## Deux plafonds qui ne sont pas les nôtres

**Huit locuteurs.** Sur une heure de plateau, la référence payante trouve de treize à quarante-deux
locuteurs — plateau, reportages, archives, voix off. Nemotron plafonne à huit voies et se trouve
donc structurellement hors-jeu sur cette matière. Pour du travail de type FranceTV en lot, pyannote
reste le seul moteur gratuit qui ne bute pas sur un plafond de comptage.

**Le calendrier hors ligne du graphe ne tourne pas.** Son export déclare le calendrier de 30,4 s
comme le seul vérifié de bout en bout. Il échoue :

```
Reshape '/pos_enc/Reshape_3' : Input shape {5000,64}, requested shape {1,1,5144,64}
```

La première lecture était une limite de longueur de séquence — 625 trames pour
`[cache | file | morceau]`. Ce n'est pas toute l'histoire, et un sondage par dichotomie sur le
fichier lui-même acceptait 321 trames de morceau, ce qui a mis sur la bonne piste. La vraie cause :
**la file grandit sans borne.** La mise à jour lui ajoute `chunk_len` trames et n'en retire que
`fifo_len`, donc un morceau plus long que la file la fait grandir — à 281 trames de morceau elle
atteint 241 au lieu de 40 et la séquence monte à 826. C'est le portage fidèle de la référence, dont
la règle n'est cohérente que si le morceau n'excède pas la file : ses trois calendriers de flux
publiés le respectent (9, 6, 3 trames) et son calendrier hors ligne, à 340, le viole. Le contexte
utilisable plafonne donc à 6,4 s, et la limite est vérifiée plutôt que supposée.

## Moteurs d'exécution : mesurés, pas raisonnés

Le premier défaut mettait **CoreML** en tête, par l'argument évident : ONNX Runtime n'a pas de
moteur Metal, donc sur Apple Silicon l'accélération passe par CoreML. Le raisonnement était juste
et la conclusion fausse. CoreML ne prend que 1193 nœuds sur 2827 et les découpe en **316
partitions**, si bien que chaque pas de flux paie autant de traversées : facteur temps réel de
1,257, c'est-à-dire **plus lent que le direct**, contre 0,528 pour le CPU seul.

Forcer huit fils sur une machine à huit cœurs performants était tout aussi contre-productif : 0,953
contre 0,528 en laissant ONNX Runtime choisir. Les deux « optimisations » nuisaient. Le CPU mène
l'ordre désormais, CUDA devant lui au titre de sa couverture d'opérateurs — non mesuré, et le
commentaire le dit.

## Le Rust fait-il gagner du RTF ? Non. Il fait gagner des endroits où tourner.

Profilé sur deux minutes d'audio, l'appel au graphe et le front-end mel chronométrés à part :

| | temps | part |
|---|---:|---:|
| graphe ONNX *(C++, identique depuis Rust)* | 90,47 s | **99,8 %** |
| front-end log-mel *(numpy, FFT en C)* | 0,06 s | 0,1 % |
| comptabilité du cache + Python | 0,10 s | 0,1 % |

Le plafond d'un portage Rust est de **0,2 % du temps** : 0,754 au lieu de 0,755. La caisse `ort`
appelle la même bibliothèque C++, et les deux morceaux de Python sont soit vectorisés, soit du
travail sur des tableaux de 264 × 512 à raison d'un appel par seconde environ.

Le Rust se justifie quand même, pour une autre raison, et il faut l'annoncer comme telle : **sur
mobile, le chemin Python n'existe pas.** On n'embarque pas raisonnablement CPython et une pile
scientifique dans une application iOS ou Android. Ce n'est pas un argument de performance mais de
*possibilité* — plus un binaire unique par plateforme, aucune résolution de dépendances à
l'installation, et aucun interpréteur à démarrer. Même moteur, même vitesse, des endroits où
tourner que Python n'atteint pas. Promettre un gain de RTF serait faux, et la mesure le dit.

## Ce qui n'est pas mesuré

- **Au-delà de huit locuteurs**, qui est le plafond déclaré de Nemotron et donc le cas le plus
  intéressant. Sa ligne sur un mixage à huit voix n'a pas abouti.
- **Les enregistrements réels au-delà de trois émissions.** Tout ce qui favorise Nemotron repose sur
  des mixages fabriqués et trois épisodes de télévision ; le seul corpus classique disponible est
  celui dont l'annotation est le problème.
- **CUDA**, faute de machine NVIDIA.
- **Les facteurs temps réel ont une dispersion d'un facteur deux** selon la charge de la machine —
  0,258 puis 0,755 pour le même calendrier. Les chiffres cités ici sont des ordres de grandeur et
  non des spécifications, et une remesure propre sur machine au repos est due.
