# Paysage

🇫🇷 Français · [🇬🇧 LANDSCAPE.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/LANDSCAPE.md)

Les projets voisins et concurrents du domaine « parole en direct → texte attribué aux locuteurs
→ synthèse », notés sur le travail pour lequel `vocal-helper` est fait : une **chaîne
producteur/consommateur asynchrone qui transforme un flux PCM vivant (micro, URL ou fichier) en
énoncés transcrits et attribués, plus une synthèse LLM courante en option**. Un projet conçu
pour autre chose — transcription par lot, diarisation hors ligne seulement — n'est pas pénalisé
pour autant : la note dit l'adéquation à *ce* créneau, rien d'autre.

Mise à jour le 2026-10-19. Ce qui a changé depuis le tableau d'août n'est pas seulement la liste
des entrées : plusieurs notes viennent désormais de **nos propres mesures** et non d'impressions,
et ces mesures en ont déplacé deux.

## Ce que nous avons mesuré nous-mêmes

| constat | où ça tombe dans le tableau |
|---|---|
| pyannote **community-1** fait jeu égal avec 3.1 à 0,116 de DER médian sur AMI, mais sa pire réunion est à **0,149 contre 0,474** : même centre, une queue quatre fois plus courte | `pyannote.audio` à 5 en hors ligne |
| Sur trois vrais plateaux de 64 min à six personnes, la sur-segmentation vaut **×2,7 (community-1), ×3,0 (3.1), ×3,7 (Sortformer), ×19,7 (sherpa-onnx)** | pourquoi `sherpa-onnx` n'a que 2 en hors ligne, malgré un excellent moteur portable |
| **Sortformer 4spk-v1 n'émet jamais un cinquième locuteur** : plafond architectural, confirmé sur 30 mix à cinq voix | `NVIDIA NeMo` plafonné à 4 en hors ligne |
| **diart** atteint 6,04 % d'erreur de locuteur contre 11,7 % pour le chemin community-1 sur un appel long à deux voix, avec un premier label en **500 ms contre 20 s** | `diart` à 5 en direct |

Ce dernier point mérite la réserve que le chiffre seul escamote : l'appel comptait **deux**
locuteurs, ce qui est le cas facile de diart et la littérature le donne régulièrement *moins*
précis que les chaînes par regroupement sur du matériel multi-locuteurs difficile. À lire comme
« le plus rapide à produire un label utilisable », pas comme « le plus exact ».

## En un coup d'œil

<!-- TABLE:START -->
| Project | Live streaming | Online diarization | Offline diarization | Local only | Rolling LLM summary | Multi-source input | Ergonomic Python API | Multi-surface |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper.cpp | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| faster-whisper | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| WhisperX | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| whisper_streaming | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| pyannote.audio | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| diart | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| NVIDIA NeMo | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ |
| Nemotron-3 Diarization | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ |
| sherpa-onnx | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| FunASR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Moonshine | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| SpeechBrain | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| vosk | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Deepgram | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Gladia | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| pyannoteAI | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
<!-- TABLE:END -->

## Carte de positionnement

<!-- FIGURE:START -->
Représentation en deux dimensions du tableau ci-dessus, produite avec
[standpoint](https://github.com/warith-harchaoui/standpoint).

![Carte de positionnement](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/landscape.png)

Les axes se lisent **horizontalement : Adaptable ↔ Polyvalent**, **verticalement : Indépendant ↔
Complet** et ils portent **72 %** de l'information des huit critères (46 % et 26 %). La carte
est donc une forme, pas un classement : deux points proches s'accordent sur l'essentiel de ce
qui a été noté et peuvent malgré tout différer sur le critère qui vous importe.

L'intéressant est le regroupement. Les services cloud (Deepgram, Gladia) sont proches de
`vocal-helper` en *capacité* et loin de lui en indépendance : c'est exactement l'échange qu'ils
proposent. Les moteurs de transcription purs se massent en bas à droite — indépendants,
polyvalents et sans aucune opinion sur qui parle. Les piles de recherche en diarisation
(pyannote.audio, NeMo, Nemotron-3, SpeechBrain) sont à gauche : profondes sur un axe, pas faites
pour être une chaîne.
<!-- FIGURE:END -->

## Positionnement

`vocal-helper` se tient là où l'**ergonomie de whisper.cpp** (local, peu cher, pas de GPU
obligatoire) rencontre la **diarisation en direct et l'analyste courant** que la plupart des
piles renvoient à une couche par lot. Il ne cherche à battre ni pyannote en diarisation ni
whisper en transcription : il les utilise et prend en charge ce que personne d'autre ne prend
— la chaîne producteur/consommateur entre les deux et la mesure de ce qu'elle coûte à la
personne qui regarde.

Concrètement, ce qui est ici et rare ailleurs :

- **Un seul paquet pour les deux régimes, avec un défaut mesuré dans chacun.** community-1 hors
  ligne, diart en direct. Deux régimes, deux gagnants, aucun supposé.
- **Une mesure qui épouse la panne** (`vocal_helper.bench`) : le cpWER facture ensemble les mots
  et le locuteur, parce qu'un diariseur qui a fondu cinq intervenants en deux obtient exactement
  le même WER qu'un qui les a séparés.
- **Une mesure du direct** (`vocal_helper.live_metrics`) : latence depuis la fin de la parole,
  taux de révision, attente par file. Le hors-ligne a le WER et le DER ; le direct n'avait rien,
  y compris chez les projets ci-dessus.
- **Des échecs bruyants.** Un étage perdu lève une exception au lieu de rendre une transcription
  vide.

## Quand prendre autre chose

La moitié honnête d'un document de paysage.

- **Vous voulez seulement la transcription d'un fichier.** Prenez `faster-whisper` ou
  `whisper.cpp` directement : tout ce qui est ici sur les files et la latence est du poids mort
  pour vous.
- **Vous voulez la meilleure diarisation hors ligne et rien d'autre.** Prenez `pyannote.audio`
  avec community-1 — c'est précisément ce que `vocal-helper` fait en dessous, donc la seule
  question est de savoir si vous voulez la chaîne autour.
- **Vous travaillez en mandarin, cantonais, japonais ou coréen.** `FunASR` est fait pour ces
  langues, pas nous.
- **Vous devez tourner sur un Raspberry Pi.** `Moonshine` vise ce matériel ; notre chemin local
  vise une machine de classe portable.
- **Vous voulez un engagement de service, pas une bibliothèque.** Deepgram, Gladia, AssemblyAI.
  L'échange est la souveraineté : l'audio quitte la machine et la facture suit les heures.
