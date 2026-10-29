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
| Transcription en direct | Streaming en direct | Diarisation en ligne | STT 100 % local | Résumé LLM glissant | Multi-source | API Python ergonomique | Multi-surface | Hors ligne |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vocal-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper.cpp | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| openai-whisper | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| faster-whisper | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| WhisperX | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| whisper_streaming | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| pyannote.audio | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| NVIDIA NeMo | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| SpeechBrain | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| vosk | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AssemblyAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Deepgram | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Otter.ai | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ |
<!-- TABLE:END -->

## Carte de positionnement

<!-- FIGURE:START -->
Représentation 2D du tableau ci-dessus.

![Carte de positionnement](https://raw.githubusercontent.com/warith-harchaoui/vocal-helper/main/assets/paysage.png)

La carte est un résumé en 2D des 8 critères : à lire comme une forme, pas comme un classement. « vocal-helper » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal : Autonomie ↔ Dynamisme** et **Vertical : Réactivité ↔ Polyvalence**.

L'axe horizontal distingue principalement les solutions qui privilégient une grande capacité à s'adapter à des situations variées et à traiter des données en temps réel, de celles qui se concentrent sur une exécution locale et une gestion des ressources plus limitée. Les options positionnées à droite, comme vocal-helper et AssemblyAI, excellent dans la gestion de flux continus et l'intégration de fonctionnalités avancées comme le résumé par intelligence artificielle, tandis que celles à gauche, comme openai-whisper et whisper.cpp, mettent l'accent sur la capacité à fonctionner sans connexion internet et à s'exécuter sur des configurations matérielles modestes.

L'axe vertical oppose, quant à lui, les solutions qui se distinguent par leur rapidité de réponse et leur efficacité dans des tâches spécifiques, à celles qui offrent une plus grande flexibilité et une gamme de fonctionnalités plus étendue. Les options situées en bas, comme Otter.ai et Deepgram, se concentrent sur une exécution rapide et une intégration aisée, tandis que celles en haut, comme vocal-helper et vosk, se caractérisent par leur capacité à gérer des scénarios complexes et à s'adapter à différents types de données.

On observe un regroupement intéressant en haut à gauche, où plusieurs options, dont whisper.cpp, openai-whisper, faster-whisper, WhisperX, whisper_streaming, SpeechBrain et vosk, partagent une forte orientation vers l'autonomie et la polyvalence. Elles offrent une grande flexibilité dans leur utilisation, mais peuvent sacrifier une certaine réactivité. À l'opposé, en bas à droite, Deepgram et Otter.ai se positionnent comme des solutions dynamiques et réactives, mais moins polyvalentes. Enfin, vocal-helper, leader de ce classement, se distingue en combinant dynamisme et polyvalence, tandis qu'AssemblyAI le rejoint dans ce coin pour des raisons différentes : AssemblyAI privilégie le dynamisme, tandis que vocal-helper excelle dans la polyvalence.
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

## Quand choisir quoi

Un comparatif qui ne recommande jamais un concurrent est une publicité, donc
cette section va dans les deux sens.

**Prenez `vocal-helper`** quand il vous faut du texte attribué aux locuteurs
*pendant que l'audio arrive encore*, sur votre propre machine, quand vous
préférez configurer une chaîne plutôt que l'assembler — et quand le même code
devra traiter un enregistrement plus tard, le backend de diarisation étant
choisi pour vous d'après la durée réelle du fichier au lieu d'être deviné.

**Prenez autre chose** quand :

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

## Sources et date de relevé

**Relevé le 2026-10-19.** Chaque note ci-dessus porte sur un logiciel qui
bouge : la date fait partie de l'affirmation.

Le tableau mêle deux sortes de preuves qui ne valent pas la même chose.

**Mesuré ici.** Les constats de *Ce que nous avons mesuré nous-mêmes* — comme
chaque DER / RTF que ce dépôt cite — viennent d'un script de ce dépôt lancé sur un
corpus nommé, jamais d'un chiffre fourni par un éditeur :

| Constat | Où il a été produit | Corpus |
|---|---|---|
| DER + RTF par scénario derrière le routeur | [`studies/router_profile_validation.py`](studies/router_profile_validation.py) | 30 mixes courts (`bagarre`) + AMI dev-slice, `pyannote.metrics` collar 0.25, médiane |
| le défaut d'embedder online (`nemo`) | [`studies/diar_embedding_backend.py`](studies/diar_embedding_backend.py) | AMI dev-slice, marge cosinus inter- moins intra-locuteur |
| community-1 contre 3.1, médiane et queue | [`CHANGELOG.md`](CHANGELOG.md), entrée `3.0.0` | AMI, 8 réunions, collar 0, recouvrement compté |
| la sur-segmentation sur de vrais plateaux, dont les ×19,7 de sherpa-onnx | [`DIARIZATION-TROUBLES.md`](DIARIZATION-TROUBLES.md) §9-§10 | trois plateaux de 64 min à six personnes |
| ce que vaut un `initial_prompt` en WER et en RTF | [`studies/whisper_prompt_lang_lock.py`](studies/whisper_prompt_lang_lock.py) | AMI dev-slice |
| la cadence de résumé (`flush_every_s=60`) | [`studies/llm_cadence_sweep_multi.py`](studies/llm_cadence_sweep_multi.py) | ≥ 4 réunions AMI |

Deux réserves dues au lecteur. L'audio de la validation du routeur et de la
ligne `sherpa` (ADR 0002) vit dans un dépôt compagnon **privé**
(`pasdebonneoudemauvaisesituation`) : ces runs sont reproductibles chez nous et
pas chez vous. Le classement des diariseurs streaming et la comparaison `diart`
viennent de deux autres dépôts non publics — `pdbms` (balayage du 2026-06-29,
N = 2089 par système) et `operator-assistance-streaming`. Ils sont cités pour la
provenance et constituent une preuve plus faible que les lignes que vous pouvez
ouvrir.

**Lu, pas mesuré.** Toutes les autres cases sont notre lecture du dépôt et de la
documentation du projet concerné à la date du relevé : ce qu'il livre et ce
qu'il annonce prendre en charge. Quand une capacité était une affaire de degré,
nous avons noté vers le bas plutôt que vers le haut ; quand un projet est fait
pour un autre métier, la note dit « adéquation à *ce* créneau », jamais
« moins bon logiciel ».

Les travaux amont sur lesquels ce paquet s'appuie réellement sont tenus en
BibTeX dans [`doc/refs.bib`](doc/refs.bib) — entre autres
[whisper.cpp](https://github.com/ggerganov/whisper.cpp),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo),
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[pyannote community-1](https://www.pyannote.ai/blog/community-1) et l'article
INTERSPEECH 2023 de Bredin pour la bande de DER publiée à laquelle la chaîne
pyannote est comparée.

Aucun chiffre de benchmark n'apparaît ci-dessus pour Deepgram, Gladia,
AssemblyAI ou pyannoteAI : nous ne les avons pas fait tourner sur ce corpus, et
imprimer le chiffre d'un éditeur à côté des nôtres aurait l'air d'une
comparaison alors que ce n'en est pas une.

## Idées à reprendre

Étudié plutôt que deviné : ce que chaque voisin fait mieux, puis ce que cette bibliothèque devrait changer pour l'égaler. Rien ici n'est un engagement : c'est la liste de lecture dont part la version suivante.

- **`WhisperX` — un alignement forcé pour des horodatages vraiment au mot** — Il lance un alignement phonémique wav2vec2 après la transcription, atteignant des horodatages au mot sous les 100 ms, puis traite par lots l'audio segmenté par VAD à environ 70× le temps réel sur un seul GPU. Les horodatages de Whisper lui-même ne sont, en comparaison, que des estimations au segment.

- **`WhisperLiveKit` — une surface d'API que les clients parlent déjà** — Il expose des points d'entrée compatibles OpenAI puis Deepgram : le code existant pointe dessus en changeant une URL de base. Un outil local qui remplace l'API hébergée sans rien réécrire retire la principale raison d'y rester.

- **Streaming Sortformer avec Simul-Whisper — le duo basse latence actuel** — Une diarisation en flux qui fige les tours de parole au fur et à mesure, au lieu de re-regrouper toute la fenêtre à chaque fois, est ce qui garde la latence plate quand une conversation s'allonge.

- **`Moonshine` puis Zipformer de sherpa-onnx — l'étage sans GPU** — Les deux sont faits pour la basse latence sans carte graphique. Un transcripteur en direct qui se replie sur un modèle de flux plus petit plutôt que d'échouer est plus utile qu'un autre exigeant CUDA ; la suite route déjà le choix de modèle par best-engine-ai-helper.
