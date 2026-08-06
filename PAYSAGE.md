# Paysage

🇫🇷 Français · [🇬🇧 LANDSCAPE.md](https://github.com/warith-harchaoui/vocal-helper/blob/main/LANDSCAPE.md)

Projets Python / open source voisins et concurrents dans l'espace
« parole en direct → texte attribué aux locuteurs → résumé », comparés
à `vocal-helper`. Les notes vont de ⭐ (1) à ⭐⭐⭐⭐⭐ (5), évaluées sur
la tâche visée par `vocal-helper` — un **pipeline producteur/consommateur
asynchrone qui transforme un flux PCM en direct (micro, URL ou fichier)
en énoncés diarisés et transcrits, plus un résumé LLM glissant
optionnel**. Un projet optimisé pour un autre usage (par ex.
transcription par lots uniquement, diarisation non temps réel, dialogue
LLM généraliste) n'est pas pénalisé — la note reflète seulement
l'adéquation à *ce* créneau.

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

La carte est un résumé en 2D des 8 critères : à lire comme une forme, pas comme un classement. « vocal-helper » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal — Précision ↔ Flexibilité** et **Vertical — Intégration ↔ Simplicité**.
<!-- FIGURE:END -->

## Positionnement

`vocal-helper` se place volontairement à l'intersection de
l'**ergonomie de whisper.cpp** (local, économique, sans GPU strictement
requis) et de la capacité de **diarisation en direct + analyste
glissant** que la plupart des piles vocales repoussent vers la couche
par lots. Il ne cherche pas à battre `pyannote` sur le DER hors ligne
ni `faster-whisper` sur le WER brut de l'ASR — il *compose* ces briques
éprouvées en un seul pipeline asynchrone dont les étapes sont
individuellement remplaçables (n'importe quelle étape sur mesure peut
être insérée sous forme de coroutine) et il expose cette composition à
travers quatre surfaces cohérentes : CLI argparse, CLI click, HTTP
FastAPI, outils MCP. Ce compromis est le principal facteur de
différenciation face à un simple notebook pyannote (pas de streaming),
la famille Whisper locale — whisper.cpp, openai-whisper, faster-whisper,
WhisperX, whisper_streaming (ASR seulement, pas de diarisation) — ou un
service cloud comme AssemblyAI / Deepgram / Otter.ai (entièrement géré,
mais hors appareil, avec des notes faibles en STT 100 % local et Hors
ligne).

Deux nuances derrière les étoiles méritent d'être précisées. La
diarisation en ligne est la contrainte la plus dure de `vocal-helper` :
il fait tourner pyannote / NeMo sous une stratégie de clustering en
ligne, là où `pyannote.audio` obtient sa meilleure note hors ligne mais
ne livre aucun pipeline de streaming par défaut. Le résumé LLM glissant
— Gemma via Ollama sur une fenêtre de 60 s — est une étape intégrée que
la plupart des piles ASR n'ont tout simplement pas, ce qui explique que
seuls les services cloud (AssemblyAI, Deepgram) s'en approchent sur
cette colonne — et seulement en envoyant votre audio hors appareil.

## Quand choisir quoi

- **`vocal-helper`** — conversation en direct → transcription diarisée
  → résumé glissant, entièrement sur l'appareil, intégrable dans
  n'importe quel service Python. Réunions, entretiens, points d'équipe,
  notes de thérapie, tableaux de modération, agents vocaux.
- **`pyannote.audio`** — diarisation par lots uniquement sur de l'audio
  enregistré, quand le DER hors ligne compte plus que la latence
  (production de podcasts, traitement d'archives).
- **`NVIDIA NeMo`** — vous faites déjà tourner une pile Triton / NIM et
  voulez Sortformer / TitaNet étroitement couplés à votre couche de
  service GPU.
- **`whisper.cpp` / `faster-whisper`** — vous avez seulement besoin
  d'ASR, sans diarisation ni analyste ; la latence n'est pas la
  contrainte la plus serrée.
- **`openai-whisper` / `WhisperX`** — vous voulez l'implémentation de
  référence exacte de Whisper pour un benchmark (`openai-whisper`), ou
  l'alignement au niveau du mot plus une diarisation par lots sur des
  enregistrements (`WhisperX`) ; le streaming n'est pas en jeu.
- **`whisper_streaming`** — vous voulez un serveur d'ASR en streaming
  prêt à l'emploi, sans diarisation ni étapes LLM.
- **`vosk` / `SpeechBrain`** — vous voulez une boîte à outils d'ASR
  locale légère ou orientée recherche, sans couche d'analyste de
  diarisation.
- **`AssemblyAI` / `Deepgram` / `Otter.ai`** — vous acceptez une
  dépendance au cloud (leurs notes sont faibles en STT 100 % local /
  Hors ligne) et voulez un SLA entièrement géré plutôt qu'un pipeline
  local.
