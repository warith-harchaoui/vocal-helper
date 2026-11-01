# Accélération de la transcription locale et budget RTF du direct

Mesuré le **2026-11-01** sur cette machine (Apple Silicon, `MTLGPUFamilyMetal4`), sur
**60 secondes de parole française réelle** extraites d'une réunion du corpus : un coup de
chauffe jeté, puis la médiane de deux passages chronométrés, même modèle de famille
`large-v3-turbo`, 6 fils CPU, langue verrouillée sur `fr`.

Script : `studies/asr_acceleration_rtf.py`. RTF = temps de calcul / durée audio ; sous 1 le
moteur va plus vite que le temps réel.

## Ce que mesure chaque moteur

| moteur | 60 s traités en | RTF | × temps réel |
|---|---:|---:|---:|
| **whisper.cpp `large-v3-turbo-q5`, Metal** | 1,99 s | **0,033** | **30×** |
| mlx-whisper `large-v3-turbo`, MLX / GPU | 3,69 s | 0,061 | 16× |
| whisper.cpp `large-v3-turbo-q5`, CPU seul | 9,83 s | 0,164 | 6× |
| faster-whisper `large-v3-turbo`, CTranslate2 int8 CPU | 39,44 s | 0,657 | 1,5× |

Trois résultats, et aucun n'était évident.

**Metal vaut 4,9×.** Même moteur, même modèle, même machine : 0,164 sur CPU contre 0,033
avec le GPU. C'est le gain d'accélération le plus simple à obtenir, et il est déjà acquis —
whisper.cpp prend son défaut `use_gpu=true`, et le journal le confirme
(`whisper_backend_init_gpu: using MTL0 backend`). Rien à faire, mais il faut le savoir :
une installation de `pywhispercpp` sans `libggml-metal` tomberait à 6× le temps réel sans
rien signaler.

**whisper.cpp bat MLX de 1,85×**, sur le propre terrain d'Apple. C'est contre-intuitif et
ça valide le choix par défaut de vocal-helper. À noter pour les produits en aval :
`notes-helper` et `parole-publique` utilisent mlx-whisper comme **source de minutage**, et
c'est un choix de précision, pas de vitesse — ses onsets de mots ont été mesurés à +0,00 s
de médiane contre la vérité terrain. On paie 1,85× de temps pour une horloge plus juste,
et c'est un échange défendable tant qu'il est dit.

**faster-whisper est hors course ici, d'un facteur 20.** Les comparatifs de 2026 annoncent
CTranslate2 « jusqu'à 4× plus rapide que whisper.cpp » ; sur Apple Silicon c'est l'inverse,
massivement, parce que CTranslate2 n'a pas de chemin GPU sur Metal et reste sur le CPU. La
campagne du 2026-06-30 (`studies/stt_faster_whisper_vs_pywhispercpp.py`) posait la question,
celle-ci la tranche : à RTF 0,657 il ne reste que 1,5× de marge, ce qui ne tient pas un
direct.

## Le budget RTF du direct — et le RTF ne s'additionne pas

En différé, un RTF de 0,66 veut dire « c'est lent ». En **direct**, il faut rester **sous
1**, sinon le pipeline prend un retard qu'il ne rattrapera jamais : chaque seconde d'audio
produit plus d'une seconde de calcul, la file grossit sans fin, et la latence part à
l'infini.

Mais la façon de combiner les étages n'est pas la somme, et la lecture évidente est fausse
**dans les deux sens**. Dans un schéma producteur/consommateur les étages tournent en
concurrence :

- deux étages sur des **unités de calcul différentes** se **recouvrent** : la chaîne coûte
  le plus lent des deux, pas les deux ;
- deux étages sur la **même unité** ne se recouvrent pas : ils font la queue, et là les
  coûts s'ajoutent.

Donc : **additionner par unité, prendre le maximum entre unités**. Sommer tout surestime
une chaîne bien placée et, plus grave, masque le vrai problème, qui n'est jamais le total
mais toujours **une seule unité saturée**. C'est ce que calcule
:func:`vocal_helper.accel.live_budget`.

| placement du mode direct gratuit | CPU | GPU | unité saturée | RTF effectif | marge |
|---|---:|---:|---|---:|---:|
| **whisper.cpp Metal + NeMo TitaNet** (le défaut) | 0,010 | 0,063 | GPU | **0,063** | **16×** |
| whisper.cpp CPU + NeMo TitaNet | 0,174 | 0,030 | CPU | 0,174 | 5,7× |
| whisper.cpp Metal + sherpa (sans torch) | 0,590 | 0,033 | CPU | 0,590 | 1,7× |
| whisper.cpp CPU + sherpa (sans torch) | **0,754** | — | CPU | **0,754** | **1,3×** |

Trois enseignements que la somme cachait.

**Le défaut met les deux étages lourds sur le GPU**, donc ils se sérialisent : 0,033 + 0,030
= 0,063, et le CPU reste à 0,010, quasi inoccupé. C'est pourtant le meilleur placement, parce
que les deux étages du GPU sont bon marché.

**Déplacer la diarisation hors du GPU n'aide pas** quand l'alternative est chère : passer à
sherpa sur CPU fait tomber le GPU à 0,033 mais monte le CPU à 0,590. Le goulot change de
main, pour le pire. On ne gagne pas en répartissant, on gagne en répartissant **vers une
unité libre ET rapide**.

**Le cas tendu est sans torch et sans GPU** : tout s'empile sur le CPU, 1,3× de marge. Une
machine un peu plus lente, un modèle un peu plus gros ou un processus concurrent font
passer au-dessus de 1, et le direct décroche définitivement. C'est aussi là que l'ANE
change quelque chose, et pas par sa vitesse : en déportant l'encodeur sur le moteur neuronal
il **libère le CPU**, et la marge du cas sans torch respire.

C'est exactement ce que borne la politique `source_policy="drop_oldest"` de
:class:`vocal_helper.PipelineConfig` : au lieu de laisser une file grossir sans fin — ou,
pire, de laisser la contre-pression remonter jusqu'à un micro où elle devient une perte
d'audio invisible dans un tampon du système — les trames les plus anciennes sont
abandonnées, comptées et rapportées. La latence reste bornée à la profondeur de la file, et
le trou dans la transcription porte un chiffre.

## Ce qu'il reste à gagner, et ce que ça coûte

Deux étages d'accélération existent au-dessus de Metal sur Apple Silicon, et **aucun des
deux n'est disponible dans le `pywhispercpp` installé ici** : le binaire ne porte aucune
référence ni symbole CoreML (vérifié à `otool` et `nm`).

| piste | gain annoncé | ce qu'il faut |
|---|---|---|
| **encodeur CoreML** | ~2× sur l'encodeur, qui domine le temps d'inférence | recompiler whisper.cpp avec `WHISPER_COREML=1`, et générer un `*-encoder.mlmodelc` par modèle |
| **backend Apple Neural Engine (ANEForge)** | ~2× de plus que CoreML, 1,0 à 1,3× de plus que Metal, et **~5× moins d'énergie** | fusionné en amont dans whisper.cpp le **2026-09-18** (PR `ggml-org/whisper.cpp#3905`, vérifié par l'API GitHub). Activé par la variable `ANEFORGE_ENCODER` au build ; sans elle le comportement est celui de whisper.cpp standard, sans surcoût. Encodeur seulement, le décodeur ne change pas. Fidélité annoncée : mêmes transcriptions que l'encodeur de référence, 99,2 % d'accord mot à mot sur un fichier long. |

Lecture honnête du rapport bénéfice-coût **pour le différé** : à RTF 0,033 on est déjà à
30× le temps réel, donc gagner 1,3× ne change pas la vie d'un corpus. Ça compte pour
**l'énergie** (un facteur 5 sur une campagne de centaines d'heures, et sur batterie), et ça
compte pour le **direct sur une machine modeste**, où la marge de 1,3× du cas sans torch
deviendrait respirable.

Donc : à faire quand le direct gratuit sortira d'Apple Silicon haut de gamme, ou quand une
campagne longue rendra l'énergie visible. Pas avant, et pas pour la vitesse seule.
