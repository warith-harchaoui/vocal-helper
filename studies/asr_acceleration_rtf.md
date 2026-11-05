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

Trois résultats et aucun n'était évident.

**Metal vaut 4,9×.** Même moteur, même modèle, même machine : 0,164 sur CPU contre 0,033
avec le GPU. C'est le gain d'accélération le plus simple à obtenir et il est déjà acquis —
whisper.cpp prend son défaut `use_gpu=true` et le journal le confirme
(`whisper_backend_init_gpu: using MTL0 backend`). Rien à faire, mais il faut le savoir :
une installation de `pywhispercpp` sans `libggml-metal` tomberait à 6× le temps réel sans
rien signaler.

**whisper.cpp bat MLX de 1,85×**, sur le propre terrain d'Apple. C'est contre-intuitif et
ça valide le choix par défaut de vocal-helper. À noter pour les produits en aval :
`notes-helper` et `parole-publique` utilisent mlx-whisper comme **source de minutage** et
c'est un choix de précision, pas de vitesse — ses onsets de mots ont été mesurés à +0,00 s
de médiane contre la vérité terrain. On paie 1,85× de temps pour une horloge plus juste
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
produit plus d'une seconde de calcul, la file grossit sans fin et la latence part à
l'infini.

Mais la façon de combiner les étages n'est pas la somme et la lecture évidente est fausse
**dans les deux sens**. Dans un schéma producteur/consommateur les étages tournent en
concurrence :

- deux étages sur des **unités de calcul différentes** se **recouvrent** : la chaîne coûte
  le plus lent des deux, pas les deux ;
- deux étages sur la **même unité** ne se recouvrent pas : ils font la queue et là les
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
= 0,063 et le CPU reste à 0,010, quasi inoccupé. C'est pourtant le meilleur placement, parce
que les deux étages du GPU sont bon marché.

**Déplacer la diarisation hors du GPU n'aide pas** quand l'alternative est chère : passer à
sherpa sur CPU fait tomber le GPU à 0,033 mais monte le CPU à 0,590. Le goulot change de
main, pour le pire. On ne gagne pas en répartissant, on gagne en répartissant **vers une
unité libre ET rapide**.

**Le cas tendu est sans torch et sans GPU** : tout s'empile sur le CPU, 1,3× de marge. Une
machine un peu plus lente, un modèle un peu plus gros ou un processus concurrent font
passer au-dessus de 1 et le direct décroche définitivement. C'est aussi là que l'ANE
change quelque chose et pas par sa vitesse : en déportant l'encodeur sur le moteur neuronal
il **libère le CPU** et la marge du cas sans torch respire.

C'est exactement ce que borne la politique `source_policy="drop_oldest"` de
:class:`vocal_helper.PipelineConfig` : au lieu de laisser une file grossir sans fin — ou,
pire, de laisser la contre-pression remonter jusqu'à un micro où elle devient une perte
d'audio invisible dans un tampon du système — les trames les plus anciennes sont
abandonnées, comptées et rapportées. La latence reste bornée à la profondeur de la file et
le trou dans la transcription porte un chiffre.

## Ce qu'il reste à gagner et ce que ça coûte

Deux étages d'accélération existent au-dessus de Metal sur Apple Silicon et **aucun des
deux n'est disponible dans le `pywhispercpp` installé ici** : le binaire ne porte aucune
référence ni symbole CoreML (vérifié à `otool` et `nm`).

| piste | gain annoncé | ce qu'il faut |
|---|---|---|
| **encodeur CoreML** | ~2× sur l'encodeur, qui domine le temps d'inférence | recompiler whisper.cpp avec `WHISPER_COREML=1` et générer un `*-encoder.mlmodelc` par modèle |
| **backend Apple Neural Engine (ANEForge)** | ~2× de plus que CoreML, 1,0 à 1,3× de plus que Metal et **~5× moins d'énergie** | fusionné en amont dans whisper.cpp le **2026-09-18** (PR `ggml-org/whisper.cpp#3905`, vérifié par l'API GitHub). Activé par la variable `ANEFORGE_ENCODER` au build ; sans elle le comportement est celui de whisper.cpp standard, sans surcoût. Encodeur seulement, le décodeur ne change pas. Fidélité annoncée : mêmes transcriptions que l'encodeur de référence, 99,2 % d'accord mot à mot sur un fichier long. |

Lecture honnête du rapport bénéfice-coût **pour le différé** : à RTF 0,033 on est déjà à
30× le temps réel, donc gagner 1,3× ne change pas la vie d'un corpus. Ça compte pour
**l'énergie** (un facteur 5 sur une campagne de centaines d'heures et sur batterie) et ça
compte pour le **direct sur une machine modeste**, où la marge de 1,3× du cas sans torch
deviendrait respirable.

Donc : à faire quand le direct gratuit sortira d'Apple Silicon haut de gamme ou quand une
campagne longue rendra l'énergie visible. Pas avant et pas pour la vitesse seule.


## La diarisation en direct : la qualité d'abord, RTF total sous 1

Priorité posée par l'utilisateur le 2026-11-01 : en direct gratuit, **la qualité de
diarisation est primordiale** et la seule contrainte dure est que le RTF **total du
système** reste sous 1.

Ça renverse l'arbitrage du router, qui choisissait l'étage en ligne « nemo, toujours » parce
que c'était le meilleur des étages en ligne. Mais l'étage en ligne est à **DER 0,586** contre
**0,122** en différé : trois à quatre fois pire, parce qu'il ne modélise pas la parole
superposée. Aucun meilleur regroupeur en ligne ne répare ça — la limite est la latence.

Et avec 16× de marge sur le défaut, il y a du budget à dépenser en qualité.

### La réponse : re-diarisation périodique hors ligne, sur fenêtre bornée

Pas un algorithme neuf : c'est l'expression en direct du **diariseur par blocs du cœur
Rust** (`core/nh-core/src/blockwise.rs`), qui avait résolu ça pour le long format. Ce code
avait gagné trois détails qu'une réécriture naïve manque et chacun fait la différence entre
« ça marche » et « ça marche presque » :

1. **l'amorce est un budget par locuteur**, pas un bloc. Un bloc d'« audio plus ancien »
   manquerait celui qui n'avait pas encore parlé et son étiquette dériverait à chaque
   rafraîchissement ;
2. **les morceaux sont séparés par du bruit de salle récolté**, pas par du silence et
   l'écart doit **dépasser le `min_duration_off` du modèle de segmentation** (0,5 s mesuré,
   d'où 0,8 s par défaut) sinon le modèle n'y coupe pas de frontière du tout, les références
   se mélangent et la relecture devient ambiguë. Le silence numérique est hors distribution,
   donc pire qu'inutile ;
3. **l'identité se récupère par relecture, pas par appariement.** On diarise
   `[amorce | fenêtre]` en un seul appel, donc le regroupement du moteur place déjà chaque
   tour de la fenêtre avec la bonne référence : il suffit de lire quelle étiquette locale a
   couvert chaque référence. Aucun plongement, aucun centroïde, aucun seuil à nous — et c'est
   **exact**, là où un recouvrement est approximatif.

Pourquoi la version évidente ne tient pas : « re-diariser tout le tampon » coûte
`rtf = 0,067 × tampon / période` et le tampon croît. Sur une heure avec une période de
60 s : **4,0**. Ça marche trois minutes et s'écroule après vingt.

Borné, le coût est constant en durée de session et croît **légèrement** avec le nombre de
locuteurs (l'amorce en dépend) :

    rtf = 0,067 × (fenêtre + amorce(k)) / période

### La table de décision, calculée

Mesures : ASR Metal 0,033, ASR CPU 0,164, VAD 0,010, pyannote hors ligne 0,067 (DER 0,122),
sherpa hors ligne 0,580 (DER 0,174), en ligne nemo 0,030 (DER 0,586). Placement pessimiste :
transcription et diarisation sur **la même** unité, donc les coûts s'ajoutent.

| option de diarisation en direct | DER | RTF total (6 voix) | marge | tient ? |
|---|---:|---:|---:|---|
| en ligne seul | **0,666 mesuré** | 0,063 | 16× | oui |
| **re-diar pyannote, f=120 s, période 60 s** | **0,348 mesuré** | 0,288 | 3,4× | **oui** |
| différé pleine fenêtre (hors d'atteinte en direct) | 0,149 mesuré | — | — | — |
| re-diar sherpa (sans torch), période 60 s | — | 2,5 | 0,4× | **non** |

**Mesuré, pas supposé**, sur cinq minutes d'AMI IS1008a avec la vérité terrain `words.rttm`,
collar 0,25 (`studies/rediar_live_der.py`). Une première version de ce document annonçait
0,122 pour la fenêtre glissante : c'était le chiffre du différé pleine fenêtre réutilisé
comme si la version bornée l'atteignait. Elle ne l'atteint pas.

**Ce qu'on achète réellement : l'erreur divisée par deux** (0,666 → 0,348) pour une marge
qui passe de 16× à 3,4×. C'est nettement moins que ce que j'avais écrit et c'est encore
l'échange demandé — la qualité d'abord, sous 1 avec de la réserve.

**Pourquoi ça s'arrête à mi-chemin et c'est structurel.** Les lignes publiées sont les
segments du **détecteur d'activité vocale** et réétiqueter ne répare pas un segment qui
contient un changement de locuteur. La passe différée choisit ses propres frontières ;
celle-ci en hérite. Le prochain levier est donc d'**émettre la segmentation du différé** au
lieu de réétiqueter celle du VAD — pas un meilleur regroupeur.

**Deux bugs trouvés par cette campagne et ils expliquent pourquoi il fallait la faire.**
Le premier : je passais `num_speakers = len(références)` au diariseur, donc dès qu'une
référence existait, chaque rafraîchissement était épinglé à **un seul locuteur** et ne
pouvait jamais en découvrir un deuxième. Auto-réalisateur : le DER restait **exactement**
celui de l'étage en ligne, à trois décimales, parce que le rafraîchissement ne pouvait pas
se contredire. C'est désormais un **plancher** (`min_speakers`), jamais un compte exact — la
doctrine que j'avais écrite pour le cloud et pas appliquée ici. Le second : le plancher de
promotion à 15 s venait du diariseur par blocs, où une fenêtre est découpée dans un
enregistrement entier ; sur une fenêtre bornée de deux minutes, un participant réel mais
discret ne l'atteint jamais, le compte reste à un et tout s'effondre. Descendu à 5 s, ce
qui écarte encore un fragment parasite (moins de deux secondes).

La politique par défaut porte **18 locuteurs** sur machine Metal avant de rompre et **9**
sans GPU — bien au-delà d'un plateau réel. Le cas qui ne tient pas est le diariseur sans
torch : sherpa coûte 8,7× pyannote, le rejouer périodiquement ne rentre pas et le dire est
plus honnête que de promettre une qualité que la machine ne rendra pas.

### Ce que l'appelant doit accepter

**Les étiquettes changent rétroactivement.** Une ligne affichée `S1` peut devenir `S2` quand
le rafraîchissement suivant arrive avec une meilleure information. C'est le prix de la
qualité hors ligne sur un flux et ça doit être un comportement **visible et voulu** de
l'affichage — une recoloration, pas une réécriture silencieuse. L'alternative, garder une
étiquette qu'on sait fausse, est pire. C'est l'invariant 2 de la cadence d'operator-assist :
la couleur peut changer après l'arrivée d'une étiquette, jamais sans.

L'étage en ligne reste le bon choix pour la priorité **inverse** : une étiquette en moins de
500 ms et aucune correction rétroactive, ce dont un agent conversationnel a besoin et un
relevé non.
