# Soumission arXiv — procédure

Cette procédure suppose que le paper est dans son état courant sur `main`
(v0.18 — task-level / heterogeneity / self-correction) et que `paper/main.pdf` compile
proprement via le workflow CI `paper-pdf.yml`. Tout se passe depuis ton
laptop ; rien à partager dans une conversation.

## 1. Construire le bundle (local, 30 secondes)

```bash
cd paper/
make clean
make           # produit main.pdf + main.bbl
make arxiv     # produit arxiv-submission.tar.gz
```

Vérifie le contenu :

```bash
tar tzf arxiv-submission.tar.gz
# attendu :
#   main.tex
#   main.bbl
```

arXiv préfère le `.bbl` pré-généré plutôt que le `.bib` brut, c'est pour ça
que le tarball ne contient pas `references.bib`. arXiv compile depuis le
`.bbl` directement sans relancer bibtex — déterministe et plus rapide.

## 2. Créer un compte arXiv (browser, ~2 min)

https://arxiv.org/user/register

- **Email** : `denis.hamon1@gmail.com` (l'email qui apparaît dans le PDF).
  Si tu utilises un autre email, arXiv peut signaler un mismatch — pas
  bloquant mais évitable.
- **Affiliation** : laisse `Independent` ou vide. Pas `OVHcloud`.
- **ORCID** : optionnel mais recommandé si tu en as un (lien permanent vers
  ta page auteur futurs travaux).

## 3. Obtenir un endorsement pour cs.LG (obstacle réel)

C'est l'étape la plus friction pour un premier dépôt. arXiv exige qu'un
nouveau soumissionnaire dans `cs.LG` soit endorsé par quelqu'un ayant 3
papers ou plus dans `cs.LG` sur les 5 dernières années.

**Options dans l'ordre de réalisme :**

1. **Tu connais quelqu'un qui peut endorser.** Ex-collègue, contact
   académique, encadrant. Tu lui demandes via le bouton "Request
   Endorsement" sur arXiv → arXiv envoie un code à 6 chiffres à la
   personne, qui te l'entre. ~5 min côté endorser.

2. **Tu demandes à arXiv directement de trouver un endorser** via le bouton
   "Search for an endorser" (depuis ta page de soumission après tentative).
   arXiv suggère des candidats basés sur la category et le sujet du paper.
   Délai variable, parfois plusieurs jours.

3. **Tu soumets dans une category moins gatekept en premier.** `stat.ML` a
   parfois des règles d'endorsement plus permissives, et tu peux toujours
   cross-list vers cs.LG après la première acceptation. Risque : si stat.ML
   est aussi gatekept (le système d'endorsement est category-par-category)
   tu n'as rien gagné.

4. **Tu déposes ailleurs en attendant.** OpenReview, ResearchGate, ou
   simplement ton GitHub (déjà fait via le workflow `paper-pdf.yml`). Moins
   prestigieux, mais valide comme preuve d'antériorité.

Réalisme : si tu n'as pas de contact endorser sous la main, prévois **~3 à
7 jours** entre la création du compte et la première soumission acceptée.

**Démarre cette étape maintenant, en parallèle des dernières expériences :**
l'endorsement est asynchrone et indépendant du contenu du papier, donc c'est
le goulot à lancer en premier pour qu'il soit résolu quand le papier est prêt.

### Message d'endorsement prêt à envoyer (neutre, sur les mérites)

Si tu sollicites une personne qui peut endorser en `cs.LG` (un contact
ayant >= 3 papers cs.LG sur 5 ans), envoie ceci -- factuel, indépendant,
aucune name-drop d'aucun labo, le travail parle de lui-même :

```
Objet : Demande d'endorsement arXiv (cs.LG) -- note de méthodologie, single-author

Bonjour [Prénom],

Je finalise une courte note de méthodologie sur l'évaluation des world
models action-conditionnés et je sollicite un endorsement arXiv pour la
catégorie cs.LG (première soumission sous mon nom). Le travail est
indépendant et open-source.

En une phrase : je propose une métrique decision-grade (Counterfactual
Planning Gap) avec un intervalle de confiance Agresti-Caffo et une règle
de verdict gatée, plus une analyse de puissance montrant combien
d'épisodes une comparaison de world models nécessite avant que son
classement soit statistiquement fiable -- une question que les
leaderboards en point-estimate ne tranchent pas.

Code, résultats et le PDF reproductibles :
https://github.com/Denis-hamon/world-model-eval-lab
PDF direct :
https://github.com/Denis-hamon/world-model-eval-lab/raw/main/paper/main.pdf

Si tu es d'accord, arXiv te demandera de saisir un code d'endorsement à
6 chiffres (~5 min de ton côté). Le code et le lien apparaîtront sur ma
page de soumission une fois que j'aurai initié la demande ; je te les
transmettrai. Aucune obligation, et merci d'avance quoi qu'il en soit.

[Nom]
```

Note de cadrage : le message reste sur les mérites du travail et ne
mentionne aucun laboratoire, modèle ni programme tiers. C'est cohérent
avec la posture d'indépendance du repo -- le travail est découvrable et
défendable seul.

## 4. Métadonnées à coller dans le formulaire de soumission

Une fois endorsé, tu vas sur https://arxiv.org/submit et tu remplis :

### Categories

- **Primary** : `cs.LG` (Machine Learning)
- **Cross-list** (secondaires, optionnels mais utiles) :
  - `cs.AI` (Artificial Intelligence)
  - `stat.ME` (Methodology) — pour le côté Agresti-Caffo CI

### Title

```
Counterfactual Planning Gap: An Interval-Gated Statistic for Diagnosing World-Model Bottlenecks under a Fixed Planner
```

### Authors

```
Denis Hamon
```

(Affiliation field : `Independent`.)

### Abstract (à coller en tant que plain text — arXiv accepte $...$ pour les maths inline mais pas les \emph, \textbf, \citep)

```
Action-conditioned world models are usually evaluated by prediction quality (reconstruction loss, frame-level FID, held-out accuracy), which is silent on the question an applied team must answer before integrating a model into a control loop: does the model, when used by a planner, produce decisions that succeed? We propose the Counterfactual Planning Gap (CPG): the success-rate difference between a fixed planner using oracle dynamics and the same planner using the learned model, on identical runs that differ only in the dynamics callable. We report it with an Agresti-Caffo plus-4 interval (which keeps the variance positive at the boundary proportions $p \in \{0,1\}$ where the Wald approximation collapses), a paired bootstrap where the design warrants it, and a five-branch verdict (MODEL BOTTLENECK, LEARNED OUTPERFORMS ORACLE, PLANNER BOTTLENECK, MODEL AS GOOD AS ORACLE, INCONCLUSIVE) gated on the lower bound of the CI rather than the point estimate, so under-powered runs cannot over-claim a diagnosis. We package CPG behind a minimal evaluation contract (wmel) and exercise it on three DeepMind Control Suite tasks. The central finding is that the verdict is heterogeneous and condition-sensitive - precisely what a calibrated metric should surface, and what a point-estimate leaderboard cannot. On Acrobot-swingup the gap looks large only because the oracle's fixed initial state is an unusually easy swing-up; sampling the task's initial-state distribution collapses the oracle to ~3% success and flips the verdict from MODEL BOTTLENECK to PLANNER BOTTLENECK. On Reacher-easy the verdict is MODEL BOTTLENECK across all arms (CPG from +0.20 to +0.33). On Cartpole-swingup at higher model capacity the larger TD-MPC2 (Hansen et al., 2024) under a Cross-Entropy-Method planner beats the oracle planner: LEARNED OUTPERFORMS ORACLE, CPG = -0.27, AC CI [-0.48, -0.02] and paired-bootstrap CI [-0.50, -0.03], both clearing zero. We also present this as self-correction: an earlier single-fixed-initial-state evaluation of this same framework reported MODEL BOTTLENECK on Acrobot; the metric's own interval-gated machinery, re-run over the task distribution, overturned that headline. Finally, because the gate is a function of the confidence interval, it doubles as a power-analysis tool: we give the per-arm episode count a comparison needs before its interval clears zero, and show that a plausible leaderboard near-tie ($0.94$ vs $0.92$ at $n = 100$) is statistically indistinguishable from noise.
```

(Source of truth: the abstract in `paper/main.tex`, v0.18 — task-level / heterogeneity / self-correction. If the paper changes again, regenerate this block from `main.tex` before submitting.)

### Comments (optional one-liner)

```
~10 pages, 5 figures, multiple tables. Code, results, and reproducibility scripts at https://github.com/Denis-hamon/world-model-eval-lab
```

### License

Recommandation : **CC BY 4.0** (Creative Commons Attribution).

- Permet la réutilisation et la traduction tant que l'attribution est
  donnée.
- Compatible avec les politiques OA de la plupart des venues académiques.
- Évite arXiv's perpetual license (très restrictive, à éviter sauf si forcé).

### Pas de figures externes, pas de package non-standard

Le bundle ne contient que `main.tex` + `main.bbl`. Aucune figure, aucun
input externe. Devrait compiler sans heurts sur le pipeline arXiv (qui
roule TeX Live 2023 ou 2024 selon la fenêtre).

## 5. Upload + soumission

1. https://arxiv.org/submit/new
2. **Submission type** : `New submission`
3. **Source** : upload `arxiv-submission.tar.gz`
4. arXiv lance son compile. Attends ~30-60 sec. Si erreur :
   - regarde le log
   - les erreurs les plus fréquentes : `lmodern` manquant (ne devrait pas
     arriver, c'est standard), package non standard. Le paper de v0.10
     avait eu `natbib` + `lmodern` à ajouter en preamble — c'est déjà fait
     en main, donc OK.
5. **Preview the PDF** : arXiv te montre le PDF compilé. Vérifie qu'il
   ressemble à `paper/main.pdf` local.
6. **Metadata review** : copie-colle les blocs de la section 4 ci-dessus.
7. **Confirm submission**.

## 6. Après la soumission

- **Moderation** : 24-72h en semaine, plus le week-end. arXiv envoie un
  email "your paper has been accepted" ou "needs revision".
- **arXiv ID** : tu reçois un identifiant style `2511.XXXXX` ou similaire,
  permanent. À ajouter à `CITATION.cff` et au paper README une fois reçu.
- **DOI** : arXiv n'assigne pas de DOI directement, mais le numéro arXiv
  est un identifiant pérenne accepté pour les citations.
- **Auto-replacement** : si tu pousses un v2 plus tard (ex: après le
  workshop submission), tu peux remplacer le PDF — l'ancien reste consultable.

## 7. Mise à jour du repo après acceptation arXiv

```bash
# Une fois que tu as l'arXiv ID (ex: 2511.12345)
# Édite paper/main.tex pour ajouter :
%   \date{arXiv:2511.12345 — \today}
# Édite CITATION.cff pour pointer le preferred-citation vers l'arXiv ID
# Édite README.md pour ajouter un badge arXiv
# Commit + push + tag v1.0.0 (le paper est désormais cité publiquement)
```

## Procédure de récupération si la moderation rejette

Causes communes de rejet :
- **Endorsement manquant** : voir section 3.
- **Off-topic pour la category** : choisir une autre category. cs.LG est
  toujours valide pour ce contenu, ne devrait pas rejeter sur ce motif.
- **Source ne compile pas** : log arXiv te dit pourquoi. Le bundle minimal
  qu'on génère devrait pas avoir ce problème.
- **Anonymous / no real author** : le paper a un auteur réel, OK.
- **Duplicate** : si tu as déjà soumis ailleurs avec un autre titre, lien
  vers l'autre version. Ici c'est une primo-soumission.

En cas de rejet, l'email arXiv contient un lien pour répondre. Tu corriges
et re-soumets.

## Une seule chose à se rappeler

**Ne partage jamais tes credentials arXiv** (ni à moi, ni dans aucun
support de transcript). L'authentification arXiv est OAuth-via-browser, tu
n'as pas besoin de générer un token API pour la soumission web.
