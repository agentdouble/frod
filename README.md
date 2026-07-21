# Fraude Detector

MVP local et explicable pour prioriser la revue de PDF d'assurance. Le pipeline ne
declare jamais qu'un document est frauduleux ou authentique : il produit un score de
revue, les indices techniques qui l'expliquent et des images localisant les zones a
controler.

## Demarrage

Prerequis : [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run fraude-detect chemin/vers/document.pdf --output output/analyse
```

Pour un PDF chiffre, preferer une variable d'environnement afin de ne pas laisser le
mot de passe dans l'historique du shell :

```bash
FRAUDE_PDF_PASSWORD='mot-de-passe' uv run fraude-detect document.pdf
```

La commande retourne notamment :

```text
output/analyse/
├── report.json
├── pages/
│   └── page-001.png
├── review/
│   └── page-001-review.png
└── forensics/
    ├── page-001-revision-diff.png
    └── page-001-image-01-ela.png
```

`review/` et `forensics/` ne contiennent des fichiers que lorsqu'une zone a ete
localisee. Les rendus originaux des pages analysees sont toujours places dans
`pages/`.

## Indices verifies dans le MVP

- historique de mises a jour incrementales encore present dans le PDF ;
- difference visuelle localisee entre les deux dernieres revisions conservees ;
- contradictions de dates et logiciels de retouche declares dans les metadonnees ;
- petite image ou texte visible superpose a une image de scan pleine page ;
- annotations PDF visibles ;
- anomalies locales de recompression dans un JPEG original embarque (ELA).

Le champ s'appelle `incremental_updates_detected`, jamais `save_count` : une
reecriture complete peut supprimer tout l'historique precedent. Une signature, un
formulaire rempli, un tampon ou une annotation sont aussi des explications legitimes.

## Lecture du score

- `0-29`, `low` : aucun signal fort detecte ;
- `30-69`, `review` : revue manuelle necessaire ;
- `70-100`, `high` : plusieurs familles d'indices se corroborent.

Le score est une priorite de revue, pas une probabilite de fraude. Une seule famille
de signaux ne peut pas produire le niveau `high`.

## Architecture

```text
src/fraude_detector/
├── cli.py                 # interface utilisateur
├── pipeline.py            # orchestration uniquement
├── models.py              # contrat JSON type
├── pdf_revisions.py       # validation des revisions conservees
├── scoring.py             # aggregation prudente
├── rendering.py           # rendus et overlays de revue
└── detectors/
    ├── pdf_structure.py
    ├── revision_diff.py
    ├── page_composition.py
    └── raster_anomaly.py
```

Chaque detecteur retourne des observations independantes et ne prend jamais la
decision finale seul. Le document source est lu sans etre reecrit.

Par defaut, l'entree est limitee a 100 Mo, l'analyse visuelle aux 25 premieres
pages et chaque rendu a 25 millions de pixels. Ces valeurs sont centralisees dans
`AnalysisConfig`.

Le rendu utilise `pypdfium2` plutot que PyMuPDF pour eviter une dependance AGPL dans
un futur produit ferme. Les licences embarquees avec les roues PDFium devront tout de
meme etre conservees lors d'une distribution binaire.

## Developpement

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
```

Les tests generent leurs PDF a la volee ; aucun document d'assurance ni donnee
personnelle n'est versionne.

## Limites connues

- une fraude aplatie puis reecrite proprement peut ne laisser aucun historique ;
- ELA est sensible aux bords, au contenu et aux recompressions legitimes ;
- les images PNG ou sans compression JPEG ne sont pas analysees par ELA ;
- la valeur metier d'un montant, d'une date ou d'une identite n'est pas verifiee ;
- une calibration serieuse exige un corpus anonymise de vrais documents legitimes et
  modifies, avec mesure du taux de faux positifs.
