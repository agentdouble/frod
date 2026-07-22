# Fraude Detector

MVP local et explicable pour prioriser la revue de PDF d'assurance. Le pipeline ne
declare jamais qu'un document est frauduleux ou authentique : il produit un score de
revue, les indices techniques qui l'expliquent et des images localisant les zones a
controler.

## Demarrage

Prerequis : [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
./start.sh
```

`start.sh` analyse le PDF ou l'image indique dans `config.yaml`. Le chemin peut etre absolu ou
relatif au dossier qui contient le fichier de configuration :

```yaml
input_path: "tests/fixtures/assurance-fraude.pdf"
```

Les images PNG, JPEG, WebP, TIFF, GIF et BMP sont analysees directement, sans
conversion en PDF, afin de conserver leurs metadonnees et leur provenance C2PA.

La commande directe reste disponible pour choisir le PDF dans le terminal :

```bash
uv run frod chemin/vers/document.pdf --output output/analyse
```

`uv run fraude-detect ...` reste disponible comme nom long equivalent.

Pour un PDF chiffre, preferer une variable d'environnement afin de ne pas laisser le
mot de passe dans l'historique du shell :

```bash
FRAUDE_PDF_PASSWORD='mot-de-passe' uv run frod document.pdf
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
    ├── page-001-image-01-ela.png
    └── ai/
        └── page-001-image-01-provenance.json
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
- provenance C2PA du PDF et des images natives, avec validation separee de la
  confiance accordee au certificat ;
- noms explicites de generateurs IA dans les metadonnees EXIF/XMP ;
- analyse passive optionnelle des photos embarquees, avec controle de stabilite
  apres JPEG 95, JPEG 75 et redimensionnement.

L'absence de C2PA, d'EXIF ou de XMP n'ajoute aucun point. Une declaration C2PA
d'origine algorithmique indique comment un media a ete produit; elle ne dit pas si
son utilisation dans un dossier d'assurance est frauduleuse. L'analyse C2PA locale
ne charge ni manifeste distant ni reponse OCSP depuis un document non fiable.

## Images generees par IA

Le chemin par defaut analyse la provenance et les metadonnees, sans telecharger de
modele. Les images sont d'abord routees par role :

- `document` : image couvrant la majorite d'une page, presumee scan et exclue des
  modeles passifs par prudence ;
- `decorative` : logo, icone ou petit tampon, exclu ;
- `photo` : seule categorie eligible a l'analyse passive ;
- `unknown` : dimensions insuffisantes, donc abstention.

L'adaptateur Community Forensics est optionnel. Ses dependances lourdes ne sont pas
installees par defaut et ses poids ne sont jamais telecharges implicitement :

```bash
uv sync --extra ai

# Avec un checkpoint deja present
uv run frod document.pdf \
  --ai-model community-forensics \
  --ai-model-path /chemin/model.safetensors

# Ou avec telechargement explicite des poids officiels epingles
uv run frod document.pdf \
  --ai-model community-forensics \
  --ai-model-download
```

Un seul modele peut produire `AI_PIXEL_TRACE_SINGLE_MODEL`, diagnostic a zero point.
Un signal score `AI_PIXEL_TRACE_CONSENSUS` exige au moins deux familles de methodes
distinctes, toutes deux elevees et stables. Les adapters supplementaires s'injectent
via `AnalysisPipeline(ai_image_adapters=...)`; le modele n'est donc jamais couple au
coeur du pipeline. Les diagnostics a zero point restent dans `report.json`, mais ne
sont ni comptes comme signaux scores ni dessines comme zones a controler.

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
├── image_assets.py        # inventaire et decodage des images PDF
├── image_roles.py         # routage photo/document/decoratif
├── image_provenance.py    # EXIF, XMP et C2PA
├── ai_images.py           # contrat des modeles et test de stabilite
├── community_forensics.py # adaptateur optionnel, chargement explicite
├── pdf_revisions.py       # validation des revisions conservees
├── scoring.py             # aggregation prudente
├── rendering.py           # rendus et overlays de revue
└── detectors/
    ├── pdf_structure.py
    ├── revision_diff.py
    ├── page_composition.py
    ├── raster_anomaly.py
    ├── image_provenance.py
    └── ai_generated_image.py
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

Deux PDF d'assurance entierement synthetiques sont versionnes comme oracles
d'integration :

- `tests/fixtures/assurance-sans-fraude.pdf` : document intact attendu en `low` ;
- `tests/fixtures/assurance-fraude.pdf` : meme document avec le montant remplace dans
  une revision incrementale et un tampon image ajoute, attendu en `high` avec des
  zones localisees.

Ils ne contiennent aucune donnee personnelle reelle. Le mot `fraude` designe ici une
alteration volontaire connue, pas une qualification juridique. Pour les regenerer :

```bash
uv run python tests/fixtures/generate_fixtures.py
```

Pour lancer manuellement les deux analyses :

```bash
uv run frod tests/fixtures/assurance-sans-fraude.pdf -o tmp/fixture-clean
uv run frod tests/fixtures/assurance-fraude.pdf -o tmp/fixture-fraude
```

## Limites connues

- une fraude aplatie puis reecrite proprement peut ne laisser aucun historique ;
- ELA est sensible aux bords, au contenu et aux recompressions legitimes ;
- les images PNG ou sans compression JPEG ne sont pas analysees par ELA ;
- les detecteurs passifs d'images IA se degradent sur scans, texte dense,
  recompressions et generateurs absents de leur corpus d'entrainement ;
- le routage actuel est geometrique, sans OCR ni classifieur de contenu : il peut
  exclure une vraie photo pleine page ou accepter une facture partielle comme photo ;
- les masques alpha PDF separes ne sont pas recomposes dans le decodage natif ;
- l'inventaire IA est borne aux 100 premieres images et le rapport passe en `partial`
  si cette limite est atteinte ;
- C2PA et les metadonnees peuvent etre supprimes lors de l'integration dans un PDF ;
- une metadonnee non signee peut etre ajoutee ou falsifiee ;
- la valeur metier d'un montant, d'une date ou d'une identite n'est pas verifiee ;
- une calibration serieuse exige un corpus anonymise de vrais documents legitimes et
  modifies, avec mesure du taux de faux positifs.
