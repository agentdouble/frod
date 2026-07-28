# Fraude Detector

MVP local et explicable pour prioriser la revue de PDF d'assurance. Le pipeline ne
declare jamais qu'un document est frauduleux ou authentique : il produit un score de
revue, les indices techniques qui l'expliquent et des images localisant les zones a
controler.

## Demarrage

Prerequis : [`uv`](https://docs.astral.sh/uv/).

```bash
./start.sh
```

Le script installe les dependances, telecharge et verifie les modeles necessaires au
premier demarrage, puis lance l'interface Streamlit. L'adresse locale a ouvrir est
affichee dans le terminal.

L'interface accepte les PDF et les images PNG, JPEG, WebP, TIFF, GIF ou BMP. Toutes
les analyses applicables sont lancees avec le meme bouton.

La ligne de commande reste disponible pour produire directement un rapport et ses
artefacts :

```bash
uv run frod chemin/vers/document.pdf --output output/analyse
```

La CLI utilise GAPL par defaut et produit le meme rapport que l'interface pour les
memes options d'analyse. `--without-gapl` permet de le desactiver explicitement.

Les modeles, fichiers importes et sorties d'analyse restent locaux et ne sont pas
suivis par Git.

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
- analyse passive GAPL des images et des photos embarquees eligibles, avec
  aggregation multi-fenetre et controle de stabilite.
- localisation experimentale de retouches sur les images autonomes avec TruFor,
  sans contribution au score Frod.

L'absence de C2PA, d'EXIF ou de XMP n'ajoute aucun point. Une declaration C2PA
d'origine algorithmique indique comment un media a ete produit; elle ne dit pas si
son utilisation dans un dossier d'assurance est frauduleuse. L'analyse C2PA locale
ne charge ni manifeste distant ni reponse OCSP depuis un document non fiable.

## Images generees par IA

Frod combine la provenance C2PA, les metadonnees EXIF/XMP et le modele local GAPL.
Pour une image autonome, GAPL analyse plusieurs fenetres completes de 224 x 224
pixels, sans inventer de pixels aux bords. Pour un PDF, seules les photographies
embarquees et eligibles sont transmises au modele.

Les scores des fenetres sont agreges en un indice global robuste aux modifications
locales. Cet indice commence a ajouter quelques points au-dessus de 50 % et ajoute
30 points Frod a partir de 90 %, ce qui declenche une revue manuelle. Il s'agit d'un
indice statistique appris, pas d'une probabilite de fraude ni d'une preuve qu'une
image a ete generee par IA.

La cartographie des fenetres et les controles de stabilite restent accessibles dans
le dossier technique de l'interface.

## Retouches locales

Le laboratoire des images autonomes execute TruFor apres l'analyse Frod. Le modele
compare les informations visuelles avec une empreinte de bruit Noiseprint++ et
produit un indice global, une carte d'anomalie et une carte de fiabilite. Frod
affiche une carte dans laquelle les anomalies sont ponderees par cette fiabilite.

Ce controle vise les retouches locales et les montages. Il est complementaire a
GAPL, qui recherche des caracteristiques apprises sur les images generees par IA.
TruFor n'est jamais execute sur un PDF, ne modifie aucun `Finding` et n'ajoute aucun
point au score. Ses seuils restent experimentaux tant qu'ils ne sont pas calibres
sur un corpus representatif de documents d'assurance.

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
├── gapl.py                 # adaptateur GAPL local
├── gapl_windows.py         # grille, indice global et contribution au score
├── trufor.py               # execution isolee et artefacts TruFor
├── trufor_worker.py        # worker court pour liberer la memoire du modele
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

Trois PDF d'assurance entierement synthetiques sont versionnes comme oracles
d'integration :

- `tests/fixtures/assurance-sans-fraude.pdf` : document intact attendu en `low` ;
- `tests/fixtures/assurance-ajout-legitime.pdf` : meme document avec un tampon de
  reception ajoute sans masquer le contenu ;
- `tests/fixtures/assurance-fraude.pdf` : meme document avec le montant remplace dans
  une revision incrementale et un tampon image ajoute, attendu en `high` avec des
  zones localisees.

Ils ne contiennent aucune donnee personnelle reelle. Le mot `fraude` designe ici une
alteration volontaire connue, pas une qualification juridique. Pour les regenerer :

```bash
uv run python tests/fixtures/generate_fixtures.py
```

Pour lancer manuellement les analyses :

```bash
uv run frod tests/fixtures/assurance-sans-fraude.pdf -o output/fixture-clean
uv run frod tests/fixtures/assurance-ajout-legitime.pdf -o output/fixture-legitimate
uv run frod tests/fixtures/assurance-fraude.pdf -o output/fixture-fraude
```

## Limites connues

- une fraude aplatie puis reecrite proprement peut ne laisser aucun historique ;
- ELA est sensible aux bords, au contenu et aux recompressions legitimes ;
- les images PNG ou sans compression JPEG ne sont pas analysees par ELA ;
- les detecteurs passifs d'images IA se degradent sur scans, texte dense,
  recompressions et generateurs absents de leur corpus d'entrainement ;
- TruFor est non calibre sur les documents d'assurance et peut manquer une retouche
  IA recente, une petite zone ou une image fortement recomprimee ;
- la licence amont de TruFor limite son utilisation aux finalites informatives et
  non lucratives ;
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
