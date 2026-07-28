# Modeles tiers optionnels

Les poids ne sont pas inclus dans ce depot et ne sont jamais telecharges pendant une
analyse par defaut.

## Community Forensics

- code et methode : <https://github.com/JeongsooP/Community-Forensics>
- poids officiels 384 : <https://huggingface.co/OwensLab/commfor-model-384>
- poids officiels 224 : <https://huggingface.co/OwensLab/commfor-model-224>
- publication : *Community Forensics: Using Thousands of Generators to Train Fake
  Image Detectors*, CVPR 2025
- licence declaree par le code et les fiches de modeles amont : MIT

Frod epingle la revision Hugging Face de chaque variante. Pour un checkpoint local,
le rapport identifie la version par un prefixe de son SHA-256. La sortie sigmoid est
nommee `synthetic_score`; elle ne doit pas etre interpretee comme une probabilite de
fraude.

## TruFor

- code et methode : <https://github.com/grip-unina/TruFor>
- poids officiels : <https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip>
- publication : *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery
  Detection and Localization*, CVPR 2023
- revision du runtime integre : `ae54475df6f41a491d7615100feb19263dec13f7`
- licence : usage informatif et non lucratif uniquement

Le runtime minimal est conserve avec les avis de licence amont. Le checkpoint n'est
pas versionne : `start.sh` telecharge l'archive officielle, verifie son MD5, extrait
`trufor.pth.tar`, puis verifie son SHA-256. TruFor est execute uniquement dans le
laboratoire des images autonomes. Son score et ses cartes ne modifient pas le score
Frod.
